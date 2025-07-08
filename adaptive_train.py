#!/usr/bin/env python3
# adaptive_train.py

import argparse
import os
import glob
import random
from pathlib import Path
import json # For potential future structured logging
from datetime import datetime, timedelta # For log_days filtering

# Rich imports
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    # Fallback print if rich is not available
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()
    console.print("[yellow]WARNING: Rich library not found. Output will be basic. Install with `pip install rich`[/yellow]")


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, # Or a custom one if needed
    BitsAndBytesConfig # For potential 4-bit/8-bit training
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
# from datasets import Dataset # If we want to use Hugging Face Datasets object

# Default paths and configurations
DEFAULT_MODEL_PATH = "./merged_model"
DEFAULT_LOG_DIR = "./interaction_logs"
DEFAULT_OUTPUT_DIR = "./merged_model_adapters" # Save adapters here

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive fine-tuning script for the merged AGI model using JSONL interaction logs.")
    parser.add_argument(
        "--analyze_jsonl_logs",
        nargs='?',
        const=-1, # Default value if flag is present without a number (e.g. analyze all, or a default like 5)
        type=int,
        metavar='N',
        help="Analyze JSONL logs and print statistics. Optionally print N random formatted training examples. If N is not given, prints stats only or a default number of examples (e.g., 5). If N is 0, prints stats only. If N is -1 (flag only), prints stats and a default number of examples."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the base model to be fine-tuned (./merged_model).",
    )
    parser.add_argument(
        "--jsonl_log_path", # Changed from --log_dir
        type=str,
        default=str(Path(__file__).parent / ".agi_terminal_cache" / "interaction_logs.jsonl"), # Assumes train script is in project root
        help="Path to the JSONL file containing interaction logs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the PEFT adapters.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate." # Common for LoRA
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA attention dimension (r)."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length for training."
    )
    parser.add_argument(
        "--use_qlora", action="store_true", help="Enable QLoRA (4-bit quantization) for training."
    )
    # log_count and log_days were removed as JSONL processing is different. Filtering can be added later.
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj", # Common Llama-like modules
        help="Comma-separated list of module names to target with LoRA (e.g., 'q_proj,v_proj')."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch", # Changed from adamw_hf, paged_adamw_8bit can be chosen for QLoRA
        help="Optimizer to use (e.g., 'adamw_hf', 'adamw_torch', 'paged_adamw_8bit', 'adafactor')."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type (e.g., 'linear', 'cosine', 'constant', 'constant_with_warmup')."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0, # Default to 0 if not specified, common for some schedulers
        help="Number of warmup steps for the learning rate scheduler."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to apply (if not using an optimizer that handles it like AdamW)."
    )
    return parser.parse_args()

# --- Helper functions for formatting JSONL data for prompt ---
def format_context_for_prompt(context_dict: dict) -> str:
    """
    Formats the context dictionary from a JSONL log entry into a string for the training prompt.
    Includes CWD, Project Root, Git info, top file types, and names of key file snippets.
    """
    if not context_dict:
        return "CONTEXT: None" # Should ideally not happen if context is always logged

    parts = []
    if "cwd" in context_dict: # Check if key exists
        parts.append(f"CWD: {context_dict['cwd']}")
    if "project_root" in context_dict: # Check if key exists
        parts.append(f"ProjectRoot: {context_dict['project_root']}")

    git_info = context_dict.get("git_info", {})
    if git_info.get("in_git_repo"):
        branch = git_info.get('branch', 'N/A')
        modified_files = git_info.get('modified_files', 0)
        status_str = f"{modified_files} modified files" if modified_files > 0 else "Clean"
        parts.append(f"Git: Branch='{branch}', Status='{status_str}'")
    else:
        parts.append("Git: Not a git repository or no info")

    file_counts = context_dict.get("file_type_counts", {})
    if file_counts:
        # Sort by count descending, take top N (e.g., 3)
        top_files = sorted(file_counts.items(), key=lambda item: item[1], reverse=True)[:3]
        files_str = ", ".join([f"{lang}:{count}" for lang, count in top_files])
        parts.append(f"FileTypes: {files_str}{'...' if len(file_counts) > 3 else ''}")

    key_snippets = context_dict.get("key_file_snippets", [])
    if key_snippets:
        # Extract filenames from snippet headers (assuming format "--- Snippet from FILENAME ...")
        filenames = []
        for snippet_header in key_snippets:
            match = re.search(r"--- Snippet from ([\w\._-]+)", snippet_header)
            if match:
                filenames.append(match.group(1))
        if filenames:
            parts.append(f"KeyFileSnippetsProvided: [{', '.join(filenames)}]")

    if not parts:
        return "CONTEXT: (No specific context details available)" # Fallback if all parts are empty
    return "SYSTEM_CONTEXT:\n" + "\n".join(f"  {p}" for p in parts)

def format_tool_interactions_for_prompt(tool_interactions: list, max_outcome_chars: int = 200) -> str:
    """
    Formats the list of tool interactions from a JSONL log entry into a structured
    string dialogue for the training prompt.
    Includes details of AGI's tool request, system's response (confirmation, outcome),
    and any AGI response after the tool use. Outcomes and secondary AGI responses
    are truncated to `max_outcome_chars` and `max_outcome_chars*2` respectively.
    """
    if not tool_interactions:
        return "" # Return empty string if no tool interactions

    turn_dialogue = ["\nINTERMEDIATE_STEPS:"] # Start with a clear section header
    for i, tool_call in enumerate(tool_interactions):
        turn_dialogue.append(f"  --- TOOL CALL {i+1} ---")
        action_type = tool_call.get('action_type', 'unknown_action')
        action_details = tool_call.get('action_details', {})
        reasoning = tool_call.get('reasoning', 'No reasoning provided.')
        user_confirmation = tool_call.get('user_confirmation', 'N/A')
        tool_outcome_summary = tool_call.get('tool_outcome_summary', 'No outcome summary.')
        agi_secondary_response = tool_call.get('agi_secondary_raw_response', None)

        turn_dialogue.append(f"    AGI_TOOL_REQUEST:")
        turn_dialogue.append(f"      Action: {action_type}")
        # Format action_details
        details_str_parts = []
        if action_type == "run_shell":
            details_str_parts.append(f"command: '{action_details.get('command')}'")
            if 'args' in action_details: details_str_parts.append(f"args: {action_details['args']}")
        elif action_type == "read_file":
            details_str_parts.append(f"filepath: '{action_details.get('filepath')}'")
            if 'max_lines' in action_details: details_str_parts.append(f"max_lines: {action_details['max_lines']}")
        elif action_type == "write_file":
            details_str_parts.append(f"filepath: '{action_details.get('filepath')}'")
            details_str_parts.append(f"content_length: {action_details.get('content_length', 'N/A')}")
        elif action_type == "web_search":
            details_str_parts.append(f"query: '{action_details.get('query')}'")
        else: # Generic fallback for unknown action_details
            details_str_parts.append(f"details: {json.dumps(action_details)}")
        turn_dialogue.append(f"      Details: {', '.join(details_str_parts)}")
        turn_dialogue.append(f"      Reasoning: {reasoning}")

        turn_dialogue.append(f"    SYSTEM_TOOL_RESPONSE:")
        turn_dialogue.append(f"      UserConfirmation: {user_confirmation}")
        # Truncate long outcomes for the prompt
        outcome_display = tool_outcome_summary
        if len(outcome_display) > max_outcome_chars:
            outcome_display = outcome_display[:max_outcome_chars] + f"... (outcome truncated, full length {len(tool_outcome_summary)})"
        turn_dialogue.append(f"      Outcome: {outcome_display}")

        if agi_secondary_response:
            # Truncate secondary AGI response if it's very long (e.g. AGI explains a large file)
            secondary_resp_display = agi_secondary_response
            if len(secondary_resp_display) > max_outcome_chars * 2: # Allow a bit more for AGI's own words
                 secondary_resp_display = secondary_resp_display[:max_outcome_chars*2] + f"... (AGI response truncated, full length {len(agi_secondary_response)})"
            turn_dialogue.append(f"    AGI_RESPONSE_AFTER_TOOL: {secondary_resp_display}")
        turn_dialogue.append(f"  --- END TOOL CALL {i+1} ---")

    return "\n".join(turn_dialogue)

# --- Log Parsing ---
def load_raw_interaction_logs(jsonl_log_path: str) -> list[dict]:
    """
    Loads raw interaction logs from a specified JSONL file.

    Each line in the JSONL file is expected to be a valid JSON object representing
    a single interaction turn. Malformed JSON lines are skipped with a warning.

    Args:
        jsonl_log_path: The path to the JSONL file containing interaction logs.

    Returns:
        A list of dictionaries, where each dictionary is a parsed JSON object
        from a line in the log file. Returns an empty list if the file is not found,
        empty, or contains no valid JSON lines.
    """
    raw_interactions = []
    log_file = Path(jsonl_log_path)
    if not log_file.exists():
        console.print(f"[yellow]JSONL log file not found at {jsonl_log_path}. Nothing to train on.[/yellow]")
        return []

    console.print(f"[info]Processing JSONL log file: {jsonl_log_path}...[/info]")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    raw_interactions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]Skipping malformed JSON line {line_num + 1} in {jsonl_log_path}: {e}[/yellow]")
                    continue
    except Exception as e:
        console.print(f"[error]Error reading or parsing JSONL log file {jsonl_log_path}: {e}[/error]")
        return []

    if not raw_interactions:
        console.print(f"[yellow]No valid interactions found in {jsonl_log_path}.[/yellow]")
        return []

    # Use Rich Progress for formatting and tokenizing
    if RICH_AVAILABLE:
        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console
        )
        task_id_format = progress_bar.add_task("Formatting logs", total=len(raw_interactions))
        progress_bar.start()

    for entry in raw_interactions:
        if RICH_AVAILABLE:
            progress_bar.update(task_id_format, advance=1)

        user_query = entry.get("user_query", "")
        agi_final_response = entry.get("agi_final_response_to_user", "")
        context_dict = entry.get("context_at_query_time", {})
        tool_interactions_list = entry.get("tool_interactions", [])

        if not user_query or not agi_final_response: # Skip entries without essential I/O
            continue

        # Construct the training text
        # This aims for the model to learn to complete the "ASSISTANT:" part given context, user query, and tool use summary
        context_str = format_context_for_prompt(context_dict)
        tool_summary_str = format_tool_interactions_for_prompt(tool_interactions_list)

        # Training text format:
        # CONTEXT:
        # - CWD: /path/to/project
        # - Git: Branch 'main', Modified files: 0
        # USER: How do I do X?
        # TOOL_INTERACTIONS: (optional)
        #   Tool Call 1: read_file(filepath='foo.py') -> Outcome: Content of foo.py...
        # ASSISTANT: You can do X by using Y from foo.py... <eos_token>

        # Need to ensure the prompt part (everything before ASSISTANT:) is distinct from the completion part.
        # A common way is to have the model generate everything after "ASSISTANT: ".

        prompt_part = f"{context_str}\nUSER: {user_query}{tool_summary_str}\nASSISTANT: "
        completion_part = f"{agi_final_response}{tokenizer.eos_token}"

        text = prompt_part + completion_part
        formatted_texts.append(text)

    if RICH_AVAILABLE:
        progress_bar.stop() # Stop formatting progress bar

    return raw_interactions # Return list of dicts

def format_interaction_for_training(interaction_entry: dict, tokenizer_eos_token: str) -> Optional[str]:
    """
    Formats a single raw interaction dictionary (from JSONL) into a complete
    string suitable for language model training.

    The format typically includes system context, user query, a summary of any
    intermediate tool interactions, and finally the AGI's response, followed by
    an EOS token. This structure helps the model learn instruction-following
    and tool use patterns within a conversational context.

    Args:
        interaction_entry: A dictionary representing one interaction turn, parsed
                           from the JSONL log.
        tokenizer_eos_token: The End-Of-Sequence token string from the tokenizer,
                             to be appended to the final formatted string.

    Returns:
        A formatted string ready for tokenization, or None if essential fields
        (user_query, agi_final_response_to_user) are missing from the entry.
    """
    user_query = interaction_entry.get("user_query", "")
    agi_final_response = interaction_entry.get("agi_final_response_to_user", "")
    context_dict = interaction_entry.get("context_at_query_time", {})
    tool_interactions_list = interaction_entry.get("tool_interactions", [])

    if not user_query or not agi_final_response: # Skip entries without essential I/O
        return None

    context_str = format_context_for_prompt(context_dict)
    tool_summary_str = format_tool_interactions_for_prompt(tool_interactions_list)

    # Using the USER: ... ASSISTANT: ... format
    # This structure helps the model learn instruction following.
    # The AGI needs to learn to generate the text after "ASSISTANT: "
    prompt_part = f"{context_str}\nUSER: {user_query}{tool_summary_str}\nASSISTANT: "
    completion_part = f"{agi_final_response}{tokenizer_eos_token if tokenizer_eos_token else ''}"

    return prompt_part + completion_part

def prepare_training_dataset(jsonl_log_path: str, tokenizer: AutoTokenizer, max_seq_length: int) -> list[dict]:
    """
    Loads raw interaction logs from the specified JSONL file, formats each valid
    interaction into a training string, and then tokenizes these strings.

    This function orchestrates the data preparation pipeline for training,
    including handling of empty or small datasets and progress reporting.

    Args:
        jsonl_log_path: Path to the JSONL file containing interaction logs.
        tokenizer: The Hugging Face AutoTokenizer instance to use for tokenization
                   and for obtaining the EOS token.
        max_seq_length: The maximum sequence length for tokenized inputs.

    Returns:
        A list of dictionaries, where each dictionary represents a tokenized
        training instance (containing 'input_ids', 'attention_mask', 'labels').
        Returns an empty list if no suitable data can be prepared.
    """
    raw_interactions = load_raw_interaction_logs(jsonl_log_path)
    if not raw_interactions:
        return []

    formatted_texts = []
    console.print(f"[info]Formatting {len(raw_interactions)} raw interactions for training...[/info]")
    if RICH_AVAILABLE:
        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(), TimeElapsedColumn(), console=console
        )
        format_task = progress_bar.add_task("Formatting data", total=len(raw_interactions))
        progress_bar.start()

    for entry in raw_interactions:
        if RICH_AVAILABLE: progress_bar.update(format_task, advance=1)
        formatted_text = format_interaction_for_training(entry, tokenizer.eos_token)
        if formatted_text:
            formatted_texts.append(formatted_text)

    if RICH_AVAILABLE: progress_bar.stop()

    if not formatted_texts:
        console.print("[yellow]No valid interactions formatted for training after processing.[/yellow]")
        return []

    min_dataset_size = 10
    if len(formatted_texts) < min_dataset_size:
        console.print(f"[warning]Warning: The number of formatted interactions ({len(formatted_texts)}) is very small (less than {min_dataset_size}). Fine-tuning may not be effective or could lead to overfitting.[/warning]")
        if input(f"Continue with {len(formatted_texts)} interactions? (yes/NO): ").lower() != "yes":
            console.print("[info]Training aborted by user due to small dataset size.[/info]")
            return []
    console.print(f"[success]Successfully formatted {len(formatted_texts)} interactions.[/success]")

    # Tokenize the formatted texts
    tokenized_dataset = []
    console.print(f"[info]Tokenizing {len(formatted_texts)} formatted interactions...[/info]")
    if RICH_AVAILABLE:
        tokenize_task = progress_bar.add_task("Tokenizing data", total=len(formatted_texts))
        progress_bar.start()

    for text in formatted_texts:
        if RICH_AVAILABLE: progress_bar.update(tokenize_task, advance=1)
        tokenized_input = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length", # Pad to max_seq_length for simplicity with DataCollatorForLanguageModeling
                                # If using a dataset object, padding can be handled by the collator more dynamically.
            # return_tensors="pt" # Trainer handles this if data is in list of dicts
        )
        # Trainer expects a list of dicts, where each dict has 'input_ids', 'attention_mask', (and 'labels' if provided)
        # For language modeling, input_ids are typically used as labels.
        tokenized_dataset.append({
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
            "labels": tokenized_input["input_ids"].copy() # Common practice for Causal LM fine-tuning
        })

    if RICH_AVAILABLE:
        progress_bar.stop()
    console.print(f"[success]Tokenized {len(tokenized_dataset)} interactions.[/success]")
    return tokenized_dataset


# --- Main Training Logic ---
def main():
    args = parse_arguments()

    if args.analyze_jsonl_logs is not None:
        console.print(f"[bold cyan]--- JSONL Log Analysis Mode ---[/bold cyan]")
        console.print(f"[info]Analyzing log file: {args.jsonl_log_path}[/info]")

        raw_logs = load_raw_interaction_logs(args.jsonl_log_path)
        if not raw_logs:
            console.print("[error]No logs loaded for analysis. Exiting.[/error]")
            return

        num_examples_to_show = args.analyze_jsonl_logs
        if args.analyze_jsonl_logs == -1: # Default if only flag is present
            num_examples_to_show = 5

        example_eos_token = "<|eos|>" # Default if tokenizer fails
        if num_examples_to_show > 0:
            try:
                # Try to load tokenizer just for eos_token, lighter than full model
                tokenizer_for_examples = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                if tokenizer_for_examples.eos_token:
                    example_eos_token = tokenizer_for_examples.eos_token
                console.print(f"[info]Using EOS token '{example_eos_token}' for example formatting.[/info]")
            except Exception as e:
                console.print(f"[warning]Could not load tokenizer from '{args.model_path}' for example formatting: {e}. Using default EOS token.[/warning]")

        analyze_and_print_stats(raw_logs, num_examples_to_show, example_eos_token)
        console.print("[info]Log analysis complete.[/info]")
        return # Exit after analysis, do not proceed to training

    console.print("[bold blue]Starting Adaptive Training Script...[/bold blue]")
    console.print(f"[info]Configuration: {args}[/info]")

    # --- 1. Load Tokenizer and Model ---
    console.print(f"[info]Loading base model from: {args.model_path}[/info]")

    # Quantization config for QLoRA
    bnb_config = None
    if args.use_qlora:
        console.print("[info]QLoRA enabled. Using 4-bit quantization.[/info]")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        # Set pad token if not present. Common practice: use eos_token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            console.print(f"[info]Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})[/info]")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config, # None if not args.use_qlora
            device_map={"":0}, # Load model on GPU 0. Adjust if multi-GPU or CPU.
                               # For QLoRA, device_map="auto" or specific GPU is common.
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16 # if not using bnb_config and want bfloat16
        )
        console.print("[success]Tokenizer and model loaded successfully.[/success]")

        if args.use_qlora:
            console.print("[info]Preparing model for k-bit training (QLoRA)...[/info]")
            model = prepare_model_for_kbit_training(model)

    except Exception as e:
        console.print(f"[error]Error loading model or tokenizer: {e}[/error]")
        console.print("[warning]Ensure './merged_model' exists and is a valid Hugging Face model directory.[/warning]")
        console.print("[info]This script should be run AFTER the user has downloaded and merged models via setup_agi_terminal.py.[/info]")
        return

    # --- 2. Configure PEFT (LoRA) ---
    # Parse lora_target_modules string into a list
    if args.lora_target_modules:
        parsed_target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
    else: # Should not happen if default is set, but as a fallback
        parsed_target_modules = ["q_proj", "v_proj"] # Minimal default

    console.print(f"[info]Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}[/info]")
    console.print(f"[info]Targeting LoRA modules: {parsed_target_modules}[/info]")
    if not parsed_target_modules:
        console.print("[warning]No LoRA target modules specified. LoRA may not be effective.[/warning]")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=parsed_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none", # Typically 'none' for LoRA
        task_type=TaskType.CAUSAL_LM,
    )

    try:
        model = get_peft_model(model, lora_config)
        console.print("[success]PEFT model (LoRA) configured successfully.[/success]")
        # model.print_trainable_parameters() # This can be verbose, make it optional or log to file
        trainable_params, all_param = model.get_nb_trainable_parameters()
        console.print(f"[info]Trainable LoRA parameters: {trainable_params} || All parameters: {all_param} || Trainable %: {100 * trainable_params / all_param:.2f}[/info]")
    except Exception as e:
        console.print(f"[error]Error configuring PEFT model: {e}[/error]")
        console.print("[warning]This might be due to incorrect target_modules for the loaded model architecture.[/warning]")
        console.print("[info]You can try to inspect `model.named_modules()` to find suitable Linear layers for LoRA.[/info]")
        return

    # --- 3. Load and Prepare Data ---
    console.print("[info]Loading and preparing dataset from interaction logs...[/info]")
    train_dataset = load_interaction_logs(
        args.log_dir,
        tokenizer,
        args.max_seq_length,
        args.log_count,
        args.log_days
    )

    if not train_dataset: # load_interaction_logs now handles printing "no data" messages
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 4. Training Arguments ---
    console.print(f"[info]Preparing training arguments. Output will be saved to: {args.output_dir}[/info]")
    os.makedirs(args.output_dir, exist_ok=True)

    # Define a custom progress bar callback for Rich if RICH_AVAILABLE
    class RichProgressCallback(transformers.TrainerCallback):
        def __init__(self):
            super().__init__()
            self.progress = None
            self.train_task_id = None

        def on_train_begin(self, args, state, control, **kwargs):
            if RICH_AVAILABLE and state.is_world_process_zero:
                self.progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                    TextColumn("Steps: {task.completed}/{task.total}"),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True # Remove progress bar when done
                )
                self.progress.start()
                self.train_task_id = self.progress.add_task("Training", total=state.max_steps)

        def on_step_end(self, args, state, control, **kwargs):
            if RICH_AVAILABLE and state.is_world_process_zero and self.progress and self.train_task_id is not None:
                self.progress.update(self.train_task_id, advance=1)

        def on_train_end(self, args, state, control, **kwargs):
            if RICH_AVAILABLE and state.is_world_process_zero and self.progress:
                self.progress.stop()
                self.progress = None
                self.train_task_id = None

    callbacks = [RichProgressCallback()] if RICH_AVAILABLE else []

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10, # Log every N steps
        save_strategy="epoch", # Save checkpoint every epoch
        fp16=not args.use_qlora,  # Use fp16 if not using QLoRA (which uses its own dtype)
        bf16=args.use_qlora and torch.cuda.is_bf16_supported(), # QLoRA often paired with bf16 compute_dtype, check support
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        # report_to="tensorboard", # Or "wandb", "none"
        disable_tqdm=RICH_AVAILABLE, # Disable default tqdm if Rich progress is used
        logging_dir=f"{args.output_dir}/logs", # Store training logs
    )
    if args.use_qlora and not torch.cuda.is_bf16_supported() and args.optimizer == "paged_adamw_8bit":
        console.print("[warning]BF16 is not supported on this GPU, but QLoRA setup might ideally use it. Training will proceed with default dtype for QLoRA compute (often float16). Consider if 'paged_adamw_8bit' is still appropriate or if another optimizer like 'adamw_torch' should be used.[/warning]")

    # --- 5. Initialize Trainer ---
    console.print("[info]Initializing Trainer...[/info]")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=None, # Add validation set if available
        data_collator=data_collator,
        callbacks=callbacks
    )

    # --- 6. Train ---
    console.print("[bold green]Starting training...[/bold green]")
    console.print("[warning]NOTE: Actual model training is computationally intensive and requires a suitable GPU environment.[/warning]")
    console.print("[info]This script provides the structure. Ensure your environment is set up for Hugging Face model training.[/info]")

    try:
        train_result = trainer.train()
        console.print(f"[success]Training completed. Metrics: {train_result.metrics}[/success]")
    except Exception as e:
        console.print(f"[error]An error occurred during trainer.train(): {e}[/error]")
        console.print("[warning]This could be due to resource limitations (CUDA OOM), configuration issues, or data problems.[/warning]")
        console.print("[info]Consider reducing batch size, sequence length, or using QLoRA if not already.[/info]")
        # No return here, proceed to attempt saving if any progress was made or if it's just a dry run.

    # --- 7. Save Model (PEFT Adapters) ---
    console.print(f"[info]Saving PEFT adapters to {args.output_dir}...[/info]")
    try:
        trainer.save_model(args.output_dir) # This saves the PEFT adapters correctly
        # model.save_pretrained(args.output_dir) # Trainer.save_model is preferred for adapters
        tokenizer.save_pretrained(args.output_dir) # Save tokenizer for completeness
        console.print(f"[success]Adapters and tokenizer saved successfully to {args.output_dir}.[/success]")
        console.print("[info]To load the fine-tuned model later:[/info]")
        console.print("[info]1. Load the base model: `model = AutoModelForCausalLM.from_pretrained(base_model_path)`[/info]")
        console.print("[info]2. Load PEFT adapters: `model = PeftModel.from_pretrained(model, adapter_path)`[/info]")
    except Exception as e:
        console.print(f"[error]Error saving model/adapters: {e}[/error]")

    console.print("[bold blue]Adaptive training script finished.[/bold blue]")

if __name__ == "__main__":
    # This is a placeholder for how the script would be run.
    # Actual execution with training will be done by the user in their environment.
    # For development purposes, one might add a --dry_run flag to skip actual training.

def analyze_and_print_stats(raw_interactions: list[dict], num_examples_to_print: int, tokenizer_eos_for_examples: str):
    """
    Calculates and prints various statistics about the loaded JSONL interaction logs.
    Optionally, it also prints a sample of formatted training prompt examples.

    Statistics include:
    - Total number of interaction turns.
    - Number and percentage of turns involving tool use.
    - Breakdown of tool usage by action type.
    - Summary of tool outcomes (success, error, cancelled) per tool type.
    - Average lengths of user queries and AGI final responses.

    Args:
        raw_interactions: A list of dictionaries, each representing a parsed JSONL entry.
        num_examples_to_print: The number of random formatted training examples to print.
                               If 0, no examples are printed.
        tokenizer_eos_for_examples: The EOS token string to use when formatting examples.
    """
    if not raw_interactions:
        console.print("[yellow]No interactions to analyze.[/yellow]")
        return

    total_turns = len(raw_interactions)
    turns_with_tools = 0
    tool_type_counts = {} # e.g., {"run_shell": 10, "read_file": 5}
    tool_outcomes = {} # e.g., {"run_shell": {"success": 8, "error": 1, "cancelled": 1}}

    total_user_query_len = 0
    total_agi_response_len = 0

    for entry in raw_interactions:
        total_user_query_len += len(entry.get("user_query", ""))
        total_agi_response_len += len(entry.get("agi_final_response_to_user", ""))

        tool_interactions_list = entry.get("tool_interactions", [])
        if tool_interactions_list:
            turns_with_tools += 1
            for tool_call in tool_interactions_list:
                action = tool_call.get("action_type", "unknown_action")
                tool_type_counts[action] = tool_type_counts.get(action, 0) + 1

                if action not in tool_outcomes:
                    tool_outcomes[action] = {"success": 0, "error": 0, "cancelled": 0, "other": 0}

                outcome_summary = tool_call.get("tool_outcome_summary", "").lower()
                user_confirmation = tool_call.get("user_confirmation", "").lower()

                if "error:" in outcome_summary or "failed" in outcome_summary or "malformed" in outcome_summary or "staticanalysisreject" in outcome_summary:
                    tool_outcomes[action]["error"] += 1
                elif user_confirmation == "cancelled":
                    tool_outcomes[action]["cancelled"] += 1
                elif "success" in outcome_summary or "processed" in outcome_summary or "executed" in outcome_summary or "identical" in outcome_summary or "up-to-date" in outcome_summary or "no output" in outcome_summary or "nothing to commit" in outcome_summary:
                     tool_outcomes[action]["success"] += 1
                else: # Catch-all for outcomes not clearly error/cancelled/success based on simple string checks
                    tool_outcomes[action]["other"] += 1


    console.print(f"\n[bold underline]Interaction Log Analysis[/bold underline]")
    console.print(f"Total Interaction Turns: {total_turns}")
    console.print(f"Turns with Tool Use: {turns_with_tools} ({turns_with_tools/total_turns:.1%} of total if total_turns > 0 else 0.0)")

    avg_user_query_len = total_user_query_len / total_turns if total_turns > 0 else 0
    avg_agi_response_len = total_agi_response_len / total_turns if total_turns > 0 else 0
    console.print(f"Average User Query Length: {avg_user_query_len:.0f} chars")
    console.print(f"Average AGI Final Response Length: {avg_agi_response_len:.0f} chars")

    if tool_type_counts:
        console.print("\n[bold]Tool Usage Breakdown:[/bold]")
        tool_table = Table(title="Tool Type Counts")
        tool_table.add_column("Tool Action", style="cyan")
        tool_table.add_column("Count", style="magenta", justify="right")
        for tool, count in sorted(tool_type_counts.items()):
            tool_table.add_row(tool, str(count))
        console.print(tool_table)

        console.print("\n[bold]Tool Outcome Summary:[/bold]")
        outcome_table = Table(title="Tool Outcome Details")
        outcome_table.add_column("Tool Action", style="cyan")
        outcome_table.add_column("Success", style="green", justify="right")
        outcome_table.add_column("Error/Reject", style="red", justify="right")
        outcome_table.add_column("Cancelled", style="yellow", justify="right")
        outcome_table.add_column("Other/Unknown", style="dim", justify="right")
        for tool, outcomes_map in sorted(tool_outcomes.items()):
            outcome_table.add_row(
                tool,
                str(outcomes_map.get("success",0)),
                str(outcomes_map.get("error",0)),
                str(outcomes_map.get("cancelled",0)),
                str(outcomes_map.get("other",0))
            )
        console.print(outcome_table)
    else:
        console.print("No tool usage found in logs.")

    if num_examples_to_print > 0 and raw_interactions:
        console.print(f"\n[bold underline]Random Formatted Training Examples (N={num_examples_to_print}):[/bold underline]")

        # Ensure num_examples_to_print is not greater than available interactions
        actual_num_to_print = min(num_examples_to_print, len(raw_interactions))
        selected_examples = random.sample(raw_interactions, actual_num_to_print)

        for i, entry in enumerate(selected_examples):
            formatted_prompt = format_interaction_for_training(entry, tokenizer_eos_for_examples)
            if formatted_prompt:
                console.print(f"\n--- Example {i+1} ---")
                console.print(Text(formatted_prompt, overflow="fold")) # Allow folding for long lines
            else:
                console.print(f"\n--- Example {i+1} (Skipped due to missing essential fields) ---")
        console.print("\n--- End of Examples ---")


    # Example of what would happen if this script is run:
    # 1. Parses arguments (defaults or from command line).
    # 2. Loads base model and tokenizer.
    # 3. Configures LoRA.
    # 4. Loads and processes log data.
    # 5. Sets up Trainer.
    # 6. (If resources allow) Runs trainer.train().
    # 7. Saves LoRA adapters.

    # To make this script runnable and test its structure without full training:
    # - Ensure placeholder ./merged_model with dummy config.json, tokenizer_config.json etc. exists.
    # - Create some dummy log files in ./interaction_logs/.
    # - Run with --num_train_epochs 0 or a very small number of steps if `trainer.train()` is called.
    # For now, the script is designed to be run by the user who has a proper environment.
    main()

```python
# Example of how to create dummy files for testing the script structure (run this separately in a Python interpreter if needed)
# Path("./merged_model").mkdir(exist_ok=True)
# Path("./interaction_logs").mkdir(exist_ok=True)
# with open("./merged_model/config.json", "w") as f: json.dump({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, f) # Minimal config
# with open("./merged_model/tokenizer_config.json", "w") as f: json.dump({"model_max_length": 512}, f) # Minimal tokenizer_config
# with open("./merged_model/vocab.json", "w") as f: json.dump({"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}, f) # Dummy vocab
# with open("./merged_model/merges.txt", "w") as f: f.write("#version: 0.2\n") # Dummy merges for BPE
# with open("./interaction_logs/interaction_20230101_120000.log", "w") as f:
#     f.write("Timestamp: 20230101_120000\nUser Input: Hello\nModel Output: Hi there!\n----------------------------------------\n")

```
