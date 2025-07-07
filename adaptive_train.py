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
    parser.add_argument(
        "--log_count", type=int, default=None, help="Process only the last N interaction log files."
    )
    parser.add_argument(
        "--log_days", type=int, default=None, help="Process logs from the last D days."
    )
    # Add more training arguments as needed (e.g., weight_decay, lr_scheduler_type)
    return parser.parse_args()

# --- Helper functions for formatting JSONL data for prompt ---
def format_context_for_prompt(context_dict: dict) -> str:
    if not context_dict:
        return "CONTEXT: None"

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
    if not tool_interactions:
        return ""

    turn_dialogue = ["\nINTERMEDIATE_STEPS:"]
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
def load_interaction_logs(jsonl_log_path: str, tokenizer: AutoTokenizer, max_seq_length: int):
    """
    Loads interaction logs from a JSONL file, formats them, and tokenizes them.
    """
    formatted_texts = []
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
        progress_bar.stop()


    if not formatted_texts:
        console.print("[yellow]No valid interactions formatted for training after processing.[/yellow]")
        return []

    min_dataset_size = 10 # Arbitrary minimum, can be adjusted
    if len(formatted_texts) < min_dataset_size:
        console.print(f"[warning]Warning: The number of formatted interactions ({len(formatted_texts)}) is very small (less than {min_dataset_size}). Fine-tuning may not be effective or could lead to overfitting.[/warning]")
        if input(f"Continue with {len(formatted_texts)} interactions? (yes/NO): ").lower() != "yes":
            console.print("[info]Training aborted by user due to small dataset size.[/info]")
            return []

    console.print(f"[success]Successfully formatted {len(formatted_texts)} interactions from JSONL.[/success]")

    # Tokenize the formatted texts
    tokenized_dataset = []
    if RICH_AVAILABLE:
        task_id_tokenize = progress_bar.add_task("Tokenizing data", total=len(formatted_texts)) # Re-add task for progress
        progress_bar.start() # Re-start if stopped

    for text in formatted_texts:
        if RICH_AVAILABLE:
            progress_bar.update(task_id_tokenize, advance=1) # Use the new task_id
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
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Common for Llama-like
    console.print(f"[info]Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}[/info]")
    console.print(f"[info]Attempting to target modules for LoRA (common Llama-like modules): {target_modules}.[/info]")
    console.print("[warning]If training fails around PEFT model configuration, these target_modules might need adjustment for your specific merged model architecture.[/warning]")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules, # Adjust based on the actual model architecture
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
        # report_to="tensorboard", # Or "wandb", "none"
        fp16=not args.use_qlora,  # Use fp16 if not using QLoRA (which uses its own dtype)
        bf16=args.use_qlora, # QLoRA often paired with bf16 compute_dtype
        # Add other arguments like weight_decay, lr_scheduler_type, warmup_steps etc.
        # optim="paged_adamw_8bit" if args.use_qlora else "adamw_hf", # QLoRA often uses paged optimizers
        disable_tqdm=RICH_AVAILABLE, # Disable default tqdm if Rich progress is used
        logging_dir=f"{args.output_dir}/logs", # Store training logs
    )

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
