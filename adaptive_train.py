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
    parser = argparse.ArgumentParser(description="Adaptive fine-tuning script for the merged AGI model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the base model to be fine-tuned (./merged_model).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        help="Directory containing interaction logs.",
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

# --- Log Parsing ---
def load_interaction_logs(log_dir: str, tokenizer: AutoTokenizer, max_seq_length: int, log_count: Optional[int] = None, log_days: Optional[int] = None):
    """
    Loads interaction logs, formats them, and tokenizes them.
    Current log format:
    Timestamp: YYYYMMDD_HHMMSS
    User Input: <text>
    Model Output: <text>
    ----------------------------------------
    """
    formatted_texts = []
    log_files = glob.glob(os.path.join(log_dir, "interaction_*.log"))

    if not log_files:
        console.print(f"[yellow]No log files found in {log_dir}. Nothing to train on.[/yellow]")
        return []

    # Sort files by name (timestamp) to get recent ones if --log_count or --log_days is used
    log_files.sort(reverse=True) # Newest first

    # Filter by --log_days
    if log_days is not None:
        filtered_by_date = []
        cutoff_date = datetime.now() - timedelta(days=log_days)
        for log_file in log_files:
            try:
                # Extract timestamp from filename: interaction_YYYYMMDD_HHMMSS.log
                filename = Path(log_file).name
                ts_str = filename.split('_')[1] # YYYYMMDD
                file_date = datetime.strptime(ts_str, "%Y%m%d") # Just compare date part for simplicity
                if file_date >= cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0): # Compare date part only
                    filtered_by_date.append(log_file)
            except (IndexError, ValueError) as e:
                console.print(f"[yellow]Could not parse date from log file {filename}: {e}. Skipping for date filter.[/yellow]")
                continue
        log_files = filtered_by_date
        console.print(f"[info]Filtered logs by date: {len(log_files)} files from the last {log_days} days remaining.[/info]")


    # Filter by --log_count (applied after date filter if both are present)
    if log_count is not None and len(log_files) > log_count:
        log_files = log_files[:log_count] # Already sorted newest first
        console.print(f"[info]Filtered logs by count: Using the latest {len(log_files)} files.[/info]")

    if not log_files:
        console.print(f"[yellow]No log files remaining after filters. Nothing to train on.[/yellow]")
        return []

    console.print(f"[info]Processing {len(log_files)} log files...[/info]")

    # Use Rich Progress for parsing
    if RICH_AVAILABLE:
        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console
        )
        task_id = progress_bar.add_task("Parsing logs", total=len(log_files))
        progress_bar.start()

    for log_file in log_files:
        if RICH_AVAILABLE:
            progress_bar.update(task_id, advance=1, description=f"Parsing {Path(log_file).name}")
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple parsing based on keywords
            parts = content.split("User Input:")
            if len(parts) < 2: continue

            input_part = parts[1]
            output_parts = input_part.split("Model Output:")
            if len(output_parts) < 2: continue

            user_input = output_parts[0].strip()
            model_output = output_parts[1].split("----------------------------------------")[0].strip()

            if user_input and model_output:
                # Format for instruction fine-tuning
                # Using a common format. The exact tokens (USER:, ASSISTANT:, \n) matter.
                # Ensure the tokenizer handles these tokens correctly or add them if they are special.
                # For merged models, the base model's tokenizer conventions are important.
                # This format aims for the model to learn to complete the "ASSISTANT:" part.
                text = f"USER: {user_input}\nASSISTANT: {model_output}{tokenizer.eos_token}"
                formatted_texts.append(text)
        except Exception as e:
            console.print(f"[yellow]Error parsing log file {Path(log_file).name}: {e}. Skipping.[/yellow]")
            continue

    if RICH_AVAILABLE:
        progress_bar.stop()

    if not formatted_texts:
        console.print("[yellow]No valid interactions parsed from logs after processing.[/yellow]")
        return []

    min_dataset_size = 10 # Arbitrary minimum, can be adjusted
    if len(formatted_texts) < min_dataset_size:
        console.print(f"[warning]Warning: The number of parsed interactions ({len(formatted_texts)}) is very small (less than {min_dataset_size}). Fine-tuning may not be effective or could lead to overfitting.[/warning]")
        if input(f"Continue with {len(formatted_texts)} interactions? (yes/NO): ").lower() != "yes":
            console.print("[info]Training aborted by user due to small dataset size.[/info]")
            return []


    console.print(f"[success]Successfully parsed {len(formatted_texts)} interactions.[/success]")

    # Tokenize the formatted texts
    tokenized_dataset = []
    if RICH_AVAILABLE:
        task_id_tokenize = progress_bar.add_task("Tokenizing data", total=len(formatted_texts))
        progress_bar.start()

    for text in formatted_texts:
        if RICH_AVAILABLE:
            progress_bar.update(task_id_tokenize, advance=1)
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
