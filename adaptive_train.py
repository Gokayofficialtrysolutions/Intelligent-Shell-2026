#!/usr/bin/env python3
# adaptive_train.py

import argparse
import os
import glob
import random
from pathlib import Path
import json # For potential future structured logging

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
    # Add more training arguments as needed (e.g., weight_decay, lr_scheduler_type)
    return parser.parse_args()

# --- Log Parsing ---
def load_interaction_logs(log_dir: str, tokenizer: AutoTokenizer, max_seq_length: int):
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
        print(f"No log files found in {log_dir}. Nothing to train on.")
        return []

    print(f"Found {len(log_files)} log files. Parsing...")

    for log_file in log_files:
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
            print(f"Error parsing log file {log_file}: {e}")
            continue

    if not formatted_texts:
        print("No valid interactions parsed from logs.")
        return []

    print(f"Successfully parsed {len(formatted_texts)} interactions.")

    # Tokenize the formatted texts
    # This is a simplified tokenization. For more advanced scenarios,
    # one might create a Hugging Face Dataset object and use .map() for tokenization.
    tokenized_dataset = []
    for text in formatted_texts:
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

    print(f"Tokenized {len(tokenized_dataset)} interactions.")
    return tokenized_dataset


# --- Main Training Logic ---
def main():
    args = parse_arguments()

    print("Starting adaptive training script...")
    print(f"Configuration: {args}")

    # --- 1. Load Tokenizer and Model ---
    print(f"Loading base model from: {args.model_path}")

    # Quantization config for QLoRA
    bnb_config = None
    if args.use_qlora:
        print("QLoRA enabled. Using 4-bit quantization.")
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
            print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config, # None if not args.use_qlora
            device_map={"":0}, # Load model on GPU 0. Adjust if multi-GPU or CPU.
                               # For QLoRA, device_map="auto" or specific GPU is common.
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16 # if not using bnb_config and want bfloat16
        )
        print("Tokenizer and model loaded successfully.")

        if args.use_qlora:
            print("Preparing model for k-bit training (QLoRA)...")
            model = prepare_model_for_kbit_training(model)

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Ensure './merged_model' exists and is a valid Hugging Face model directory.")
        print("This script should be run AFTER the user has downloaded and merged models.")
        return

    # --- 2. Configure PEFT (LoRA) ---
    # Target modules can vary by model architecture. Common ones are query, key, value layers.
    # Need to inspect the model to find appropriate module names (e.g., 'q_proj', 'k_proj', 'v_proj' for Llama-like models)
    # For now, let's assume common ones for demonstration. This might need adjustment.
    target_modules = ["q_proj", "v_proj"] # Placeholder - MUST BE CHECKED FOR THE MERGED MODEL
    # A more robust way would be to inspect model.named_modules()
    # Or use a utility function to find all linear layers if targeting them broadly.

    print(f"Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Attempting to target modules: {target_modules}. THIS MAY NEED ADJUSTMENT FOR YOUR SPECIFIC MERGED MODEL.")

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
        print("PEFT model (LoRA) configured successfully.")
        model.print_trainable_parameters()
    except Exception as e:
        print(f"Error configuring PEFT model: {e}")
        print("This might be due to incorrect target_modules for the loaded model architecture.")
        # Example inspection:
        # print("Model architecture inspection (first few layers):")
        # for name, module in model.named_modules():
        #     if 'proj' in name or 'attention' in name or 'mlp' in name: # Common keywords
        #         print(name)
        #     # Or print all to find suitable Linear layers
        return

    # --- 3. Load and Prepare Data ---
    print("Loading and preparing dataset from interaction logs...")
    train_dataset = load_interaction_logs(args.log_dir, tokenizer, args.max_seq_length)

    if not train_dataset:
        print("No training data available. Exiting.")
        return

    # Data collator for language modeling. It will handle creating labels from input_ids if not already present.
    # If labels are input_ids (as done in load_interaction_logs), MLM=False ensures labels are not further masked.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 4. Training Arguments ---
    print(f"Preparing training arguments. Output will be saved to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

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
    )

    # --- 5. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=None, # Add validation set if available
        data_collator=data_collator,
    )

    # --- 6. Train ---
    print("Starting training...")
    print("NOTE: Actual model training is computationally intensive and requires a suitable GPU environment.")
    print("This script provides the structure. Ensure your environment is set up for Hugging Face model training.")

    # This is where the actual training happens.
    # In a sandboxed environment without a GPU or sufficient resources, this will be slow or fail.
    # For the purpose of this agent's task, we are setting up the script.
    # The user will run this script in their own environment.
    try:
        trainer.train()
        print("Training completed.")
    except Exception as e:
        print(f"An error occurred during trainer.train(): {e}")
        print("This could be due to resource limitations (CUDA OOM), configuration issues, or data problems.")
        print("Consider reducing batch size, sequence length, or using QLoRA if not already.")
        # No return here, proceed to attempt saving if any progress was made or if it's just a dry run.

    # --- 7. Save Model (PEFT Adapters) ---
    print(f"Saving PEFT adapters to {args.output_dir}...")
    try:
        # Trainer might save checkpoints already. This explicitly saves the final adapter.
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir) # Save tokenizer for completeness
        print(f"Adapters and tokenizer saved successfully to {args.output_dir}.")
        print("To load the fine-tuned model later:")
        print("1. Load the base model: model = AutoModelForCausalLM.from_pretrained(base_model_path)")
        print("2. Load PEFT adapters: model = PeftModel.from_pretrained(model, adapter_path)")
    except Exception as e:
        print(f"Error saving model/adapters: {e}")

    print("Adaptive training script finished.")

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
