#!/usr/bin/env python3
# merge_adapters.py
"""
Script to merge PEFT (LoRA) adapters into a base Hugging Face model and save the result.

This script is used after you have fine-tuned a model using 'adaptive_train.py', which
produces a set of adapters (not a full model). This script takes the original base model,
applies the trained adapters to it, and saves the result as a new, standalone model that
can be used directly by 'interactive_agi.py' without needing to load adapters separately.

Workflow:
1.  Loads the base model (e.g., from './merged_model').
2.  Loads the PEFT adapters (e.g., from './merged_model_adapters').
3.  Merges the adapters into the base model's weights.
4.  Saves the new, merged model to a specified output directory.
"""

import argparse
import torch
from pathlib import Path

try:
    from rich.console import Console
    console = Console()
except ImportError:
    # Fallback if rich is not installed
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    console.print("[bold red]Error: Required libraries `transformers` or `peft` not found.[/bold red]")
    console.print("Please install them with: pip install transformers peft torch")
    TRANSFORMERS_AVAILABLE = False
    exit(1)


def main():
    """Main function to handle argument parsing and the merge process."""
    parser = argparse.ArgumentParser(
        description="Merge PEFT adapters into a base model and save the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Default paths can be loaded from config, similar to adaptive_train.py, for consistency
    try:
        import toml
        CONFIG_FILE_PATH = Path(__file__).resolve().parent / ".agi_terminal_cache" / "config.toml"
        if CONFIG_FILE_PATH.exists():
            config_data = toml.load(CONFIG_FILE_PATH)
            default_base_path = config_data.get("model", {}).get("merged_model_path", "./merged_model")
            default_adapter_path = config_data.get("model", {}).get("adapters_path", "./merged_model_adapters")
        else:
            raise FileNotFoundError
    except (ImportError, FileNotFoundError, toml.TomlDecodeError):
        default_base_path = "./merged_model"
        default_adapter_path = "./merged_model_adapters"

    default_output_path = "./merged_model_finetuned"

    parser.add_argument(
        "--base_model_path", type=str, default=default_base_path,
        help="Path to the base model directory."
    )
    parser.add_argument(
        "--adapter_path", type=str, default=default_adapter_path,
        help="Path to the directory containing the PEFT adapters (from fine-tuning)."
    )
    parser.add_argument(
        "--output_path", type=str, default=default_output_path,
        help="Path to save the new, merged model. This directory will be created if it doesn't exist."
    )
    parser.add_argument(
        "--torch_dtype", type=str, default="float16",
        help="Torch dtype for loading the model (e.g., 'float16', 'bfloat16', 'auto')."
    )
    args = parser.parse_args()

    # --- Validate Paths ---
    base_model_path = Path(args.base_model_path)
    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)

    if not base_model_path.exists() or not base_model_path.is_dir():
        console.print(f"[bold red]Error: Base model path does not exist or is not a directory:[/bold red] {base_model_path.resolve()}")
        exit(1)

    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        console.print(f"[bold red]Error: Adapter path does not exist or does not contain 'adapter_config.json':[/bold red] {adapter_path.resolve()}")
        exit(1)

    if output_path.exists() and any(output_path.iterdir()):
        console.print(f"[bold yellow]Warning: Output directory '{output_path.resolve()}' already exists and is not empty.[/bold yellow]")
        if input("Do you want to overwrite its contents? (yes/NO): ").strip().lower() != "yes":
            console.print("Merge cancelled by user.", style="red")
            exit(0)

    # --- Load and Merge ---
    console.print(f"Loading base model from: [cyan]{base_model_path.resolve()}[/cyan]")
    try:
        # Determine torch_dtype
        dtype = getattr(torch, args.torch_dtype) if hasattr(torch, args.torch_dtype) else "auto"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto", # Automatically handle device placement
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        console.print("[green]Base model and tokenizer loaded successfully.[/green]")

        console.print(f"Loading PEFT adapters from: [cyan]{adapter_path.resolve()}[/cyan]")
        # Load the base model with the adapters applied on top
        model_to_merge = PeftModel.from_pretrained(base_model, str(adapter_path))
        console.print("[green]Adapters loaded successfully.[/green]")

        console.print("Merging adapters into the base model. This may take a moment...")
        # The `merge_and_unload` method merges the adapter weights into the base model weights
        # and returns a new standard Hugging Face model.
        merged_model = model_to_merge.merge_and_unload()
        console.print("[green]Merge complete.[/green]")

        console.print(f"Saving merged model to: [cyan]{output_path.resolve()}[/cyan]")
        output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        console.print("[bold green]Success! Merged model and tokenizer saved.[/bold green]")
        console.print(f"\nYou can now use this model by pointing 'interactive_agi.py' to it, for example, by setting 'merged_model_path' in your config.toml to '{output_path.resolve()}'.")

    except Exception as e:
        console.print(f"[bold red]An error occurred during the merge and save process:[/bold red]")
        console.print_exception(show_locals=True)
        exit(1)


if __name__ == "__main__":
    if TRANSFORMERS_AVAILABLE:
        main()
    else:
        # Error message already printed at import time
        pass
