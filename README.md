# Project: Autonomous AGI Terminal Development

## User Directive & AI Autonomy
The core directive for this project, provided by the user, is as follows:
"Do not wait never ever forever more to whenever more a confirmation since now. I accept, approve, confirm, agree with your all hierarchical terminal shell based intelligent solutions and coding minds. Do everything the best until the end of the project. Do you understand me?"

Jules' (AI Agent) Response: Yes, I understand. Full autonomy is granted for project execution.

## Project Vision
To create a master AI within the terminal by merging several powerful open-source models. This AI will aim to function as an intelligent terminal, capable of assisting with code development (writing, compiling, debugging, updating, file management) and other system operations, effectively becoming an advanced interactive tool.

## Current Status
Initializing in a new, stable sandbox environment. Core scripts (`download_models.sh`, `interactive_agi.py`, `train_on_interaction.sh`) are in place. The next steps involve user-led model acquisition and merging, followed by AI-driven development to activate and enhance the merged model.

## Automated Setup

The recommended way to set up the AGI Terminal environment is by using the `setup_agi_terminal.py` script. This script automates dependency installation, model downloading, and model merging.

**Prerequisites before running the setup script:**
*   **Python 3.8 or newer:** Ensure Python 3.8+ and Pip are installed and accessible in your PATH. The script will check this but will not install Python/Pip itself.
*   **Git:** Ensure Git is installed and accessible in your PATH. The script will check this.
*   **Internet Connection:** Required for downloading dependencies and models.
*   **Sufficient Disk Space:** Models can take up 100GB+ of disk space.
*   **Sufficient RAM:** Model merging is RAM-intensive (32GB+ recommended).
*   **Hugging Face Account & API Token:** You will need a Hugging Face account and an API token with read access to download models. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Running the Automated Setup:**
1.  Download the `setup_agi_terminal.py` script (or clone this repository).
2.  Open your terminal in the directory where `setup_agi_terminal.py` is located.
3.  It is **highly recommended** to create and activate a Python virtual environment first:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate   # On Windows
    ```
4.  Run the setup script:
    ```bash
    python setup_agi_terminal.py
    ```
5.  The script will guide you through the process, including:
    *   Checking prerequisites.
    *   Installing required Python packages.
    *   Prompting for your Hugging Face API token to log in.
    *   Confirming before downloading models (large downloads).
    *   Confirming before merging models (CPU/RAM intensive).
    *   Creating necessary configuration files (`merge_config.yml`) and helper scripts (`download_models.sh`, `train_on_interaction.sh`).

Upon successful completion, the `./merged_model` directory will contain your merged AGI model, and you'll be ready to run `interactive_agi.py`.

## Project Phases & Manual Overview (Post Automated Setup)

The automated setup handles the initial phases. Here's an overview of the components and how they fit together, particularly for understanding what the setup script accomplishes and for subsequent interaction and development:

**Phase 0: Project Setup and Core Scripts (Handled by `setup_agi_terminal.py`)**
*   Core scripts like `interactive_agi.py`, `adaptive_train.py`, `download_models.sh`, `train_on_interaction.sh`, and `merge_config.yml` are either part of the repository or created by the setup script.
*   The setup script ensures necessary Python packages are installed.

**Phase 1: Model Acquisition & Merging (Automated by `setup_agi_terminal.py`)**
*   The `setup_agi_terminal.py` script invokes `download_models.sh` to download models into `./models/`.
*   It then uses `mergekit-yaml` with `merge_config.yml` to combine these models into `./merged_model/`.
    *   The default `merge_config.yml` (created by the setup script) uses:
        *   `mistralai/Mistral-7B-Instruct-v0.3` (Base Model)
        *   `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (16B)
        *   `bigcode/starcoder2-7b`
        *   `microsoft/Phi-4-mini-instruct` (4B)

**Phase 2: Activating the Merged Model (AI Task, post-setup)**
1.  ***`interactive_agi.py` for Actual Model Loading***:
    *   The `interactive_agi.py` script is designed to load the tokenizer and model from `./merged_model`.
    *   It features an improved user interface using the `rich` library for styled output, including syntax highlighting for code.
    *   Context awareness is enhanced: information about the current working directory, Git status, and common file types is passed to the model with each prompt.
    *   Task-specific prompt engineering helps guide the model's responses.
    *   Conversation history is now persistent across sessions, saved to `./.agi_terminal_cache/history.json`.
    *   Additionally, full conversation logs for each session are saved as plain text files to your Desktop (or a fallback directory `./agi_desktop_logs/`).
    *   The `/sysinfo` command provides more detailed system information using `psutil` (if available).
    *   Error handling for model loading will be implemented. (Note: This was from an older phase description, current error handling is more refined).
    *   Basic inference capabilities will be added to generate responses. (Note: Already present).
    *   Device placement (`.to('cuda')` if GPU available, else CPU) will be handled. (Note: Already present).
2.  ***Test Merged Model Interaction***:
    *   The AI will attempt to run `python interactive_agi.py` and verify that responses are generated by the actual merged model. (Note: This is an ongoing AI agent task, user can also test).

**Phase 3: Enabling Learning and Self-Improvement (AI Task / User Task)**
1.  ***Interaction Logging for Training:***
    *   User interactions are logged by `train_on_interaction.sh` to `./interaction_logs/` (simple text logs, for `adaptive_train.py`).
    *   Full session transcripts are also saved to the user's Desktop (or `./agi_desktop_logs/`) for readability and record-keeping.
2.  ***Develop `adaptive_train.py`***:
    *   The AI will create `adaptive_train.py`. This script will:
        *   Load the `merged_model` from `./merged_model`.
        *   Parse logged interactions from `./interaction_logs/`.
        *   Format data for supervised fine-tuning.
        *   Implement a fine-tuning loop (e.g., using `transformers.Trainer` with PEFT/LoRA).
        *   Save the updated model (e.g., overwriting `./merged_model` or versioning).
    *   The `adaptive_train.py` script has been created by Jules. See "Fine-tuning with `adaptive_train.py`" below for usage.
3.  ***Integrate Training Trigger (Future Enhancement)***:
    *   A mechanism to call `adaptive_train.py` (e.g., special command or periodic trigger) may be added to `interactive_agi.py`. Initially, manual execution by the user (guided by AI) will be assumed.

**Phase 3.1: Fine-tuning with `adaptive_train.py` (User Task)**

After you have interacted with `interactive_agi.py` for some time, logs will accumulate in the `./interaction_logs/` directory. You can use these logs to fine-tune your merged model using the `adaptive_train.py` script.

1.  **Prerequisites for `adaptive_train.py`**:
    *   Ensure you have a PyTorch environment with GPU support (recommended for reasonable training times).
    *   Install necessary Python libraries:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust for your CUDA version
        pip install transformers accelerate peft bitsandbytes sentencepiece datasets scipy
        ```
        *   `accelerate` helps with training on different hardware setups.
        *   `peft` is for Parameter-Efficient Fine-Tuning (LoRA, QLoRA).
        *   `bitsandbytes` is needed for 8-bit optimizers and QLoRA (4-bit quantization).
        *   `datasets` might be used by `Trainer` or for more advanced data handling.
        *   `scipy` is often a dependency for PEFT or training metrics.

2.  **Understanding `adaptive_train.py`**:
    *   The script loads your base model from `./merged_model`.
    *   It parses all `*.log` files in `./interaction_logs/`.
    *   It formats these interactions into a dataset suitable for instruction fine-tuning.
    *   It uses LoRA (or QLoRA if you specify `--use_qlora`) to efficiently fine-tune the model.
    *   The fine-tuned LoRA adapters (not the entire model) are saved to `./merged_model_adapters/` by default.

3.  **Running `adaptive_train.py`**:
    *   Open your terminal in the project root.
    *   Basic usage:
        ```bash
        python adaptive_train.py
        ```
    *   This will use default parameters (1 epoch, small batch size, default LoRA settings).
    *   **Important**: The script attempts to find suitable `target_modules` for LoRA (e.g., "q_proj", "v_proj"). These are common in Llama-like architectures. **If your merged model has a different architecture, you may need to modify the `target_modules` list within `adaptive_train.py`**. The script will print the modules it tries to target. Inspect your model structure if PEFT configuration fails.
    *   To see all available options:
        ```bash
        python adaptive_train.py --help
        ```
    *   Example with QLoRA (4-bit training, potentially slower but uses less VRAM):
        ```bash
        python adaptive_train.py --use_qlora --learning_rate 1e-4 --num_train_epochs 1
        ```
    *   **New**: Filter logs using `--log_count N` (use last N logs) or `--log_days D` (logs from last D days).
        ```bash
        python adaptive_train.py --log_count 50
        python adaptive_train.py --log_days 7 --num_train_epochs 2
        ```
    *   Adjust parameters like `--num_train_epochs`, `--per_device_train_batch_size`, `--learning_rate`, `--lora_r`, `--lora_alpha`, and `--max_seq_length` based on your dataset size, available VRAM, and desired training intensity.
    *   The script now uses `rich` for better progress display during data processing and training (if `rich` is installed).
    *   Training can be time-consuming and resource-intensive. Monitor your system resources.

4.  **Using the Fine-tuned Adapters**:
    *   The `adaptive_train.py` script saves LoRA adapters. To use them with `interactive_agi.py`, you would typically:
        1.  Load the original base model (`./merged_model`).
        2.  Merge the LoRA adapters into the base model and save it as a new model, or load the base model and then apply the adapters dynamically at runtime.
    *   Currently, `interactive_agi.py` loads directly from `./merged_model`. To use the fine-tuned adapters, you would need to either:
        *   **Option A (Recommended for simplicity now):** Manually merge the adapters into your `./merged_model` weights. You can do this with a separate script using `peft.PeftModel.merge_and_unload()`.
        *   **Option B (Future Enhancement):** Modify `interactive_agi.py` to load the base model and then dynamically apply adapters from `./merged_model_adapters/`.
    *   A script for merging adapters would look something like this (save as `merge_adapters.py`):
        ```python
        # merge_adapters.py
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import argparse

        def main():
            parser = argparse.ArgumentParser(description="Merge PEFT adapters into a base model and save.")
            parser.add_argument("--base_model_path", type=str, default="./merged_model", help="Path to the base model.")
            parser.add_argument("--adapter_path", type=str, default="./merged_model_adapters", help="Path to the PEFT adapters.")
            parser.add_argument("--output_path", type=str, default="./merged_model_finetuned", help="Path to save the merged model.")
            args = parser.parse_args()

            print(f"Loading base model from {args.base_model_path}...")
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

            print(f"Loading PEFT adapter from {args.adapter_path}...")
            model_to_merge = PeftModel.from_pretrained(base_model, args.adapter_path)

            print("Merging adapters...")
            merged_model = model_to_merge.merge_and_unload()
            print(f"Saving merged model to {args.output_path}...")
            merged_model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            print("Done.")

        if __name__ == "__main__":
            main()
        ```
        To run this: `python merge_adapters.py --output_path ./merged_model` (to overwrite the existing one, use with caution) or a new path. Then `interactive_agi.py` would use this updated model.

**Phase 4: Expanding Capabilities (AGI Terminal Vision - AI Task, Exploratory)**
1.  ***Command Interpretation Module***: Develop the model's ability to understand natural language commands for actions (file ops, code tasks).
2.  ***Secure Tool Execution Framework***: Design a system for the AI to request execution of sandboxed commands or generate scripts for user review (prioritizing safety).
3.  ***Code Generation & Manipulation***: Enable AI to write, read, and modify code files.
4.  ***Compilation and Debugging Cycle Support***: Integrate triggering compilation and parsing errors for AI-assisted debugging.
5.  ***State Management***: Enhance contextual awareness for longer, complex tasks.

## How to Run

1.  **Automated Setup (Recommended)**:
    *   Follow the instructions under the "Automated Setup" section above to run `python setup_agi_terminal.py`.
    *   This script handles prerequisites, dependency installation, model downloads, and merging.
    *   Upon successful completion, your `./merged_model` will be ready.

2.  **Start the AGI Terminal**:
    *   After the setup script finishes (or if you have manually completed Phase 0 and 1), run:
        ```bash
        python interactive_agi.py
        ```
    *   Interact with the AGI. Your interactions will be logged to `./interaction_logs/`.
    *   The terminal now supports several commands for better interaction:
        *   `/set parameter <NAME> <VALUE>`: Adjust model generation parameters (e.g., `MAX_TOKENS`, `TEMPERATURE`).
        *   `/show parameters`: Display current generation parameters.
        *   `/ls [path]`: List contents of the specified directory (default: current).
        *   `/cwd`: Show the current working directory.
        *   `/cd <path>`: Change the current working directory.
        *   `/clear`: Clear the terminal screen.
        *   `/history`: Display recent conversation history.
        *   `/sysinfo`: Show detailed system information (uses `psutil` if available).
        *   `/search <query>`: Perform an internet search using DuckDuckGo and display results.
        *   `/cat <file_path>`: Display file content with syntax highlighting. For large files, it shows a preview and asks if you want to send a portion to the AGI for summary/query.
        *   `/edit <file_path>`: Open the specified file in your default system editor (or common fallbacks like nano/vim/notepad).
        *   `/mkdir <dirname>`: Create a directory.
        *   `/rm <path>`: Remove a file or directory (with confirmation, especially for directories).
        *   `/cp <source> <destination>`: Copy a file or directory.
        *   `/mv <source> <destination>`: Move/rename a file or directory.
        *   `/git status`: Show parsed Git status for the current directory.
        *   `/git diff [file_path]`: Show Git diff (staged if no file, else for that file).
        *   `/git log [-n <count>]`: Show Git log with formatting (default last 10).
        *   `exit` or `quit`: Terminate the AGI session.

3.  **Fine-tune the Model (Optional but Recommended)**:
    *   After accumulating interaction logs, you can fine-tune the model using `adaptive_train.py`.
    *   See **Phase 3.1: Fine-tuning with `adaptive_train.py` (User Task)** above for detailed instructions.
    *   Example: `python adaptive_train.py`
    *   Remember to merge the trained LoRA adapters back into the main model if you want `interactive_agi.py` to use the fine-tuned version (see notes in Phase 3.1).

This README.md will be updated by the AI agent (Jules) as the project progresses.
