# Project: Autonomous AGI Terminal Development

**Creator: Gökay Yaşar Üzümcü**

---

## 1. Project Overview

Welcome to the AGI Terminal! This project aims to create a sophisticated, locally-run AI development assistant. By merging several powerful open-source language models, it forms a versatile "MergedAGI" capable of understanding context, generating code, assisting with debugging, interacting with your file system, and more, all within your terminal.

The system is designed for developers, researchers, and AI enthusiasts who want a highly capable AI assistant that operates on their own machine, offering greater privacy, customization, and potential for offline use.

**Core Features:**

*   **Merged AI Model:** Combines the strengths of multiple specialized open-source LLMs.
*   **Interactive Terminal:** A rich command-line interface for interacting with the MergedAGI.
*   **Context Awareness:** The AGI is provided with context about your current working directory, Git status, and key files to make its responses more relevant.
*   **Tool Usage Framework:** The AGI can:
    *   Execute whitelisted shell commands (e.g., `ls`, `pwd`, `grep`).
    *   Read and suggest changes to files.
    *   Perform internet searches via DuckDuckGo.
    *   Execute Python code snippets in a restricted environment.
    *   Manage Git operations (branch, checkout, commit, push).
    *   Formulate and attempt multi-step plans using these tools.
*   **Adaptive Learning:** Includes a script (`adaptive_train.py`) to fine-tune the MergedAGI on your interaction history, allowing it to learn your preferences and improve over time.
*   **Customizable:** You can change the models being merged, fine-tune with your own data, and extend its capabilities.
*   **Local & Private:** Runs entirely on your machine (once models are downloaded).

This `README.md` serves as your comprehensive guide to setting up, using, and customizing the AGI Terminal.

## 2. System Prerequisites

Before you begin, ensure your system meets the following requirements:

*   **Operating System:**
    *   **Linux (Recommended):** Debian/Ubuntu, Fedora, or other modern distributions. Most shell scripts and tools are optimized for a Linux-like environment.
    *   **macOS:** Should work, but some shell commands or dependencies might require minor adjustments (e.g., using `gsed` instead of `sed`). Homebrew is recommended for installing packages like `git-lfs`.
    *   **Windows:** Requires **Windows Subsystem for Linux (WSL) 2** with a Linux distribution installed. Direct execution on Windows CMD or PowerShell is not supported due to reliance on bash scripts and POSIX paths.
*   **Python:**
    *   Version **3.8 or newer**. Python 3.10+ is recommended.
    *   `pip` (Python package installer) must be available for your Python installation.
    *   **Virtual Environment (Strongly Recommended):** To avoid conflicts with system-wide Python packages, it's crucial to create and activate a Python virtual environment (e.g., using `venv` or `conda`) before installing dependencies.
        ```bash
        # Example using venv
        python3 -m venv .venv
        source .venv/bin/activate  # For Linux/macOS/WSL
        # If using plain Windows Python (not recommended for this project): .venv\Scripts\activate
        ```
*   **Git:**
    *   Latest stable version of Git must be installed and accessible in your system's PATH.
*   **Git LFS (Large File Storage):**
    *   Required for downloading the AI model files. Install it and run `git lfs install` once globally to initialize it for your user account.
        *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install git-lfs`
        *   Fedora: `sudo dnf install git-lfs`
        *   macOS (Homebrew): `brew install git-lfs`
        *   Windows: Download installer from [https://git-lfs.github.com/](https://git-lfs.github.com/)
        *   After installation, run: `git lfs install`
*   **Build Tools (Potentially):**
    *   Some Python packages (especially those related to PyTorch or `bitsandbytes` for QLoRA) might require compilation. Ensure you have a C/C++ compiler toolchain (e.g., `build-essential` on Debian/Ubuntu, Xcode Command Line Tools on macOS).
*   **CUDA Toolkit (Optional, for GPU Acceleration):**
    *   If you have an NVIDIA GPU and want to accelerate model inference and training:
        *   Install the NVIDIA CUDA Toolkit appropriate for your GPU drivers.
        *   PyTorch will need to be installed with CUDA support (the setup script attempts a generic install; manual install might be needed for specific CUDA versions).
        *   QLoRA fine-tuning (`--use_qlora` in `adaptive_train.py`) heavily relies on CUDA.

## 3. Disk Space & RAM Requirements

Running and fine-tuning large language models is resource-intensive.

*   **Disk Space:**
    *   **Python Dependencies:** 5-20GB (PyTorch with CUDA can be very large).
    *   **AI Models (Downloaded):** Each model can range from ~2GB (e.g., TinyLlama-1.1B) to 30GB+ (e.g., DeepSeek-Coder-V2-Lite, GPT-NeoX-20B). The default selection in `download_models.sh` can easily consume **70-150GB**.
    *   **Merged Model:** The size of the merged model will be comparable to the sum of the sizes of the models being merged if not sharded, or the size of the largest model if sharded outputs are managed carefully. Mergekit also uses temporary disk space during the merge process. Expect **tens of GBs** for the merged model and temporary files.
    *   **Interaction Logs & Adapters:** These will grow over time but are initially small.
    *   **Recommendation:** At least **200-250GB of free disk space** is recommended for a smooth experience with the default model set. More if you plan to download additional large models.
*   **RAM:**
    *   **Model Loading & Inference (`interactive_agi.py`):**
        *   For smaller merged models (e.g., based on 7B parameter models): 16GB+ RAM. 32GB is safer, especially if other applications are running.
        *   Larger merged models (e.g., including 16B+ components) will require significantly more RAM (32GB, 48GB, or even 64GB+).
    *   **Model Merging (`setup_agi_terminal.py` via `mergekit`):**
        *   This is very RAM-intensive. `mergekit` loads models into RAM to perform the merge.
        *   For the default set of models (mostly 7B scale): **32GB RAM is a minimum**. 64GB is highly recommended for stability and speed, especially if merging larger models like DeepSeek-Coder-V2-Lite (16B).
        *   Insufficient RAM during merge can lead to crashes or extremely slow swapping.
    *   **Fine-tuning (`adaptive_train.py`):**
        *   **GPU VRAM:** If using GPU for fine-tuning (highly recommended), VRAM is critical.
            *   7B models with LoRA: 12-24GB VRAM might be feasible.
            *   QLoRA can reduce VRAM requirements significantly (e.g., 7B models might fit in 8-12GB VRAM).
        *   **System RAM:** Still important for data loading and processing, even with GPU training. 16-32GB system RAM is a good baseline.

**Monitor your system resources closely during setup and operation.**

## 4. Setup Instructions

Follow these steps carefully to set up the AGI Terminal environment on your machine.

**Step 1: Clone the Repository**

```bash
git clone <repository_url> # Replace <repository_url> with the actual URL of this project
cd <repository_directory_name>
```

**Step 2: Create and Activate a Python Virtual Environment (Highly Recommended)**

This isolates project dependencies.

```bash
# Ensure you are in the project's root directory
python3 -m venv .venv
source .venv/bin/activate  # For Linux/macOS/WSL
# If using plain Windows Python (not recommended for this project): .venv\Scripts\activate
```
You should see `(.venv)` at the beginning of your terminal prompt.

**Step 3: Install Git LFS (if not already done globally)**

Refer to Section 2 for installation instructions specific to your OS. After installing, run:
```bash
git lfs install
```
This initializes Git LFS for your user account. The `download_models.sh` script also runs this.

**Step 4: Run the Automated Setup Script (`setup_agi_terminal.py`)**

This script automates the remaining setup process. It will:
*   Check for prerequisites (Python, pip, git, git-lfs).
*   Install required Python packages (PyTorch, Transformers, mergekit, etc.).
*   Execute `download_models.sh` to download the AI models.
*   Execute `mergekit` to merge the downloaded models based on `merge_config.yml`.

To run the script:
```bash
python setup_agi_terminal.py
```

The script will guide you with prompts and status messages. For a fully non-interactive setup (e.g., in an automated script), you can use the `--force-yes` flag to bypass all confirmations:
```bash
python setup_agi_terminal.py --force-yes
```
*   It will ask for confirmation before installing Python packages (which includes PyTorch and can take time and space).
*   It will ask for confirmation before starting model downloads (very time and space consuming).
*   It will ask for confirmation before starting model merging (CPU and RAM intensive).

**Important Notes for `setup_agi_terminal.py`:**

*   **PyTorch Installation:** The script attempts a standard `pip install torch torchvision torchaudio`. If you have a specific CUDA version or need a CPU-only version (to save space/time if you don't have a compatible GPU), you might need to install PyTorch manually *before* running `setup_agi_terminal.py`. Refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for specific commands. If PyTorch installation fails, the script will warn you and ask if you want to proceed with other packages.
*   **Model Downloads:** The `download_models.sh` script (called by the setup script) uses `git clone` and `git lfs pull`. This can take many hours and requires substantial disk space (see Section 3). Ensure your internet is stable. If a download is interrupted, you might be able to resume by navigating to the specific model's directory (e.g., `./models/mistral7b_v03/`) and running `git lfs pull` manually, or by re-running `download_models.sh` (it attempts to skip already completed clones).
*   **Model Merging:** This step uses `mergekit` and is CPU and RAM intensive. It can also take a long time. Ensure your system has adequate resources (especially RAM, see Section 3).

**Step 5: Verify Setup**

After `setup_agi_terminal.py` completes:
*   **Check for Python Packages:** Ensure there were no critical errors during `pip install`.
*   **Check Model Downloads:** The `./models/` directory should contain subdirectories for each downloaded model (e.g., `./models/mistral7b_v03/`, `./models/olmo7b_instruct/`, etc.). These directories should be populated with model files (often large `.safetensors` or `.bin` files).
*   **Check Merged Model:** The `./merged_model/` directory should exist and contain the files for your merged AGI model (configuration files, tokenizer files, and model weight files like `pytorch_model.bin` or `model.safetensors.index.json` with shards).
*   **Review Logs:** Check the terminal output from `setup_agi_terminal.py` for any errors or warnings.

If any step failed, review the error messages, check your system resources (disk space, RAM, internet), and consult the Troubleshooting section (Section 7).

## 5. Running the AGI Terminal (`interactive_agi.py`)

Once the setup is complete and you have a `./merged_model/` directory, you can start the AGI Terminal.

**To start:**

```bash
python interactive_agi.py
```

You should see a startup banner and then a prompt like `You> `.

**Interacting with the AGI:**

*   Simply type your query or instruction and press Enter.
*   The AGI will process your input (considering current context) and generate a response.
*   If the AGI decides to use a tool (e.g., run a shell command, write a file), it will typically ask for your confirmation before proceeding with sensitive actions.

**Built-in Slash Commands:**

Type these commands directly into the prompt:

*   `/help`: Display a list of available slash commands and their usage.
*   `/exit` or `/quit`: Terminate the AGI session.
*   `/clear`: Clear the terminal screen.
*   `/history`: Display recent conversation history.
*   `/set parameter <NAME> <VALUE>`: Adjust model generation parameters (e.g., `MAX_TOKENS`, `TEMPERATURE`). Example: `/set parameter TEMPERATURE 0.7`.
*   `/show parameters`: Display current AGI generation parameters.
*   `/config show`: Display current application configuration from `config.toml`.
*   `/config get <section.key>`: Get a specific configuration value.
*   `/config set <section.key> <value>`: Set a configuration value (e.g., `/config set display.syntax_theme native`). Some changes may require restart.
*   `/cwd`: Show the current working directory.
*   `/cd <path>`: Change the current working directory.
*   `/ls [path]`: List contents of the specified directory (default: current).
*   `/mkdir <dirname>`: Create a directory.
*   `/rm <path>`: Remove a file or directory (with confirmation).
*   `/cp <source> <destination>`: Copy a file or directory.
*   `/mv <source> <destination>`: Move/rename a file or directory.
*   `/cat <file_path>`: Display file content with syntax highlighting. For large files, it may show a preview and ask if you want the AGI to summarize a portion.
*   `/edit <file_path>`: Open the specified file in your default system editor (or common fallbacks like nano/vim/notepad).
*   `/search <query>`: Perform an internet search using DuckDuckGo and display results.
*   `/git status`: Show parsed Git status for the current directory.
*   `/git diff [file_path]`: Show Git diff (staged if no file, else for that file).
*   `/git log [-n <count>]`: Show Git log with formatting (default last 10).
*   `/read_script <filename>`: Displays content of whitelisted project scripts (e.g., `interactive_agi.py`). Useful for asking the AGI about its own code.
*   `/suggest_code_change <file_path>`: (Highly Experimental) Allows you to describe a change to a whitelisted project file. The AGI will suggest the change (e.g., in diff format), which you must then review and apply manually.
*   `/analyze_dir [path]`: (Experimental) Asks the AGI to analyze a directory's structure and provide a JSON response, displayed as a tree.
*   `/sysinfo`: Show detailed system information.
*   `/save_script <name> <cmd1> ; <cmd2> ...`: Save a sequence of commands as a user script.
*   `/run_script <name>`: Execute a saved user script.
*   `/list_scripts`: List all saved user scripts.
*   `/delete_script <name>`: Delete a saved user script.
*   `/list_models`: List models defined in `download_models.sh` (for reference).

**Interaction Logs:**

*   **Plain Text Session Logs:** By default, a plain text transcript of your session is saved to your Desktop (or a fallback directory `./agi_desktop_logs/`). This can be configured via `config.toml`.
*   **JSONL Interaction Logs (for Training):** Detailed structured logs of each turn (user query, AGI responses, tool usage, context) are saved to `./.agi_terminal_cache/interaction_logs.jsonl`. These are the primary data source used by `adaptive_train.py`.

## 6. Training / Fine-tuning (`adaptive_train.py`)

The AGI Terminal includes `adaptive_train.py` to fine-tune your MergedAGI using the accumulated interaction logs from `./.agi_terminal_cache/interaction_logs.jsonl`. This allows the AGI to learn from your specific interactions, improving its responses and tool usage over time.

**Prerequisites for Fine-tuning:**

*   **GPU with Sufficient VRAM:** Highly recommended. Fine-tuning is computationally intensive.
    *   NVIDIA GPU with CUDA support.
    *   VRAM requirements depend on model size and batch size (see Section 3). QLoRA can significantly reduce VRAM needs.
*   **PyTorch with CUDA:** Ensure your PyTorch installation supports your GPU.
*   **Dependencies:** `transformers`, `accelerate`, `peft`, `bitsandbytes` (especially for QLoRA), `scipy`, `datasets`. These should be installed by `setup_agi_terminal.py`.

**Running `adaptive_train.py`:**

1.  **Accumulate Interaction Logs:** Use `interactive_agi.py` normally. Your interactions will be logged.
2.  **Run the Script:**
    ```bash
    python adaptive_train.py [options]
    ```
    Key options (see `python adaptive_train.py --help` for all):
    *   `--model_path`: Path to your base model (e.g., `./merged_model`). Default: `./merged_model`.
    *   `--jsonl_log_path`: Path to `interaction_logs.jsonl`. Default: `./.agi_terminal_cache/interaction_logs.jsonl`.
    *   `--output_dir`: Where to save the trained LoRA/QLoRA adapters. Default: `./merged_model_adapters`.
    *   `--num_train_epochs`: Number of training epochs (e.g., 1-3).
    *   `--learning_rate`: Learning rate (e.g., 2e-4, 5e-5).
    *   `--use_qlora`: Enable QLoRA for 4-bit training (reduces memory).
    *   **Data Filtering Options:** Use flags like `--train-on-successful-tool-use all` or `--train-on-halted-plans` to focus training on specific types of interactions.

**Example:**
```bash
python adaptive_train.py --num_train_epochs 1 --learning_rate 5e-5 --use_qlora --train-on-successful-tool-use all
```

**Using the Fine-tuned Adapters:**

*   `adaptive_train.py` saves PEFT adapters (LoRA layers) to the specified output directory (e.g., `./merged_model_adapters`). These adapters are small and contain only the *changes* to the model.
*   To use these changes, you must merge them with the original base model to create a new, fully fine-tuned model.

**Step 1: Merge the Adapters**

This project includes a `merge_adapters.py` script to handle this process.

```bash
# Basic usage, with default paths:
python merge_adapters.py

# You can also specify paths. The defaults are read from your config.toml.
python merge_adapters.py \\
  --base_model_path ./merged_model \\
  --adapter_path ./merged_model_adapters \\
  --output_path ./merged_model_finetuned
```
This will create a new directory (e.g., `./merged_model_finetuned`) containing the complete, fine-tuned model.

**Step 2: Use the Merged Model**

To make `interactive_agi.py` use your newly fine-tuned model, you need to update its configuration.

1.  Open the configuration file at `./.agi_terminal_cache/config.toml`.
2.  Find the `[model]` section and change the `merged_model_path` to point to your new directory.
    ```toml
    [model]
    # Old path:
    # merged_model_path = "./merged_model"
    # New path:
    merged_model_path = "./merged_model_finetuned"
    ```
3.  Alternatively, you can use the `/config` command within the terminal:
    ```
    /config set model.merged_model_path ./merged_model_finetuned
    ```
4.  Restart `interactive_agi.py`. It will now load and use your improved model.

**Analysis Mode:**
Before training, you can analyze your logs and see how they would be formatted:
```bash
python adaptive_train.py --analyze_jsonl_logs 10 # Show stats and 10 example formatted prompts
python adaptive_train.py --analyze_jsonl_logs --train-on-successful-plans # See stats if only successful plans are used
```

## 7. Troubleshooting

*   **`setup_agi_terminal.py` fails during Python package installation:**
    *   **Disk Space:** Ensure you have enough free disk space (especially for PyTorch).
    *   **Network Issues:** Check your internet connection.
    *   **PyTorch:** If `torch` installation is the problem, try installing it manually first with options specific to your system (CPU-only or specific CUDA version) from [pytorch.org](https://pytorch.org/get-started/locally/). Then re-run `setup_agi_terminal.py`.
    *   **Compiler Errors:** Some packages might need C/C++ compilers. Install `build-essential` (Linux) or Xcode Command Line Tools (macOS).
    *   **Permissions:** Ensure you have write permissions in the project directory and your Python environment.
*   **`download_models.sh` fails:**
    *   **Disk Space:** This is the most common issue. Free up significant space.
    *   **Internet Connection:** Downloads are large; ensure stability.
    *   **Git LFS Not Installed/Initialized:** Run `git lfs install` and ensure `git-lfs` command is in your PATH.
    *   **Typos in Model URLs (if customized):** Double-check URLs in `download_models.sh`.
    *   **Resume Interrupted Downloads:** Navigate to the specific model directory (e.g., `./models/mistral7b_v03/`) and run `git lfs pull`.
*   **`mergekit` (model merging) fails:**
    *   **RAM:** This is very RAM-intensive. Ensure you have enough (see Section 3). Close other applications.
    *   **Disk Space:** `mergekit` can use temporary disk space.
    *   **Model Compatibility:** If you customized `merge_config.yml` with very different models, merging might fail even with `--allow-crimes`. Try merging fewer or more similar models.
    *   **Corrupted Downloads:** Ensure models in `./models/` are complete.
*   **`interactive_agi.py` fails to load model:**
    *   Ensure `./merged_model/` directory exists and contains valid model files.
    *   Check `~/.agi_terminal_cache/config.toml` to ensure `model.merged_model_path` points to the correct location.
    *   Check console output for specific error messages from Transformers/PyTorch.
    *   You might be out of RAM to load the model.
*   **`adaptive_train.py` fails:**
    *   **GPU/CUDA Issues:** Ensure CUDA toolkit and NVIDIA drivers are correctly installed and PyTorch recognizes your GPU (`torch.cuda.is_available()` should be true).
    *   **VRAM:** Fine-tuning requires significant VRAM. Reduce batch size (`--per_device_train_batch_size`), enable QLoRA (`--use_qlora`), or use a smaller model.
    *   **Dependencies:** Ensure `peft`, `bitsandbytes` (for QLoRA), `accelerate` are installed.
    *   **Log File Path:** Verify `--jsonl_log_path` is correct.
*   **General Slowness:**
    *   Model inference and merging are demanding. Ensure your hardware meets recommendations.
    *   Using CPU for inference with large models will be slow.

## 8. Customization

*   **Changing Models for Merging:**
    1.  Edit `download_models.sh`:
        *   Modify the `MODELS_TO_DOWNLOAD` associative array. Add new model keys and their Hugging Face repository URLs.
        *   Ensure the chosen models are publicly accessible via `git clone`.
    2.  Edit `merge_config.yml`:
        *   Update the `slices.sources` list to include the new model keys (e.g., `model: ./models/your_new_model_key`).
        *   Adjust the `base_model` if necessary.
        *   Experiment with different `merge_method` and `parameters` for best results.
    3.  Re-run `python setup_agi_terminal.py`. It will skip already downloaded models and attempt to download new ones, then re-run the merge. Alternatively, run `bash download_models.sh` and then `mergekit-yaml merge_config.yml ./merged_model [options]` manually.
*   **Adjusting Generation Parameters:**
    *   Temporarily: Use `/set parameter <NAME> <VALUE>` in `interactive_agi.py`.
    *   Permanently: Edit `./.agi_terminal_cache/config.toml` under the `[generation]` section.
*   **Customizing Training:**
    *   Use the various CLI options for `adaptive_train.py` to filter data, change epochs, learning rate, LoRA parameters, etc.
    *   Modify the `format_interaction_for_training` function in `adaptive_train.py` if you want to change how log entries are structured into training prompts.
*   **Adding Slash Commands to `interactive_agi.py`:**
    *   Define a new function for your command.
    *   Add an `elif user_input_str.lower().startswith("/yourcommand")` block in the `process_single_input_turn` function to call your function.
    *   Update the `/help` command output.

## 9. Directory Structure Overview

```
AGI_TERMINAL_PROJECT_ROOT/
├── .venv/                     # Python virtual environment (if created here)
├── .agi_terminal_cache/       # Cache directory for runtime data
│   ├── config.toml            # Application configuration
│   ├── history.json           # Conversation history
│   └── interaction_logs.jsonl # Detailed logs for fine-tuning
├── models/                    # Downloaded base models
│   ├── mistral7b_v03/         # Example: Contains files for mistralai/Mistral-7B-Instruct-v0.3
│   ├── olmo7b_instruct/       # Example: Contains files for allenai/OLMo-7B-Instruct
│   └── ...                    # Other downloaded models
├── merged_model/              # Output directory for the merged AGI model by default
│   ├── config.json
│   ├── pytorch_model.bin (or .safetensors files)
│   ├── tokenizer_config.json
│   └── ...
├── merged_model_adapters/     # Default output directory for fine-tuned LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.bin (or .safetensors)
│   └── ...
├── agi_desktop_logs/          # Fallback for plain text session logs if Desktop not found
│   └── AGI_Terminal_Log_YYYYMMDD_HHMMSS.txt
├── setup_agi_terminal.py      # Main setup script
├── download_models.sh         # Script to download base models
├── merge_config.yml           # Configuration for mergekit
├── interactive_agi.py         # Main script to run the AGI terminal
├── adaptive_train.py          # Script for fine-tuning the merged model
├── README.md                  # This file
└── .gitignore                 # Git ignore file
```

---

This comprehensive `README.md` should provide users with the necessary information to set up and use the AGI Terminal. Remember to replace `<repository_url>` and `<repository_directory_name>` with actual values when users clone the project.
