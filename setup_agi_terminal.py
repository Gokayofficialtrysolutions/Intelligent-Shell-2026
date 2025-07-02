#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import getpass
import shutil

MIN_PYTHON_VERSION = (3, 8)
REQUIRED_PYTHON_PACKAGES = [
    "huggingface_hub",
    "mergekit",
    "torch", # Consider adding torchvision, torchaudio depending on torch install method
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes", # May have OS-specific build issues if not from wheel
    "sentencepiece",
    "scipy",
    "rich",
    "psutil"
]

# --- Utility Functions ---
def print_notice(message):
    print(f"\n--- {message} ---\n")

def print_success(message):
    print(f"[SUCCESS] {message}")

def print_warning(message):
    print(f"[WARNING] {message}")

def print_error(message):
    print(f"[ERROR] {message}")
    sys.exit(1)

def run_command(command, capture_output=False, text=True, check=True, shell=False):
    """Helper to run shell commands."""
    print(f"Executing: {' '.join(command) if isinstance(command, list) else command}")
    try:
        process = subprocess.run(command, capture_output=capture_output, text=text, check=check, shell=shell)
        return process
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {e}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        raise
    except FileNotFoundError:
        print_error(f"Command not found: {command[0]}. Please ensure it's installed and in your PATH.")
        raise

# --- Prerequisite Checks ---
def check_python_version():
    print_notice("Checking Python version...")
    if sys.version_info < MIN_PYTHON_VERSION:
        print_error(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required. You have {sys.version_info.major}.{sys.version_info.minor}.")
    print_success(f"Python version {sys.version_info.major}.{sys.version_info.minor} is sufficient.")

def check_pip():
    print_notice("Checking for pip...")
    try:
        run_command([sys.executable, "-m", "pip", "--version"], capture_output=True)
        print_success("pip is available.")
    except:
        print_error("pip is not available. Please ensure pip is installed for your Python environment.")

def check_git():
    print_notice("Checking for git...")
    if not shutil.which("git"):
        print_error("git is not found. Please install git and ensure it's in your PATH.")
    try:
        run_command(["git", "--version"], capture_output=True)
        print_success("git is available.")
    except:
        # This case should ideally be caught by shutil.which, but as a fallback:
        print_error("git is not found or not executable. Please install git and ensure it's in your PATH.")

def check_virtual_environment():
    print_notice("Checking for virtual environment...")
    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        print_warning("You are not running in a Python virtual environment (e.g., venv, conda).")
        print_warning("It is highly recommended to use a virtual environment to avoid conflicts with system packages.")
        if input("Do you want to continue without a virtual environment? (yes/NO): ").lower() != "yes":
            print_error("Setup aborted by user. Please create and activate a virtual environment and re-run this script.")
        else:
            print_warning("Continuing without a virtual environment as per user confirmation.")
    else:
        print_success("Running in a virtual environment.")


# --- Dependency Installation ---
def install_python_packages():
    print_notice(f"Installing required Python packages: {', '.join(REQUIRED_PYTHON_PACKAGES)}")
    # Special handling for torch - recommend user installs manually if issues arise, or use --index-url for CUDA
    print_warning("PyTorch installation can be complex depending on your system (CPU/GPU, CUDA version).")
    print_warning("This script will attempt a standard pip install for torch.")
    print_warning("If you have a specific CUDA version, you might need to install PyTorch manually first using the correct command from https://pytorch.org/")

    # Confirm before installing torch
    if input(f"Proceed with installing PyTorch and other packages ({', '.join(REQUIRED_PYTHON_PACKAGES)})? (YES/no): ").lower() == "no":
        print_error("Package installation aborted by user.")

    # Install torch separately first, then the rest.
    # This is a generic torch install. For CUDA, user might need specific index URL.
    try:
        run_command([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        print_success("PyTorch (cpu/generic cuda) installation attempted.")
    except Exception as e:
        print_warning(f"Standard PyTorch installation failed or had issues: {e}")
        print_warning("This might be okay if you have a compatible PyTorch version already installed, or you may need to install it manually with specific CUDA options.")
        if input("Continue with other packages despite potential PyTorch issues? (yes/NO): ").lower() != "yes":
            print_error("Setup aborted due to PyTorch installation issue.")

    # Install remaining packages
    remaining_packages = [p for p in REQUIRED_PYTHON_PACKAGES if p != "torch"]
    if remaining_packages:
        try:
            run_command([sys.executable, "-m", "pip", "install"] + remaining_packages)
            print_success(f"Installed: {', '.join(remaining_packages)}.")
        except Exception as e:
            print_error(f"Failed to install some Python packages: {e}")
            print_error("Please try installing them manually or check your pip configuration.")
    print_success("Python package installation phase complete.")

# --- Hugging Face Login ---
def huggingface_login():
    print_notice("Hugging Face Hub Login")

    # Check if already logged in
    try:
        process = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True, check=False)
        if process.returncode == 0 and "username" in process.stdout.lower(): # Simple check
            print_success(f"Already logged in to Hugging Face as: {process.stdout.strip().splitlines()[0]}")
            if input("Do you want to re-login or use current login? (type 'relogin' or press Enter to use current): ").lower() != 'relogin':
                return
    except FileNotFoundError:
        print_warning("`huggingface-cli` not found yet (expected if installing it now). Will proceed to login attempt.")
    except Exception as e:
        print_warning(f"Could not check current Hugging Face login status: {e}")


    print("You need to log in to Hugging Face Hub to download models.")
    print("You can get your Hugging Face API token from https://huggingface.co/settings/tokens")
    hf_token = getpass.getpass("Enter your Hugging Face API token (will not be displayed): ")

    if not hf_token:
        print_error("No token provided. Cannot proceed with Hugging Face login.")

    try:
        # Try non-interactive login
        run_command(["huggingface-cli", "login", "--token", hf_token], capture_output=True)
        print_success("Successfully logged in to Hugging Face Hub using the provided token.")
    except Exception as e:
        print_warning(f"Automated Hugging Face login failed: {e}")
        print_warning("Please try logging in manually by running: huggingface-cli login")
        print_warning("You can paste the token when prompted.")
        if input("Have you logged in manually in another terminal? (yes/no): ").lower() != "yes":
            print_error("Hugging Face login is required to download models.")

def main():
    print_notice("AGI Terminal Setup Script")
    print("This script will guide you through setting up the AGI Terminal environment.")
    print("It will check prerequisites, install dependencies, and download models.")

    # --- Phase A: Prerequisite Checks & Initial Setup ---
    check_python_version()
    check_pip()
    check_git()
    check_virtual_environment() # Warns and asks user, doesn't create venv itself

    if input("Proceed with Python package installations? (YES/no): ").lower() == "no":
        print_error("Setup aborted by user before package installation.")

    install_python_packages()
    huggingface_login()

    print_success("Phase A: Core Structure & Prerequisites completed.")

    # --- Phase B: Script/Config Generation & Model Operations ---
    print_notice("Starting Phase B: Script/Config Generation & Model Operations")

    create_auxiliary_scripts_and_configs()

    if input("Proceed with downloading models? This may take a long time and significant disk space. (YES/no): ").lower() == "no":
        print_error("Model download aborted by user.")
    run_model_download()

    if input("Proceed with merging models? This is CPU and RAM intensive and can take a long time. (YES/no): ").lower() == "no":
        print_error("Model merging aborted by user.")
    run_model_merge()

    print_success("Phase B: Script/Config Generation & Model Operations completed.")

    # --- Phase C: Final Steps & Verification ---
    print_notice("Starting Phase C: Final Setup Verification")

    # Create cache directory for history etc.
    cache_dir = Path("./.agi_terminal_cache")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Ensured cache directory exists: {cache_dir.resolve()}")
    except OSError as e:
        print_warning(f"Could not create cache directory {cache_dir.resolve()}: {e}")

    # Ensure interactive_agi.py and adaptive_train.py are present (they should be in the repo)
    # and make adaptive_train.py executable as well if it exists.
    scripts_to_check_exec = ["interactive_agi.py", "adaptive_train.py"]
    for script_name in scripts_to_check_exec:
        if os.path.exists(script_name):
            make_executable(script_name) # adaptive_train.py might also be run directly
        else:
            print_warning(f"{script_name} not found in the current directory. This script is expected to be part of the repository.")

    # Placeholder for a mock test if we add a CLI flag to interactive_agi.py
    # For now, just print instructions.
    print_notice("Setup Complete!")
    print_success("The AGI Terminal environment setup script has finished.")
    print("What's next?")
    print("1. The merged model is located in: ./merged_model/")
    print("2. To start interacting with the AGI, run:")
    print("   python interactive_agi.py")
    print("3. Your interactions will be logged in ./interaction_logs/")
    print("4. To fine-tune the model on these interactions, run:")
    print("   python adaptive_train.py")
    print("   (Ensure you have reviewed its --help options and prerequisites, especially GPU availability).")
    print("\nIf you encountered any errors, please review the messages above and address them.")
    print("Consider re-running this setup script if issues were resolved, or performing steps manually if needed.")


# --- Auxiliary Script and Config Content ---

DOWNLOAD_MODELS_SH_CONTENT = """#!/bin/bash
# Script to download models from Hugging Face Hub
# Called by setup_agi_terminal.py

mkdir -p models

declare -A models_to_download=(
  ["llama3"]="meta-llama/Meta-Llama-3-8B-Instruct"
  ["mistral7b_v03"]="mistralai/Mistral-7B-Instruct-v0.3"
  ["deepseek_coder_v2_lite"]="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" # 16B
  ["bloom"]="bigscience/bloom-1b7"
  ["gpt_neox"]="EleutherAI/gpt-neox-20b" # ~40GB
  ["gpt4all_j"]="nomic-ai/gpt4all-j"
  ["starcoder2_7b"]="bigcode/starcoder2-7b"
  ["phi4_mini_instruct"]="microsoft/Phi-4-mini-instruct" # 4B
)

echo "Starting model downloads invoked by setup_agi_terminal.py..."
echo "This may take a very long time and significant disk space."
echo "Ensure you have sufficient disk space and a stable internet connection."
echo "Meta-Llama-3-8B-Instruct requires approved access from Meta on Hugging Face."
# Confirmation should be handled by the calling Python script.

for model_key in "${!models_to_download[@]}"; do
  repo_id="${models_to_download[$model_key]}"
  target_dir="./models/${model_key}"

  echo "----------------------------------------------------------------------"
  echo "Downloading $repo_id to $target_dir..."
  echo "----------------------------------------------------------------------"

  if [ -d "$target_dir" ] && [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
    echo "Directory $target_dir already exists and is not empty. Skipping download for $repo_id."
    echo "If you want to re-download, please delete the directory $target_dir first."
    continue
  fi
  mkdir -p "$target_dir" # Ensure target_dir exists even if empty check passed weirdly

  huggingface-cli download "$repo_id" \\
    --local-dir "$target_dir" \\
    --local-dir-use-symlinks False \\
    --resume-download \\
    --quiet # Use --verbose for more detailed output

  if [ $? -eq 0 ]; then
    echo "Successfully downloaded $repo_id to $target_dir."
  else
    echo "Error downloading $repo_id. Please check the error messages above."
    echo "For Llama 3, ensure you have requested and been granted access on Hugging Face."
  fi
done

echo "----------------------------------------------------------------------"
echo "All model download attempts finished."
echo "Please check each ./models/<model_name> directory for completion."
echo "Current disk usage for ./models directory:"
du -sh ./models
echo "----------------------------------------------------------------------"
"""

MERGE_CONFIG_YML_CONTENT = """# merge_config.yml
# Configuration for mergekit, created by setup_agi_terminal.py
# Based on latest model recommendations.

slices:
  - sources:
      - model: ./models/mistral7b_v03 # mistralai/Mistral-7B-Instruct-v0.3
      - model: ./models/deepseek_coder_v2_lite # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (16B)
      - model: ./models/starcoder2_7b # bigcode/starcoder2-7b
      - model: ./models/phi4_mini_instruct # microsoft/Phi-4-mini-instruct (4B)
merge_method: linear
base_model: ./models/mistral7b_v03 # Using Mistral-7B-Instruct-v0.3 as the base
parameters: {}
dtype: float16

# To include other downloaded models like llama3 (./models/llama3),
# bloom (./models/bloom), or gpt_neox (./models/gpt_neox),
# add them to the `sources` list above. Example:
#      - model: ./models/llama3
#      - model: ./models/gpt_neox
"""

TRAIN_ON_INTERACTION_SH_CONTENT = """#!/bin/bash
# Placeholder training script: train_on_interaction.sh
# Called by interactive_agi.py, created by setup_agi_terminal.py

USER_INPUT="$1"
MODEL_OUTPUT="$2"

mkdir -p ./interaction_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./interaction_logs/interaction_${TIMESTAMP}.log"

echo "Timestamp: $TIMESTAMP" >> "$LOG_FILE"
echo "User Input: $USER_INPUT" >> "$LOG_FILE"
echo "Model Output: $MODEL_OUTPUT" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

echo "[train_on_interaction.sh] Logged interaction to: $LOG_FILE"
exit 0
"""

def write_file(filepath, content):
    print(f"Creating/overwriting file: {filepath}")
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        print_success(f"File {filepath} written successfully.")
    except IOError as e:
        print_error(f"Failed to write file {filepath}: {e}")
        raise

def make_executable(filepath):
    print(f"Making file executable: {filepath}")
    try:
        os.chmod(filepath, 0o755)
        print_success(f"File {filepath} is now executable.")
    except OSError as e:
        print_error(f"Failed to make {filepath} executable: {e}")
        raise

def create_auxiliary_scripts_and_configs():
    print_notice("Creating auxiliary scripts and configuration files...")
    write_file("download_models.sh", DOWNLOAD_MODELS_SH_CONTENT)
    make_executable("download_models.sh")

    write_file("merge_config.yml", MERGE_CONFIG_YML_CONTENT)

    write_file("train_on_interaction.sh", TRAIN_ON_INTERACTION_SH_CONTENT)
    make_executable("train_on_interaction.sh")
    print_success("Auxiliary scripts and configs created.")

def run_model_download():
    print_notice("Running model download script (download_models.sh)...")
    print_warning("This will download multiple large language models and can take a very long time and significant disk space (potentially 100GB+).")
    print_warning("Ensure you have a stable internet connection and sufficient free disk space.")

    # The download script itself also prints warnings.
    # Confirmation is handled before calling this function in main().
    try:
        run_command(["bash", "./download_models.sh"], capture_output=False) # Stream output
        print_success("Model download script finished.")
    except Exception as e:
        print_error(f"Model download script failed: {e}")
        print_error("Please check the output from the script for specific error messages.")
        print_error("You may need to resolve issues (e.g., disk space, Hugging Face access for Llama3) and re-run the setup, or attempt manual downloads.")
        raise # Re-raise to stop the setup

def run_model_merge():
    print_notice("Running model merge process (mergekit)...")
    print_warning("This process is CPU and RAM intensive and can take a long time.")
    print_warning("Ensure your system has adequate resources (32GB+ RAM highly recommended, more for larger merges).")

    # Confirmation is handled before calling this function in main().
    merge_command = [
        "mergekit-yaml", "merge_config.yml", "./merged_model",
        "--out-shard-size", "2B", # From original README
        "--allow-crimes",         # From original README
        "--lazy-unpickle"         # From original README
    ]
    try:
        # Check if mergekit-yaml is available (it should be if pip install worked)
        if not shutil.which("mergekit-yaml"):
            print_error("`mergekit-yaml` command not found. Please ensure mergekit was installed correctly.")
            print_error("You might need to ensure the Python scripts directory (e.g., ~/.local/bin) is in your PATH.")
            raise FileNotFoundError("mergekit-yaml not found")

        run_command(merge_command, capture_output=False) # Stream output
        print_success("Model merge process finished. Merged model should be in './merged_model'.")
    except Exception as e:
        print_error(f"Model merge process failed: {e}")
        print_error("Please check the output from mergekit for specific error messages.")
        print_error("Common issues include insufficient RAM, incorrect model paths in merge_config.yml (though this script generates it), or model compatibility problems.")
        raise # Re-raise to stop the setup


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code != 0: # Non-zero exit code indicates an error message was already printed
            print("\nSetup process terminated.")
        else:
            print("\nSetup process finished or exited gracefully.")
    except KeyboardInterrupt:
        print_error("\nSetup process interrupted by user (Ctrl+C).")
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        print_error("Setup failed.")
