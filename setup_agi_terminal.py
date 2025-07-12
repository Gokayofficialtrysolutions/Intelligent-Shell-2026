#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import getpass
import shutil
from pathlib import Path

# --- Script Configuration ---
MIN_PYTHON_VERSION = (3, 8)
# List of Python packages required by the AGI Terminal and its setup.
REQUIRED_PYTHON_PACKAGES = [
    # "huggingface_hub", # No longer directly needed for login by this script
    "mergekit",
    "torch", # Consider adding torchvision, torchaudio depending on torch install method
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes", # May have OS-specific build issues if not from wheel
    "sentencepiece",
    "scipy",
    "rich",
    "psutil",
    "toml"
]

# --- Utility Functions ---
def print_notice(message):
    """Prints a noticeable header message."""
    print(f"\n--- {message} ---\n")

def print_success(message):
    """Prints a success message."""
    print(f"[SUCCESS] {message}")

def print_warning(message):
    """Prints a warning message."""
    print(f"[WARNING] {message}")

def print_error(message):
    """Prints an error message and exits the script."""
    print(f"[ERROR] {message}")
    sys.exit(1)

def run_command(command, capture_output=False, text=True, check=True, shell=False, sensitive_output=False):
    """
    Helper to run shell commands.
    """
    cmd_str = ' '.join(command) if isinstance(command, list) else command
    print(f"Executing: {cmd_str}")
    try:
        process = subprocess.run(command, capture_output=capture_output, text=text, check=check, shell=shell)
        return process
    except subprocess.CalledProcessError as e:
        err_msg = f"Command failed: {cmd_str}\nReturn code: {e.returncode}"
        if capture_output and not sensitive_output:
            if e.stdout:
                err_msg += f"\nStdout:\n{e.stdout.strip()}"
            if e.stderr:
                err_msg += f"\nStderr:\n{e.stderr.strip()}"
        print_error(err_msg)
    except FileNotFoundError:
        print_error(f"Command not found: {command[0]}. Please ensure it's installed and in your PATH.")
        raise

# --- Prerequisite Checks ---
def check_python_version():
    """Checks if the current Python version meets the minimum requirement."""
    print_notice("Checking Python version...")
    if sys.version_info < MIN_PYTHON_VERSION:
        print_error(
            f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required. "
            f"You are using {sys.version_info.major}.{sys.version_info.minor}."
        )
    print_success(f"Python version {sys.version_info.major}.{sys.version_info.minor} is sufficient.")

def check_pip():
    """Checks if pip is available and runnable."""
    print_notice("Checking for pip...")
    try:
        run_command([sys.executable, "-m", "pip", "--version"], capture_output=True)
        print_success("pip is available.")
    except subprocess.CalledProcessError:
        print_error("pip is not available or not runnable. Please ensure pip is installed and configured for your Python environment.")
    except FileNotFoundError:
        print_error("Python executable not found to run pip. Please check your Python installation.")

def check_git_and_lfs():
    """Checks for git and git-lfs installations."""
    print_notice("Checking for git and git-lfs...")
    # Check for git
    if not shutil.which("git"):
        print_error("git is not found. Please install git and ensure it's in your system's PATH.")
    try:
        run_command(["git", "--version"], capture_output=True)
        print_success("git is available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("git is not found or not executable. Please install git and ensure it's in your system's PATH.")

    # Check for git-lfs
    if not shutil.which("git-lfs"):
        print_error(
            "git-lfs is not found. This is required for downloading large model files.\n"
            "Please install git-lfs and ensure it's in your system's PATH.\n"
            "Installation instructions:\n"
            "  - Debian/Ubuntu: sudo apt-get install git-lfs\n"
            "  - Fedora: sudo dnf install git-lfs\n"
            "  - macOS (Homebrew): brew install git-lfs\n"
            "  - Windows: Download from https://git-lfs.github.com/\n"
            "After installation, run 'git lfs install' in your terminal to initialize it globally."
        )
    try:
        run_command(["git", "lfs", "--version"], capture_output=True)
        print_success("git-lfs is available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("git-lfs is installed but not runnable or found in PATH. Please ensure git-lfs is correctly installed and configured.")

def check_virtual_environment():
    """Checks if the script is running in a Python virtual environment and warns if not."""
    print_notice("Checking for Python virtual environment...")
    in_virtual_env = hasattr(sys, 'real_prefix') or \
                     (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or \
                     (os.environ.get("VIRTUAL_ENV") is not None)

    if not in_virtual_env:
        print_warning(
            "You are not running in a Python virtual environment (e.g., venv, conda).\n"
            "It is STRONGLY recommended to use a virtual environment to manage dependencies\n"
            "and avoid conflicts with other Python projects or system packages."
        )
        if not sys.stdin.isatty():
            print_warning("Non-interactive session detected. Assuming you want to proceed without a virtual environment if this warning appears.")
        elif input("Do you want to continue setup without a virtual environment? (yes/NO): ").lower() != "yes":
            print_error(
                "Setup aborted by user. Please create and activate a Python virtual environment, "
                "then re-run this script."
            )
        else:
            print_warning("Continuing setup without a virtual environment as per user confirmation. This is not recommended for most users.")
    else:
        print_success("Running in a Python virtual environment.")

def get_user_confirmation(prompt_message, default_to_yes_in_non_interactive=True):
    """
    Handles user confirmation prompts.
    """
    if not sys.stdin.isatty() and default_to_yes_in_non_interactive:
        print_warning(f"Non-interactive session: Defaulting to YES for: '{prompt_message}'")
        return True

    while True:
        response = input(f"{prompt_message} (yes/NO): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n", ""]:
            return False
        else:
            print("Invalid input. Please answer 'yes' or 'no'.")

# --- Dependency Installation ---
def install_python_packages():
    """Installs required Python packages using pip."""
    print_notice(f"Installing required Python packages: {', '.join(REQUIRED_PYTHON_PACKAGES)}")

    print_warning(
        "PyTorch (torch) installation can be complex depending on your system (CPU/GPU, CUDA version).\n"
        "This script will attempt a standard pip install for 'torch', 'torchvision', and 'torchaudio'.\n"
        "For specific CUDA versions or CPU-only builds to save space/time, you might need to install\n"
        "PyTorch manually first using the correct command from https://pytorch.org/get-started/locally/"
    )
    print_warning(
        "PyTorch and its dependencies can consume significant disk space (many GBs).\n"
        "Ensure you have at least 20-30GB of free disk space before proceeding with this step."
    )

    if not get_user_confirmation(
        f"Proceed with installing PyTorch and other packages ({len(REQUIRED_PYTHON_PACKAGES) + 2} total including torch extras)?",
        default_to_yes_in_non_interactive=True
    ):
        print_error("Package installation aborted by user.")

    pytorch_packages = ["torch", "torchvision", "torchaudio"]
    try:
        print_notice(f"Attempting to install {', '.join(pytorch_packages)}...")
        run_command([sys.executable, "-m", "pip", "install"] + pytorch_packages)
        print_success(f"{', '.join(pytorch_packages)} installation command executed.")
        print_warning("Please check the output above for actual success or failure of PyTorch installation.")
    except Exception as e:
        print_warning(
            f"Installation of {', '.join(pytorch_packages)} failed or had issues: {e}\n"
            "This might be due to disk space, network issues, or system compatibility.\n"
            "You can try installing a CPU-only version if you don't need GPU support, e.g.:\n"
            f"'{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'"
        )
        if not get_user_confirmation(
            "Attempt to continue installing other packages despite potential PyTorch issues?",
            default_to_yes_in_non_interactive=False
        ):
            print_error("Setup aborted due to PyTorch installation issues.")

    remaining_packages = [pkg for pkg in REQUIRED_PYTHON_PACKAGES if pkg.lower() != "torch"]
    if remaining_packages:
        print_notice(f"Attempting to install remaining packages: {', '.join(remaining_packages)}...")
        try:
            run_command([sys.executable, "-m", "pip", "install"] + remaining_packages)
            print_success(f"Installation command for remaining packages executed.")
            print_warning("Please check the output above for actual success or failure of these packages.")
        except Exception as e:
            print_error(
                f"Failed to install some of the remaining Python packages: {', '.join(remaining_packages)}.\n"
                f"Error: {e}\n"
                "Please try installing them manually or check your pip and network configuration."
            )

    print_success("Python package installation phase complete (commands executed).")
    print_warning("It is crucial to verify the output above to ensure all packages installed correctly, especially PyTorch.")

def main():
    """
    Main function to orchestrate the setup process.
    """
    print_notice("AGI Terminal Setup Script - Automated Environment Configuration")
    print("This script will guide you through setting up the AGI Terminal environment.")
    print("It will perform the following main steps:")
    print("  1. Check for essential prerequisites (Python, pip, git, git-lfs).")
    print("  2. Install required Python packages (including PyTorch, Transformers, mergekit).")
    print("  3. Create necessary configuration files and helper scripts.")
    print("  4. Download selected open-source AI models using 'git lfs'.")
    print("  5. Merge these models using 'mergekit'.")
    print("Please ensure you have a stable internet connection and sufficient disk space (100GB+ recommended for models).")
    print("-" * 70)

    # --- Phase A: Prerequisite Checks & Initial Setup ---
    print_notice("Phase A: Verifying Prerequisites and Core Python Environment")
    check_python_version()
    check_pip()
    check_git_and_lfs()
    check_virtual_environment()

    if not get_user_confirmation(
        "Proceed with Python package installations (this includes PyTorch and can take significant time and disk space)?",
        default_to_yes_in_non_interactive=True
    ):
        print_error("Setup aborted by user before Python package installation.")
    install_python_packages()
    print_success("Phase A: Prerequisites and Core Python Environment setup commands executed.")
    print_warning("Review output above to ensure packages like PyTorch installed correctly before proceeding.")
    if not get_user_confirmation("Continue to Phase B (Script Generation, Model Download & Merge)?", default_to_yes_in_non_interactive=True):
        print_error("Setup aborted by user before Phase B.")

    # --- Phase B: Script/Config Generation & Model Operations ---
    print_notice("Phase B: Generating Auxiliary Scripts, Downloading & Merging Models")
    create_auxiliary_scripts_and_configs()

    print_warning(
        "The next step is downloading AI models. This is the most time-consuming and disk-space-intensive part.\n"
        "Models can range from a few GBs to tens of GBs EACH. Total space can easily exceed 100GB.\n"
        "Ensure your internet connection is stable and you have monitored your disk space."
    )
    if not get_user_confirmation("Proceed with downloading models?", default_to_yes_in_non_interactive=True):
        print_error("Model download aborted by user.")
    run_model_download()

    print_warning(
        "Model merging is CPU and RAM intensive. It can also take a long time.\n"
        "Ensure your system has adequate resources (32GB+ RAM highly recommended, more for larger merges)."
    )
    if not get_user_confirmation("Proceed with merging models?", default_to_yes_in_non_interactive=True):
        print_error("Model merging aborted by user.")
    run_model_merge()

    print_success("Phase B: Auxiliary Scripts, Model Download & Merge commands executed.")
    print_warning("Review output above to ensure models downloaded and merged correctly.")
    if not get_user_confirmation("Continue to Phase C (Final Verification)?", default_to_yes_in_non_interactive=True):
        print_error("Setup aborted by user before Phase C.")

    # --- Phase C: Final Steps & Verification ---
    print_notice("Phase C: Final Setup Verification & Next Steps")
    cache_dir = Path("./.agi_terminal_cache")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Ensured AGI Terminal cache directory exists: {cache_dir.resolve()}")
    except OSError as e:
        print_warning(f"Could not create AGI Terminal cache directory {cache_dir.resolve()}: {e}")

    scripts_to_verify = ["interactive_agi.py", "adaptive_train.py"]
    for script_name in scripts_to_verify:
        if not os.path.exists(script_name):
            print_warning(f"Main script '{script_name}' not found in the current directory. It is expected to be part of the repository.")

    print_notice("AGI Terminal Setup Script Finished!")
    print_success("All planned setup commands have been executed.")
    print("-" * 70)
    print("IMPORTANT: Review all output above for any errors or warnings, especially during package installation, model download, and model merge phases.")
    print("-" * 70)
    print("What's next?")
    print("1. Merged Model: If successful, your merged model is located in the './merged_model/' directory.")
    print("2. Run AGI Terminal: To start interacting with your AGI, run:")
    print("   python interactive_agi.py")
    print("3. Interaction Logs: Your interactions will be logged in the './interaction_logs/' directory.")
    print("4. Fine-tuning: To fine-tune the model on these interactions (requires GPU and further setup):")
    print("   python adaptive_train.py --help (to see options)")
    print("\nIf you encountered errors not automatically handled:")
    print("  - Check your internet connection, disk space, and system resources (RAM).")
    print("  - Consult the project's README.md for troubleshooting tips (once created).")
    print("  - You may need to perform some steps manually or re-run this script after resolving issues.")
    print("-" * 70)

# --- Auxiliary Script and Config Content ---
MERGE_CONFIG_YML_CONTENT = """# merge_config.yml
# Configuration for mergekit
# Models are expected to be in ./models/<model_key>/

slices:
  - sources:
      - model: ./models/mistral7b_v03 # mistralai/Mistral-7B-Instruct-v0.3 (Base for this merge)
      - model: ./models/olmo7b_instruct # allenai/OLMo-7B-Instruct (Replaced Llama3)
      - model: ./models/deepseek_coder_v2_lite # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
      - model: ./models/starcoder2_7b # bigcode/starcoder2-7b
      - model: ./models/phi3_mini_instruct # microsoft/Phi-3-mini-4k-instruct
merge_method: linear
base_model: ./models/mistral7b_v03
parameters: {}
dtype: float16
"""

def write_file(filepath, content):
    """Writes content to a file, overwriting it if it exists."""
    print(f"Creating/overwriting file: {filepath}")
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        print_success(f"File {filepath} written successfully.")
    except IOError as e:
        print_error(f"Failed to write file {filepath}: {e}")
        raise

def make_executable(filepath):
    """Makes a file executable (adds +x permission)."""
    print(f"Making file executable: {filepath}")
    try:
        os.chmod(filepath, 0o755)
        print_success(f"File {filepath} is now executable.")
    except OSError as e:
        print_error(f"Failed to make {filepath} executable: {e}")
        raise

def create_auxiliary_scripts_and_configs():
    """Creates/updates helper scripts and configuration files."""
    print_notice("Creating auxiliary scripts and configuration files...")
    if os.path.exists("download_models.sh"):
        make_executable("download_models.sh")
    else:
        print_warning("download_models.sh not found. This script is expected to exist in the repository.")

    write_file("merge_config.yml", MERGE_CONFIG_YML_CONTENT)
    # The `train_on_interaction.sh` script is deprecated and its generation is removed.
    print_success("Auxiliary scripts and configuration files generated/updated successfully.")

def run_model_download():
    """
    Executes the 'download_models.sh' script to download AI models.
    """
    script_path = "./download_models.sh"
    print_notice(f"Executing model download script: {script_path}")

    if not os.path.exists(script_path):
        print_error(f"Model download script '{script_path}' not found. Please ensure it's in the project root.")
        return

    try:
        run_command(["bash", script_path], capture_output=False)
        print_success(f"Model download script '{script_path}' finished execution.")
        print_warning("Please review the output from the download script carefully for any errors or incomplete downloads.")
    except subprocess.CalledProcessError as e:
        print_error(
            f"The model download script '{script_path}' failed.\n"
            "Common reasons include: \n"
            "  - Insufficient disk space.\n"
            "  - Network connectivity issues (interrupted downloads).\n"
            "  - git or git-lfs not installed or configured correctly.\n"
            "  - Incorrect model repository URLs or permissions (though these are public).\n"
            "Please check the script's output above for specific error messages from git/git-lfs."
        )
    except FileNotFoundError:
        print_error(f"'bash' command not found. Cannot execute '{script_path}'. Please ensure bash is installed and in PATH.")

def run_model_merge():
    """
    Executes the 'mergekit-yaml' command to merge downloaded models.
    """
    print_notice("Attempting to merge downloaded models using mergekit...")

    mergekit_executable = "mergekit-yaml"
    config_file = "merge_config.yml"
    output_directory = "./merged_model"

    if not shutil.which(mergekit_executable):
        print_error(
            f"'{mergekit_executable}' command not found. This is part of the 'mergekit' Python package.\n"
            "Ensure 'mergekit' was installed correctly in your Python environment.\n"
            "You might need to ensure the Python scripts directory (e.g., ~/.local/bin or venv/bin) is in your system's PATH."
        )
        return

    if not os.path.exists(config_file):
        print_error(f"Merge configuration file '{config_file}' not found. This file should be created by this setup script.")
        return

    print_warning(
        "Mergekit may create temporary files that also consume significant disk space during the merge process.\n"
        "Ensure you have ample free space beyond the storage for the models themselves."
    )

    merge_command_args = [
        mergekit_executable,
        config_file,
        output_directory,
        "--out-shard-size", "2B",
        "--allow-crimes",
        "--lazy-unpickle",
    ]

    try:
        run_command(merge_command_args, capture_output=False)
        print_success(f"Model merge process finished. If successful, the merged model is in '{output_directory}'.")
        print_warning("Review the output from mergekit carefully for any errors or warnings.")
    except subprocess.CalledProcessError as e:
        print_error(
            f"Model merge process using '{mergekit_executable}' failed.\n"
            "Common reasons include:\n"
            "  - Insufficient RAM (merging is memory-intensive).\n"
            "  - Insufficient disk space (for merged model and temporary files).\n"
            "  - Errors in 'merge_config.yml' (e.g., incorrect model paths, incompatible models specified).\n"
            "  - Models not downloaded completely or corrupted.\n"
            "  - Compatibility issues between models if '--allow-crimes' is not sufficient or appropriate.\n"
            "Please check the detailed error messages from mergekit in the output above."
        )
    except FileNotFoundError:
        print_error(f"'{mergekit_executable}' was not found by subprocess despite initial check. Ensure it's executable and in PATH.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code != 0:
            print("\nSetup process terminated.")
        else:
            print("\nSetup process finished or exited gracefully.")
    except KeyboardInterrupt:
        print_error("\nSetup process interrupted by user (Ctrl+C).")
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        print_error("Setup failed.")
