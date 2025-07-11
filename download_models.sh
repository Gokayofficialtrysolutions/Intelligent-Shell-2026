#!/bin/bash
# Script to download AI models from Hugging Face Hub using git clone and git lfs.
# This script is intended to be called by setup_agi_terminal.py or run manually.

# --- Script Behavior ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Return value of a pipeline is the status of the last command to exit with a non-zero status,
# or zero if no command exited with a non-zero status.
set -o pipefail

# --- Welcome and Instructions ---
echo "======================================================================"
echo "AI Model Download Script for AGI Terminal"
echo "======================================================================"
echo "This script will download multiple large language models using git and git-lfs."
echo "IMPORTANT:"
echo "  - DISK SPACE: Ensure you have SIGNIFICANT free disk space (100GB+ is common for multiple models)."
echo "  - TIME: Downloading can take a very long time, depending on your internet speed and model sizes."
echo "  - STABILITY: A stable internet connection is crucial. Interrupted LFS downloads can be problematic."
echo "  - GIT LFS: This script requires git and git-lfs to be installed and configured."
echo "======================================================================"
echo

# --- Prerequisite Checks ---
# Check for git
if ! command -v git &> /dev/null; then
    echo "[ERROR] git command not found. Please install git and ensure it's in your PATH."
    exit 1
fi

# Check for git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "[ERROR] git-lfs command not found. This is essential for downloading model files."
    echo "Please install git-lfs. Common methods:"
    echo "  - Debian/Ubuntu: sudo apt-get update && sudo apt-get install git-lfs"
    echo "  - Fedora: sudo dnf install git-lfs"
    echo "  - macOS (Homebrew): brew install git-lfs"
    echo "  - Windows: Download from https://git-lfs.github.com/"
    echo "After installing, run 'git lfs install' once in your terminal to initialize it globally."
    exit 1
fi

# Initialize git-lfs for the current user (makes sure it's ready)
# --skip-repo is used because we are not in a specific repo context yet.
# --system can be used if admin rights are available and system-wide init is desired.
echo "[INFO] Ensuring git-lfs is initialized for the user..."
git lfs install --skip-repo

# Check for GIT_LFS_SKIP_SMUDGE environment variable
if [ "${GIT_LFS_SKIP_SMUDGE:-0}" == "1" ]; then
    echo "[WARNING] The environment variable GIT_LFS_SKIP_SMUDGE is set to 1."
    echo "This will prevent LFS files from being downloaded automatically during 'git clone'."
    echo "This script relies on LFS files being downloaded. Please unset this variable or ensure LFS files are pulled manually."
    # Consider exiting or prompting user here if this is critical. For now, a warning.
fi

# --- Model Definitions ---
# Associative array mapping a local directory key to the Hugging Face model repository URL.
# These models are chosen for their capabilities and open licenses (or replacements for gated ones).
declare -A MODELS_TO_DOWNLOAD=(
  # Primary models for the AGI Terminal merge:
  ["mistral7b_v03"]="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"    # Base model for merge
  ["olmo7b_instruct"]="https://huggingface.co/allenai/OLMo-7B-Instruct"            # Replacement for Llama3
  ["deepseek_coder_v2_lite"]="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" # Coding focused
  ["starcoder2_7b"]="https://huggingface.co/bigcode/starcoder2-7b"                  # Coding focused
  ["phi3_mini_instruct"]="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct" # Capable small model

  # Optional additional models (can be uncommented or added to merge_config.yml later):
  # ["tinyllama_chat"]="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Small chat model
  # ["bloom"]="https://huggingface.co/bigscience/bloom-1b7"                       # Multilingual older model
  # ["gpt_neox"]="https://huggingface.co/EleutherAI/gpt-neox-20b"                 # Large, powerful older model (very resource intensive)
)

# Target directory for all model downloads
MODELS_BASE_DIR="./models"
mkdir -p "${MODELS_BASE_DIR}" # Ensure base directory exists

echo "[INFO] Starting model download process. Models will be saved in '${MODELS_BASE_DIR}/<model_key>/'."

# --- Download Loop ---
for model_key in "${!MODELS_TO_DOWNLOAD[@]}"; do
  repo_url="${MODELS_TO_DOWNLOAD[$model_key]}"
  target_dir="${MODELS_BASE_DIR}/${model_key}" # e.g., ./models/olmo7b_instruct

  echo "----------------------------------------------------------------------"
  echo "[INFO] Processing model: ${model_key} (${repo_url})"
  echo "[INFO] Target directory: ${target_dir}"
  echo "----------------------------------------------------------------------"

  if [ -d "${target_dir}/.git" ]; then
    echo "[INFO] Git repository already exists in ${target_dir}."
    echo "[INFO] Checking LFS status and attempting to pull any missing files..."
    # Navigate to directory, try to fetch LFS files, then navigate back.
    # This helps if a previous download was interrupted.
    ( # Start a subshell to avoid cd affecting the main script's PWD
      cd "${target_dir}" || { echo "[ERROR] Failed to cd into ${target_dir}. Skipping LFS pull."; exit 1; }
      echo "[INFO] Current directory: $(pwd)"
      echo "[INFO] Fetching LFS objects (git lfs fetch)..."
      git lfs fetch origin # Fetch LFS objects from origin
      echo "[INFO] Checking out LFS files (git lfs checkout)..."
      git lfs checkout     # Ensure all fetched LFS files are checked out to the working directory
      echo "[INFO] LFS pull attempt complete for ${target_dir}."
    )
    echo "[SUCCESS] Model ${model_key} likely up-to-date. If you suspect issues, remove the directory and re-run."
    continue # Skip to the next model
  elif [ -d "${target_dir}" ]; then
    echo "[WARNING] Directory ${target_dir} exists but is not a git repository or seems incomplete."
    echo "[WARNING] Removing existing directory ${target_dir} to ensure a clean clone."
    rm -rf "${target_dir}"
  fi

  echo "[INFO] Cloning repository structure for ${model_key} from ${repo_url}..."
  # GIT_TERMINAL_PROMPT=0 prevents git from prompting for credentials if repo is private (should not happen for these public ones)
  # --depth 1 clones only the latest commit history, making the initial clone much faster for large repos.
  # LFS files are versioned independently, so --depth 1 doesn't affect which LFS files are downloaded.
  if GIT_TERMINAL_PROMPT=0 git clone --depth 1 "${repo_url}" "${target_dir}"; then
    echo "[SUCCESS] Cloned repository structure for ${model_key} to ${target_dir}."

    echo "[INFO] Downloading LFS files for ${model_key} in ${target_dir}..."
    ( # Start a subshell
      cd "${target_dir}" || { echo "[ERROR] Failed to cd into ${target_dir} for LFS pull. Model download incomplete."; exit 1; }
      # `git lfs pull` is a convenience command that runs `git lfs fetch` followed by `git lfs checkout`.
      if git lfs pull; then
        echo "[SUCCESS] Successfully downloaded LFS files for ${model_key}."
      else
        echo "[ERROR] 'git lfs pull' failed for ${model_key} in ${target_dir}."
        echo "This could be due to network issues, insufficient disk space, or LFS server problems."
        echo "You can try running 'cd ${target_dir} && git lfs pull' manually to see more detailed errors or resume."
        # Optionally, one might exit 1 here to stop the whole script if one model fails.
        # For now, it will continue to try other models.
      fi
    )
  else
    echo "[ERROR] Failed to clone repository ${repo_url} for model ${model_key}."
    echo "Please check the error messages above, your internet connection, and the repository URL."
    # Continue to the next model, but this one failed.
  fi
  echo # Add a blank line for readability before next model
done

# --- Completion Summary ---
echo "======================================================================"
echo "[INFO] All model download attempts finished."
echo "======================================================================"
echo "Please review the output above for any errors for individual models."
echo "Verify that each target directory in '${MODELS_BASE_DIR}/' contains the expected model files."
echo "Large files (LFS objects) should now be present."
echo
echo "Current disk usage for the '${MODELS_BASE_DIR}' directory:"
du -sh "${MODELS_BASE_DIR}"
echo "======================================================================"
echo "[SCRIPT COMPLETE]"
echo "======================================================================"
