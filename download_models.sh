#!/bin/bash

# Script to download models from Hugging Face Hub
# Requires huggingface-cli to be installed and logged in:
# pip install huggingface_hub
# huggingface-cli login

# Create a directory for models if it doesn't exist
mkdir -p models

# Define models to download (repository_id and optional revision/branch)
# Format: "user/model_name[:revision]"
# Using smaller versions for some to manage disk space and memory for merging
# Llama 3 is gated, you must request access on its Hugging Face page first.
declare -A models_to_download=(
  ["llama3"]="meta-llama/Meta-Llama-3-8B-Instruct" # Retaining Llama 3 as it wasn't in the user's update list
  ["mistral7b_v03"]="mistralai/Mistral-7B-Instruct-v0.3" # Updated from v0.1
  ["deepseek_coder_v2_lite"]="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" # Updated from 6.7b-instruct, this is 16B
  ["bloom"]="bigscience/bloom-1b7" # Using a smaller Bloom model, retained
  ["gpt_neox"]="EleutherAI/gpt-neox-20b" # This is very large (around 40GB), retained
  ["gpt4all_j"]="nomic-ai/gpt4all-j" # Base model, not instruct. Apache 2.0, retained
  ["starcoder2_7b"]="bigcode/starcoder2-7b" # Updated from Starcoder2 3B
  ["phi4_mini_instruct"]="microsoft/Phi-4-mini-instruct" # Updated from microsoft/phi-2, this is 4B
)

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null
then
    echo "huggingface-cli could not be found. Please install it first:"
    echo "pip install huggingface_hub"
    echo "Then login:"
    echo "huggingface-cli login"
    exit 1
fi

# Check if logged in to huggingface-cli
# Attempt to get user info, redirect stderr to null to avoid printing token if successful
if ! huggingface-cli whoami > /dev/null 2>&1; then
  echo "You are not logged in to Hugging Face CLI. Please log in first:"
  echo "huggingface-cli login"
  echo "Note: For meta-llama/Meta-Llama-3-8B-Instruct, you also need to request access on its Hugging Face model card."
  exit 1
fi

echo "Starting model downloads. This may take a very long time and significant disk space."
echo "Estimated disk space for selected models could exceed 100GB (GPT-NeoX 20B alone is ~40GB)."
echo "Ensure you have sufficient disk space and a stable internet connection."
echo "Meta-Llama-3-8B-Instruct requires approved access from Meta on Hugging Face."
read -p "Do you want to proceed with downloading all models? (y/N): " confirm_download

if [[ "$confirm_download" != [yY] && "$confirm_download" != [yY][eE][sS] ]]; then
  echo "Download cancelled by user."
  exit 0
fi


for model_key in "${!models_to_download[@]}"; do
  repo_id="${models_to_download[$model_key]}"
  target_dir="./models/${model_key}"

  echo "----------------------------------------------------------------------"
  echo "Downloading $repo_id to $target_dir..."
  echo "----------------------------------------------------------------------"

  if [ -d "$target_dir" ]; then
    # Check if directory is empty or contains only a .git folder (from failed git clone)
    # A more robust check might be needed, e.g., for specific model files.
    if [ -z "$(ls -A "$target_dir" | grep -v '.git')" ]; then
        echo "Directory $target_dir exists but is empty. Attempting download."
    else
        echo "Directory $target_dir already exists and appears to contain files. Skipping download for $repo_id."
        echo "If you want to re-download, please delete the directory $target_dir first."
        continue
    fi
  fi

  # Using huggingface-cli download which is generally better for large models
  # It handles LFS files correctly and can resume.
  # The --local-dir-use-symlinks False is important for Mergekit compatibility later
  # as it might have issues with symlinks for model files.
  huggingface-cli download "$repo_id" \
    --local-dir "$target_dir" \
    --local-dir-use-symlinks False \
    --resume-download \
    --quiet # Use --verbose for more detailed output

  if [ $? -eq 0 ]; then
    echo "Successfully downloaded $repo_id to $target_dir."
  else
    echo "Error downloading $repo_id. Please check the error messages above."
    echo "For Llama 3, ensure you have requested and been granted access on Hugging Face."
    echo "For other models, check if the repository ID is correct and public, or if you need specific access."
  fi
done

echo "----------------------------------------------------------------------"
echo "All model download attempts finished."
echo "Please check each ./models/<model_name> directory for completion."
echo "Note: GPT-NeoX 20B is particularly large (approx 40GB)."
echo "Total disk space used will be significant."
echo "----------------------------------------------------------------------"

# Suggestion for user to check disk space
echo "Current disk usage for ./models directory:"
du -sh ./models

# Make the script executable
chmod +x download_models.sh

echo "Run './download_models.sh' to start downloading the models."
