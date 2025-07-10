#!/bin/bash
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
  fi
done

echo "----------------------------------------------------------------------"
echo "All model download attempts finished."
echo "Please check each ./models/<model_name> directory for completion."
echo "Current disk usage for ./models directory:"
du -sh ./models
echo "----------------------------------------------------------------------"
