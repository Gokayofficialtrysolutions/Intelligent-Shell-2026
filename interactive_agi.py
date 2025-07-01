# interactive_agi.py

import subprocess
import os
import sys
from pathlib import Path

# Placeholder for loading the actual merged model
# In a real scenario, this would involve:
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

class AGIPPlaceholder:
    def __init__(self, model_path_str="./merged_model"):
        self.model_path = Path(model_path_str)
        self.is_model_ready = False # Will be True if placeholder conditions met or actual model loads
        self.tokenizer = None
        self.model = None

        # Check if the merged_model directory exists and seems populated
        # This is a basic check. A real check would ensure all necessary files are present.
        if self.model_path.exists() and any(self.model_path.iterdir()):
            print(f"INFO: Found model directory at '{self.model_path}'.")
            print("INFO: Current version uses MOCK responses for interaction.")
            print("INFO: Actual model loading and inference from './merged_model' is intended for a later step.")
            # Placeholder: Assume ready for mock responses if directory exists.
            # Actual model loading attempt would go here:
            # print(f"INFO: Attempting to load actual model from {self.model_path} (this is a placeholder)...")
            # try:
            #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            #     self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            #     # if torch.cuda.is_available():
            #     #     self.model = self.model.to("cuda")
            #     self.is_model_ready = True # Set to true if actual model loads
            #     print(f"INFO: Successfully initialized tokenizer and model from {self.model_path} (conceptually).")
            # except Exception as e:
            #     print(f"WARNING: Failed to load actual model from {self.model_path}: {e}")
            #     print("INFO: Falling back to mock responses only.")
            # For now, even if model path exists, we use mock responses until explicitly enabling real inference.
            self.is_model_ready = True # Allow mock responses if dir exists
        else:
            print(f"WARNING: Model directory '{self.model_path}' not found or is empty.")
            print("INFO: Please ensure models are downloaded and merged using mergekit to './merged_model'.")
            print("INFO: Continuing with mock responses only.")
            # If you want the script to fail without a model directory:
            # sys.exit(f"Error: Merged model directory '{self.model_path}' not found. Exiting.")
            self.is_model_ready = True # Still allow mock responses for now

    def generate_response(self, prompt: str) -> str:
        if not self.is_model_ready: # This check is more for a state where even mock is not desired
            return "Critical Error: AGI system not ready."

        # Actual model inference logic (currently placeholder)
        # if self.model and self.tokenizer:
        #     try:
        #         inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        #         # if torch.cuda.is_available():
        #         #     inputs = {k: v.to("cuda") for k, v in inputs.items()}
        #         # with torch.no_grad():
        #         #     outputs = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
        #         # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #         # return response
        #     except Exception as e:
        #         print(f"ERROR: Actual model inference failed: {e}")
        #         return "Error during generation with actual model. Falling back to mock."

        # Placeholder response logic
        prompt_lower = prompt.lower()
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Mock AGI: Hello there! How can I assist you today?"
        elif "how are you" in prompt_lower:
            return "Mock AGI: I am a collection of code and data, currently in a mock response mode. I function as programmed!"
        elif "what is your name" in prompt_lower:
            return "Mock AGI: I am a merged AI, currently in a placeholder phase. You can call me 'AGI'."
        elif "what can you do" in prompt_lower:
            return "Mock AGI: In my final form, I'll be able to assist with various tasks. Right now, I provide these mock responses and log our interactions for future learning."
        else:
            return f"Mock AGI: I've processed your input: '{prompt}'. This is a mock reply. The system will evolve!"

def call_training_script(user_input: str, model_output: str):
    script_path_str = "./train_on_interaction.sh"
    script_path = Path(script_path_str)

    if not script_path.exists():
        print(f"WARNING: Training script '{script_path}' not found! Skipping training call.")
        return

    try:
        # Ensure the script is executable (useful if permissions were somehow reset)
        if not os.access(script_path, os.X_OK):
            subprocess.run(['chmod', '+x', script_path_str], check=True)

        # Using Popen for potentially more complex stdout/stderr handling if needed later
        process = subprocess.Popen(
            [script_path_str, user_input, model_output],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=30) # Added a timeout

        if process.returncode == 0:
            if stdout:
                print(f"[Training Script Output]:\n{stdout.strip()}")
        else:
            print(f"ERROR: Training script '{script_path}' failed with exit code {process.returncode}.")
            if stdout:
                print(f"STDOUT:\n{stdout.strip()}")
            if stderr:
                print(f"STDERR:\n{stderr.strip()}")

    except subprocess.TimeoutExpired:
        print(f"ERROR: Training script '{script_path}' timed out.")
    except Exception as e:
        print(f"ERROR: Failed to execute or communicate with training script '{script_path}': {e}")

def main():
    print("Initializing AGI System (Interactive Mode)...")
    # This is where the path to the merged model will be critical later.
    agi_model = AGIPPlaceholder(model_path_str="./merged_model")

    # Check if the merged model directory exists, provide guidance if not.
    if not agi_model.model_path.exists() or not any(agi_model.model_path.iterdir()):
        print("\n---")
        print("IMPORTANT: The './merged_model' directory, expected to contain the AI model,")
        print("is missing or empty. This script will use MOCK responses.")
        print("To enable actual AI responses:")
        print("  1. Ensure you have run './download_models.sh' successfully.")
        print("  2. Configure 'merge_config.yml' with the correct model paths.")
        print("  3. Run mergekit: 'mergekit-yaml merge_config.yml ./merged_model'")
        print("---")

    print("\nAGI Interactive Terminal (Mock Mode)")
    print("Type 'exit', 'quit', or press Ctrl+D to end.")
    print("Press Ctrl+C for forceful interruption.")
    print("------------------------------------")

    try:
        while True:
            try:
                user_input = input("You: ")
            except EOFError: # Handle Ctrl+D for graceful exit
                print("\nExiting (EOF detected)...")
                break

            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting AGI session.")
                break

            if not user_input.strip(): # Skip empty input
                continue

            agi_response = agi_model.generate_response(user_input)
            print(f"AGI Output: {agi_response}")

            call_training_script(user_input, agi_response)
            print("------------------------------------")

    except KeyboardInterrupt: # Handle Ctrl+C for graceful exit
        print("\nExiting (KeyboardInterrupt detected)...")
    finally:
        print("AGI session terminated.")

if __name__ == "__main__":
    # Ensures that the script is executable itself, though typically Python scripts are run with `python script.py`
    # os.chmod(__file__, 0o755) # This might cause issues depending on environment, usually not needed for .py
    main()
