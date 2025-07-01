# interactive_agi.py

import subprocess
import os
import sys
from pathlib import Path

# Attempt to import PyTorch and Transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: PyTorch or Transformers library not found. Real model loading will fail.")
    print("Please install them: pip install torch transformers sentencepiece") # sentencepiece is often needed

# --- Configuration for Parameter Control ---
PARAM_MAP = {
    "MAX_TOKENS": "max_new_tokens",
    "TEMPERATURE": "temperature",
    "TOP_P": "top_p",
    "TOP_K": "top_k",
    "REPETITION_PENALTY": "repetition_penalty"
}

DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 1024,
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 0,  # Disabled by default if top_p is active and > 0
    "repetition_penalty": 1.15,
    "do_sample": True, # Must be True for temperature, top_p, top_k to work
    # pad_token_id will be set from tokenizer
}

class MergedAGI:
    def __init__(self, model_path_str="./merged_model"):
        self.model_path = Path(model_path_str)
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.is_model_loaded = False
        self.generation_params = DEFAULT_GENERATION_PARAMS.copy() # Start with defaults
        self.model_max_length = 2048 # Default, will try to update from model config

        if not TRANSFORMERS_AVAILABLE:
            print("ERROR: Transformers/PyTorch not available. Cannot load merged model.")
            return

        if not self.model_path.exists() or not self.model_path.is_dir():
            print(f"WARNING: Model directory '{self.model_path}' not found or is not a directory.")
            print("INFO: Real model loading skipped. AGI will use mock responses if available as fallback.")
            return

        print(f"INFO: Found model directory at '{self.model_path}'. Attempting to load model...")
        try:
            # trust_remote_code=True can be a security risk if loading untrusted models.
            # For models from Hugging Face Hub, it's often needed for custom architectures.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            print("INFO: Tokenizer loaded successfully.")

            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
            print("INFO: Model loaded successfully.")

            if torch.cuda.is_available():
                self.device = "cuda"
                self.model.to(self.device)
                print(f"INFO: Model moved to {self.device}.")
            else:
                print("INFO: CUDA not available. Using CPU. This might be slow for large models.")

            self.model.eval() # Set model to evaluation mode

            # Set pad_token_id if not already set by tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print(f"INFO: Set tokenizer.pad_token_id to eos_token_id ({self.tokenizer.eos_token_id})")

            self.generation_params["pad_token_id"] = self.tokenizer.pad_token_id

            # Try to get model's max length
            try:
                self.model_max_length = self.model.config.max_position_embeddings
                print(f"INFO: Model max sequence length: {self.model_max_length}")
            except AttributeError:
                print(f"WARNING: Could not determine model's max_position_embeddings. Using default: {self.model_max_length}")


            self.is_model_loaded = True
            print("INFO: Merged AGI model initialized successfully.")

        except OSError as e:
            print(f"ERROR: OSError during model loading (e.g. files missing from '{self.model_path}'): {e}")
        except Exception as e:
            print(f"ERROR: Failed to load model or tokenizer from '{self.model_path}': {e}")
            print("INFO: Ensure the model directory is correct and contains all necessary files (config, weights, tokenizer).")

        if not self.is_model_loaded:
             print("INFO: Falling back to mock responses due to model loading failure.")


    def generate_response(self, prompt: str) -> str:
        if not self.is_model_loaded or not self.tokenizer or not self.model:
            return "Critical Error: Actual AGI model not loaded. Cannot generate response."

        try:
            # Ensure prompt length + max_new_tokens is within model_max_length
            # This is a simplified check; precise token counting for prompt is better.
            effective_max_prompt_len = self.model_max_length - self.generation_params.get("max_new_tokens", 256)
            if effective_max_prompt_len <= 0: # max_new_tokens is too large
                return "Error: max_new_tokens is too large for model's context window. Please reduce it."

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True, # Pad to longest in batch if batching, else just ensure tensor
                truncation=True,
                max_length=effective_max_prompt_len
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_params)

            # Decode only the newly generated tokens
            response_text = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response_text.strip()

        except Exception as e:
            print(f"ERROR: Exception during model generation: {e}")
            return "Error: Could not generate response from model."

    def set_parameter(self, param_name_str: str, param_value_str: str) -> str:
        param_name_upper = param_name_str.upper()
        actual_param_name = PARAM_MAP.get(param_name_upper)

        if not actual_param_name:
            return f"Error: Unknown parameter '{param_name_str}'. Known params: {', '.join(PARAM_MAP.keys())}"

        try:
            original_value = self.generation_params.get(actual_param_name)

            if actual_param_name in ["max_new_tokens", "top_k"]:
                new_value = int(param_value_str)
                if new_value <= 0:
                    return f"Error: {param_name_upper} must be > 0."
                if actual_param_name == "max_new_tokens" and new_value >= self.model_max_length:
                    return f"Error: {param_name_upper} ({new_value}) must be less than model's max length ({self.model_max_length})."
            elif actual_param_name in ["temperature", "top_p", "repetition_penalty"]:
                new_value = float(param_value_str)
                if actual_param_name == "temperature" and not (0.01 <= new_value <= 5.0): # Wider range for experimentation
                    return "Error: TEMPERATURE must be between 0.01 and 5.0."
                if actual_param_name == "top_p" and not (0.01 <= new_value <= 1.0):
                    return "Error: TOP_P must be between 0.01 and 1.0."
                if actual_param_name == "repetition_penalty" and new_value < 1.0:
                     return "Error: REPETITION_PENALTY must be >= 1.0."
            else: # Should not happen if PARAM_MAP is correct
                return f"Error: Parameter '{actual_param_name}' has unhandled type."

            self.generation_params[actual_param_name] = new_value
            # Special handling for do_sample based on temperature
            if actual_param_name == "temperature":
                if new_value == 0.0: # Effectively greedy
                     self.generation_params["do_sample"] = False
                     print("INFO: Temperature is 0.0, setting do_sample=False (greedy decoding).")
                else:
                     self.generation_params["do_sample"] = True


            return f"Set {param_name_upper} to {new_value} (was {original_value if original_value is not None else 'default'})"

        except ValueError:
            return f"Error: Invalid value '{param_value_str}' for {param_name_upper}. Expected numeric type."
        except Exception as e:
            return f"Error setting parameter {param_name_upper}: {e}"

    def show_parameters(self) -> str:
        if not self.is_model_loaded:
            return "Model not loaded. Parameters are at default values but not actively used by a real model."

        output = ["Current Generation Parameters:"]
        for key, user_name in PARAM_MAP.items(): # Iterate PARAM_MAP to show user-friendly names
            value = self.generation_params.get(user_name)
            output.append(f"  {key}: {value}")
        # Show other relevant params not in PARAM_MAP directly
        output.append(f"  DO_SAMPLE: {self.generation_params.get('do_sample')}")
        output.append(f"  PAD_TOKEN_ID: {self.generation_params.get('pad_token_id')}")
        output.append(f"  Model Max Sequence Length: {self.model_max_length}")
        return "\n".join(output)


class AGIPPlaceholder:
    def __init__(self, model_path_str="./merged_model"): # model_path_str is for consistency
        self.model_path = Path(model_path_str) # Still check for user guidance
        # No actual model loading here
        print("INFO: AGIPPlaceholder initialized (mock responses).")

    def generate_response(self, prompt: str) -> str:
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

    # Dummy methods for parameter control in placeholder
    def set_parameter(self, param_name_str: str, param_value_str: str) -> str:
        return f"Mock AGI: Parameter setting ({param_name_str}={param_value_str}) noted. Real model not active."

    def show_parameters(self) -> str:
        return "Mock AGI: Currently in mock mode. Real model parameters are not active."


def call_training_script(user_input: str, model_output: str):
    script_path_str = "./train_on_interaction.sh"
    script_path = Path(script_path_str)

    if not script_path.exists():
        print(f"WARNING: Training script '{script_path}' not found! Skipping training call.")
        return

    try:
        if not os.access(script_path, os.X_OK): # Ensure executable
            subprocess.run(['chmod', '+x', script_path_str], check=True)

        # Use shlex to handle quotes in arguments properly if they were to be introduced
        # For now, direct list is fine as train_on_interaction.sh expects simple args.
        process = subprocess.Popen(
            [script_path_str, user_input, model_output],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=30)

        if process.returncode == 0:
            if stdout: print(f"[Training Script Output]:\n{stdout.strip()}")
        else:
            print(f"ERROR: Training script '{script_path}' failed (code {process.returncode}).")
            if stdout: print(f"STDOUT:\n{stdout.strip()}")
            if stderr: print(f"STDERR:\n{stderr.strip()}")

    except subprocess.TimeoutExpired:
        print(f"ERROR: Training script '{script_path}' timed out.")
    except Exception as e:
        print(f"ERROR: Failed to execute training script '{script_path}': {e}")


def main():
    print("Initializing AGI System (Interactive Mode)...")

    # Attempt to load the real model
    agi_interface = MergedAGI(model_path_str="./merged_model")

    # Fallback to placeholder if real model failed to load
    if not agi_interface.is_model_loaded:
        print("INFO: MergedAGI could not load the model. Falling back to AGIPPlaceholder.")
        # Provide guidance if model directory seems missing, even if MergedAGI already warned.
        if not agi_interface.model_path.exists() or not any(agi_interface.model_path.iterdir()):
             print("\n---\nIMPORTANT: The './merged_model' directory, expected to contain the AI model,")
             print("is missing or empty. This script will use MOCK responses.")
             print("To enable actual AI responses:")
             print("  1. Ensure you have run './download_models.sh' successfully.")
             print("  2. Configure 'merge_config.yml' with the correct model paths.")
             print("  3. Run mergekit: 'mergekit-yaml merge_config.yml ./merged_model'\n---")
        agi_interface = AGIPPlaceholder(model_path_str="./merged_model")
        terminal_mode = "Mock Mode"
    else:
        terminal_mode = "Merged Model Mode"

    print(f"\nAGI Interactive Terminal ({terminal_mode})")
    print("Type '/set parameter <NAME> <VALUE>' to change generation settings (e.g., /set parameter MAX_TOKENS 512).")
    print("Type '/show parameters' to see current settings.")
    print("Type 'exit', 'quit', or press Ctrl+D to end.")
    print("Press Ctrl+C for forceful interruption.")
    print("------------------------------------")

    try:
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                print("\nExiting (EOF detected)...")
                break

            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting AGI session.")
                break

            if not user_input.strip():
                continue

            # Handle parameter commands
            if user_input.lower().startswith("/set parameter "):
                parts = user_input.strip().split(maxsplit=3) # Use maxsplit=3 for `/set parameter NAME VALUE`
                if len(parts) == 4:
                    _, _, param_name, param_value = parts
                    response = agi_interface.set_parameter(param_name, param_value)
                    print(f"AGI System: {response}")
                else:
                    print("AGI System: Invalid command. Usage: /set parameter <NAME> <VALUE>")
            elif user_input.lower() == "/show parameters":
                response = agi_interface.show_parameters()
                print(f"AGI System:\n{response}")
            else:
                agi_response = agi_interface.generate_response(user_input)
                print(f"AGI Output: {agi_response}")
                if agi_interface.is_model_loaded or isinstance(agi_interface, AGIPPlaceholder): # Log for both real and mock
                    call_training_script(user_input, agi_response)

            print("------------------------------------")

    except KeyboardInterrupt:
        print("\nExiting (KeyboardInterrupt detected)...")
    finally:
        print("AGI session terminated.")

if __name__ == "__main__":
    main()
