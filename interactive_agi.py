# interactive_agi.py

import subprocess
import os
import sys
from pathlib import Path
import re

# Rich imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback print if rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    print("WARNING: Rich library not found. Output will be basic.")
    print("Please install it: pip install rich")

console_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "prompt": "bold magenta",
    "agiprompt": "bold blue"
})
console = Console(theme=console_theme)

# --- Context Analyzer ---
class ContextAnalyzer:
    def __init__(self):
        pass # No complex initialization needed for this version

    def get_cwd_context(self) -> str:
        return os.getcwd()

    def get_git_context(self) -> dict:
        git_context = {"branch": "N/A", "modified_files": 0, "in_git_repo": False}
        try:
            # Check if .git directory exists
            if not Path(".git").is_dir():
                return git_context

            git_context["in_git_repo"] = True

            # Get current branch
            branch_process = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False)
            if branch_process.returncode == 0:
                git_context["branch"] = branch_process.stdout.strip()

            # Get number of modified files (unstaged or staged)
            status_process = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
            if status_process.returncode == 0:
                # Each line in porcelain output is a modified/new/deleted file
                # We are interested in lines not starting with '??' (untracked) for "modified" count here
                # but for simplicity, let's count all changes that aren't clean.
                modified_lines = [line for line in status_process.stdout.strip().splitlines() if line.strip()]
                git_context["modified_files"] = len(modified_lines)

        except FileNotFoundError:
            # Git command not found
            git_context["branch"] = "Git_Not_Found"
        except Exception:
            # Other git errors, keep default N/A values
            pass
        return git_context

    def get_file_type_counts(self, N=5) -> dict:
        counts = {}
        common_extensions = {
            '.py': 'Python', '.js': 'JavaScript', '.html': 'HTML', '.css': 'CSS',
            '.md': 'Markdown', '.json': 'JSON', '.txt': 'Text', '.java': 'Java',
            '.c': 'C', '.cpp': 'C++', '.go': 'Go', '.rs': 'Rust', '.sh': 'Shell'
        }
        try:
            for item in os.listdir("."):
                if os.path.isfile(item):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in common_extensions:
                        lang_name = common_extensions[ext]
                        counts[lang_name] = counts.get(lang_name, 0) + 1
            # Get top N file types by count
            sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            return dict(sorted_counts[:N])
        except OSError:
            return {}


    def get_full_context_string(self) -> str:
        cwd = self.get_cwd_context()
        git_info = self.get_git_context()
        file_counts = self.get_file_type_counts()

        context_parts = [f"CWD='{cwd}'"]
        if git_info["in_git_repo"]:
            context_parts.append(f"GitBranch='{git_info['branch']}'")
            if git_info["modified_files"] > 0:
                 context_parts.append(f"GitModified={git_info['modified_files']}")

        if file_counts:
            file_str_parts = [f"{lang}:{count}" for lang, count in file_counts.items()]
            context_parts.append(f"Files=({', '.join(file_str_parts)})")

        return "Context: " + ", ".join(context_parts)

from collections import deque # For conversation history
from rich.table import Table # For /ls command
from datetime import datetime # For history timestamps

# Global context analyzer instance
context_analyzer = ContextAnalyzer()

from collections import deque # For conversation history
from rich.table import Table # For /ls command and sysinfo
from datetime import datetime # For history timestamps
import json # For saving/loading history
import platform # Already used, but good to note for sysinfo
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    console.print("WARNING: `psutil` library not found. Detailed system info in /sysinfo will be limited.", style="warning")
    console.print("Install with: pip install psutil", style="info")


# --- Configuration & Globals ---
HISTORY_FILE_PATH = Path("./.agi_terminal_cache/history.json")
HISTORY_MAX_LEN = 100 # Store last 100 exchanges (user + assistant = 1 exchange)

# Global context analyzer instance
context_analyzer = ContextAnalyzer()

# Global conversation history
conversation_history = deque(maxlen=HISTORY_MAX_LEN)

# Attempt to import PyTorch and Transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    console.print("WARNING: PyTorch or Transformers library not found. Real model loading will fail.", style="warning")
    console.print("Please install them: pip install torch transformers sentencepiece", style="info")

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
            console.print("ERROR: Transformers/PyTorch not available. Cannot load merged model.", style="error")
            return

        if not self.model_path.exists() or not self.model_path.is_dir():
            console.print(f"WARNING: Model directory '{self.model_path}' not found or is not a directory.", style="warning")
            console.print("INFO: Real model loading skipped. AGI will use mock responses if available as fallback.", style="info")
            return

        console.print(f"INFO: Found model directory at '{self.model_path}'. Attempting to load model...", style="info")
        try:
            # trust_remote_code=True can be a security risk if loading untrusted models.
            # For models from Hugging Face Hub, it's often needed for custom architectures.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            console.print("INFO: Tokenizer loaded successfully.", style="success")

            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
            console.print("INFO: Model loaded successfully.", style="success")

            if torch.cuda.is_available():
                self.device = "cuda"
                self.model.to(self.device)
                console.print(f"INFO: Model moved to {self.device}.", style="info")
            else:
                console.print("INFO: CUDA not available. Using CPU. This might be slow for large models.", style="info")

            self.model.eval() # Set model to evaluation mode

            # Set pad_token_id if not already set by tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                console.print(f"INFO: Set tokenizer.pad_token_id to eos_token_id ({self.tokenizer.eos_token_id})", style="info")

            self.generation_params["pad_token_id"] = self.tokenizer.pad_token_id

            # Try to get model's max length
            try:
                self.model_max_length = self.model.config.max_position_embeddings
                console.print(f"INFO: Model max sequence length: {self.model_max_length}", style="info")
            except AttributeError:
                console.print(f"WARNING: Could not determine model's max_position_embeddings. Using default: {self.model_max_length}", style="warning")


            self.is_model_loaded = True
            console.print("INFO: Merged AGI model initialized successfully.", style="success")

        except OSError as e:
            console.print(f"ERROR: OSError during model loading from '{self.model_path}'. This often indicates missing files or permission issues.", style="error")
            console.print(f"Details: {e}", style="dim error")
        except ImportError as e: # More specific for missing optional dependencies for a model
            console.print(f"ERROR: ImportError during model loading. A required library might be missing for this model type.", style="error")
            console.print(f"Details: {e}", style="dim error")
        except Exception as e:
            console.print(f"ERROR: An unexpected error occurred while loading model or tokenizer from '{self.model_path}'.", style="error")
            console.print(f"Details: {type(e).__name__}: {e}", style="dim error")
            console.print("INFO: Ensure the model directory is correct, complete, and not corrupted. Also check console for other related errors (e.g., Transformers/PyTorch).", style="info")

        if not self.is_model_loaded:
             console.print("INFO: Falling back to mock responses due to model loading failure.", style="info")


    def generate_response(self, prompt: str) -> str:
        if not self.is_model_loaded or not self.tokenizer or not self.model:
            return "[bold red]Critical Error:[/bold red] Actual AGI model not loaded. Cannot generate response."

        try:
            # Get current context string
            current_context_str = context_analyzer.get_full_context_string()

            # Prepend context to the user's prompt
            # The model will see something like:
            # Context: CWD='/path/to/project', GitBranch='main', Files=(py:5, js:2) \nUser Query: <user_input_here>
            # This helps the model understand the environment.
            # We might want to make the separator clearer or add specific tags.

            # --- Task-Specific Prompt Engineering ---
            query_lower = prompt.lower()
            task_prefix = "General Query: " # Default prefix

            code_keywords = ["code", "script", "function", "class", "algorithm", "module", "generate python", "write go", "debug this"]
            explanation_keywords = ["explain", "what is", "describe", "how does", "tell me about"]
            # Add more keyword sets as needed

            if any(keyword in query_lower for keyword in code_keywords):
                task_prefix = "Coding Task: "
            elif any(keyword in query_lower for keyword in explanation_keywords):
                task_prefix = "Explanation Request: "

            # Construct the full prompt with context and task-specific prefix
            full_prompt = f"{current_context_str}\n{task_prefix}{prompt}"
            # console.print(f"[Dim]Full prompt being sent to model:\n{full_prompt}[/Dim]") # For debugging

            # Ensure prompt length + max_new_tokens is within model_max_length
            # This is a simplified check; precise token counting for prompt is better.
            effective_max_prompt_len = self.model_max_length - self.generation_params.get("max_new_tokens", 256)
            if effective_max_prompt_len <= 0: # max_new_tokens is too large
                return "[bold red]Error:[/bold red] max_new_tokens is too large for model's context window. Please reduce it."

            # Tokenize the full prompt (context + user query)
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True, # Pad to longest in batch if batching, else just ensure tensor
                truncation=True,
                max_length=effective_max_prompt_len # Ensure this respects the model's actual limit
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_params)

            # Decode only the newly generated tokens
            response_text = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response_text.strip()

        except Exception as e:
            console.print(f"ERROR: Exception during model generation: {e}", style="error")
            return error_message

    def set_parameter(self, param_name_str: str, param_value_str: str) -> str:
        param_name_upper = param_name_str.upper()
        actual_param_name = PARAM_MAP.get(param_name_upper)

        if not actual_param_name:
            return f"[bold red]Error:[/bold red] Unknown parameter '{param_name_str}'. Known params: {', '.join(PARAM_MAP.keys())}"

        try:
            original_value = self.generation_params.get(actual_param_name)

            if actual_param_name in ["max_new_tokens", "top_k"]:
                new_value = int(param_value_str)
                if new_value <= 0:
                    return f"[bold red]Error:[/bold red] {param_name_upper} must be > 0."
                if actual_param_name == "max_new_tokens" and new_value >= self.model_max_length:
                    return f"[bold red]Error:[/bold red] {param_name_upper} ({new_value}) must be less than model's max length ({self.model_max_length})."
            elif actual_param_name in ["temperature", "top_p", "repetition_penalty"]:
                new_value = float(param_value_str)
                if actual_param_name == "temperature" and not (0.01 <= new_value <= 5.0): # Wider range for experimentation
                    return "[bold red]Error:[/bold red] TEMPERATURE must be between 0.01 and 5.0."
                if actual_param_name == "top_p" and not (0.01 <= new_value <= 1.0):
                    return "[bold red]Error:[/bold red] TOP_P must be between 0.01 and 1.0."
                if actual_param_name == "repetition_penalty" and new_value < 1.0:
                     return "[bold red]Error:[/bold red] REPETITION_PENALTY must be >= 1.0."
            else: # Should not happen if PARAM_MAP is correct
                return f"[bold red]Error:[/bold red] Parameter '{actual_param_name}' has unhandled type."

            self.generation_params[actual_param_name] = new_value
            # Special handling for do_sample based on temperature
            if actual_param_name == "temperature":
                if new_value == 0.0: # Effectively greedy
                     self.generation_params["do_sample"] = False
                     console.print("INFO: Temperature is 0.0, setting do_sample=False (greedy decoding).", style="info")
                else:
                     self.generation_params["do_sample"] = True

            return f"[green]Set[/green] {param_name_upper} to {new_value} (was {original_value if original_value is not None else 'default'})"

        except ValueError:
            return f"[bold red]Error:[/bold red] Invalid value '{param_value_str}' for {param_name_upper}. Expected numeric type."
        except Exception as e:
            return f"[bold red]Error setting parameter {param_name_upper}:[/bold red] {e}"

    def show_parameters(self) -> str:
        if not self.is_model_loaded:
            return "[yellow]Model not loaded.[/yellow] Parameters are at default values but not actively used by a real model."

        output_text = Text("Current Generation Parameters:\n", style="bold")
        for key, user_name in PARAM_MAP.items(): # Iterate PARAM_MAP to show user-friendly names
            value = self.generation_params.get(user_name)
            output_text.append(f"  {key}: {value}\n")
        # Show other relevant params not in PARAM_MAP directly
        output_text.append(f"  DO_SAMPLE: {self.generation_params.get('do_sample')}\n")
        output_text.append(f"  PAD_TOKEN_ID: {self.generation_params.get('pad_token_id')}\n")
        output_text.append(f"  Model Max Sequence Length: {self.model_max_length}\n")
        return output_text


class AGIPPlaceholder:
    def __init__(self, model_path_str="./merged_model"): # model_path_str is for consistency
        self.model_path = Path(model_path_str) # Still check for user guidance
        # No actual model loading here
        console.print("INFO: AGIPPlaceholder initialized (mock responses).", style="info")

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
        return f"[dim]Mock AGI: Parameter setting ({param_name_str}={param_value_str}) noted. Real model not active.[/dim]"

    def show_parameters(self) -> str:
        return "[dim]Mock AGI: Currently in mock mode. Real model parameters are not active.[/dim]"


def call_training_script(user_input: str, model_output: str):
    script_path_str = "./train_on_interaction.sh"
    script_path = Path(script_path_str)

    if not script_path.exists():
        console.print(f"WARNING: Training script '{script_path}' not found! Skipping training call.", style="warning")
        return

    try:
        if not os.access(script_path, os.X_OK): # Ensure executable
            subprocess.run(['chmod', '+x', script_path_str], check=True)

        process = subprocess.Popen(
            [script_path_str, user_input, model_output],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=30) # Add timeout

        if process.returncode == 0:
            if stdout: console.print(f"[dim cyan][Training Script Output]:[/dim cyan]\n{stdout.strip()}")
        else:
            console.print(f"ERROR: Training script '{script_path}' failed (code {process.returncode}).", style="error")
            if stdout: console.print(f"STDOUT:\n{stdout.strip()}", style="dim")
            if stderr: console.print(f"STDERR:\n{stderr.strip()}", style="dim red")

    except subprocess.TimeoutExpired:
        console.print(f"ERROR: Training script '{script_path}' timed out.", style="error")
    except Exception as e:
        console.print(f"ERROR: Failed to execute training script '{script_path}': {e}", style="error")

def display_startup_banner():
    banner_text = """
 █████╗  ██████╗ ██╗     ██████╗ ██╗███╗   ██╗ █████╗ ██╗
██╔══██╗██╔════╝ ██║    ██╔════╝ ██║████╗  ██║██╔══██╗██║
███████║██║  ███╗██║    ██║  ███╗██║██╔ ██╗ ██║███████║██║
██╔══██║██║   ██║██║    ██║   ██║██║██║╚██╗██║██╔══██║██║
██║  ██║╚██████╔╝██║    ╚██████╔╝██║██║ ╚████║██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
        Interactive Terminal - Merged Model Edition
"""
    console.print(Panel(Text(banner_text, justify="center", style="bold blue on black"), title="[bold green]AGI Terminal[/bold green]", expand=False))

def detect_code_blocks(text: str) -> list:
    """Detects code blocks and separates them from plain text."""
    parts = []
    last_end = 0
    # Regex to find ``` optionally followed by a language name, then content, then ```
    # It captures the language name (optional) and the code content.
    # Handles optional spaces around language name.
    # Language name can be alphanumeric with hyphens or underscores.
    for match in re.finditer(r"```(?:([\w\-_]+)\s*)?\n(.*?)\n```", text, re.DOTALL):
        text_before = text[last_end:match.start()]
        if text_before:
            parts.append({"type": "text", "content": text_before})

        lang = (match.group(1) or "plaintext").lower().strip()
        code_content = match.group(2).strip()

        parts.append({"type": "code", "content": code_content, "lang": lang})
        last_end = match.end()

    text_after = text[last_end:]
    if text_after:
        parts.append({"type": "text", "content": text_after})
    return parts

def main():
    display_startup_banner()
    console.print("Initializing AGI System (Interactive Mode)...", style="info")

    agi_interface = MergedAGI(model_path_str="./merged_model")

    if not agi_interface.is_model_loaded:
        console.print("INFO: MergedAGI could not load the model. Falling back to AGIPPlaceholder.", style="warning")
        if not agi_interface.model_path.exists() or not any(agi_interface.model_path.iterdir()):
            msg = Text()
            msg.append("\n---\nIMPORTANT: The './merged_model' directory, expected to contain the AI model,\n", style="bold yellow")
            msg.append("is missing or empty. This script will use MOCK responses.\n", style="yellow")
            msg.append("To enable actual AI responses:\n", style="yellow")
            msg.append("  1. Ensure you have run './download_models.sh' (or `python setup_agi_terminal.py`) successfully.\n")
            msg.append("  2. Configure 'merge_config.yml' with the correct model paths (if not using setup script).\n")
            msg.append("  3. Run mergekit: 'mergekit-yaml merge_config.yml ./merged_model'\n---")
            console.print(Panel(msg, title="[bold red]Model Not Found[/bold red]", border_style="red"))
        agi_interface = AGIPPlaceholder(model_path_str="./merged_model")
        terminal_mode = "[bold yellow]Mock Mode[/bold yellow]"
    else:
        terminal_mode = "[bold green]Merged Model Mode[/bold green]"

    console.print(f"\nAGI Interactive Terminal ({terminal_mode})")
    console.print("Type '/set parameter <NAME> <VALUE>' to change generation settings (e.g., /set parameter MAX_TOKENS 512).")
    console.print("Type '/show parameters' to see current settings.")
    console.print("Type 'exit', 'quit', or press Ctrl+D to end.")
    console.print("Press Ctrl+C for forceful interruption.")
    console.print("-" * console.width)

    try:
        while True:
            try:
                user_input = console.input("[prompt]You: [/prompt]")
            except EOFError:
                console.print("\nExiting (EOF detected)...", style="info")
                break
            except KeyboardInterrupt:
                console.print("\nKeyboardInterrupt: Type 'exit' or Ctrl+D to quit, or press Ctrl-C again to force exit.", style="warning")
                try: # Second Ctrl-C to force exit
                    console.input("") # Dummy input to catch second Ctrl-C
                except KeyboardInterrupt:
                    console.print("\nForcing exit...", style="error")
                    break
                continue


            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("Exiting AGI session.", style="info")
                break

            if not user_input.strip():
                continue

            # Add user input to history before processing
            conversation_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()})

            if user_input.lower().startswith("/set parameter "):
                parts = user_input.strip().split(maxsplit=3)
                if len(parts) == 4:
                    _, _, param_name, param_value = parts
                    response = agi_interface.set_parameter(param_name, param_value)
                    console.print(f"AGI System: {response}")
                else:
                    console.print("AGI System: [red]Invalid command.[/red] Usage: /set parameter <NAME> <VALUE>")
            elif user_input.lower() == "/show parameters":
                response = agi_interface.show_parameters()
                console.print(Panel(response, title="[bold blue]AGI Parameters[/bold blue]", border_style="blue"))
            elif user_input.lower().startswith("/ls"):
                parts = user_input.strip().split(maxsplit=1)
                path_to_list = parts[1] if len(parts) > 1 else "."
                list_directory_contents(path_to_list)
            elif user_input.lower() == "/cwd":
                console.print(Panel(os.getcwd(), title="[bold blue]Current Working Directory[/bold blue]", border_style="blue"))
            elif user_input.lower().startswith("/cd "):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) > 1:
                    change_directory(parts[1])
                else:
                    console.print("[red]Usage: /cd <path>[/red]")
            elif user_input.lower() == "/clear":
                console.clear()
                display_startup_banner() # Re-display banner after clearing
            elif user_input.lower() == "/history":
                display_conversation_history()
            elif user_input.lower() == "/sysinfo":
                display_system_info()
            else:
                with console.status("[yellow]AGI is thinking...[/yellow]", spinner="dots"):
                    agi_response_text = agi_interface.generate_response(user_input)

                conversation_history.append({"role": "assistant", "content": agi_response_text, "timestamp": datetime.now().isoformat()})
                response_parts = detect_code_blocks(agi_response_text)

                console.print(f"[agiprompt]AGI Output:[/agiprompt]")
                for part in response_parts:
                    if part["type"] == "text":
                        console.print(Text(part["content"]))
                    elif part["type"] == "code":
                        lang_for_syntax = part["lang"] if part["lang"] else "plaintext"
                        try:
                            code_syntax = Syntax(part["content"], lang_for_syntax, theme="monokai", line_numbers=True, word_wrap=True)
                            console.print(code_syntax)
                        except Exception:
                            code_syntax = Syntax(part["content"], "plaintext", theme="monokai", line_numbers=True, word_wrap=True)
                            console.print(code_syntax)
                            console.print(f"(Note: language '{part['lang']}' not recognized for syntax highlighting, shown as plaintext)", style="dim italic")

                if agi_interface.is_model_loaded or isinstance(agi_interface, AGIPPlaceholder):
                    call_training_script(user_input, agi_response_text)

            console.print("-" * console.width)

    except KeyboardInterrupt:
        console.print("\nExiting due to KeyboardInterrupt...", style="info")
    finally:
        console.print("AGI session terminated.", style="info")


# --- Command Implementations ---
def list_directory_contents(path_str: str):
    try:
        target_path = Path(path_str).resolve()
        if not target_path.exists():
            console.print(f"[red]Error: Path does not exist: {target_path}[/red]")
            return
        if not target_path.is_dir():
            console.print(f"[red]Error: Not a directory: {target_path}[/red]")
            return

        table = Table(title=f"Contents of [cyan]{target_path}[/cyan]", show_lines=True)
        table.add_column("Name", style="bold yellow", min_width=20)
        table.add_column("Type", style="blue")
        table.add_column("Size (Bytes)", style="magenta", justify="right")
        table.add_column("Modified", style="green")

        for item in sorted(list(target_path.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                stat = item.stat()
                item_type = "Dir" if item.is_dir() else "File"
                size = stat.st_size if item.is_file() else ""
                modified_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                table.add_row(item.name, item_type, str(size), modified_time)
            except OSError:
                 table.add_row(item.name, "N/A (Permission Denied)", "N/A", "N/A")


        console.print(table)
    except FileNotFoundError:
        console.print(f"[red]Error: Directory or path component not found: {path_str}[/red]")
    except PermissionError:
        console.print(f"[red]Error: Permission denied for directory: {path_str}[/red]")
    except Exception as e:
        console.print(f"[red]Error listing directory '{path_str}': {type(e).__name__} - {e}[/red]")

def change_directory(path_str: str):
    try:
        resolved_path = Path(path_str).resolve()
        if not resolved_path.is_dir():
            console.print(f"[red]Error: Not a directory or path does not exist: {resolved_path}[/red]")
            return
        os.chdir(resolved_path)
        console.print(f"Changed directory to: [cyan]{os.getcwd()}[/cyan]")
    except FileNotFoundError: # Should be caught by resolve() or is_dir() mostly
        console.print(f"[red]Error: Directory not found: {path_str}[/red]")
    except PermissionError:
        console.print(f"[red]Error: Permission denied to change directory to: {path_str}[/red]")
    except Exception as e:
        console.print(f"[red]Error changing directory to '{path_str}': {type(e).__name__} - {e}[/red]")

def display_conversation_history():
    if not conversation_history:
        console.print("[yellow]No history yet.[/yellow]")
        return

    history_panel_content = Text()
    for entry in conversation_history:
        role_style = "prompt" if entry['role'] == 'user' else "agiprompt"
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')
        history_panel_content.append(f"[{timestamp}] ", style="dim")
        history_panel_content.append(f"{entry['role'].capitalize()}: ", style=role_style)
        history_panel_content.append(entry['content'] + "\n")

    console.print(Panel(history_panel_content, title="[bold blue]Conversation History[/bold blue]", border_style="blue", expand=False))

def display_system_info():
    info_text = Text()
    info_text.append("OS: ", style="bold")
    info_text.append(f"{platform.system()} {platform.release()} ({platform.machine()})\n")
    # This function is now correctly structured to use a Rich Table.
    # The previous Text object approach was an intermediate step.
    table = Table(title="System Information", show_header=True, header_style="bold magenta", border_style="blue")
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value")

    table.add_row("OS", f"{platform.system()} {platform.release()} ({platform.machine()})")
    table.add_row("Python Version", f"{sys.version.split()[0]}")
    table.add_row("CPU Cores", str(os.cpu_count()))

    if PSUTIL_AVAILABLE:
        try:
            # CPU Info
            cpu_freq = psutil.cpu_freq()
            freq_current_str = f"{cpu_freq.current:.2f} MHz" if cpu_freq and cpu_freq.current > 0 else "N/A"
            freq_max_str = f"{cpu_freq.max:.2f} MHz" if cpu_freq and hasattr(cpu_freq, 'max') and cpu_freq.max > 0 else "N/A"
            table.add_row("CPU Frequency", f"Current: {freq_current_str}, Max: {freq_max_str}")
            table.add_row("CPU Usage", f"{psutil.cpu_percent(interval=0.1)}%") # Short interval for responsiveness

            # Memory Info
            mem = psutil.virtual_memory()
            table.add_row("Total Memory", f"{mem.total / (1024**3):.2f} GiB")
            table.add_row("Available Memory", f"{mem.available / (1024**3):.2f} GiB")
            table.add_row("Used Memory", f"{mem.used / (1024**3):.2f} GiB ({mem.percent}%)")

            # Disk Info (root partition)
            disk = psutil.disk_usage('/')
            table.add_row("Total Disk (/)", f"{disk.total / (1024**3):.2f} GiB")
            table.add_row("Free Disk (/)", f"{disk.free / (1024**3):.2f} GiB ({disk.percent}% used)")

            # Network Info (basic)
            net_io = psutil.net_io_counters()
            table.add_row("Network Sent", f"{net_io.bytes_sent / (1024**2):.2f} MiB")
            table.add_row("Network Received", f"{net_io.bytes_recv / (1024**2):.2f} MiB")

        except Exception as e:
            table.add_row("[red]psutil Error[/red]", str(e))
    else:
        # Fallback for basic memory info if psutil not available
        try:
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            mem_gib = mem_bytes / (1024.**3)
            table.add_row("Total Memory (approx)", f"{mem_gib:.2f} GiB")
        except Exception: # Broad except as os.sysconf might not be available/applicable
            table.add_row("Total Memory (approx)", "N/A (psutil not installed for details)")

    console.print(table) # Display the table within a Panel

def load_history():
    if HISTORY_FILE_PATH.exists():
        try:
            with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                history_list = json.load(f)
                # Manually extend the deque to respect its maxlen
                for item in history_list:
                    conversation_history.append(item) # deque will handle maxlen
            console.print(f"Loaded {len(conversation_history)} history entries from {HISTORY_FILE_PATH}", style="dim info")
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"Error loading history from {HISTORY_FILE_PATH}: {e}", style="warning")
    else:
        console.print(f"No history file found at {HISTORY_FILE_PATH}. Starting fresh.", style="dim info")

def save_history():
    try:
        HISTORY_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(conversation_history), f, indent=2) # Convert deque to list for saving
        # console.print(f"Saved {len(conversation_history)} history entries to {HISTORY_FILE_PATH}", style="dim info") # Optional: too verbose on every save
    except IOError as e:
        console.print(f"Error saving history to {HISTORY_FILE_PATH}: {e}", style="warning")


if __name__ == "__main__":
    load_history() # Load history at startup
    try:
        main()
    except SystemExit: # Catch sys.exit for graceful termination without re-printing message
        pass
    finally:
        save_history() # Save history on exit (graceful or via interrupt)
        console.print("AGI session terminated. History saved.", style="info") # Overwrites the one in main()
