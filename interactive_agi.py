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
import urllib.parse # For encoding search queries
import shutil # For file operations like rm (recursive), cp, mv

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # console.print("WARNING: `psutil` library not found. Detailed system info in /sysinfo will be limited.", style="warning")
    # console.print("Install with: pip install psutil", style="info")
    # Quieten this for now as it's printed by setup script if needed.


# --- Configuration & Globals ---
HISTORY_FILE_PATH = Path("./.agi_terminal_cache/history.json")
HISTORY_MAX_LEN = 100 # Store last 100 exchanges (user + assistant = 1 exchange)

# Global context analyzer instance
context_analyzer = ContextAnalyzer()

# Global conversation history
conversation_history = deque(maxlen=HISTORY_MAX_LEN)

# --- Desktop Path for Session Logs ---
def get_desktop_path() -> Path:
    """Determines the user's desktop path."""
    home = Path.home()
    # Common desktop paths; order can matter if multiple exist (e.g. via symlinks)
    desktop_paths = [
        home / "Desktop",
        home / "Pulpit",  # Polish
        home / "Bureau",  # French
        home / "Schreibtisch",  # German
        home / "Escritorio",  # Spanish
        home / "Scrivania" # Italian
    ]

    # For Linux, also check XDG user dirs
    if platform.system() == "Linux":
        try:
            xdg_desktop_dir_bytes = subprocess.check_output(
                ["xdg-user-dir", "DESKTOP"], text=False, stderr=subprocess.DEVNULL
            )
            xdg_desktop_dir_str = xdg_desktop_dir_bytes.decode("utf-8").strip()
            if xdg_desktop_dir_str: # Ensure it's not empty
                desktop_paths.insert(0, Path(xdg_desktop_dir_str)) # Prioritize XDG
        except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
            pass # xdg-user-dir might not be available or configured

    # For Windows
    if platform.system() == "Windows":
        # USERPROFILE should point to C:\Users\<username>
        # Desktop is usually directly under this.
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            desktop_paths.insert(0, Path(user_profile) / "Desktop") # Prioritize

    for path in desktop_paths:
        if path.is_dir():
            return path

    # Fallback if no standard desktop path is found
    fallback_path = Path("./agi_desktop_logs")
    console.print(f"WARNING: Could not determine standard Desktop path. Logging session to: {fallback_path.resolve()}", style="warning")
    fallback_path.mkdir(parents=True, exist_ok=True)
    return fallback_path

class SessionLogger:
    def __init__(self, log_directory: Path):
        self.log_file = None
        if not log_directory.exists():
            try:
                log_directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                console.print(f"Error creating log directory {log_directory}: {e}", style="error")
                # If log dir creation fails, don't attempt to log.
                return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_directory / f"AGI_Terminal_Log_{timestamp}.txt"
        try:
            # Touch the file to ensure it's creatable early
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"--- AGI Terminal Session Log ---\n")
                f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("------------------------------------\n\n")
            console.print(f"INFO: Logging this session to: {self.log_file.resolve()}", style="info")
        except Exception as e:
            console.print(f"Error creating session log file {self.log_file}: {e}", style="error")
            self.log_file = None # Disable logging if file can't be created

    def log_entry(self, role: str, content: str):
        if not self.log_file:
            return # Logging disabled

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # Clean up potential rich formatting for plain text log
                plain_content = re.sub(r'\[/?\w+\]', '', content) # Basic removal of [style] tags
                f.write(f"[{timestamp}] {role.upper()}: {plain_content}\n\n")
        except Exception as e:
            # Don't crash the app if logging fails, just print a warning once.
            if hasattr(self, "_log_error_printed") and self._log_error_printed:
                pass
            else:
                console.print(f"Error writing to session log {self.log_file}: {e}", style="warning")
                self._log_error_printed = True # Print only once

# Global session logger instance (will be initialized in main)
session_logger = None


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
    global session_logger # Allow modification of the global instance
    desktop_path = get_desktop_path()
    session_logger = SessionLogger(desktop_path)

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
            if session_logger:
                session_logger.log_entry("User", user_input)

            if user_input.lower().startswith("/set parameter "):
                parts = user_input.strip().split(maxsplit=3)
                if len(parts) == 4:
                    _, _, param_name, param_value = parts
                    response = agi_interface.set_parameter(param_name, param_value)
                    console.print(f"AGI System: {response}")
                    # No specific logging for system responses to desktop log, but they are in internal history.
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
            elif user_input.lower().startswith("/search "):
                query = user_input[len("/search "):].strip()
                if query:
                    perform_internet_search(query)
                else:
                    console.print("[red]Usage: /search <your query>[/red]")
            elif user_input.lower().startswith("/mkdir "):
                path_to_create = user_input[len("/mkdir "):].strip()
                if path_to_create:
                    create_directory_command(path_to_create)
                else:
                    console.print("[red]Usage: /mkdir <directory_name>[/red]")
            elif user_input.lower().startswith("/rm "):
                path_to_remove = user_input[len("/rm "):].strip()
                if path_to_remove:
                    remove_path_command(path_to_remove)
                else:
                    console.print("[red]Usage: /rm <file_or_directory_path>[/red]")
            elif user_input.lower().startswith("/cp "):
                parts = user_input.strip().split()
                if len(parts) == 3:
                    source, destination = parts[1], parts[2]
                    copy_path_command(source, destination)
                else:
                    console.print("[red]Usage: /cp <source> <destination>[/red]")
            elif user_input.lower().startswith("/mv "):
                parts = user_input.strip().split()
                if len(parts) == 3:
                    source, destination = parts[1], parts[2]
                    move_path_command(source, destination)
                else:
                    console.print("[red]Usage: /mv <source> <destination>[/red]")
            elif user_input.lower().startswith("/cat "):
                file_path_to_cat = user_input[len("/cat "):].strip()
                if file_path_to_cat:
                    cat_file_command(file_path_to_cat, agi_interface) # Pass agi_interface for summarization
                else:
                    console.print("[red]Usage: /cat <file_path>[/red]")
            elif user_input.lower().startswith("/edit "):
                file_path_to_edit = user_input[len("/edit "):].strip()
                if file_path_to_edit:
                    edit_file_command(file_path_to_edit)
                else:
                    console.print("[red]Usage: /edit <file_path>[/red]")
            elif user_input.lower().startswith("/git "):
                git_command_parts = user_input.strip().split(maxsplit=2) # /git <subcommand> [args]
                git_subcommand = git_command_parts[1].lower() if len(git_command_parts) > 1 else None
                git_args = git_command_parts[2] if len(git_command_parts) > 2 else None

                if git_subcommand == "status":
                    git_status_command()
                elif git_subcommand == "diff":
                    git_diff_command(git_args) # git_args might be None or a file path
                elif git_subcommand == "log":
                    # Args for log could be -n <count> or other git log options.
                    # For simplicity, we'll just pass them along or parse -n.
                    git_log_command(git_args) # git_args might be like "-n 5" or None
                else:
                    console.print(f"[red]Unknown git subcommand: {git_subcommand}[/red]")
                    console.print("[info]Available git commands: status, diff [file], log [-n count][/info]")
            else:
                with console.status("[yellow]AGI is thinking...[/yellow]", spinner="dots"):
                    agi_response_text = agi_interface.generate_response(user_input)

                conversation_history.append({"role": "assistant", "content": agi_response_text, "timestamp": datetime.now().isoformat()})
                if session_logger:
                    session_logger.log_entry("AGI", agi_response_text)

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
def create_directory_command(path_str: str):
    try:
        target_path = Path(path_str)
        # Use exist_ok=False to mimic standard mkdir behavior (error if exists)
        # For interactive use, often exist_ok=True is friendlier if re-running.
        # Let's stick to erroring if it exists for now, can be changed based on UX preference.
        target_path.mkdir(parents=True, exist_ok=False)
        console.print(f"Directory created: [green]{target_path.resolve()}[/green]")
    except FileExistsError:
        console.print(f"[yellow]Directory already exists: {Path(path_str).resolve()}[/yellow]")
    except PermissionError:
        console.print(f"[red]Error: Permission denied to create directory: {path_str}[/red]")
    except Exception as e:
        console.print(f"[red]Error creating directory '{path_str}': {type(e).__name__} - {e}[/red]")

def remove_path_command(path_str: str):
    target_path = Path(path_str).resolve() # Resolve first to show absolute path in prompt
    if not target_path.exists():
        console.print(f"[red]Error: Path does not exist: {target_path}[/red]")
        return

    confirm_prompt = f"Are you sure you want to remove '{target_path}'?"
    if target_path.is_dir():
        confirm_prompt += " [bold red](This is a directory and will be removed recursively!)[/bold red]"

    try:
        if RICH_AVAILABLE:
            from rich.prompt import Confirm
            if not Confirm.ask(confirm_prompt, default=False, console=console):
                console.print("Removal cancelled.", style="yellow")
                return
        else: # Fallback for basic confirmation
            if input(f"{confirm_prompt} (yes/NO): ").lower() != "yes":
                console.print("Removal cancelled.", style="yellow")
                return

        if target_path.is_file() or target_path.is_symlink():
            target_path.unlink()
            console.print(f"File removed: [green]{target_path}[/green]")
        elif target_path.is_dir():
            shutil.rmtree(target_path)
            console.print(f"Directory removed recursively: [green]{target_path}[/green]")
        else:
            # This case should be rare if .exists() was true
            console.print(f"[red]Error: Path is not a file or directory: {target_path}[/red]")

    except PermissionError:
        console.print(f"[red]Error: Permission denied to remove: {path_str}[/red]")
    except Exception as e:
        console.print(f"[red]Error removing '{path_str}': {type(e).__name__} - {e}[/red]")

def copy_path_command(source_str: str, destination_str: str):
    source_path = Path(source_str).resolve()
    # For destination, don't resolve yet, as it might be a new filename in an existing dir
    destination_path = Path(destination_str)


    if not source_path.exists():
        console.print(f"[red]Error: Source path does not exist: {source_path}[/red]")
        return

    try:
        # Ensure destination parent directory exists if destination is a full path
        if not destination_path.parent.exists() and not destination_path.is_dir() : # if dest is like /a/b/newfile.txt, parent must exist
             # Check if it's a directory path being specified or a file in a non-existent dir.
             # If the destination_str ends with / or \ or if source is a dir, assume dest is a dir.
            if str(destination_str).endswith(os.sep) or source_path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True) # Create dest dir if it's explicitly a dir path
            # else: it's a file path, parent must exist. shutil.copy2 will fail if parent dir doesn't exist for a file.
            # This logic can be complex if destination_path itself is intended to be a new directory name.
            # For simplicity, shutil.copytree handles dest dir creation. For files, parent must exist.

        if source_path.is_file():
            shutil.copy2(source_path, destination_path)
            console.print(f"File copied from [cyan]{source_path}[/cyan] to [cyan]{destination_path.resolve()}[/cyan]", style="success")
        elif source_path.is_dir():
            # For copytree, if destination_path is an existing directory, files are copied inside it.
            # If it does not exist, it will be created.
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            console.print(f"Directory copied from [cyan]{source_path}[/cyan] to [cyan]{destination_path.resolve()}[/cyan]", style="success")
        else:
            console.print(f"[red]Error: Source path is not a file or directory: {source_path}[/red]")
    except FileNotFoundError: # e.g. destination parent directory doesn't exist for a file copy
        console.print(f"[red]Error: File not found or destination path invalid. Ensure destination directory exists if copying a file. {destination_str}[/red]")
    except PermissionError:
        console.print(f"[red]Error: Permission denied during copy operation.[/red]")
    except shutil.SameFileError:
        console.print(f"[yellow]Warning: Source and destination are the same file: {source_path}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error copying '{source_str}' to '{destination_str}': {type(e).__name__} - {e}[/red]")

def move_path_command(source_str: str, destination_str: str):
    source_path = Path(source_str).resolve()
    # Destination path for move can be a directory or a new name.
    destination_path_obj = Path(destination_str)


    if not source_path.exists():
        console.print(f"[red]Error: Source path does not exist: {source_path}[/red]")
        return

    try:
        # If destination is an existing directory, move source inside it.
        # Otherwise, move/rename source to destination_str.
        # shutil.move handles this logic.
        final_dest_path_str = str(destination_path_obj.resolve() if destination_path_obj.is_dir() else destination_path_obj)

        shutil.move(str(source_path), destination_str) # shutil.move prefers strings for src and dst

        # Try to resolve the final path of the moved item for the message
        # This can be tricky if destination_str was a directory
        final_resolved_dest = Path(destination_str)
        if final_resolved_dest.is_dir() and not Path(destination_str, source_path.name).exists(): # If dest was a dir, and source was moved into it
             final_resolved_dest = Path(destination_str, source_path.name) # Construct the full path
        elif not final_resolved_dest.exists() and Path(destination_str).parent.joinpath(source_path.name).exists(): # Renamed in same dir
             final_resolved_dest = Path(destination_str).parent.joinpath(source_path.name)
        elif Path(destination_str).exists():
             final_resolved_dest = Path(destination_str)


        console.print(f"Moved [cyan]{source_path}[/cyan] to [cyan]{final_resolved_dest.resolve()}[/cyan]", style="success")
    except PermissionError:
        console.print(f"[red]Error: Permission denied during move operation.[/red]")
    except shutil.Error as e: # Catches things like "Destination path '...' already exists" if it's a file and not a dir
        console.print(f"[red]Error moving '{source_str}' to '{destination_str}': {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error moving '{source_str}' to '{destination_str}': {type(e).__name__} - {e}[/red]")

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

    console.print(table)

# --- Git Command Implementations ---
def git_status_command():
    try:
        # Check if in a git repo first (otherwise `git status` prints an error to stderr)
        check_repo_process = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, check=False)
        if check_repo_process.returncode != 0 or check_repo_process.stdout.strip() != "true":
            console.print("[yellow]Not inside a Git repository.[/yellow]")
            return

        process = subprocess.run(["git", "status", "--porcelain", "-b"], capture_output=True, text=True, check=True)
        output = process.stdout.strip()
        lines = output.splitlines()

        if not lines:
            console.print("[green]Git status: Clean working directory.[/green]")
            return

        branch_line = ""
        staged_files = []
        unstaged_files = []
        untracked_files = []

        if lines[0].startswith("##"):
            branch_line = lines[0]
            lines = lines[1:] # Rest are file statuses

        # Parse branch info (e.g., "## main...origin/main [ahead 1]")
        branch_name = branch_line.split("...")[0][3:].strip() # Remove "## "
        remote_info = branch_line.split("...")[-1].strip() if "..." in branch_line else ""

        for line in lines:
            status_code = line[:2]
            filepath = line[3:]
            if status_code.startswith("??"):
                untracked_files.append(filepath)
            elif status_code[0] in "MADRCU": # Staged changes (first char)
                staged_files.append(f"({status_code[0].strip()}) {filepath}")
            elif status_code[1] in "MD": # Unstaged changes (second char)
                 unstaged_files.append(f"({status_code[1].strip()}) {filepath}")
            # This parsing is simplified; porcelain can be complex.

        table = Table(title=f"Git Status: [cyan]{branch_name}[/cyan] [dim]({remote_info})[/dim]", show_lines=False, box=None)
        table.add_column("Category", style="bold yellow")
        table.add_column("Files")

        if staged_files:
            table.add_row("Staged", "\n".join(staged_files))
        if unstaged_files:
            table.add_row("Unstaged", "\n".join(unstaged_files))
        if untracked_files:
            table.add_row("Untracked", "\n".join(untracked_files))

        if not staged_files and not unstaged_files and not untracked_files:
             table.add_row("Status", "[green]Clean working directory.[/green]")

        console.print(table)

    except FileNotFoundError:
        console.print("[red]Error: git command not found. Is Git installed and in PATH?[/red]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running git status: {e.stderr}[/red]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred with /git status: {type(e).__name__} - {e}[/red]")

def git_diff_command(file_path_str: Optional[str] = None):
    try:
        # Check if in a git repo
        check_repo_process = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, check=False)
        if check_repo_process.returncode != 0 or check_repo_process.stdout.strip() != "true":
            console.print("[yellow]Not inside a Git repository.[/yellow]")
            return

        git_command = ["git", "diff"]
        title = "Git Diff (Staged Changes)"
        if file_path_str:
            git_command.extend(["--", file_path_str])
            title = f"Git Diff for [cyan]{file_path_str}[/cyan] (Working Directory vs Index)"
        else: # If no file, diff staged
            git_command.append("HEAD")


        process = subprocess.run(git_command, capture_output=True, text=True, check=False) # check=False as diff returns 1 if there are changes

        if process.returncode != 0 and process.stderr and "fatal:" in process.stderr.lower():
            console.print(f"[red]Git diff error: {process.stderr.strip()}[/red]")
            return

        if not process.stdout.strip():
            console.print(f"[green]No differences found for '{file_path_str if file_path_str else 'staged changes'}'[/green]")
            return

        console.print(Panel(Syntax(process.stdout, "diff", theme="monokai", line_numbers=True, word_wrap=True),
                            title=f"[bold blue]{title}[/bold blue]", border_style="blue"))
    except FileNotFoundError:
        console.print("[red]Error: git command not found. Is Git installed and in PATH?[/red]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred with /git diff: {type(e).__name__} - {e}[/red]")

def git_log_command(args_str: Optional[str] = None):
    try:
        # Check if in a git repo
        check_repo_process = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, check=False)
        if check_repo_process.returncode != 0 or check_repo_process.stdout.strip() != "true":
            console.print("[yellow]Not inside a Git repository.[/yellow]")
            return

        git_command = ["git", "log", "--pretty=format:%C(yellow)%h%Creset %C(green)%ad%Creset | %s %C(cyan)%d%Creset %C(bold blue)[%an]%Creset", "--date=short"]
        count = 10 # Default count

        if args_str:
            # Basic parsing for -n <count> or just <count>
            match_n = re.match(r"-n\s*(\d+)", args_str)
            match_num_only = re.match(r"(\d+)", args_str)
            if match_n:
                count = int(match_n.group(1))
            elif match_num_only:
                 count = int(match_num_only.group(1))
            # For more complex arg parsing, argparse would be better if this were a standalone script

        git_command.append(f"-{count}")

        process = subprocess.run(git_command, capture_output=True, text=True, check=True)
        if not process.stdout.strip():
            console.print("[yellow]No git history found.[/yellow]")
            return

        # Rich console will render the ANSI escape codes from git's pretty format
        console.print(Panel(Text.from_ansi(process.stdout), title=f"[bold blue]Git Log (Last {count} Commits)[/bold blue]", border_style="blue"))

    except FileNotFoundError:
        console.print("[red]Error: git command not found. Is Git installed and in PATH?[/red]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running git log: {e.stderr}[/red]")
    except ValueError: # For int conversion of count
        console.print("[red]Invalid count for /git log. Please provide a number (e.g., /git log -n 15).[/red]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred with /git log: {type(e).__name__} - {e}[/red]")


# --- File Content Interaction Commands ---
MAX_CAT_LINES = 200 # Lines to display for large files before asking to summarize
MAX_CAT_CHARS_FOR_SUMMARY = 5000 # Max chars to send for summary

def get_lexer_for_filename(filename: str) -> str:
    """Tries to guess the lexer name for rich.syntax from filename."""
    ext_to_lexer = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp",
        ".go": "go", ".rs": "rust", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
        ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
        ".json": "json", ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
        ".md": "markdown", ".rst": "rst", ".txt": "text",
        ".sql": "sql", ".rb": "ruby", ".php": "php", ".swift": "swift",
        ".kt": "kotlin", ".kts": "kotlin",
        "dockerfile": "dockerfile", ".dockerfile": "dockerfile",
        ".conf": "ini", ".ini": "ini", ".cfg":"ini",
        ".diff": "diff", ".patch": "diff",
    }
    suffix = Path(filename).suffix.lower()
    name = Path(filename).name.lower() # For files like 'Dockerfile'
    if name in ext_to_lexer: # Check full name first for files like Dockerfile
        return ext_to_lexer[name]
    return ext_to_lexer.get(suffix, "text") # Default to text

def cat_file_command(file_path_str: str, agi_interface: MergedAGI):
    try:
        target_path = Path(file_path_str).resolve()
        if not target_path.exists():
            console.print(f"[red]Error: File not found: {target_path}[/red]")
            return
        if not target_path.is_file():
            console.print(f"[red]Error: Not a file: {target_path}[/red]")
            return

        file_size = target_path.stat().st_size
        lexer = get_lexer_for_filename(target_path.name)

        content_to_display = ""
        is_truncated_display = False

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            content = "".join(lines)

            if len(lines) > MAX_CAT_LINES or file_size > MAX_CAT_CHARS_FOR_SUMMARY * 2: # Heuristic for large files
                is_truncated_display = True
                content_to_display = "".join(lines[:MAX_CAT_LINES])
                if len(lines) > MAX_CAT_LINES * 2: # If significantly larger, show head and tail
                     content_to_display += f"\n\n[dim]... (file truncated, {len(lines) - MAX_CAT_LINES*2} lines hidden) ...[/dim]\n\n"
                     content_to_display += "".join(lines[-MAX_CAT_LINES:])
                elif len(lines) > MAX_CAT_LINES: # Moderately larger, just show head
                     content_to_display += f"\n\n[dim]... (file truncated, {len(lines) - MAX_CAT_LINES} more lines not shown) ...[/dim]\n"
                console.print(f"[yellow]File is large ({len(lines)} lines). Displaying a portion.[/yellow]")
            else:
                content_to_display = content

        except UnicodeDecodeError:
            console.print(f"[red]Error: Could not decode file {target_path} as UTF-8. It might be a binary file or use a different encoding.[/red]")
            # Optionally, try latin-1 or offer to show hex dump for binaries
            if Confirm.ask(f"Attempt to display as binary/latin-1?", default=False, console=console):
                try:
                    with open(target_path, 'r', encoding='latin-1') as f:
                        lines = f.readlines(MAX_CAT_LINES + 1) # Read a bit more to check size
                    content = "".join(lines[:MAX_CAT_LINES])
                    content_to_display = content
                    if len(lines) > MAX_CAT_LINES:
                        is_truncated_display = True
                        content_to_display += "\n\n[dim]... (file truncated, shown with latin-1 encoding) ...[/dim]\n"
                    lexer = "text" # Force plaintext for non-utf8
                except Exception as e_latin1:
                    console.print(f"[red]Error reading as latin-1: {e_latin1}[/red]")
                    return
            else:
                return
        except Exception as e:
            console.print(f"[red]Error reading file '{target_path}': {type(e).__name__} - {e}[/red]")
            return

        console.print(Panel(Syntax(content_to_display, lexer, theme="monokai", line_numbers=True, word_wrap=True),
                            title=f"[bold blue]Content of {target_path.name}[/bold blue] ({file_size} bytes)",
                            border_style="blue"))

        if is_truncated_display or file_size > MAX_CAT_CHARS_FOR_SUMMARY:
            if Confirm.ask("Send a portion of this file to the AGI for summary or query?", default=False, console=console):
                # Prepare content for AGI (first N chars for simplicity, could be more sophisticated)
                content_for_agi = content[:MAX_CAT_CHARS_FOR_SUMMARY]
                if len(content) > MAX_CAT_CHARS_FOR_SUMMARY:
                    content_for_agi += "\n[...content truncated...]"

                agi_prompt = f"Regarding the following content from file '{target_path.name}':\n\n{content_for_agi}\n\nPlease summarize it or answer questions I might have."

                console.print(f"\n[info]Sending summary request for {target_path.name} to AGI...[/info]")
                with console.status("[yellow]AGI is processing file content...[/yellow]", spinner="dots"):
                    agi_response_text = agi_interface.generate_response(agi_prompt)

                # Log interaction
                if session_logger:
                    session_logger.log_entry("User", f"/cat {file_path_str} (sent to AGI)")
                    session_logger.log_entry("AGI", agi_response_text)
                conversation_history.append({"role": "user", "content": f"/cat {file_path_str} (sent to AGI: {target_path.name})", "timestamp": datetime.now().isoformat()})
                conversation_history.append({"role": "assistant", "content": agi_response_text, "timestamp": datetime.now().isoformat()})

                # Display AGI response
                response_parts = detect_code_blocks(agi_response_text)
                console.print(f"[agiprompt]AGI Analysis of {target_path.name}:[/agiprompt]")
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

    except Exception as e:
        console.print(f"[red]Error in /cat command for '{file_path_str}': {type(e).__name__} - {e}[/red]")


def edit_file_command(file_path_str: str):
    target_path = Path(file_path_str) # Don't resolve yet, editor might create it

    editor = os.environ.get('EDITOR')
    if not editor:
        system = platform.system()
        if system == "Windows":
            editor = "notepad"
        elif system in ["Linux", "Darwin"]: # Darwin is macOS
            # Prefer nano if available, then vim, then vi
            if shutil.which("nano"): editor = "nano"
            elif shutil.which("vim"): editor = "vim"
            elif shutil.which("vi"): editor = "vi"
            else: # A very basic fallback, though most systems will have one of the above
                console.print("[yellow]No default editor found (EDITOR variable not set, nano/vim/vi not found). Please set EDITOR environment variable.[/yellow]")
                return
        else: # Unknown system
            console.print(f"[yellow]Unsupported OS ({system}) for default editor detection. Please set EDITOR environment variable.[/yellow]")
            return

    console.print(f"Attempting to open '{target_path}' with editor '{editor}'...", style="info")
    try:
        # For terminal editors, subprocess.run usually works well.
        # For GUI editors on some systems, they might detach, or shell=True might be needed
        # but shell=True is a security risk if file_path_str is not sanitized (though less so here).
        # For simplicity and security, avoid shell=True if possible.
        process = subprocess.run([editor, str(target_path)], check=False)
        if process.returncode != 0:
            console.print(f"[yellow]Editor '{editor}' exited with code {process.returncode}.[/yellow]")
        else:
            console.print(f"Editor '{editor}' closed.", style="info")
            if not target_path.exists():
                 console.print(f"[yellow]Note: File '{target_path}' was not created/saved by the editor.[/yellow]")

    except FileNotFoundError:
        console.print(f"[red]Error: Editor '{editor}' not found. Please ensure it's installed and in your PATH or set the EDITOR environment variable.[/red]")
    except Exception as e:
        console.print(f"[red]Error opening editor '{editor}' for '{target_path}': {type(e).__name__} - {e}[/red]")


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

# --- Internet Search Implementation ---
def parse_duckduckgo_html_results(html_content: str, num_results: int = 5) -> list[dict]:
    """
    Parses DuckDuckGo HTML results to extract titles, snippets, and URLs.
    This is specific to DDG's HTML version and might break if their structure changes.
    """
    results = []
    # Regex to find result blocks. DDG HTML uses <div class="result"> or similar,
    # but class names can change. We'll look for common patterns.
    # A result typically has a link with class result__a, a snippet with result__snippet.
    # This is a simplified parser focusing on common structures.

    # Pattern for each result block (very approximate and fragile)
    # Looks for <a class="result__a" href="<url>">title</a> ... <a class="result__snippet">snippet</a>
    # The actual classes are often "result__a", "result__snippet", and title is within the <a> tag.
    # The URL in href usually needs to be de-proxied from duckduckgo.

    # Simplified regex focusing on common patterns seen in DDG HTML results
    # This will need refinement and testing.
    # Each result block is roughly: <h2 class="result__title">...<a href="URL">TITLE</a>...</h2>...<a class="result__snippet" href="URL">SNIPPET</a>
    # Or variations like: <div class="result"> <a class="result__a" href="...">TITLE</a> <div class="result__snippet">SNIPPET</div> </div>

    # Let's try to find result blocks and then extract parts.
    # This regex attempts to find a block starting with a result link and containing a snippet.
    # It is very basic and might need significant refinement.
    # For stability, a proper HTML parsing library (like BeautifulSoup) would be much better,
    # but per instructions, trying with regex/string methods first.

    # Regex to capture link, title, and snippet for each result
    # This pattern assumes a structure like:
    # <a class="result__a" href="<url>"> <text_title> </a> ... <a class="result__snippet" ...> <text_snippet> </a>
    # It's simplified and might need to be more robust.

    # A common pattern for DDG HTML results:
    # <div class="web-result">
    #   <div class="result__header">
    #     <a class="result__a" href="<actual_url_after_ddg_redirect_parsing>">TITLE</a>
    #   </div>
    #   <div class="result__body">
    #     <a class="result__snippet" href="...">SNIPPET</a>
    #   </div>
    # </div>
    # The URL in result__a needs de-proxying. Example: /l/?kh=-1&amp;uddg=DECODED_URL

    # More robust: find result blocks first, then parse within them
    result_blocks = re.findall(r'(<div class="web-result".*?</div>\s*</div>)', html_content, re.DOTALL)
    if not result_blocks: # Fallback to an older possible structure
        result_blocks = re.findall(r'(<div class="result".*?</div>\s*</div>)', html_content, re.DOTALL)


    for block in result_blocks:
        if len(results) >= num_results:
            break

        title_match = re.search(r'<a class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
        url_match = re.search(r'<a class="result__a"[^>]*href="([^"]*)"', block, re.DOTALL)
        snippet_match = re.search(r'<a class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)
        if not snippet_match: # Try another common snippet class
             snippet_match = re.search(r'<div class="result__snippet"[^>]*>(.*?)</div>', block, re.DOTALL)


        if title_match and url_match and snippet_match:
            raw_url = url_match.group(1)
            # Decode DDG's URL
            parsed_url_params = urllib.parse.parse_qs(urllib.parse.urlsplit(raw_url).query)
            actual_url = parsed_url_params.get('uddg', [raw_url])[0] # Get 'uddg' param if exists

            title = re.sub('<[^<]+?>', '', title_match.group(1)).strip() # Remove HTML tags
            snippet = re.sub('<[^<]+?>', '', snippet_match.group(1)).strip() # Remove HTML tags

            results.append({'title': title, 'snippet': snippet, 'url': actual_url})

    if not results and len(html_content) > 500 : # If main regex failed, try a very broad one as a last resort
        # This is a very greedy and less accurate fallback.
        # Looks for any link with some text after it.
        console.print("[dim]Main search parser failed, trying broad fallback...", style="warning")
        fallback_matches = re.findall(r'<a href="(http[^"]+)">(.*?)</a>.*?<p>(.*?)</p>', html_content, re.DOTALL | re.IGNORECASE)
        for url, title, snippet in fallback_matches:
            if len(results) >= num_results:
                break
            title = re.sub('<[^<]+?>', '', title).strip()
            snippet = re.sub('<[^<]+?>', '', snippet).strip()
            if len(title) > 10 and len(snippet) > 20 : # Basic quality check
                 results.append({'title': title, 'snippet': snippet, 'url': url})


    return results[:num_results]


def perform_internet_search(query: str):
    """Performs an internet search using DuckDuckGo HTML version and displays results."""
    console.print(f"Searching the web for: \"{query}\"...", style="info")

    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    # This is where the view_text_website tool would be called in a real agent environment
    # For now, we'll simulate it or expect it to be available via a different mechanism
    # if this script is run by an agent executor.
    # Since Jules has `view_text_website`, let's assume it can be called.
    # However, Jules' tools are not directly callable from the Python code it writes.
    # This function needs to be structured so Jules can call its tool.

    # To make this testable if run directly (and for Jules to adapt):
    # We can use requests here for direct execution, but Jules would replace this part.
    html_content = ""
    try:
        # This block is for direct execution / testing.
        # Jules would replace this with its `view_text_website` tool call.
        import requests
        headers = { # Mimic a common browser to avoid simple blocks
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        html_content = response.text
        console.print(f"Successfully fetched search results page (length: {len(html_content)} chars).", style="dim success")
    except ImportError:
        console.print("[bold red]Error:[/bold red] The 'requests' library is needed for direct search test. Jules would use its internal tools.", style="error")
        console.print("Please run `pip install requests` if you want to test this search directly.", style="info")
        return
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching search results:[/bold red] {e}", style="error")
        return
    except Exception as e: # Catch any other unexpected error
        console.print(f"[bold red]An unexpected error occurred during web fetch:[/bold red] {e}", style="error")
        return

    if not html_content:
        console.print("No content received from search page.", style="warning")
        return

    parsed_results = parse_duckduckgo_html_results(html_content)

    if not parsed_results:
        console.print("Could not parse any search results. The page structure might have changed or no results found.", style="warning")
        # console.print(f"[dim]Raw HTML snippet for debugging (first 1000 chars):\n{html_content[:1000]}[/dim]") # For debugging
        return

    console.print(f"\n[bold green]Search Results for \"{query}\":[/bold green]")
    for i, res in enumerate(parsed_results):
        panel_content = Text()
        panel_content.append(f"{i+1}. {res['title']}\n", style="bold link " + str(res['url'])) # Make URL clickable if terminal supports
        panel_content.append(f"   {res['snippet']}\n", style="italic")
        panel_content.append(f"   Source: {res['url']}", style="dim")
        console.print(Panel(panel_content, expand=False, border_style="blue"))

# --- Command Implementations ---
def list_directory_contents(path_str: str):
