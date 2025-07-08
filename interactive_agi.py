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

        # Add Project Root information if it's set and different from CWD
        # (PROJECT_ROOT_PATH is resolved, cwd from get_cwd_context() might not be, so resolve it for comparison)
        resolved_cwd = Path(cwd).resolve()
        if PROJECT_ROOT_PATH and PROJECT_ROOT_PATH != resolved_cwd:
            context_parts.append(f"ProjectRoot='{PROJECT_ROOT_PATH}'")

        if file_counts:
            file_str_parts = [f"{lang}:{count}" for lang, count in file_counts.items()]
            context_parts.append(f"Files=({', '.join(file_str_parts)})")

        # Add file snippets
        snippets = self.get_key_file_snippets()
        if snippets:
            context_parts.append("\nKey File Snippets:\n" + "\n".join(snippets))

        return "Context:\n" + "\n".join(context_parts)

    def get_key_file_snippets(self, max_files: int = 2, max_lines_per_file_half: int = 5, max_total_snippet_chars: int = 1000) -> list[str]:
        """Gets snippets from key files in the current directory."""
        snippets = []
        chars_added = 0

        # Prioritize README.md
        key_files_candidates = []
        readme_path = Path("README.md")
        if readme_path.exists() and readme_path.is_file():
            key_files_candidates.append(readme_path)

        # Then Python files, then other common text files, sort by modification time (most recent first)
        other_candidates = []
        py_files = []
        common_text_exts = ['.txt', '.sh', '.js', '.json', '.yml', '.toml', '.css', '.html']

        for item_path in Path(".").iterdir():
            if item_path.is_file() and item_path.name: # Ensure name is not empty for suffix to be meaningful
                if item_path.suffix == '.py' and item_path.name != "interactive_agi.py" and item_path.name != "setup_agi_terminal.py": # Exclude self
                    py_files.append(item_path)
                elif item_path.name.lower() != "readme.md" and item_path.suffix.lower() in common_text_exts: # Avoid re-adding README, check suffix after name
                    other_candidates.append(item_path)

        # Sort by modification time, most recent first
        py_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        other_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        key_files_candidates.extend(py_files)
        key_files_candidates.extend(other_candidates)

        files_processed_count = 0
        for file_path in key_files_candidates:
            if files_processed_count >= max_files:
                break
            if chars_added >= max_total_snippet_chars:
                break

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                if not lines: continue

                snippet_parts = []
                # Head
                head_snippet = "".join(lines[:max_lines_per_file_half])
                snippet_parts.append(f"--- Snippet from {file_path.name} (first {max_lines_per_file_half} lines) ---\n{head_snippet.strip()}")

                # Tail (if file is larger than twice the half-lines to show, to avoid overlap)
                if len(lines) > max_lines_per_file_half * 2:
                    tail_snippet = "".join(lines[-max_lines_per_file_half:])
                    snippet_parts.append(f"--- (last {max_lines_per_file_half} lines) ---\n{tail_snippet.strip()}")

                full_snippet_text = "\n".join(snippet_parts)

                if chars_added + len(full_snippet_text) > max_total_snippet_chars and snippets: # If not first snippet, and adding this exceeds total
                    remaining_chars = max_total_snippet_chars - chars_added
                    if remaining_chars > 50: # Only add if a meaningful part can be added
                        snippets.append(full_snippet_text[:remaining_chars] + "\n[...snippet truncated...]")
                    chars_added = max_total_snippet_chars # Mark as full
                    break

                snippets.append(full_snippet_text)
                chars_added += len(full_snippet_text)
                files_processed_count += 1

            except Exception: # Ignore errors reading individual files for snippets
                pass

        return snippets

    def get_full_context_dict(self) -> dict:
        """Returns the context information as a dictionary."""
        cwd = self.get_cwd_context()
        git_info = self.get_git_context()
        file_counts = self.get_file_type_counts()

        context_dict = {
            "cwd": cwd,
            "git_info": git_info, # This is already a dict
            "file_type_counts": file_counts, # This is already a dict
            "key_file_snippets": self.get_key_file_snippets() # This is a list of strings
        }

        # Add project root if available and different from CWD
        resolved_cwd = Path(cwd).resolve()
        if PROJECT_ROOT_PATH and PROJECT_ROOT_PATH != resolved_cwd:
            context_dict["project_root"] = str(PROJECT_ROOT_PATH)

        return context_dict

from collections import deque # For conversation history
from rich.table import Table # For /ls command
from datetime import datetime # For history timestamps

# Global context analyzer instance
context_analyzer = ContextAnalyzer()

from collections import deque # For conversation history
from rich.table import Table # For /ls command and sysinfo
from rich.tree import Tree # For /analyze_dir command
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

# TOML import for config file
try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False
    console.print("WARNING: `toml` library not found. Configuration file (`config.toml`) cannot be used.", style="warning")
    console.print("Install with: pip install toml", style="info")


# --- Configuration & Globals ---
CACHE_DIR = Path("./.agi_terminal_cache")
HISTORY_FILE_PATH = CACHE_DIR / "history.json"
CONFIG_FILE_PATH = CACHE_DIR / "config.toml"

DEFAULT_CONFIG = {
    "generation": {
        "default_temperature": 0.4,
        "default_max_tokens": 1024,
        "default_top_p": 0.9,
        "default_repetition_penalty": 1.15,
    },
    "history": {
        "max_len": 100, # Max internal history deque length
    },
    "display": {
        "syntax_theme": "monokai",
    },
    "logging": {
        "desktop_logging_enabled": True,
        "desktop_log_path_override": "", # Empty means use default desktop path
        "jsonl_logging_enabled": True, # New setting for JSONL interaction logs
    }
}
APP_CONFIG = {} # Will be loaded from file or defaults

# Global context analyzer instance
context_analyzer = ContextAnalyzer()

# Global conversation history - maxlen will be set from config
conversation_history = deque()

# --- Desktop Path for Session Logs ---
def get_desktop_path() -> Path:
    """
    Determines the user's desktop path.
    Uses APP_CONFIG['logging']['desktop_log_path_override'] if set.
    """
    override_path_str = APP_CONFIG.get("logging", {}).get("desktop_log_path_override", "")
    if override_path_str:
        override_path = Path(override_path_str).expanduser()
        if override_path.is_dir(): # Check if user provided path is a valid directory
            console.print(f"INFO: Using custom desktop log path from config: {override_path}", style="info")
            return override_path
        else:
            console.print(f"WARNING: Custom desktop log path '{override_path_str}' is not a valid directory. Using default.", style="warning")
            # Fall through to default detection if override is invalid

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

# --- User Command Scripts (Macros) ---
USER_SCRIPTS_FILE_PATH = CACHE_DIR / "user_scripts.json"

def load_user_scripts() -> dict:
    """Loads user-defined command scripts from the JSON file."""
    if USER_SCRIPTS_FILE_PATH.exists():
        try:
            with open(USER_SCRIPTS_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[warning]Error loading user scripts from {USER_SCRIPTS_FILE_PATH}: {e}[/warning]")
    return {}

def save_user_scripts(scripts: dict):
    """Saves the user-defined command scripts to the JSON file."""
    try:
        USER_SCRIPTS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_SCRIPTS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(scripts, f, indent=2)
    except IOError as e:
        console.print(f"[error]Could not save user scripts to {USER_SCRIPTS_FILE_PATH}: {e}[/error]")


# --- JSONL Interaction Logging ---
JSONL_LOG_FILE_PATH = CACHE_DIR / "interaction_logs.jsonl"
JSONL_LOGGING_ENABLED = True # Will be updated from config
SESSION_ID = "" # Will be set at startup
TURN_ID_COUNTER = 0

def log_interaction_to_jsonl(interaction_data: dict):
    if not JSONL_LOGGING_ENABLED:
        return
    try:
        JSONL_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSONL_LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            json.dump(interaction_data, f)
            f.write('\n')
    except Exception as e:
        console.print(f"[warning]Failed to write to JSONL interaction log: {e}[/warning]")


# --- Project Root Detection ---
PROJECT_ROOT_PATH: Optional[Path] = None

def find_project_root(start_path: Path) -> Optional[Path]:
    """
    Searches upwards from start_path for project markers like .git or .agi_project_root.
    Returns the path to the project root directory if found, otherwise None.
    """
    current_dir = start_path.resolve()
    # Limit upward search to avoid going to the very root of the filesystem unnecessarily
    # or getting stuck if home dir is not accessible in some edge cases.
    # Max depth of 10 parent directories seems reasonable for most project structures.
    for _ in range(10):
        # Check for .git directory (common for git repositories)
        git_dir = current_dir / ".git"
        if git_dir.is_dir():
            return current_dir

        # Check for a custom marker file (e.g., if not a git repo but user wants to define scope)
        marker_file = current_dir / ".agi_project_root"
        if marker_file.is_file():
            return current_dir

        # Move to parent directory
        parent_dir = current_dir.parent
        if parent_dir == current_dir: # Reached filesystem root
            break
        current_dir = parent_dir

    return None


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
        self.last_detected_task_type = "general" # Initialize last detected task type

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
            task_type = "general" # Default task type
            task_prefix = "General Query: "

            # Order matters: more specific keyword sets should come first.
            code_generation_keywords = ["generate code", "write a script", "create a function", "implement class", "python code for", "javascript for"]
            code_debugging_keywords = ["debug this code", "fix this error", "what's wrong with this snippet", "error in my code"]
            code_explanation_keywords = ["explain this code", "what does this function do", "code walkthrough"]
            summarization_keywords = ["summarize", "tl;dr", "give me a summary", "briefly explain this document"]
            git_keywords = ["git status", "git diff", "git log", "git commit", "git branch"] # For when user asks AGI about git, not just runs /git
            explanation_keywords = ["explain", "what is", "describe", "how does", "tell me about", "define"] # General explanations

            if any(keyword in query_lower for keyword in code_generation_keywords):
                task_type = "code_generation"
                task_prefix = "Code Generation Task: "
            elif any(keyword in query_lower for keyword in code_debugging_keywords):
                task_type = "code_debugging"
                task_prefix = "Code Debugging Task: "
            elif any(keyword in query_lower for keyword in code_explanation_keywords):
                task_type = "code_explanation"
                task_prefix = "Code Explanation Request: "
            elif any(keyword in query_lower for keyword in summarization_keywords):
                task_type = "summarization"
                task_prefix = "Summarization Request: "
            elif any(keyword in query_lower for keyword in git_keywords) and not prompt.strip().startswith("/git"): # Avoid if it's already a /git command
                task_type = "git_query"
                task_prefix = "Git Related Query: "
            elif any(keyword in query_lower for keyword in explanation_keywords):
                task_type = "explanation"
                task_prefix = "Explanation Request: "

            # Store task_type for potential use in response styling (though not directly returned by this func yet)
            self.last_detected_task_type = task_type


            # Construct the full prompt with context and task-specific prefix

            # --- Tool Execution Prompt Engineering ---
            # Check if the query might imply a shell command the AGI could suggest
            shell_command_keywords = ["list files", "show directory", "what is my current path", "print working directory", "show disk space", "display memory usage", "what's the date", "system information", "who am i"]
            prompt_for_tool_use = False
            if any(keyword in query_lower for keyword in shell_command_keywords):
                prompt_for_tool_use = True

            if prompt_for_tool_use:
                # Guide the AGI to consider using a shell command
                tool_instruction = (
                    "You have access to a 'run_shell' tool. If you can answer the query by suggesting a safe, read-only shell command from the allowed list, "
                    "please respond ONLY with a JSON object in the following format:\n"
                    "{\n"
                    "  \"action\": \"run_shell\",\n"
                    "  \"command\": \"<command_executable_name>\",\n"
                    "  \"args\": [\"<argument1>\", \"<argument2>\", ...],\n"
                    "  \"reasoning\": \"<brief_explanation_for_this_command>\"\n"
                    "}\n"
                    "Details:\n"
                    "- The 'command' field MUST be one of the following whitelisted commands: " + f"{', '.join(SHELL_COMMAND_WHITELIST)}. \n"
                    "- The 'args' field MUST be a list of strings, where each string is a separate argument to the command. Do not include the command itself in 'args'.\n"
                    "- Arguments should be simple and not contain shell metacharacters (like pipes |, redirection >, variables $, etc.).\n"
                    "- Example for listing files: {\"action\": \"run_shell\", \"command\": \"ls\", \"args\": [\"-la\", \"/tmp\"], \"reasoning\": \"List all files with details in /tmp directory.\"}\n"
                    "- Example for showing first 10 lines of a file: {\"action\": \"run_shell\", \"command\": \"head\", \"args\": [\"-n\", \"10\", \"my_file.txt\"], \"reasoning\": \"Show the first 10 lines of my_file.txt.\"}\n"
                    "- Example for counting lines in a file: {\"action\": \"run_shell\", \"command\": \"wc\", \"args\": [\"-l\", \"data.csv\"], \"reasoning\": \"Count the number of lines in data.csv.\"}\n"
                    "- Example for searching text in a file: {\"action\": \"run_shell\", \"command\": \"grep\", \"args\": [\"-i\", \"warning\", \"application.log\"], \"reasoning\": \"Search for 'warning' (case-insensitive) in application.log.\"}\n"
                    "\n"
                    "You also have access to a 'read_file' tool. If you need to read the content of a file to answer the user's query, "
                    "respond ONLY with a JSON object in the following format:\n"
                    "{\n"
                    "  \"action\": \"read_file\",\n"
                    "  \"filepath\": \"<path_to_file_relative_to_project_root>\",\n"
                    "  \"max_lines\": <optional_integer_max_lines_to_read_for_context_usually_around_100_to_200>,\n"
                    "  \"reasoning\": \"<brief_explanation_why_you_need_to_read_this_file>\"\n"
                    "}\n"
                    "Details for 'read_file':\n"
                    "- 'filepath' MUST be a relative path to a file (relative to project root). Do not use absolute paths.\n"
                    "- 'max_lines' is optional. If the file is very large, the system may truncate it or provide a summary. Suggesting max_lines helps focus.\n"
                    "- After you request a file read, the system will provide its content (or a part of it) in the next turn, along with the original user query. You should then use that information to formulate your final answer.\n"
                    "- Example: {\"action\": \"read_file\", \"filepath\": \"src/utils.py\", \"max_lines\": 100, \"reasoning\": \"To understand the helper functions available before answering how to implement the feature.\"}\n"
                    "\n"
                    "Additionally, you have a 'write_file' tool. If you need to create a new file or suggest modifications to an existing one, "
                    "respond ONLY with a JSON object in the following format:\n"
                    "{\n"
                    "  \"action\": \"write_file\",\n"
                    "  \"filepath\": \"<path_to_file_relative_to_project_root>\",\n"
                    "  \"content\": \"<full_content_to_be_written>\",\n"
                    "  \"reasoning\": \"<brief_explanation_for_this_file_creation_or_modification>\"\n"
                    "}\n"
                    "Details for 'write_file':\n"
                    "- 'filepath' MUST be a relative path (relative to project root). Do not use absolute paths.\n"
                    "- 'content' MUST be the complete, new content for the file.\n"
                    "- ALL write operations require explicit user confirmation. If the file exists, a diff will be shown to the user. If it's a new file, the full content will be shown for confirmation.\n"
                    "- Example (new file): {\"action\": \"write_file\", \"filepath\": \"notes/todo.txt\", \"content\": \"- Buy milk\\n- Plan project\", \"reasoning\": \"Create a new todo list file.\"}\n"
                    "- Example (suggesting modification to existing file - system shows diff to user): {\"action\": \"write_file\", \"filepath\": \"app/config.py\", \"content\": \"DEBUG = False\\nSECRET_KEY = 'new_secret'\", \"reasoning\": \"Disable debug mode and update secret key.\"}\n"
                    "\n"
                    "Finally, you have a 'web_search' tool. If you need to search the internet to answer the user's query, "
                    "respond ONLY with a JSON object in the following format:\n"
                    "{\n"
                    "  \"action\": \"web_search\",\n"
                    "  \"query\": \"<your_search_query_string>\",\n"
                    "  \"reasoning\": \"<brief_explanation_why_internet_search_is_needed>\"\n"
                    "}\n"
                    "Details for 'web_search':\n"
                    "- The system will perform the search and provide you with a summary of results in the next turn.\n"
                    "- You should then use that information to formulate your final answer to the user.\n"
                    "- Example: {\"action\": \"web_search\", \"query\": \"current weather in London\", \"reasoning\": \"To fetch the current weather conditions in London for the user.\"}\n"
                    "\n"
                    "You also have Git-related tools. To suggest creating a new Git branch, use:\n"
                    "{\n"
                    "  \"action\": \"git_branch_create\",\n"
                    "  \"branch_name\": \"<new_branch_name>\",\n"
                    "  \"base_branch\": \"<optional_source_branch_name>\",\n"
                    "  \"reasoning\": \"<why_create_this_branch>\"\n"
                    "}\n"
                    "Details for 'git_branch_create':\n"
                    "- 'branch_name' is required. 'base_branch' is optional; if omitted, the new branch starts from the current HEAD.\n"
                    "- User confirmation will be required before the branch is created.\n"
                    "- Example: {\"action\": \"git_branch_create\", \"branch_name\": \"feature/new-widget\", \"base_branch\": \"develop\", \"reasoning\": \"To start development on the new widget feature, branching off develop.\"}\n"
                    "\n"
                    "To suggest checking out a Git branch (existing or new with -b), use:\n"
                    "{\n"
                    "  \"action\": \"git_checkout\",\n"
                    "  \"branch_name\": \"<branch_name_to_checkout_or_create>\",\n"
                    "  \"create_new\": <true_or_false>, \n"
                    "  \"reasoning\": \"<why_checkout_this_branch>\"\n"
                    "}\n"
                    "Details for 'git_checkout':\n"
                    "- 'branch_name' is required. 'create_new' is a boolean (true for 'checkout -b', false for regular checkout), defaults to false if omitted.\n"
                    "- User confirmation will be required.\n"
                    "- Example (checkout existing): {\"action\": \"git_checkout\", \"branch_name\": \"main\", \"create_new\": false, \"reasoning\": \"Switch to the main branch.\"}\n"
                    "- Example (create and checkout new): {\"action\": \"git_checkout\", \"branch_name\": \"bugfix/login-issue\", \"create_new\": true, \"reasoning\": \"Create and switch to a new branch for fixing the login issue.\"}\n"
                    "\n"
                    "To request execution of a Python code snippet, use:\n"
                    "{\n"
                    "  \"action\": \"execute_python_code\",\n"
                    "  \"code\": \"<python_code_snippet_as_a_string>\",\n"
                    "  \"reasoning\": \"<why_this_code_should_be_run>\"\n"
                    "}\n"
                    "Details for 'execute_python_code':\n"
                    "- The 'code' MUST be a self-contained Python 3 snippet. It will be executed in a restricted environment.\n"
                    "- DO NOT attempt direct file I/O (e.g., `open()`) or network requests in the code. Use `read_file`, `write_file`, or `web_search` tools for those purposes.\n"
                    "- DO NOT use `import os`, `import sys`, `import subprocess`, or other system-level imports.\n"
                    "- Only a limited set of safe built-in functions and potentially a few safe standard library modules (like `math`, `json`, `datetime`, `random`, `re` if explicitly allowed by admin) can be used. Assume imports are disallowed unless told otherwise for specific safe modules.\n"
                    "- All code execution requires explicit user confirmation. The user will see the exact code you provide.\n"
                    "- Output (stdout, stderr) and exceptions from the code will be returned to you in the next turn.\n"
                    "- Example: {\"action\": \"execute_python_code\", \"code\": \"data = [1, 2, 3, 4, 5]\\nprint(sum(data))\\nprint('Max value:', max(data))\", \"reasoning\": \"To calculate and print the sum and max of a list as requested.\"}\n"
                    "\n"
                    "To suggest a Git commit, use:\n"
                    "{\n"
                    "  \"action\": \"git_commit\",\n"
                    "  \"commit_message\": \"<your_commit_message>\",\n"
                    "  \"stage_all\": <true_or_false>, \n"
                    "  \"reasoning\": \"<why_this_commit_is_needed>\"\n"
                    "}\n"
                    "Details for 'git_commit':\n"
                    "- 'commit_message' is required. 'stage_all' is a boolean (defaults to false); if true, it's like `git commit -a -m \"message\"` (stages all tracked, modified files).\n"
                    "- If 'stage_all' is false, ensure files are already staged by the user or through prior operations.\n"
                    "- User confirmation will be required.\n"
                    "- Example: {\"action\": \"git_commit\", \"commit_message\": \"Fix: Correct typo in README\", \"stage_all\": true, \"reasoning\": \"To commit the typo correction in the README file.\"}\n"
                    "\n"
                    "To suggest a Git push, use:\n"
                    "{\n"
                    "  \"action\": \"git_push\",\n"
                    "  \"remote_name\": \"<optional_remote_name>\",\n"
                    "  \"branch_name\": \"<optional_branch_to_push>\",\n"
                    "  \"reasoning\": \"<why_push_is_needed>\"\n"
                    "}\n"
                    "Details for 'git_push':\n"
                    "- 'remote_name' is optional (defaults to 'origin'). 'branch_name' is optional (defaults to the current Git HEAD branch).\n"
                    "- Force pushing is NOT supported via this tool for safety.\n"
                    "- User confirmation will be required.\n"
                    "- Example: {\"action\": \"git_push\", \"remote_name\": \"origin\", \"branch_name\": \"feature/login\", \"reasoning\": \"To push the completed login feature to the remote repository.\"}\n"
                    "\n"
                    "General Guidelines for Tool Use and Error Handling:\n"
                    "- If you request a tool action (e.g., read_file, run_shell, execute_python_code) and the system informs you in the next turn that the action failed (e.g., file not found, command error, code execution exception, search failed), you MUST acknowledge this failure in your response to the user.\n"
                    "- Explain the problem clearly and concisely based on the feedback provided by the system.\n"
                    "- If appropriate, ask the user for clarification (e.g., \"The file you specified was not found, could you please check the path?\" or \"The command resulted in an error, perhaps try a different approach?\").\n"
                    "- If a user's request is too ambiguous for you to confidently choose a tool or its parameters, ask for clarification BEFORE attempting to use a tool.\n"
                    "- Your primary goal is to be a helpful and coherent assistant. Use the information from tool outcomes (both success and failure) to inform your final response to the user for that turn.\n"
                    "- For complex user requests that require multiple operations (e.g., reading several files, processing data, then writing a result):\n"
                    "    - Briefly outline your plan in your reasoning for the first tool call, or in a natural language response if you need to clarify the overall goal with the user first.\n"
                    "    - Request tools one at a time.\n"
                    "    - After the system provides the outcome of a tool action, carefully consider that outcome to decide your next step. This might be another tool call (the next part of your plan) or generating the final answer.\n"
                    "    - If a step in your plan fails, inform the user about the failure and why your overall plan might be affected or needs to change.\n"
                    "\n"
                    "If the query cannot be answered with one of these tools ('run_shell', 'read_file', 'write_file', 'web_search', 'git_branch_create', 'git_checkout', 'execute_python_code', 'git_commit', 'git_push'), or if it requires actions not supported, answer directly as a helpful assistant without using the JSON format.\n"
                    "Choose only ONE action per turn if you decide to use a tool. Do not combine them in one JSON response.\n"
                )
                full_prompt = f"{current_context_str}\n\nTool Instructions:\n{tool_instruction}\n\nUser Query:\n{task_prefix}{prompt}"
            else:
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
            expected_type = "integer" if actual_param_name in ["max_new_tokens", "top_k"] else "float"
            return f"[bold red]Error:[/bold red] Invalid value '{param_value_str}' for {param_name_upper}. Expected an {expected_type}."
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
    global session_logger, APP_CONFIG, conversation_history, DEFAULT_GENERATION_PARAMS, PROJECT_ROOT_PATH, SESSION_ID, JSONL_LOGGING_ENABLED

    # --- Initialize Session ID ---
    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S_") + Path(sys.argv[0]).stem # e.g., 20231027_103000_interactive_agi

    # --- Initialize Project Root ---
    # Done early as other parts might use it (e.g. config loading from project root in future)
    PROJECT_ROOT_PATH = find_project_root(Path.cwd())
    if PROJECT_ROOT_PATH:
        console.print(f"INFO: Project root detected at: [cyan]{PROJECT_ROOT_PATH}[/cyan]", style="info")
    else:
        PROJECT_ROOT_PATH = Path.cwd().resolve() # Fallback to current working directory
        console.print(f"WARNING: No project root marker (.git or .agi_project_root) found. "
                      f"Using current working directory as project scope: [cyan]{PROJECT_ROOT_PATH}[/cyan]", style="warning")

    # Load config first, as it affects other initializations
    load_config()

    # Now that APP_CONFIG is loaded, initialize things that depend on it
    # (conversation_history deque maxlen is handled in load_config)
    # (DEFAULT_GENERATION_PARAMS is updated in load_config)

    desktop_path = get_desktop_path() # Uses APP_CONFIG for override
    session_logger = SessionLogger(desktop_path)
    if APP_CONFIG.get("logging", {}).get("desktop_logging_enabled", True) == False:
        session_logger.enabled = False # Ensure logger respects config if disabled
        console.print("[info]Desktop session logging is disabled via config.[/info]")


    display_startup_banner() # Uses APP_CONFIG for syntax_theme implicitly via console
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

    # --- Main Input Processing Function ---
def process_single_input_turn(user_input_str: str,
                              agi_interface_instance: Union[MergedAGI, AGIPPlaceholder],
                              turn_log_data_ref: dict,
                              is_scripted_input: bool = False) -> str: # Returns "CONTINUE" or "EXIT"
    """
    Processes a single input string as one turn of interaction.
    This function contains the main logic for handling user slash commands or
    sending input to the AGI, including tool processing and logging.
    It's called by the main loop and by the /run_script command.

    Args:
        user_input_str: The input string to process.
        agi_interface_instance: The active AGI interface.
        turn_log_data_ref: Reference to the dictionary for logging this turn's data.
        is_scripted_input: True if this input comes from a /run_script command.

    Returns:
        "CONTINUE" if the terminal should continue to the next input.
        "EXIT" if an 'exit' or 'quit' command was processed.
    """
    global final_text_for_user_display # To ensure it's assigned if an early exit happens
    final_text_for_user_display = "" # Reset for the turn

    # Log the raw user query for this turn (already set by main loop or /run_script)
    # turn_log_data_ref["user_query"] = user_input_str
    # turn_log_data_ref["timestamp_user_query"] = datetime.now().isoformat() (also set by caller)

    if user_input_str.strip().lower() in ["exit", "quit"]:
        console.print("Exiting AGI session.", style="info")
        # For JSONL logging, we might want to log this exit action.
        turn_log_data_ref["agi_final_response_to_user"] = "User initiated exit."
        turn_log_data_ref["timestamp_final_response"] = datetime.now().isoformat()
        log_interaction_to_jsonl(turn_log_data_ref)
        return "EXIT"

    if not user_input_str.strip():
        return "CONTINUE" # Skip empty inputs

    # Add user input to internal conversation history and plain text session log
    # This happens regardless of whether it's a command or AGI query.
    conversation_history.append({"role": "user", "content": user_input_str, "timestamp": datetime.now().isoformat()})
    if session_logger and not is_scripted_input: # Avoid double logging from /run_script's own print
        session_logger.log_entry("User", user_input_str)

    # Default assumption: action_taken_by_tool_framework will be false unless a tool is successfully parsed and handled.
    action_taken_by_tool_framework = False

    # --- User Command Processing ---
    # This large if/elif block handles slash commands.
    # If a command is handled, it typically prints its own output and this function will return "CONTINUE".
    # If no slash command matches, it falls through to the 'else' which processes input via AGI.

    if user_input_str.lower().startswith("/set parameter "):
        parts = user_input_str.strip().split(maxsplit=3)
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
            elif user_input.lower().startswith("/read_script "):
                script_name_to_read = user_input[len("/read_script "):].strip()
                allowed_scripts = ["interactive_agi.py", "setup_agi_terminal.py", "adaptive_train.py", "download_models.sh", "train_on_interaction.sh", "merge_config.yml"]
                if script_name_to_read and script_name_to_read in allowed_scripts:
                    # Check if file exists in current dir or one level up (common for dev setups)
                    script_path = Path(script_name_to_read)
                    if not script_path.exists():
                        script_path = Path("..") / script_name_to_read # Check parent if not in CWD

                    if script_path.exists() and script_path.is_file():
                         cat_file_command(str(script_path), agi_interface)
                    else:
                         console.print(f"[red]Error: Script '{script_name_to_read}' not found at expected locations.[/red]")
                elif script_name_to_read:
                    console.print(f"[red]Error: Reading script '{script_name_to_read}' is not allowed. Allowed scripts: {', '.join(allowed_scripts)}[/red]")
                else:
                    console.print("[red]Usage: /read_script <script_filename>[/red]")
            elif user_input.lower().startswith("/config"):
                config_args = user_input.strip()[len("/config"):].strip() # Get args after /config
                config_command_handler(config_args)
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
            elif user_input.lower() == "/help":
                display_help_command()
                # For /help and other user commands that don't involve AGI, we might log them differently or skip full AGI log
                # For now, let's assume they might have a minimal log entry or are handled outside the main AGI interaction logging path
                # For simplicity in this step, we'll focus on interactions that go to the AGI.
                # So, if a command is handled here, it bypasses the main AGI JSONL logging for the turn.
                # This could be refined later if needed. A simple log_interaction_to_jsonl call could be made here too.
            elif user_input.lower().startswith("/save_script "):
                parts = user_input.strip().split(maxsplit=2)
                if len(parts) < 3:
                    console.print("[red]Usage: /save_script <script_name> <command1> ; <command2> ; ...[/red]")
                else:
                    script_name = parts[1]
                    commands_str = parts[2]
                    # Basic script name validation
                    if not re.match(r"^[a-zA-Z0-9_-]+$", script_name):
                        console.print(f"[red]Invalid script name: '{script_name}'. Use alphanumeric characters, underscores, or hyphens.[/red]")
                    elif not commands_str.strip():
                        console.print(f"[red]Cannot save an empty script for '{script_name}'.[/red]")
                    else:
                        commands_list = [cmd.strip() for cmd in commands_str.split(';') if cmd.strip()]
                        if not commands_list:
                             console.print(f"[red]No valid commands found in script string for '{script_name}'. Use ';' to separate commands.[/red]")
                        else:
                            scripts = load_user_scripts()
                            scripts[script_name] = commands_list
                            save_user_scripts(scripts)
                            console.print(f"[green]Script '{script_name}' saved with {len(commands_list)} command(s).[/green]")
            elif user_input_str.lower().startswith("/run_script "):
                parts = user_input_str.strip().split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    console.print("[red]Usage: /run_script <script_name>[/red]")
                else:
                    script_name_to_run = parts[1].strip()
                    scripts = load_user_scripts()
                    if script_name_to_run not in scripts:
                        console.print(f"[yellow]Script '{script_name_to_run}' not found.[/yellow]")
                    else:
                        console.print(f"[info]Running script: '{script_name_to_run}'...[/info]")
                        script_commands = scripts[script_name_to_run]
                        for i, command_str in enumerate(script_commands):
                            console.print(f"\n[magenta]--- Script '{script_name_to_run}' Command {i+1}/{len(script_commands)} ---[/magenta]")
                            console.print(f"[dim]Executing: [/dim][bold cyan]{command_str}[/bold cyan]")

                            action = ""
                            if RICH_AVAILABLE: # More controlled input if Rich is available
                                action_prompt = Text.assemble(
                                    ("Press ", "magenta"), ("ENTER", "bold white"), (" to execute, ", "magenta"),
                                    ("S", "bold white"), (" to skip, ", "magenta"),
                                    ("A", "bold white"), (" to abort script: ", "magenta")
                                )
                                user_choice = console.input(action_prompt).lower().strip()
                            else: # Basic input
                                user_choice = input("Press ENTER to execute, S to skip, A to abort script: ").lower().strip()

                            if user_choice == 's':
                                console.print("[yellow]Skipped.[/yellow]")
                                continue
                            elif user_choice == 'a':
                                console.print("[yellow]Script aborted by user.[/yellow]")
                                break
                            elif user_choice == "": # Enter pressed
                                global TURN_ID_COUNTER # Need to modify global counter
                                TURN_ID_COUNTER +=1
                                script_turn_data = {
                                    "session_id": SESSION_ID, # Use global session ID
                                    "turn_id": TURN_ID_COUNTER,
                                    "timestamp_user_query": datetime.now().isoformat(),
                                    "user_query": command_str, # The command from the script
                                    "tool_interactions": [],
                                    "script_info": {"parent_script": script_name_to_run, "command_index": i}
                                }
                                # Call the main processing function for this command
                                # is_scripted_input=True prevents double logging of user input to session_logger
                                status = process_single_input_turn(command_str, agi_interface_instance, script_turn_data, is_scripted_input=True)
                                if status == "EXIT": # If the scripted command was 'exit' or 'quit'
                                    console.print(f"[yellow]Script '{script_name_to_run}' encountered an exit command. Script terminated.[/yellow]")
                                    # We need to decide if 'exit' in a script exits the whole app or just the script.
                                    # For now, let's make it terminate the script only. The main loop will continue.
                                    # If we wanted it to exit the app, this function would need to propagate "EXIT"
                                    break
                            else:
                                console.print("[yellow]Invalid choice. Skipping command.[/yellow]")
                                continue
                        else: # For loop completed without break
                            console.print(f"[info]Script '{script_name_to_run}' finished.[/info]")
            elif user_input.lower() == "/list_scripts":
                scripts = load_user_scripts()
                if not scripts:
                    console.print("[yellow]No user scripts saved yet. Use /save_script to create one.[/yellow]")
                else:
                    table = Table(title="[bold blue]Saved User Scripts[/bold blue]")
                    table.add_column("Script Name", style="cyan", no_wrap=True)
                    table.add_column("# Commands", style="magenta", justify="right")
                    table.add_column("Commands (first few shown)", style="white")
                    for name, cmds in sorted(scripts.items()):
                        cmds_preview = " ; ".join(cmds[:3])
                        if len(cmds) > 3:
                            cmds_preview += " ; ..."
                        table.add_row(name, str(len(cmds)), cmds_preview)
                    console.print(table)
            elif user_input.lower().startswith("/delete_script "):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    console.print("[red]Usage: /delete_script <script_name>[/red]")
                else:
                    script_name_to_delete = parts[1].strip()
                    scripts = load_user_scripts()
                    if script_name_to_delete in scripts:
                        del scripts[script_name_to_delete]
                        save_user_scripts(scripts)
                        console.print(f"[green]Script '{script_name_to_delete}' deleted.[/green]")
                    else:
                        console.print(f"[yellow]Script '{script_name_to_delete}' not found.[/yellow]")
            elif user_input.lower().startswith("/suggest_code_change "): # New handler
                file_path_to_suggest = user_input[len("/suggest_code_change "):].strip()
                if file_path_to_suggest:
                    suggest_code_change_command(file_path_to_suggest, agi_interface)
                else:
                    console.print("[red]Usage: /suggest_code_change <file_path>[/red]")
            else: # Default: AGI generates a response or suggests a command
                # Capture context before AGI call
                current_turn_interaction_data["context_at_query_time"] = context_analyzer.get_full_context_dict()
                current_turn_interaction_data["agi_initial_processing_details"] = {
                    "generation_params_used": agi_interface.generation_params.copy(), # Log a copy
                    "detected_task_type": agi_interface.last_detected_task_type # This is updated before generation
                }

                with console.status("[yellow]AGI is thinking...[/yellow]", spinner="dots"):
                    # Note: generate_response uses context_analyzer internally, so context is applied there.
                    # The user_input here is the raw one.
                    agi_initial_raw_response = agi_interface.generate_response(user_input)

                current_turn_interaction_data["agi_initial_raw_response"] = agi_initial_raw_response
                agi_response_text = agi_initial_raw_response # This might be overwritten if a tool is used and re-prompts AGI

                action_taken_by_tool_framework = False # True if a tool action is successfully initiated
                                                       # False if it's a direct answer or tool fails before execution.

                # This variable will hold the final text displayed to the user for this turn.
                # It starts as the AGI's initial response, but can be updated by tool processing outcomes.
                final_text_for_user_display = agi_initial_raw_response # Default if no tool or tool fails early

                # --- Try to process AGI response as a potential tool request ---
                try:
                    # Attempt to parse for tool use first
                    # AGI might return JSON directly or within ```json ... ``` markdown.
                    json_match = re.search(r"```json\n(.*?)\n```", agi_initial_raw_response, re.DOTALL)
                    json_str_to_parse = json_match.group(1) if json_match else agi_response_text

                    first_brace = json_str_to_parse.find('{')
                    last_brace = json_str_to_parse.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str_to_parse = json_str_to_parse[first_brace : last_brace+1]

                    data = json.loads(json_str_to_parse)

                    if isinstance(data, dict) and data.get("action") == "run_shell":
                        command_executable = data.get("command")
                        command_args_list = data.get("args", [])
                        reasoning = data.get("reasoning", "No reasoning provided.")

                        # Ensure command_args_list is actually a list of strings
                        if not isinstance(command_args_list, list) or not all(isinstance(arg, str) for arg in command_args_list):
                            console.print(f"[warning]AGI suggested command with invalid 'args' format. Expected a list of strings. Payload: {data}[/warning]")
                            action_taken_by_tool_framework = False
                            agi_response_text = f"AGI's suggested command had malformed arguments. Reasoning: {reasoning}"
                        elif command_executable and command_executable in SHELL_COMMAND_WHITELIST:
                            command_to_display = f"{command_executable} {' '.join(command_args_list)}"
                            console.print(Panel(Text(f"AGI suggests running command: [bold cyan]{command_to_display}[/bold cyan]\nReason: {reasoning}", style="yellow"), title="[bold blue]Shell Command Suggestion[/bold blue]"))

                            confirmed = False
                            if RICH_AVAILABLE:
                                from rich.prompt import Confirm
                                confirmed = Confirm.ask("Execute this command?", default=False, console=console)
                            else:
                                confirmed = input(f"Execute: {command_to_display}? (yes/NO): ").lower() == "yes"

                            tool_interaction_log_entry = {
                                "tool_request_timestamp": datetime.now().isoformat(), # Approx time of request processing
                                "action_type": "run_shell",
                                "action_details": {"command": command_executable, "args": command_args_list},
                                "reasoning": reasoning,
                                "user_confirmation": "confirmed" if confirmed else "cancelled"
                            }

                            if confirmed:
                                stdout_s, stderr_s, retcode = execute_shell_command(command_executable, command_args_list)
                                outcome_parts = []
                                if stdout_s and stdout_s.strip(): outcome_parts.append(f"Stdout: {stdout_s.strip()}")
                                if stderr_s and stderr_s.strip(): outcome_parts.append(f"Stderr: {stderr_s.strip()}")
                                if not outcome_parts and retcode == 0 : outcome_parts.append("Command executed with no output.")
                                elif not outcome_parts and retcode !=0 : outcome_parts.append(f"Command failed with return code {retcode} and no output.")

                                tool_outcome_summary = f"ReturnCode: {retcode}\n" + "\n".join(outcome_parts)
                                # Truncate for log if very long
                                if len(tool_outcome_summary) > 500:
                                    tool_outcome_summary = tool_outcome_summary[:497] + "..."
                                tool_interaction_log_entry["tool_outcome_summary"] = tool_outcome_summary
                                final_text_for_user_display = f"System: Executed command '{command_to_display}'. Output was printed above."
                            else:
                                console.print("Command execution cancelled by user.", style="yellow")
                                if session_logger: session_logger.log_entry("System", f"Cancelled execution of: {command_to_display}")
                                tool_interaction_log_entry["tool_outcome_summary"] = f"Command '{command_to_display}' execution cancelled by user."
                                final_text_for_user_display = "System: Command execution cancelled."

                            tool_interaction_log_entry["tool_outcome_timestamp"] = datetime.now().isoformat()
                            current_turn_interaction_data["tool_interactions"].append(tool_interaction_log_entry)
                            action_completed = True

                        elif command_executable: # Command suggested but not whitelisted
                            command_to_display = f"{command_executable} {' '.join(command_args_list)}"
                            console.print(f"[warning]AGI suggested a non-whitelisted command: '{command_to_display}'. Execution denied for safety.[/warning]")
                            if session_logger: session_logger.log_entry("AGI_Suggestion (Denied)", f"Command: {command_to_display}, Reason: {reasoning}")

                            current_turn_interaction_data["tool_interactions"].append({
                                "tool_request_timestamp": datetime.now().isoformat(),
                                "action_type": "run_shell",
                                "action_details": {"command": command_executable, "args": command_args_list},
                                "reasoning": reasoning,
                                "user_confirmation": "denied_by_system_whitelist",
                                "tool_outcome_timestamp": datetime.now().isoformat(),
                                "tool_outcome_summary": "Command execution denied by system (not whitelisted)."
                            })
                            action_taken_by_tool_framework = False # Let the error message display
                            agi_response_text = f"The AGI suggested a command that is not on the whitelist: '{command_to_display}'. Reasoning: {reasoning} (Displaying as text instead)."
                        else: # No command_executable provided
                            console.print(f"[warning]AGI suggestion for 'run_shell' did not include a 'command'. Payload: {data}[/warning]")
                            current_turn_interaction_data["tool_interactions"].append({
                                "tool_request_timestamp": datetime.now().isoformat(),
                                "action_type": "run_shell",
                                "action_details": data, # Log the malformed data
                                "reasoning": reasoning,
                                "user_confirmation": "n/a_malformed_request",
                                "tool_outcome_timestamp": datetime.now().isoformat(),
                                "tool_outcome_summary": "Malformed request: missing 'command'."
                            })
                            action_taken_by_tool_framework = False # Let the error message display
                            agi_response_text = f"AGI's suggested command was malformed (missing command). Reasoning: {reasoning}"

                    elif isinstance(data, dict) and data.get("action") == "read_file":
                        filepath_str = data.get("filepath")
                        max_lines = data.get("max_lines") # Optional
                        reasoning = data.get("reasoning", "No reasoning provided for file read.")

                        if not filepath_str:
                            console.print(f"[warning]AGI 'read_file' request missing 'filepath'. Payload: {data}[/warning]")
                            agi_response_text = "AGI's file read request was malformed (missing filepath)."
                            action_taken_by_tool_framework = False # Fall through to display error
                        else:
                            console.print(Panel(Text(f"AGI requests to read file: [bold cyan]{filepath_str}[/bold cyan]\nReason: {reasoning}", style="yellow"), title="[bold blue]File Read Request[/bold blue]"))

                            # Execute the read_file tool action
                            file_content_for_agi, read_error = handle_read_file_request(filepath_str, max_lines)

                            if read_error:
                                # Inform AGI about the error
                                error_prompt = (
                                    f"Context:\nAn attempt to read the file '{filepath_str}' failed.\n"
                                    f"Error: {read_error}\n\n"
                                    f"Original User Query: {user_input}\n\n"
                                    "Please inform the user about the error and proceed based on available information or ask for clarification."
                                )
                                with console.status("[yellow]AGI is processing file read error...[/yellow]", spinner="dots"):
                                    agi_response_text = agi_interface.generate_response(error_prompt)
                                # Let this response be displayed normally by falling through action_taken_by_tool_framework = False
                                # (or True if we consider this an action completion)
                                # For now, let it display the AGI's error-handling response.
                                action_taken_by_tool_framework = False # To display the AGI's response to the error
                            else:
                                # Construct new prompt for AGI with file content
                                subsequent_prompt = (
                                    f"{context_analyzer.get_full_context_string()}\n\n"
                                    f"File Content ('{filepath_str}'):\n```text\n{file_content_for_agi}\n```\n\n"
                                    f"Original User Query that led to reading this file: \"{user_input}\"\n\n"
                                    "Based on the provided file content and the original query, please formulate your answer to the user."
                                )
                                console.print(f"[info]File '{filepath_str}' read. Sending content to AGI for processing against original query...[/info]")
                                with console.status("[yellow]AGI is processing file content for the original query...[/yellow]", spinner="dots"):
                                    agi_response_text = agi_interface.generate_response(subsequent_prompt)
                                # This agi_response_text is the final one for the user for this turn.
                                # It will be displayed by the standard mechanism below.
                                action_taken_by_tool_framework = False # Let the normal display path handle this final response.
                                                                    # The actual "tool action" was reading the file and re-prompting.
                                                                    # We're essentially transforming the AGI's output here.
                            # Log the AGI's initial request to read the file
                            if session_logger:
                                session_logger.log_entry("AGI_Request_ReadFile", f"File: {filepath_str}, MaxLines: {max_lines}, Reason: {reasoning}, Error: {read_error}")
                            # Removed redundant conversation_history.append for "assistant_tool_request"
                            # The full details are in tool_interaction_log_entry for JSONL.
                            # The final AGI response (after getting file content or error) will be logged by the generic handler below.

                    elif isinstance(data, dict) and data.get("action") == "write_file":
                        filepath_str = data.get("filepath")
                        content_to_write = data.get("content") # Content can be empty string
                        reasoning = data.get("reasoning", "No reasoning provided for file write.")

                        if filepath_str is None or content_to_write is None: # Check for None, empty string for content is allowed
                            missing_fields = []
                            if filepath_str is None: missing_fields.append("'filepath'")
                            if content_to_write is None: missing_fields.append("'content'")
                            error_msg = f"AGI 'write_file' request missing required field(s): {', '.join(missing_fields)}. Payload: {data}"
                            console.print(f"[warning]{error_msg}[/warning]")
                            agi_response_text = f"Your 'write_file' request was malformed ({error_msg}). Please provide all required fields."
                            action_taken_by_tool_framework = False # Fall through to display error to user via AGI
                        else:
                            console.print(Panel(Text(f"AGI requests to write file: [bold cyan]{filepath_str}[/bold cyan]\nReason: {reasoning}", style="yellow"), title="[bold blue]File Write Request[/bold blue]"))

                            write_message, write_error = handle_write_file_request(filepath_str, content_to_write)

                            # Prepare message for AGI based on write outcome
                            if write_error:
                                outcome_summary = f"An attempt to write the file '{filepath_str}' failed. Error: {write_error}"
                                tool_outcome_for_log = f"Error: {write_error}"
                            else: # write_message contains success or cancellation message
                                outcome_summary = f"File write operation for '{filepath_str}': {write_message}"
                                tool_outcome_for_log = write_message # Success or cancellation message

                            tool_interaction_log_entry = {
                                "tool_request_timestamp": tool_request_start_time,
                                "action_type": "write_file",
                                "action_details": {"filepath": filepath_str, "content_length": len(content_to_write)}, # Not logging full content here
                                "reasoning": reasoning,
                                "user_confirmation": "confirmed" if "successfully" in tool_outcome_for_log else ("cancelled" if "cancelled" in tool_outcome_for_log else "n/a"), # Infer from message
                                "tool_outcome_timestamp": datetime.now().isoformat(),
                                "tool_outcome_summary": tool_outcome_for_log
                            }

                            # Re-prompt AGI with the outcome
                            subsequent_prompt_for_agi = (
                                f"{context_analyzer.get_full_context_string()}\n\n"
                                f"Outcome of your 'write_file' request for '{filepath_str}':\n{outcome_summary}\n\n"
                                f"Original User Query that may have led to this write request: \"{user_input}\"\n\n"
                                "Based on this outcome, please formulate your response to the user or decide on the next step."
                            )
                            tool_interaction_log_entry["context_for_next_agi_step"] = subsequent_prompt_for_agi # Log the prompt

                            console.print(f"[info]File write attempt for '{filepath_str}' processed. Outcome: {outcome_summary}. Re-prompting AGI...[/info]")
                            with console.status("[yellow]AGI is processing file write outcome...[/yellow]", spinner="dots"):
                                agi_secondary_raw_response = agi_interface.generate_response(subsequent_prompt_for_agi)

                            tool_interaction_log_entry["agi_secondary_raw_response"] = agi_secondary_raw_response
                            final_text_for_user_display = agi_secondary_raw_response # This is now the final response

                            current_turn_interaction_data["tool_interactions"].append(tool_interaction_log_entry)
                            action_taken_by_tool_framework = True # A tool action sequence completed.

                            # Log the AGI's initial request to write the file and the outcome to plain text log
                            if session_logger:
                                session_logger.log_entry("AGI_Request_WriteFile", f"File: {filepath_str}, Reason: {reasoning}, Outcome: {tool_outcome_for_log}")
                            # JSONL log is built in current_turn_interaction_data

                    elif isinstance(data, dict) and data.get("action") == "web_search":
                        tool_request_start_time = datetime.now().isoformat()
                        search_query = data.get("query")
                        reasoning = data.get("reasoning", "No reasoning provided for web search.")

                        if not search_query:
                            error_msg = f"AGI 'web_search' request missing 'query' field. Payload: {data}"
                            console.print(f"[warning]{error_msg}[/warning]")
                            agi_response_text = f"Your 'web_search' request was malformed ({error_msg}). Please provide a query."
                            action_taken_by_tool_framework = False # Fall through to display error to user via AGI
                        else:
                            console.print(Panel(Text(f"AGI requests web search for: [bold cyan]{search_query}[/bold cyan]\nReason: {reasoning}", style="yellow"), title="[bold blue]Web Search Request[/bold blue]"))

                            search_results_str, search_error = handle_web_search_request(search_query)

                            outcome_summary_for_log = "Search successful."
                            if search_error:
                                outcome_summary_for_log = f"Search failed: {search_error}"
                            elif not search_results_str or "No search results found." in search_results_str : # handle_web_search_request can return "No search results found." as non-error
                                outcome_summary_for_log = "Search returned no results or no usable results."

                            # Content for AGI: either results or an error/status message
                            content_for_agi = search_results_str if not search_error else f"Web search for '{search_query}' failed. Error: {search_error}"
                            if not content_for_agi and not search_error: # e.g. "No search results found."
                                content_for_agi = "No search results were found for your query."


                            subsequent_prompt = (
                                f"{context_analyzer.get_full_context_string()}\n\n"
                                f"Search Results for your query \"{search_query}\":\n{content_for_agi}\n\n"
                                f"Original User Query that may have led to this web search: \"{user_input}\"\n\n"
                                "Based on these search results (or lack thereof), please formulate your response to the user."
                            )
                            console.print(f"[info]Web search for '{search_query}' processed. Re-prompting AGI...[/info]")
                            with console.status("[yellow]AGI is processing web search results...[/yellow]", spinner="dots"):
                                agi_response_text = agi_interface.generate_response(subsequent_prompt)

                            action_taken_by_tool_framework = False # Let the normal display path handle this final response.

                            if session_logger:
                                session_logger.log_entry("AGI_Request_WebSearch", f"Query: {search_query}, Reason: {reasoning}, Outcome: {outcome_summary_for_log}")
                            conversation_history.append({
                                "role": "assistant_tool_request",
                                "content": f"Requested web search for: '{search_query}'. Reason: {reasoning}. Outcome: {outcome_summary_for_log}",
                                "timestamp": datetime.now().isoformat()
                            })
                            # Final AGI response logged by generic handler.

                    elif isinstance(data, dict) and data.get("action") == "git_branch_create":
                        tool_request_start_time = datetime.now().isoformat()
                        branch_name = data.get("branch_name")
                        base_branch = data.get("base_branch") # Optional
                        reasoning = data.get("reasoning", "No reasoning provided for git branch create.")

                        tool_interaction_log_entry = {
                            "tool_request_timestamp": tool_request_start_time,
                            "action_type": "git_branch_create",
                            "action_details": {"branch_name": branch_name, "base_branch": base_branch},
                            "reasoning": reasoning,
                        }

                        if not branch_name:
                            error_msg = f"AGI 'git_branch_create' request missing 'branch_name' field. Payload: {data}"
                            console.print(f"[warning]{error_msg}[/warning]")
                            final_text_for_user_display = f"Your 'git_branch_create' request was malformed ({error_msg})."
                            action_taken_by_tool_framework = False # Let error display
                            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
                            tool_interaction_log_entry["tool_outcome_summary"] = "Malformed request: missing 'branch_name'."
                        else:
                            console.print(Panel(Text(f"AGI requests to create git branch: [bold cyan]{branch_name}[/bold cyan]" +
                                                     (f" from base [bold cyan]{base_branch}[/bold cyan]" if base_branch else "") +
                                                     f"\nReason: {reasoning}", style="yellow"), title="[bold blue]Git Branch Create Request[/bold blue]"))

                            outcome_message, outcome_error = handle_git_branch_create_request(branch_name, base_branch)

                            tool_interaction_log_entry["user_confirmation"] = "confirmed" if "successfully" in (outcome_message or "") or "processed" in (outcome_message or "") else \
                                                                           ("cancelled" if "cancelled" in (outcome_message or "") else "n/a_error_or_not_applicable")
                            tool_interaction_log_entry["tool_outcome_timestamp"] = datetime.now().isoformat()

                            if outcome_error:
                                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                                outcome_summary_for_agi = f"An attempt to create branch '{branch_name}' failed. Error: {outcome_error}"
                            else:
                                tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                                outcome_summary_for_agi = f"Git branch operation for '{branch_name}': {outcome_message}"

                            subsequent_prompt_for_agi = (
                                f"{context_analyzer.get_full_context_string()}\n\n"
                                f"Outcome of your 'git_branch_create' request for '{branch_name}':\n{outcome_summary_for_agi}\n\n"
                                f"Original User Query: \"{user_input}\"\n\n"
                                "Based on this outcome, please formulate your response to the user or decide on the next step."
                            )
                            tool_interaction_log_entry["context_for_next_agi_step"] = subsequent_prompt_for_agi

                            console.print(f"[info]Git branch create attempt for '{branch_name}' processed. Outcome: {outcome_summary_for_agi}. Re-prompting AGI...[/info]")
                            with console.status("[yellow]AGI is processing git branch create outcome...[/yellow]", spinner="dots"):
                                agi_secondary_raw_response = agi_interface.generate_response(subsequent_prompt_for_agi)

                            tool_interaction_log_entry["agi_secondary_raw_response"] = agi_secondary_raw_response
                            final_text_for_user_display = agi_secondary_raw_response
                            action_taken_by_tool_framework = True

                        current_turn_interaction_data["tool_interactions"].append(tool_interaction_log_entry)

                    elif isinstance(data, dict) and data.get("action") == "git_checkout":
                        tool_request_start_time = datetime.now().isoformat()
                        branch_name = data.get("branch_name")
                        create_new = data.get("create_new", False) # Default to False if not provided
                        reasoning = data.get("reasoning", "No reasoning provided for git checkout.")

                        tool_interaction_log_entry = {
                            "tool_request_timestamp": tool_request_start_time,
                            "action_type": "git_checkout",
                            "action_details": {"branch_name": branch_name, "create_new": create_new},
                            "reasoning": reasoning,
                        }

                        if not branch_name:
                            error_msg = f"AGI 'git_checkout' request missing 'branch_name' field. Payload: {data}"
                            console.print(f"[warning]{error_msg}[/warning]")
                            final_text_for_user_display = f"Your 'git_checkout' request was malformed ({error_msg})."
                            action_taken_by_tool_framework = False
                            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
                            tool_interaction_log_entry["tool_outcome_summary"] = "Malformed request: missing 'branch_name'."
                        else:
                            action_desc_for_print = "Create and checkout new branch" if create_new else "Checkout branch"
                            console.print(Panel(Text(f"AGI requests to {action_desc_for_print.lower()}: [bold cyan]{branch_name}[/bold cyan]\nReason: {reasoning}", style="yellow"),
                                                title="[bold blue]Git Checkout Request[/bold blue]"))

                            outcome_message, outcome_error = handle_git_checkout_request(branch_name, create_new)

                            tool_interaction_log_entry["user_confirmation"] = "confirmed" if outcome_message and "Successfully" in outcome_message else \
                                                                           ("cancelled" if outcome_message and "cancelled" in outcome_message else "n/a_error_or_not_applicable")
                            tool_interaction_log_entry["tool_outcome_timestamp"] = datetime.now().isoformat()

                            if outcome_error:
                                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                                outcome_summary_for_agi = f"An attempt to {action_desc_for_print.lower()} '{branch_name}' failed. Error: {outcome_error}"
                            else:
                                tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                                outcome_summary_for_agi = f"Git checkout operation for '{branch_name}': {outcome_message}"

                            subsequent_prompt_for_agi = (
                                f"{context_analyzer.get_full_context_string()}\n\n"
                                f"Outcome of your 'git_checkout' request for '{branch_name}' (create_new={create_new}):\n{outcome_summary_for_agi}\n\n"
                                f"Original User Query: \"{user_input}\"\n\n"
                                "Based on this outcome, please formulate your response to the user or decide on the next step."
                            )
                            tool_interaction_log_entry["context_for_next_agi_step"] = subsequent_prompt_for_agi

                            console.print(f"[info]Git checkout attempt for '{branch_name}' processed. Outcome: {outcome_summary_for_agi}. Re-prompting AGI...[/info]")
                            with console.status("[yellow]AGI is processing git checkout outcome...[/yellow]", spinner="dots"):
                                agi_secondary_raw_response = agi_interface.generate_response(subsequent_prompt_for_agi)

                            tool_interaction_log_entry["agi_secondary_raw_response"] = agi_secondary_raw_response
                            final_text_for_user_display = agi_secondary_raw_response
                            action_taken_by_tool_framework = True

                        current_turn_interaction_data["tool_interactions"].append(tool_interaction_log_entry)

                    elif isinstance(data, dict) and data.get("action") == "execute_python_code":
                        tool_request_start_time = datetime.now().isoformat()
                        code_snippet = data.get("code")
                        reasoning = data.get("reasoning", "No reasoning provided for code execution.")

                        tool_interaction_log_entry = {
                            "tool_request_timestamp": tool_request_start_time,
                            "action_type": "execute_python_code",
                            "action_details": {"code": code_snippet}, # Log the actual code
                            "reasoning": reasoning,
                        }

                        if not isinstance(code_snippet, str) or not code_snippet.strip():
                            error_msg = f"AGI 'execute_python_code' request missing or invalid 'code' field (must be non-empty string). Payload: {data}"
                            console.print(f"[warning]{error_msg}[/warning]")
                            final_text_for_user_display = f"Your 'execute_python_code' request was malformed ({error_msg})."
                            action_taken_by_tool_framework = False
                            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
                            tool_interaction_log_entry["tool_outcome_summary"] = "Malformed request: missing or invalid 'code'."
                        else:
                            # User confirmation happens inside handle_execute_python_code_request
                            stdout_s, stderr_s, exception_s = handle_execute_python_code_request(code_snippet)

                            # Determine user confirmation from outcome (stdout_s will contain cancel message)
                            if stdout_s == "Code execution cancelled by user.":
                                tool_interaction_log_entry["user_confirmation"] = "cancelled"
                            else:
                                tool_interaction_log_entry["user_confirmation"] = "confirmed" # Assumed if not cancelled

                            tool_interaction_log_entry["tool_outcome_timestamp"] = datetime.now().isoformat()

                            outcome_parts = []
                            if stdout_s: outcome_parts.append(f"Stdout:\n{stdout_s.strip()}")
                            if stderr_s: outcome_parts.append(f"Stderr:\n{stderr_s.strip()}")
                            if exception_s: outcome_parts.append(f"Exception:\n{exception_s.strip()}")

                            tool_outcome_for_log = "\n---\n".join(outcome_parts) if outcome_parts else "No output or exception."
                            if stdout_s == "Code execution cancelled by user.": # Override for cleaner log if cancelled
                                tool_outcome_for_log = "Code execution cancelled by user."

                            tool_interaction_log_entry["tool_outcome_summary"] = tool_outcome_for_log

                            # Prepare summary for AGI (might be slightly different from full log, e.g., more concise)
                            outcome_summary_for_agi = tool_outcome_for_log
                            if len(outcome_summary_for_agi) > 1000: # Truncate very long outputs for AGI prompt
                                outcome_summary_for_agi = outcome_summary_for_agi[:1000] + "\n[...output truncated for AGI prompt...]"

                            subsequent_prompt_for_agi = (
                                f"{context_analyzer.get_full_context_string()}\n\n"
                                f"Outcome of your 'execute_python_code' request:\n```python\n{code_snippet}\n```\nExecution Result:\n{outcome_summary_for_agi}\n\n"
                                f"Original User Query: \"{user_input}\"\n\n"
                                "Based on this outcome, please formulate your response to the user or decide on the next step."
                            )
                            tool_interaction_log_entry["context_for_next_agi_step"] = subsequent_prompt_for_agi

                            console.print(f"[info]Python code execution processed. Outcome logged. Re-prompting AGI...[/info]")
                            with console.status("[yellow]AGI is processing code execution outcome...[/yellow]", spinner="dots"):
                                agi_secondary_raw_response = agi_interface.generate_response(subsequent_prompt_for_agi)

                            tool_interaction_log_entry["agi_secondary_raw_response"] = agi_secondary_raw_response
                            final_text_for_user_display = agi_secondary_raw_response
                            action_taken_by_tool_framework = True

                        current_turn_interaction_data["tool_interactions"].append(tool_interaction_log_entry)

                    elif isinstance(data, dict) and data.get("action") == "git_commit":
                        tool_request_start_time = datetime.now().isoformat()
                        commit_message = data.get("commit_message")
                        stage_all = data.get("stage_all", False) # Default to False
                        reasoning = data.get("reasoning", "No reasoning provided for git commit.")

                        tool_interaction_log_entry = {
                            "tool_request_timestamp": tool_request_start_time,
                            "action_type": "git_commit",
                            "action_details": {"commit_message": commit_message, "stage_all": stage_all},
                            "reasoning": reasoning,
                        }

                        if not commit_message or not commit_message.strip():
                            error_msg = f"AGI 'git_commit' request missing or empty 'commit_message' field. Payload: {data}"
                            console.print(f"[warning]{error_msg}[/warning]")
                            final_text_for_user_display = f"Your 'git_commit' request was malformed ({error_msg})."
                            action_taken_by_tool_framework = False
                            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
                            tool_interaction_log_entry["tool_outcome_summary"] = "Malformed request: missing or empty 'commit_message'."
                        else:
                            # User confirmation happens inside handle_git_commit_request
                            outcome_message, outcome_error = handle_git_commit_request(commit_message, stage_all)

                            tool_interaction_log_entry["user_confirmation"] = "confirmed" if outcome_message and ("successfully" in outcome_message.lower() or "nothing to commit" in outcome_message.lower()) else \
                                                                           ("cancelled" if outcome_message and "cancelled" in outcome_message.lower() else "n/a_error_or_not_applicable")
                            tool_interaction_log_entry["tool_outcome_timestamp"] = datetime.now().isoformat()

                            if outcome_error:
                                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                                outcome_summary_for_agi = f"An attempt to commit with message '{commit_message}' failed. Error: {outcome_error}"
                            else: # outcome_message is not None
                                tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                                outcome_summary_for_agi = f"Git commit operation with message '{commit_message}': {outcome_message}"

                            subsequent_prompt_for_agi = (
                                f"{context_analyzer.get_full_context_string()}\n\n"
                                f"Outcome of your 'git_commit' request (stage_all={stage_all}):\nMessage: \"{commit_message}\"\nResult: {outcome_summary_for_agi}\n\n"
                                f"Original User Query: \"{user_input}\"\n\n"
                                "Based on this outcome, please formulate your response to the user or decide on the next step."
                            )
                            tool_interaction_log_entry["context_for_next_agi_step"] = subsequent_prompt_for_agi

                            console.print(f"[info]Git commit attempt processed. Outcome: {outcome_summary_for_agi}. Re-prompting AGI...[/info]")
                            with console.status("[yellow]AGI is processing git commit outcome...[/yellow]", spinner="dots"):
                                agi_secondary_raw_response = agi_interface.generate_response(subsequent_prompt_for_agi)

                            tool_interaction_log_entry["agi_secondary_raw_response"] = agi_secondary_raw_response
                            final_text_for_user_display = agi_secondary_raw_response
                            action_taken_by_tool_framework = True

                        current_turn_interaction_data["tool_interactions"].append(tool_interaction_log_entry)

                    elif isinstance(data, dict) and data.get("action") == "git_push":
                        tool_request_start_time = datetime.now().isoformat()
                        remote_name = data.get("remote_name") # Optional
                        branch_name = data.get("branch_name") # Optional
                        reasoning = data.get("reasoning", "No reasoning provided for git push.")

                        tool_interaction_log_entry = {
                            "tool_request_timestamp": tool_request_start_time,
                            "action_type": "git_push",
                            "action_details": {"remote_name": remote_name, "branch_name": branch_name},
                            "reasoning": reasoning,
                        }

                        # No specific fields to pre-validate here other than their existence, which .get handles.
                        # handle_git_push_request will determine defaults if they are None.

                        # User confirmation happens inside handle_git_push_request
                        outcome_message, outcome_error = handle_git_push_request(remote_name, branch_name)

                        tool_interaction_log_entry["user_confirmation"] = "confirmed" if outcome_message and "successfully" in outcome_message.lower() or "processed" in outcome_message.lower() or "up-to-date" in outcome_message.lower() else \
                                                                       ("cancelled" if outcome_message and "cancelled" in outcome_message.lower() else "n/a_error_or_not_applicable")
                        tool_interaction_log_entry["tool_outcome_timestamp"] = datetime.now().isoformat()

                        if outcome_error:
                            tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                            outcome_summary_for_agi = f"An attempt to push to remote '{remote_name or 'default'}' for branch '{branch_name or 'current'}' failed. Error: {outcome_error}"
                        else: # outcome_message is not None
                            tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                            outcome_summary_for_agi = f"Git push operation to remote '{remote_name or 'default'}' for branch '{branch_name or 'current'}': {outcome_message}"

                        subsequent_prompt_for_agi = (
                            f"{context_analyzer.get_full_context_string()}\n\n"
                            f"Outcome of your 'git_push' request:\n{outcome_summary_for_agi}\n\n"
                            f"Original User Query: \"{user_input}\"\n\n"
                            "Based on this outcome, please formulate your response to the user or decide on the next step."
                        )
                        tool_interaction_log_entry["context_for_next_agi_step"] = subsequent_prompt_for_agi

                        console.print(f"[info]Git push attempt processed. Outcome: {outcome_summary_for_agi}. Re-prompting AGI...[/info]")
                        with console.status("[yellow]AGI is processing git push outcome...[/yellow]", spinner="dots"):
                            agi_secondary_raw_response = agi_interface.generate_response(subsequent_prompt_for_agi)

                        tool_interaction_log_entry["agi_secondary_raw_response"] = agi_secondary_raw_response
                        final_text_for_user_display = agi_secondary_raw_response
                        action_taken_by_tool_framework = True

                        current_turn_interaction_data["tool_interactions"].append(tool_interaction_log_entry)

                # If an action was successfully processed by a tool handler above (which involves re-prompting AGI)
                # action_taken_by_tool_framework would be True, and final_text_for_user_display is AGI's secondary response.
                # If JSON was malformed or action unknown, action_taken_by_tool_framework is False,
                # and final_text_for_user_display is AGI's initial (problematic) response or a system error message.

                except json.JSONDecodeError:
                    # Not a JSON response for tool use, treat as normal chat
                    action_taken_by_tool_framework = False # No tool action was completed or properly initiated
                    # final_text_for_user_display remains agi_initial_raw_response
                except Exception as e: # Catch-all for other errors during tool processing
                    console.print(f"[warning]Could not fully process AGI response for potential tool use: {type(e).__name__} - {e}[/warning]")
                    action_taken_by_tool_framework = False # Tool processing failed
                    # final_text_for_user_display may have been updated by the failing tool block to an error message,
                    # or it remains agi_initial_raw_response.
                else: # Parsed JSON was not a dictionary or had no "action" key
                    action_taken_by_tool_framework = False
                    # final_text_for_user_display remains agi_initial_raw_response

                # Display the final response to the user (either direct AGI response or AGI response after tool use)
                # The variable `final_text_for_user_display` holds what needs to be shown.
                # `action_taken_by_tool_framework` helps distinguish if a tool was involved in producing this final text.
                # If a tool was successfully initiated AND it doesn't re-prompt (like run_shell when confirmed),
                # final_text_for_user_display might be a system message.
                # If a tool re-prompts AGI (read_file, write_file, web_search), final_text_for_user_display is AGI's secondary response.
                # If no tool, it's AGI's initial response.

                # Log to conversation_history (internal deque)
                # This should always be the text that was actually displayed or the final synthesis from AGI.
                conversation_history.append({"role": "assistant", "content": final_text_for_user_display, "timestamp": datetime.now().isoformat()})
                if session_logger and getattr(session_logger, 'enabled', True): # Check if enabled
                    session_logger.log_entry("AGI", final_text_for_user_display) # Log final user-facing text

                response_parts = detect_code_blocks(final_text_for_user_display)

                # Determine panel style based on detected task type from the *last* AGI interaction
                # (could be initial or secondary if a tool was used)
                panel_title_text = "[agiprompt]AGI Output[/agiprompt]"
                panel_border_style_color = "blue" # Default
                task_type_for_style = agi_interface.last_detected_task_type # This reflects the type for the *last* call to generate_response

                if task_type_for_style == "code_generation":
                    panel_title_text = "[agiprompt]AGI Code Generation[/agiprompt]"
                    panel_border_style_color = "green"
                # ... (other styling rules remain the same) ...
                elif task_type_for_style == "code_debugging":
                    panel_title_text = "[agiprompt]AGI Code Debugging[/agiprompt]"
                    panel_border_style_color = "yellow"
                elif task_type_for_style == "explanation" or task_type_for_style == "code_explanation":
                    panel_title_text = "[agiprompt]AGI Explanation[/agiprompt]"
                    panel_border_style_color = "cyan"
                elif task_type_for_style == "summarization":
                    panel_title_text = "[agiprompt]AGI Summary[/agiprompt]"
                    panel_border_style_color = "magenta"
                elif task_type_for_style == "git_query":
                    panel_title_text = "[agiprompt]AGI Git Query Response[/agiprompt]"
                    panel_border_style_color = "bright_black"

                output_renderable = Text()
                contains_code_blocks_in_final_output = False
                for part_idx, part in enumerate(response_parts):
                    if part["type"] == "text":
                        output_renderable.append(part["content"])
                    elif part["type"] == "code":
                        contains_code_blocks_in_final_output = True
                        if part_idx > 0 and response_parts[part_idx-1]["type"] == "text" and not response_parts[part_idx-1]["content"].endswith("\n"):
                            output_renderable.append("\n")
                        current_theme = APP_CONFIG.get("display", {}).get("syntax_theme", "monokai")
                        lang_for_syntax = part["lang"] if part["lang"] else "plaintext"
                        try:
                            code_syntax = Syntax(part["content"], lang_for_syntax, theme=current_theme, line_numbers=True, word_wrap=True)
                            output_renderable.append(code_syntax)
                        except Exception as e_rich_syntax:
                            console.print(f"[dim warning]Rich Syntax Error for lang '{lang_for_syntax}': {e_rich_syntax}. Falling back to plaintext.[/dim warning]")
                            code_syntax_fallback = Syntax(part["content"], "plaintext", theme=current_theme, line_numbers=True, word_wrap=True)
                            output_renderable.append(code_syntax_fallback)
                    if part_idx < len(response_parts) -1:
                         output_renderable.append("\n")

                console.print(Panel(output_renderable, title=panel_title_text, border_style=panel_border_style_color, expand=False))

                current_turn_interaction_data["timestamp_final_response"] = datetime.now().isoformat()
                current_turn_interaction_data["agi_final_response_to_user"] = final_text_for_user_display
                current_turn_interaction_data["final_response_formatting_details"] = {
                    "panel_title": panel_title_text, # Store the actual text used
                    "panel_border_style": panel_border_style_color,
                    "contains_code_blocks": contains_code_blocks_in_final_output
                }

                if agi_interface.is_model_loaded or isinstance(agi_interface, AGIPPlaceholder):
                    # Pass the final user-facing AGI text to training script
                    call_training_script(user_input, final_text_for_user_display)

                log_interaction_to_jsonl(current_turn_interaction_data)


            console.print("-" * console.width)

    except KeyboardInterrupt:
        console.print("\nExiting due to KeyboardInterrupt...", style="info")
    # The main __main__ block's finally clause handles saving history and the final "AGI session terminated" message.

# --- Shell Command Whitelist & Execution ---
SHELL_COMMAND_WHITELIST = [
    "ls", "pwd", "echo", "date", "uname", "df", "free", "whoami", "uptime", "hostname",
    "head", "tail", "wc", "grep"
]

# --- Experimental Code Change Suggestion ---
# Whitelist of files the AGI can suggest changes for (mostly its own project files)
SUGGEST_CHANGE_WHITELIST = ["interactive_agi.py", "setup_agi_terminal.py", "adaptive_train.py", "merge_config.yml", "download_models.sh", "train_on_interaction.sh", "README.md"]

def suggest_code_change_command(file_path_str: str, agi_interface: MergedAGI):
    target_path = Path(file_path_str)

    if target_path.name not in SUGGEST_CHANGE_WHITELIST:
        console.print(f"[red]Error: Suggestions for modifying '{target_path.name}' are not allowed for safety.[/red]")
        console.print(f"[info]Allowed files for suggestions: {', '.join(SUGGEST_CHANGE_WHITELIST)}[/info]")
        return

    if not target_path.exists() or not target_path.is_file():
        console.print(f"[red]Error: File not found or is not a regular file: {target_path.resolve()}[/red]")
        return

    console.print(f"[info]Selected file for change suggestion: {target_path.resolve()}[/info]")

    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        console.print(f"[red]Error reading file '{target_path.name}': {e}[/red]")
        return

    # Display a portion of the file for context if it's large
    max_display_lines = 50
    display_lines = file_content.splitlines()
    if len(display_lines) > max_display_lines:
        console.print(Panel("\n".join(display_lines[:max_display_lines]) + "\n... (file content truncated for display)",
                              title=f"Start of {target_path.name}", border_style="dim"))
    else:
        cat_file_command(str(target_path), agi_interface) # Use existing cat to show it with syntax highlighting

    console.print(f"\n[prompt]Describe the change you want to make to '{target_path.name}' (or type 'cancel'):[/prompt]")
    change_description = console.input("> ")

    if not change_description or change_description.lower() == 'cancel':
        console.print("Code change suggestion cancelled.", style="yellow")
        return

    # Prepare prompt for AGI
    # Truncate file content if very large, to keep prompt size reasonable
    max_content_chars_for_prompt = 8000 # Example limit
    content_for_prompt = file_content
    if len(file_content) > max_content_chars_for_prompt:
        content_for_prompt = file_content[:max_content_chars_for_prompt] + "\n\n[...FILE CONTENT TRUNCATED FOR PROMPT...]"
        console.print(f"[yellow]Warning: File content is large, sending a truncated version ({max_content_chars_for_prompt} chars) to AGI.[/yellow]")

    prompt_text = (
        f"You are an expert software developer. The user wants to make a change to the file '{target_path.name}'.\n"
        f"Current file content of '{target_path.name}':\n```\n{content_for_prompt}\n```\n\n"
        f"User's requested change: \"{change_description}\"\n\n"
        "Please provide the suggested code modification in a `diff -u` format. "
        "The diff should be relative to the provided file content. "
        "Output ONLY the diff content, enclosed in a ```diff ... ``` markdown block. "
        "If you cannot represent the change as a diff, describe the necessary find/replace operations or line changes clearly."
    )

    console.print("\n[info]Requesting code change suggestion from AGI...[/info]")
    with console.status("[yellow]AGI is thinking about the code change...[/yellow]", spinner="dots"):
        suggestion_response = agi_interface.generate_response(prompt_text)

    if session_logger:
        session_logger.log_entry("User", f"/suggest_code_change {file_path_str} - Request: {change_description}")
        session_logger.log_entry("AGI_Suggestion", suggestion_response)
    conversation_history.append({"role": "user", "content": f"/suggest_code_change {file_path_str} - Request: {change_description}", "timestamp": datetime.now().isoformat()})
    conversation_history.append({"role": "assistant", "content": f"Suggested change for {target_path.name}:\n{suggestion_response}", "timestamp": datetime.now().isoformat()})

    console.print(f"\n[agiprompt]AGI's Suggested Change for {target_path.name}:[/agiprompt]")

    # Try to display as diff, otherwise plain text
    diff_match = re.search(r"```diff\n(.*?)\n```", suggestion_response, re.DOTALL)
    if diff_match:
        diff_content = diff_match.group(1)
        console.print(Panel(Syntax(diff_content, "diff", theme=APP_CONFIG.get("display",{}).get("syntax_theme","monokai"), line_numbers=True),
                            title="Suggested Diff", border_style="yellow"))
    else:
        # If not a diff block, display as regular text/code blocks
        response_parts = detect_code_blocks(suggestion_response)
        for part in response_parts:
            if part["type"] == "text":
                console.print(Text(part["content"]))
            elif part["type"] == "code":
                current_theme = APP_CONFIG.get("display", {}).get("syntax_theme", "monokai")
                lang_for_syntax = part["lang"] if part["lang"] else "plaintext"
                try:
                    code_syntax = Syntax(part["content"], lang_for_syntax, theme=current_theme, line_numbers=True, word_wrap=True)
                    console.print(code_syntax)
                except Exception as e_rich_syntax:
                    console.print(f"[dim warning]Rich Syntax Error for lang '{lang_for_syntax}': {e_rich_syntax}. Falling back to plaintext.[/dim warning]")
                    code_syntax_fallback = Syntax(part["content"], "plaintext", theme=current_theme, line_numbers=True, word_wrap=True)
                    console.print(code_syntax_fallback)

    console.print("\n[bold yellow]IMPORTANT: This is only a suggestion. Review it carefully. You must apply these changes manually if you agree.[/bold yellow]")


def execute_shell_command(command_executable: str, command_args: list[str]):
    """
    Executes a whitelisted shell command with arguments using shell=False.
    Prints the output directly to the console and returns captured stdout/stderr.
    """
    if command_executable not in SHELL_COMMAND_WHITELIST:
        msg = f"Error: Command '{command_executable}' is not in the allowed list of safe commands."
        console.print(f"[bold red]{msg}[/bold red]")
        if session_logger: session_logger.log_entry("System", f"Denied execution (not whitelisted): {command_executable} {' '.join(command_args)}")
        return None, msg, -1 # stdout, stderr, returncode

    full_command_list = [command_executable] + command_args
    command_to_display = f"{command_executable} {' '.join(command_args)}" # For display purposes

    console.print(f"Executing: [bold cyan]{command_to_display}[/bold cyan]", style="info")
    try:
        process = subprocess.run(
            full_command_list,
            shell=False,  # Crucial for security when taking args from AGI
            capture_output=True,
            text=True,
            timeout=15,
            check=False # We will check returncode manually
        )

        output_panel_title = f"[bold green]Output of: {command_to_display}[/bold green]"
        output_content = ""
        if process.stdout:
            output_content += f"[bold]Stdout:[/bold]\n{process.stdout.strip()}\n"
        else:
            output_content += "[dim]No output (stdout)[/dim]\n"

        if process.stderr:
            output_content += f"\n[bold red]Stderr:[/bold red]\n{process.stderr.strip()}"

        if process.returncode != 0:
             output_content += f"\n[bold yellow]Return code:[/bold yellow] {process.returncode}"

        console.print(Panel(Text(output_content.strip()), title=output_panel_title))

        if session_logger:
            log_message = (
                f"Executed: {command_to_display}\n"
                f"Return Code: {process.returncode}\n"
                f"Stdout: {process.stdout.strip()}\n"
                f"Stderr: {process.stderr.strip()}"
            )
            session_logger.log_entry("System", log_message)
        return process.stdout, process.stderr, process.returncode

    except FileNotFoundError: # command_executable not found
        err_msg = f"Error: Command '{command_executable}' not found. Is it installed and in PATH?"
        console.print(f"[red]{err_msg}[/red]")
        if session_logger: session_logger.log_entry("System", f"Command not found: {command_to_display}")
        return None, err_msg, -1
    except subprocess.TimeoutExpired:
        err_msg = f"Error: Command '{command_to_display}' timed out."
        console.print(f"[red]{err_msg}[/red]")
        if session_logger: session_logger.log_entry("System", f"Command timed out: {command_to_display}")
        return None, err_msg, -1
    except PermissionError: # Should be less common with shell=False for the command itself
        err_msg = f"Error: Permission denied when trying to execute '{command_to_display}'."
        console.print(f"[red]{err_msg}[/red]")
        if session_logger: session_logger.log_entry("System", f"Permission error executing: {command_to_display}")
        return None, err_msg, -1
    except Exception as e:
        err_msg = f"Error executing command '{command_to_display}': {type(e).__name__} - {e}"
        console.print(f"[red]{err_msg}[/red]")
        if session_logger: session_logger.log_entry("System", f"Error executing {command_to_display}: {e}")
        return None, err_msg, -1

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

# --- Config Management ---
def load_config() -> dict:
    global APP_CONFIG # Allow modification of global APP_CONFIG
    config = DEFAULT_CONFIG.copy() # Start with defaults

    if TOML_AVAILABLE and CONFIG_FILE_PATH.exists():
        try:
            loaded_toml = toml.load(CONFIG_FILE_PATH)
            # Simple deep merge for one level of nesting (like 'generation', 'history')
            for main_key, sub_dict in DEFAULT_CONFIG.items():
                if main_key in loaded_toml and isinstance(loaded_toml[main_key], dict):
                    # If the key exists in loaded_toml and is a dictionary, merge its sub-keys
                    if main_key not in config: config[main_key] = {} # Ensure section exists
                    for sub_key, default_val in sub_dict.items():
                        config[main_key][sub_key] = loaded_toml[main_key].get(sub_key, default_val)
                elif main_key in loaded_toml: # For top-level keys if any added later (not in DEFAULT_CONFIG structure)
                    config[main_key] = loaded_toml[main_key]
                # If main_key from DEFAULT_CONFIG is not in loaded_toml, its default sub_dict remains.

            console.print(f"INFO: Configuration loaded from {CONFIG_FILE_PATH}", style="info")
        except toml.TomlDecodeError as e:
            console.print(f"ERROR: Invalid TOML in {CONFIG_FILE_PATH}: {e}. Using default configuration and attempting to save a valid one.", style="error")
            config = DEFAULT_CONFIG.copy() # Reset to pure defaults on decode error
            save_config(config) # Save a fresh default config
        except IOError as e:
            console.print(f"ERROR: Could not read {CONFIG_FILE_PATH}: {e}. Using default configuration.", style="error")
    else:
        if TOML_AVAILABLE:
            console.print(f"INFO: No config file found at {CONFIG_FILE_PATH}. Creating with default values.", style="info")
            save_config(config) # Create default config file
        else:
            console.print("INFO: TOML library not available. Using default internal configuration.", style="info")

    APP_CONFIG = config # Update global APP_CONFIG

    # Apply loaded config to relevant globals
    global conversation_history, DEFAULT_GENERATION_PARAMS, session_logger

    # Re-initialize deque with new maxlen if it changed
    current_history_list = list(conversation_history) # Preserve current items if any
    new_maxlen = APP_CONFIG.get("history", {}).get("max_len", DEFAULT_CONFIG["history"]["max_len"])
    conversation_history = deque(maxlen=new_maxlen) # Set new max_len from config
    conversation_history.extend(current_history_list) # Add back items, deque handles overflow

    # Update DEFAULT_GENERATION_PARAMS based on loaded config
    # MergedAGI instances copy this at their __init__. If an instance exists, it won't get this update
    # unless we explicitly update it or it re-reads from APP_CONFIG.
    # For now, new MergedAGI instances will pick this up.
    gen_config = APP_CONFIG.get("generation", {})
    DEFAULT_GENERATION_PARAMS["temperature"] = gen_config.get("default_temperature", DEFAULT_GENERATION_PARAMS["temperature"])
    DEFAULT_GENERATION_PARAMS["max_new_tokens"] = gen_config.get("default_max_tokens", DEFAULT_GENERATION_PARAMS["max_new_tokens"])
    DEFAULT_GENERATION_PARAMS["top_p"] = gen_config.get("default_top_p", DEFAULT_GENERATION_PARAMS["top_p"])
    DEFAULT_GENERATION_PARAMS["repetition_penalty"] = gen_config.get("default_repetition_penalty", DEFAULT_GENERATION_PARAMS["repetition_penalty"])

    # Update syntax theme for Rich (used in /cat and AGI code block display)
    # This requires a mechanism to pass the theme to Syntax objects or re-initialize console if theme affects it globally.
    # For Syntax objects, they take `theme` as an arg. We can pass APP_CONFIG['display']['syntax_theme']

    # Update desktop logging settings
    # This needs to happen *before* SessionLogger is initialized if path override is to take effect.
    # The enabled flag can be set after initialization.
    # get_desktop_path() now reads APP_CONFIG, so it's fine.
    # session_logger.enabled is set after its init based on config.

    global JSONL_LOGGING_ENABLED
    JSONL_LOGGING_ENABLED = APP_CONFIG.get("logging", {}).get("jsonl_logging_enabled", True)
    if JSONL_LOGGING_ENABLED:
        console.print(f"INFO: JSONL interaction logging is ENABLED. Logs will be saved to: {JSONL_LOG_FILE_PATH}", style="info")
    else:
        console.print("INFO: JSONL interaction logging is DISABLED via config.", style="info")

    # --- Ensure .gitignore for cache directory ---
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gitignore_path = CACHE_DIR / ".gitignore"
        ignore_entry = "interaction_logs.jsonl"
        if gitignore_path.exists():
            with open(gitignore_path, 'r+', encoding='utf-8') as f:
                current_ignores = f.read().splitlines()
                if ignore_entry not in current_ignores:
                    f.seek(0, os.SEEK_END) # Go to end of file
                    if f.tell() > 0 and current_ignores and current_ignores[-1]: # If not empty and last line not blank
                        f.write('\n')
                    f.write(f"{ignore_entry}\n")
        else:
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(f"# Ignore cache-specific files\n")
                f.write("history.json\n") # Good to ignore the regular history too if not already
                f.write("config.toml\n")  # And the config if it's auto-generated with defaults
                f.write(f"{ignore_entry}\n")
        # Also good practice to have a .gitignore in the main project root that ignores .agi_terminal_cache/
        # but this script manages the .gitignore *inside* the cache dir.
    except Exception as e:
        console.print(f"[warning]Could not create or update .gitignore in {CACHE_DIR}: {e}[/warning]")

    return config

def save_config(config_to_save: dict):
    if not TOML_AVAILABLE:
        console.print("WARNING: TOML library not available. Cannot save configuration.", style="warning")
        return
    try:
        CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure .agi_terminal_cache exists
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            toml.dump(config_to_save, f)
    except IOError as e:
        console.print(f"ERROR: Could not save configuration to {CONFIG_FILE_PATH}: {e}", style="error")
    except Exception as e: # Catch other toml errors
        console.print(f"ERROR: Failed to serialize configuration for saving: {e}", style="error")

def config_command_handler(args_str: Optional[str]):
    global APP_CONFIG
    parts = args_str.split(maxsplit=2) if args_str else []
    subcommand = parts[0].lower() if parts else "show"

    if subcommand == "show":
        table = Table(title="Current Application Configuration", show_header=True, header_style="bold magenta", box=None, padding=(0,1))
        table.add_column("Section", style="cyan", overflow="fold")
        table.add_column("Setting", style="yellow", overflow="fold")
        table.add_column("Value", style="green", overflow="fold")

        sorted_app_config = dict(sorted(APP_CONFIG.items()))
        for section, settings in sorted_app_config.items():
            if isinstance(settings, dict):
                sorted_settings = dict(sorted(settings.items()))
                for i, (key, value) in enumerate(sorted_settings.items()):
                    table.add_row(section if i == 0 else "", key, str(value))
            else:
                table.add_row("general", section, str(settings))
        console.print(table)

    elif subcommand == "get" and len(parts) > 1:
        key_path = parts[1]
        keys = key_path.split('.')
        value_ptr = APP_CONFIG
        try:
            for k in keys:
                value_ptr = value_ptr[k]
            console.print(Panel(Text(str(value_ptr)), title=key_path))
        except KeyError:
            console.print(f"[red]Error: Config key '{key_path}' not found.[/red]")
        except TypeError:
            console.print(f"[red]Error: Invalid key path '{key_path}'. Part of it is not a section.[/red]")

    elif subcommand == "set":
        if len(parts) < 3:
            console.print("[red]Usage: /config set <section.key> <value>[/red]")
            return
        key_path = parts[1]
        new_value_str = parts[2]

        keys = key_path.split('.')
        config_section = APP_CONFIG # This is a reference to the global dict
        temp_ptr = config_section

        try:
            for k_idx, k_name in enumerate(keys[:-1]):
                temp_ptr = temp_ptr[k_name]

            final_key = keys[-1]
            if final_key not in temp_ptr:
                console.print(f"[red]Error: Config key '{key_path}' not found for setting.[/red]")
                return

            current_value = temp_ptr[final_key]
            typed_value = None

            # Attempt type conversion based on default or existing value type
            # This ensures we try to keep the type consistent.
            # More robust type handling might be needed for arbitrary new keys.
            default_val_type = type(DEFAULT_CONFIG)
            for k in keys: default_val_type = type(default_val_type.get(k, {})) if isinstance(default_val_type, dict) else type(None)

            if isinstance(current_value, bool) or default_val_type == bool:
                if new_value_str.lower() in ['true', 'yes', '1', 'on']: typed_value = True
                elif new_value_str.lower() in ['false', 'no', '0', 'off']: typed_value = False
                else: raise ValueError("Boolean value must be true/false/yes/no/1/0/on/off")
            elif isinstance(current_value, int) or default_val_type == int:
                typed_value = int(new_value_str)
            elif isinstance(current_value, float) or default_val_type == float:
                typed_value = float(new_value_str)
            else:
                typed_value = new_value_str # Default to string

            temp_ptr[final_key] = typed_value
            save_config(APP_CONFIG)
            console.print(f"[green]Config '{key_path}' set to '{typed_value}'.[/green]")

            # Apply immediate changes if possible
            if key_path == "history.max_len":
                global conversation_history
                current_history_list = list(conversation_history)
                conversation_history = deque(maxlen=typed_value)
                conversation_history.extend(current_history_list)
                console.print("[info]Conversation history max length updated. Change will apply fully on next restart for history loading if new max is smaller.[/info]")
            elif key_path.startswith("generation."):
                 # Update DEFAULT_GENERATION_PARAMS for new MergedAGI instances
                gen_key = final_key # e.g. "default_temperature"
                param_key_in_gen_params = gen_key.replace("default_", "") # e.g. "temperature"
                if param_key_in_gen_params in DEFAULT_GENERATION_PARAMS:
                    DEFAULT_GENERATION_PARAMS[param_key_in_gen_params] = typed_value
                console.print(f"[info]Default generation parameter '{param_key_in_gen_params}' updated. Active AGI instances may need parameter reset or restart to see change.[/info]")
            elif key_path == "logging.desktop_logging_enabled" and session_logger:
                session_logger.enabled = typed_value
            elif key_path == "logging.desktop_log_path_override" and session_logger:
                 console.print("[info]Desktop log path override changed. This will take effect on next restart.[/info]")


        except KeyError:
            console.print(f"[red]Error: Config key '{key_path}' not found for setting.[/red]")
        except (ValueError, TypeError) as e:
            console.print(f"[red]Error setting value for '{key_path}': Invalid value or type. {e}[/red]")
        except Exception as e:
            console.print(f"[red]Unexpected error setting config: {e}[/red]")

    else:
        console.print("[red]Invalid /config command. Usage:\n  /config show\n  /config get <section.key>\n  /config set <section.key> <value>[/red]")


# --- Structured Output Command (Experimental) ---
def analyze_dir_command(path_str: Optional[str], agi_interface: MergedAGI):
    target_path_str = path_str if path_str else "."
    resolved_path = Path(target_path_str).resolve()

    console.print(f"[info]Analyzing directory structure for: {resolved_path}[/info]")

    dir_listing = []
    max_items = 30 # Limit number of items to keep context manageable
    item_count = 0

    try:
        if not resolved_path.is_dir():
            console.print(f"[red]Error: Not a directory or path does not exist: {resolved_path}[/red]")
            return

        for item in resolved_path.iterdir():
            if item_count >= max_items:
                dir_listing.append({"name": "...", "type": "truncated"})
                break
            item_type = "directory" if item.is_dir() else "file"
            entry = {"name": item.name, "type": item_type}
            if item.is_file():
                try:
                    entry["size"] = item.stat().st_size
                except OSError:
                    entry["size"] = -1 # Indicate error or inaccessible
            dir_listing.append(entry)
            item_count += 1

    except PermissionError:
        console.print(f"[red]Error: Permission denied for directory: {resolved_path}[/red]")
        return
    except Exception as e:
        console.print(f"[red]Error listing directory for analysis '{resolved_path}': {type(e).__name__} - {e}[/red]")
        return

    if not dir_listing:
        console.print(f"[yellow]Directory is empty or no accessible items found: {resolved_path}[/yellow]")
        # Still, we can ask the AGI about an empty directory if the user wishes
        # For now, let's proceed to ask anyway, or one could return here.

    # Construct prompt for AGI
    # Convert basic listing to a string format for the prompt
    listing_str_parts = []
    for item in dir_listing:
        if item["type"] == "directory":
            listing_str_parts.append(f"- {item['name']}/ (directory)")
        elif item["type"] == "file":
            size_str = f" ({item['size']} bytes)" if item['size'] != -1 else ""
            listing_str_parts.append(f"- {item['name']}{size_str} (file)")
        else: # Truncated message
            listing_str_parts.append(f"- {item['name']}")

    listing_for_prompt = "\n".join(listing_str_parts)

    prompt_text = (
        f"Analyze the following directory listing for '{resolved_path}'. "
        "Provide your analysis as a JSON object. The JSON object should have a root key 'directory_analysis'. "
        "Under this key, include 'path', 'item_count', and a 'summary' (a brief textual description of the directory content or purpose, if inferable). "
        "Also include a 'structure' key, which should be a list of items. Each item in the 'structure' list should be an object with 'name', 'type' ('file' or 'directory'), "
        "and optionally 'size' for files, and for directories, optionally a 'children' list if you want to represent nesting (though the provided listing is flat). "
        "Focus on interpreting the provided flat list. If the listing was truncated, mention it in the summary.\n\n"
        f"Directory Listing:\n{listing_for_prompt}\n\n"
        "JSON Response:"
    )

    console.print("[info]Requesting directory analysis from AGI...[/info]")
    with console.status("[yellow]AGI is analyzing directory structure...[/yellow]", spinner="dots"):
        raw_response = agi_interface.generate_response(prompt_text)

    if session_logger:
        session_logger.log_entry("User", f"/analyze_dir {target_path_str}")
        session_logger.log_entry("AGI", raw_response)
    conversation_history.append({"role": "user", "content": f"/analyze_dir {target_path_str}", "timestamp": datetime.now().isoformat()})
    conversation_history.append({"role": "assistant", "content": raw_response, "timestamp": datetime.now().isoformat()})

    console.print(f"[agiprompt]AGI Analysis (Raw):[/agiprompt]")
    # Display raw response first, then try to parse and display structured
    # This helps debugging if JSON is malformed
    console.print(Panel(raw_response, title="Raw AGI Response", border_style="dim yellow", expand=False))

    try:
        # Attempt to extract JSON part if the model includes explanatory text before/after
        json_match = re.search(r"```json\n(.*?)\n```", raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no markdown block, assume the whole response might be JSON (or fail parsing)
            # A more robust way would be to find first '{' and last '}'
            first_brace = raw_response.find('{')
            last_brace = raw_response.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = raw_response[first_brace : last_brace+1]
            else:
                json_str = raw_response


        parsed_json = json.loads(json_str)

        if isinstance(parsed_json, dict) and "directory_analysis" in parsed_json:
            analysis = parsed_json["directory_analysis"]
            console.print("\n[bold green]Structured Directory Analysis:[/bold green]")

            analysis_table = Table(show_header=False, box=None)
            analysis_table.add_column("Key", style="bold cyan")
            analysis_table.add_column("Value")
            analysis_table.add_row("Path", Text(analysis.get("path", str(resolved_path))))
            analysis_table.add_row("Item Count (in provided list)", str(analysis.get("item_count", len(dir_listing))))
            analysis_table.add_row("AGI Summary", Text(analysis.get("summary", "N/A")))
            console.print(analysis_table)

            if "structure" in analysis and isinstance(analysis["structure"], list):
                tree = Tree(f"[bold blue]Directory Tree (from AGI JSON):[/bold blue] {analysis.get('path', str(resolved_path))}")

                def add_nodes_to_tree(branch, items):
                    for item_data in items:
                        name = item_data.get("name", "Unknown Item")
                        item_type = item_data.get("type", "unknown")
                        display_name = f"[green]{name}[/green]" if item_type == "file" else f"[bold yellow]{name}/[/bold yellow]"
                        if item_type == "file" and "size" in item_data:
                            display_name += f" [dim]({item_data['size']} bytes)[/dim]"

                        sub_branch = branch.add(display_name)
                        if "children" in item_data and isinstance(item_data["children"], list):
                            add_nodes_to_tree(sub_branch, item_data["children"])

                add_nodes_to_tree(tree, analysis["structure"])
                console.print(tree)
            else:
                console.print("[yellow]No 'structure' list found in AGI's JSON analysis or it's not a list.[/yellow]")
        else:
            console.print("[yellow]AGI response was valid JSON, but not in the expected 'directory_analysis' format. Displaying raw.[/yellow]")
            # Already displayed raw above, or could display json_str prettily
            # console.print(json.dumps(parsed_json, indent=2))


    except json.JSONDecodeError:
        console.print("[yellow]AGI did not return valid JSON for directory analysis. Raw output displayed above.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error processing AGI's structured response: {type(e).__name__} - {e}[/red]")


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
        console.print(f"[red]Error running git log: {e.stderr}[/red]")
    except ValueError: # For int conversion of count
        console.print("[red]Invalid count for /git log. Please provide a number (e.g., /git log -n 15).[/red]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred with /git log: {type(e).__name__} - {e}[/red]")


# --- AGI Tool Helper Functions ---
AGI_READ_FILE_DEFAULT_MAX_LINES = 200 # Default max lines for AGI file read if not specified by AGI
AGI_READ_FILE_MAX_CHARS = 10000     # Absolute max characters to return to AGI from a file read

def handle_read_file_request(filepath_str: str, max_lines_from_agi: Optional[int]) -> tuple[Optional[str], Optional[str]]:
    """
    Handles an AGI's request to read a file after validating the path and user permissions.

    Reads the file content, applying limits on lines and characters.
    Ensures the path is relative and within the defined project root.

    Args:
        filepath_str: The relative path to the file, as requested by AGI.
        max_lines_from_agi: Optional integer specifying the maximum number of lines
                            the AGI prefers to read. System defaults and caps apply.

    Returns:
        A tuple `(file_content_string, None)` on successful read, where
        `file_content_string` contains the file content (possibly truncated with a message).
        A tuple `(None, error_message_string)` on failure (e.g., file not found,
        access denied, decoding error).
    """
    try:
        requested_path = Path(filepath_str)

        # Security Validation 1: Ensure it's a relative path
        if requested_path.is_absolute():
            return None, f"File path must be relative, but got absolute path: {filepath_str}"

        # Resolve the path based on current working directory
        # Path.resolve() also handles '..' components safely.
        abs_path = (Path.cwd() / requested_path).resolve()
        cwd_resolved = Path.cwd().resolve()

        # Security Validation 2: Check if the resolved path is within the defined PROJECT_ROOT_PATH.
        # PROJECT_ROOT_PATH is resolved at startup, so it's an absolute path.
        if PROJECT_ROOT_PATH is None: # Should not happen if main() initializes it correctly
             console.print("[bold red]CRITICAL: PROJECT_ROOT_PATH is not set. File access denied for safety.[/bold red]")
             return None, "Project scope is not defined. Cannot read file."

        # Ensure abs_path is within the project root.
        # Using str.startswith for broader Python version compatibility (Path.is_relative_to is Py3.9+)
        if not str(abs_path).startswith(str(PROJECT_ROOT_PATH)):
            # Also check if PROJECT_ROOT_PATH itself is a parent of abs_path,
            # which means abs_path is project_root/some/path.
            # The check `str(abs_path).startswith(str(PROJECT_ROOT_PATH))` correctly handles this.
            # However, if abs_path IS PROJECT_ROOT_PATH, it should also be allowed.
            # The startswith check covers this too.
            # The only edge case is if PROJECT_ROOT_PATH is /foo and abs_path is /foobar - this would pass.
            # To be more robust: check that abs_path starts with PROJECT_ROOT_PATH AND the next char is '/' or it's identical.
            if abs_path != PROJECT_ROOT_PATH and not str(abs_path).startswith(str(PROJECT_ROOT_PATH) + os.sep):
                 return None, f"Access denied: File path '{filepath_str}' resolves to '{abs_path}', which is outside the defined project root '{PROJECT_ROOT_PATH}'."

        if not abs_path.exists():
            return None, f"File not found at resolved path: {abs_path}"
        if not abs_path.is_file():
            return None, f"Path is not a file: {abs_path}"

        # Determine max lines to read
        max_lines_to_read = AGI_READ_FILE_DEFAULT_MAX_LINES
        if max_lines_from_agi is not None:
            try:
                agn_max_l = int(max_lines_from_agi)
                if agn_max_l > 0:
                    max_lines_to_read = min(agn_max_l, AGI_READ_FILE_DEFAULT_MAX_LINES * 2) # Allow AGI to ask for more, but cap it reasonably
            except ValueError:
                pass # Ignore invalid max_lines from AGI, use default

        file_content_lines = []
        chars_read = 0
        lines_actually_read = 0
        truncated_message = ""

        with open(abs_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines_to_read:
                    truncated_message = f"\n[...content truncated after {max_lines_to_read} lines...]"
                    break
                if chars_read + len(line) > AGI_READ_FILE_MAX_CHARS:
                    # If adding this line exceeds char limit, take a partial line if possible, then break.
                    remaining_chars = AGI_READ_FILE_MAX_CHARS - chars_read
                    if remaining_chars > 0:
                        file_content_lines.append(line[:remaining_chars])
                    truncated_message = f"\n[...content truncated due to character limit ({AGI_READ_FILE_MAX_CHARS} chars)...]"
                    break

                file_content_lines.append(line)
                chars_read += len(line)
                lines_actually_read = i + 1

        final_content = "".join(file_content_lines).strip() # Strip trailing newlines from the snippet itself
        if truncated_message:
            final_content += truncated_message

        # Log how much was read
        console.print(f"[dim info]Read {lines_actually_read} lines ({chars_read} chars) from '{filepath_str}'. Max lines requested/default: {max_lines_to_read}. Char limit: {AGI_READ_FILE_MAX_CHARS}.[/dim info]")

        return final_content, None

    except UnicodeDecodeError:
        return None, f"Could not decode file '{filepath_str}' as UTF-8. It might be binary or use a different encoding."
    except IOError as e:
        return None, f"IOError reading file '{filepath_str}': {e}"
    except Exception as e: # Catch-all for unexpected errors during file handling
        console.print(f"[bold red]Unexpected error in handle_read_file_request for '{filepath_str}': {type(e).__name__} - {e}[/bold red]")
        return None, f"An unexpected error occurred while trying to read file: {type(e).__name__}"

import difflib # For generating diffs in handle_write_file_request

def handle_write_file_request(filepath_str: str, content_to_write: str) -> tuple[Optional[str], Optional[str]]:
    """
    Handles an AGI's request to write content to a file, subject to path validation and user confirmation.

    If the file exists, a diff of changes is shown to the user for confirmation.
    If the file is new, its full proposed content is shown for confirmation.
    Ensures the path is relative, within the project root, and the parent directory exists.
    Standardizes newlines to '\n' and ensures non-empty files end with a newline.

    Args:
        filepath_str: The relative path to the file (within project root) where content should be written.
        content_to_write: The full string content intended for the file.

    Returns:
        A tuple `(success_message, None)` if the file is written successfully or if the operation
        is cancelled by the user (message will indicate cancellation).
        A tuple `(None, error_message_string)` if a validation error, I/O error, or other
        exception occurs during the process.
    """
    try:
        if PROJECT_ROOT_PATH is None: # Should have been initialized in main()
            console.print("[bold red]CRITICAL: PROJECT_ROOT_PATH is not set. File write access denied for safety.[/bold red]")
            return None, "Project scope (PROJECT_ROOT_PATH) is not defined. Cannot write file."

        requested_path = Path(filepath_str)
        if requested_path.is_absolute():
            return None, f"File path must be relative, but got absolute path: {filepath_str}"

        # Resolve path relative to project root
        abs_path = (PROJECT_ROOT_PATH / requested_path).resolve()

        # Security Check: Ensure abs_path is within PROJECT_ROOT_PATH
        if not (abs_path == PROJECT_ROOT_PATH.resolve() or str(abs_path).startswith(str(PROJECT_ROOT_PATH.resolve()) + os.sep)):
            return None, f"Access denied: File path '{filepath_str}' resolves to '{abs_path}', which is outside the defined project root '{PROJECT_ROOT_PATH.resolve()}'."

        if abs_path.is_dir():
            return None, f"Path '{filepath_str}' points to an existing directory. Cannot write file content to a directory."

        parent_dir = abs_path.parent
        if not parent_dir.exists():
            return None, f"Parent directory '{parent_dir}' for file '{filepath_str}' does not exist. Please create it first."
        if not parent_dir.is_dir():
            return None, f"The parent path '{parent_dir}' for file '{filepath_str}' is not a directory."

        user_confirmed = False
        final_content_for_file = content_to_write
        # Standardize newlines for diffing and writing: use '\n'
        if "\r\n" in final_content_for_file:
            final_content_for_file = final_content_for_file.replace("\r\n", "\n")
        if "\r" in final_content_for_file:
            final_content_for_file = final_content_for_file.replace("\r", "\n")

        # Ensure content ends with a newline if it's not empty, for cleaner diffs and typical file formats.
        if final_content_for_file and not final_content_for_file.endswith('\n'):
            content_for_display_and_diff = final_content_for_file + '\n'
        elif not final_content_for_file: # Empty content
            content_for_display_and_diff = "" # An empty file is just empty
        else: # Already ends with newline
            content_for_display_and_diff = final_content_for_file


        if abs_path.exists() and abs_path.is_file():
            try:
                existing_content = abs_path.read_text(encoding='utf-8')
                # Standardize newlines for existing content for diffing
                if "\r\n" in existing_content: existing_content = existing_content.replace("\r\n", "\n")
                if "\r" in existing_content: existing_content = existing_content.replace("\r", "\n")

                existing_content_lines = existing_content.splitlines(keepends=True)
                content_to_write_lines = content_for_display_and_diff.splitlines(keepends=True)

                if existing_content == content_for_display_and_diff: # Check after newline standardization
                     return f"No changes detected for '{filepath_str}'. Content is identical.", None

                diff = difflib.unified_diff(
                    existing_content_lines,
                    content_to_write_lines,
                    fromfile=f"a/{filepath_str}",
                    tofile=f"b/{filepath_str}",
                    lineterm='\n' # Important for difflib
                )
                diff_output = "".join(diff)

                if not diff_output: # Should be caught by direct content comparison above, but as fallback:
                    diff_output = f"--- a/{filepath_str}\n+++ b/{filepath_str}\n@@ -1 +1 @@\n-{existing_content.strip()}\n+{content_for_display_and_diff.strip()}"


                console.print(Panel(Syntax(diff_output, "diff", theme=APP_CONFIG.get("display",{}).get("syntax_theme","monokai"), line_numbers=False, word_wrap=True),
                                    title=f"[bold yellow]Suggested changes for {filepath_str}[/bold yellow]", border_style="yellow"))
                if RICH_AVAILABLE:
                    from rich.prompt import Confirm
                    user_confirmed = Confirm.ask(f"Apply these changes to '{filepath_str}'?", default=False, console=console)
                else:
                    user_confirmed = input(f"Apply these changes to '{filepath_str}'? (yes/NO): ").lower() == "yes"

            except Exception as e_read_diff:
                return None, f"Error reading existing file for diff '{filepath_str}': {e_read_diff}"
        else: # New file
            lexer = get_lexer_for_filename(filepath_str)
            console.print(Panel(Syntax(content_for_display_and_diff, lexer, theme=APP_CONFIG.get("display",{}).get("syntax_theme","monokai"), line_numbers=True, word_wrap=True),
                                title=f"[bold green]Content for new file: {filepath_str}[/bold green]", border_style="green"))
            if RICH_AVAILABLE:
                from rich.prompt import Confirm
                user_confirmed = Confirm.ask(f"Create new file '{filepath_str}' with this content?", default=False, console=console)
            else:
                user_confirmed = input(f"Create new file '{filepath_str}'? (yes/NO): ").lower() == "yes"

        if user_confirmed:
            try:
                # Write the version that has standardized newlines and a trailing newline if content exists
                with open(abs_path, 'w', encoding='utf-8', newline='\n') as f: # newline='\n' ensures only \n is written
                    f.write(content_for_display_and_diff)

                return f"File '{filepath_str}' written successfully.", None
            except IOError as e:
                return None, f"IOError writing to file '{filepath_str}': {e}"
            except Exception as e_write:
                return None, f"Unexpected error writing to file '{filepath_str}': {e_write}"
        else:
            return "Write operation cancelled by user.", None

    except Exception as e:
        console.print(f"[bold red]Unexpected error in handle_write_file_request for '{filepath_str}': {type(e).__name__} - {e}[/bold red]")
        return None, f"An unexpected error occurred processing the write request: {type(e).__name__}"

def handle_git_branch_create_request(branch_name: str, base_branch: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Handles an AGI's request to create a new Git branch, after user confirmation.

    Validates the proposed branch name for common invalid patterns.
    Constructs and executes the `git branch` command. Interprets Git's output
    to determine success, including cases where the branch might already exist
    (which Git sometimes reports via stderr with a zero exit code).

    Args:
        branch_name: The name for the new branch.
        base_branch: Optional name of an existing branch from which to start the new branch.
                     If None or empty, the new branch starts from the current HEAD.

    Returns:
        A tuple `(success_message, None)` on successful operation (including messages
        from Git like "Branch 'X' set up to track 'Y'.") or if the operation was
        cancelled by the user.
        A tuple `(None, error_message_string)` if validation fails, Git command fails,
        or an unexpected error occurs.
    """
    if not branch_name:
        return None, "Branch name cannot be empty."

    # Basic validation for branch name (simplified)
    # Git has more complex rules, but this catches common issues.
    # Ref: man git-check-ref-format
    if re.search(r"[\s~^:*?\[\]\\]|@{|\.\.", branch_name) or \
       branch_name.startswith('/') or branch_name.endswith('/') or \
       branch_name.endswith(".lock"):
        return None, f"Invalid branch name: '{branch_name}'. Contains invalid characters or patterns."

    git_command_list = ["git", "branch", branch_name]
    confirm_message = f"Create new branch '{branch_name}'"
    if base_branch and base_branch.strip():
        # Further validation for base_branch could be added if needed
        git_command_list.append(base_branch.strip())
        confirm_message += f" from base branch '{base_branch.strip()}'"
    confirm_message += "?"

    if RICH_AVAILABLE:
        from rich.prompt import Confirm
        confirmed = Confirm.ask(confirm_message, default=False, console=console)
    else:
        confirmed = input(f"{confirm_message} (yes/NO): ").lower() == "yes"

    if not confirmed:
        return "Branch creation cancelled by user.", None

    try:
        process = subprocess.run(git_command_list, shell=False, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            # Check stderr for common "warning: " or if branch already exists type messages from git.
            # Git branch might return 0 even if branch already exists but just prints to stderr.
            # "git branch <name>" successfully creates or does nothing if exists (stderr: "fatal: A branch named '...' already exists.")
            # Let's check stderr for "fatal" or "error"
            if "fatal:" in process.stderr.lower() or "error:" in process.stderr.lower():
                 return None, f"Git error: {process.stderr.strip()}"
            success_msg = f"Branch '{branch_name}' operation processed."
            if process.stdout: success_msg += f"\nGit stdout: {process.stdout.strip()}"
            if process.stderr: success_msg += f"\nGit stderr (warnings): {process.stderr.strip()}" # e.g. if it already existed and command did nothing.
            return success_msg.strip(), None
        else:
            # Return code is non-zero, this is definitely an error.
            return None, f"Git command failed with return code {process.returncode}. Error: {process.stderr.strip() if process.stderr else process.stdout.strip()}"
    except FileNotFoundError:
        return None, "Git command not found. Is Git installed and in PATH?"
    except Exception as e:
        console.print(f"[bold red]Unexpected error in handle_git_branch_create_request: {type(e).__name__} - {e}[/bold red]")
        return None, f"An unexpected error occurred: {type(e).__name__}"

def handle_git_checkout_request(branch_name: str, create_new: bool) -> tuple[Optional[str], Optional[str]]:
    """
    Handles an AGI's request to checkout a Git branch, potentially creating it if specified.

    Validates the branch name if `create_new` is true.
    Constructs and executes `git checkout <branch>` or `git checkout -b <branch>`
    after user confirmation. Interprets Git's output for success or failure.
    Git often prints informational messages (e.g., "Switched to branch '...'") to
    stderr on success, so stderr is checked.

    Args:
        branch_name: The name of the branch to checkout or create.
        create_new: If True, attempts to create the branch and check it out (`-b` flag).
                    If False, attempts to checkout an existing branch.

    Returns:
        A tuple `(success_message, None)` on successful operation (often Git's output)
        or if the operation was cancelled by the user.
        A tuple `(None, error_message_string)` if validation fails, Git command fails,
        or an unexpected error occurs.
    """
    if not branch_name:
        return None, "Branch name cannot be empty for checkout."

    # Validate branch_name, especially if creating new
    if create_new:
        if re.search(r"[\s~^:*?\[\]\\]|@{|\.\.", branch_name) or \
           branch_name.startswith('/') or branch_name.endswith('/') or \
           branch_name.endswith(".lock"):
            return None, f"Invalid new branch name: '{branch_name}'. Contains invalid characters or patterns."

    git_command_list = ["git", "checkout"]
    action_desc = "Checkout branch"
    if create_new:
        git_command_list.append("-b")
        action_desc = "Create and checkout new branch"
    git_command_list.append(branch_name)

    confirm_message = f"{action_desc} '{branch_name}'?"

    if RICH_AVAILABLE:
        from rich.prompt import Confirm
        confirmed = Confirm.ask(confirm_message, default=False, console=console)
    else:
        confirmed = input(f"{confirm_message} (yes/NO): ").lower() == "yes"

    if not confirmed:
        return f"{action_desc} operation cancelled by user.", None

    try:
        process = subprocess.run(git_command_list, shell=False, capture_output=True, text=True, check=False)
        # Git checkout often prints to stderr even on success (e.g., "Switched to a new branch '...'")
        # So, check returncode first.
        if process.returncode == 0:
            success_msg = process.stderr.strip() if process.stderr.strip() else process.stdout.strip() # Prefer stderr for messages like "Switched to..."
            if not success_msg : success_msg = f"Successfully performed: {' '.join(git_command_list)}"
            return success_msg, None
        else:
            # Error occurred
            error_msg = process.stderr.strip() if process.stderr else process.stdout.strip()
            if not error_msg: error_msg = f"Git command failed with return code {process.returncode} but no specific error message."
            return None, f"Git error: {error_msg}"

    except FileNotFoundError:
        return None, "Git command not found. Is Git installed and in PATH?"
    except Exception as e:
        console.print(f"[bold red]Unexpected error in handle_git_checkout_request: {type(e).__name__} - {e}[/bold red]")
        return None, f"An unexpected error occurred during git checkout: {type(e).__name__}"

def handle_git_commit_request(commit_message: str, stage_all: bool) -> tuple[Optional[str], Optional[str]]:
    """
    Handles an AGI's request to make a Git commit, after user confirmation.

    Validates that the commit message is not empty.
    Constructs `git commit -m "<message>"` or `git commit -a -m "<message>"`
    based on the `stage_all` flag. Displays the command and full message for confirmation.
    Interprets Git's output, including handling the "nothing to commit" case as
    an informational success rather than an error.

    Args:
        commit_message: The commit message string.
        stage_all: If True, stages all tracked and modified files before committing (`-a` flag).

    Returns:
        A tuple `(success_message, None)` on successful commit (including "nothing to commit")
        or if the operation was cancelled by the user.
        A tuple `(None, error_message_string)` if validation fails, Git command execution
        results in an error (other than "nothing to commit"), or an unexpected error occurs.
    """
    if not commit_message or not commit_message.strip():
        return None, "Commit message cannot be empty."

    git_command_list = ["git", "commit"]
    if stage_all:
        git_command_list.append("-a")
    git_command_list.extend(["-m", commit_message])

    command_str_display = " ".join(git_command_list) # For display only, not for execution

    # Display the proposed command and message clearly to the user
    commit_panel_content = Text.assemble(
        "AGI suggests the following Git commit:\n\n",
        (f"{command_str_display}", "bold cyan"),
        "\n\nCommit Message:\n",
        (f"{commit_message}", "italic")
    )
    console.print(Panel(commit_panel_content,
                        title="[bold blue]Git Commit Request[/bold blue]", border_style="blue", expand=False))

    if RICH_AVAILABLE:
        from rich.prompt import Confirm
        confirmed = Confirm.ask("Proceed with this commit?", default=False, console=console)
    else:
        confirmed = input(f"Proceed with commit: (yes/NO): ").lower() == "yes" # Simplified prompt for non-rich

    if not confirmed:
        return "Commit operation cancelled by user.", None

    try:
        process = subprocess.run(git_command_list, shell=False, capture_output=True, text=True, check=False)

        if process.returncode == 0:
            output = process.stdout.strip() if process.stdout.strip() else "Commit successful (no detailed output from git)."
            if process.stderr.strip():
                output += f"\nGit Stderr (info/warnings): {process.stderr.strip()}"
            return output, None
        else:
            error_msg = process.stderr.strip() if process.stderr.strip() else process.stdout.strip()
            if not error_msg: error_msg = f"Git commit command failed with return code {process.returncode} but no specific error message."
            # Common case: nothing to commit
            if "nothing to commit" in error_msg.lower() or "no changes added to commit" in error_msg.lower():
                return f"Nothing to commit. {error_msg}", None # Return as success message with Git's info
            return None, f"Git commit error: {error_msg}"

    except FileNotFoundError:
        return None, "Git command not found. Is Git installed and in PATH?"
    except Exception as e:
        console.print(f"[bold red]Unexpected error in handle_git_commit_request: {type(e).__name__} - {e}[/bold red]")
        return None, f"An unexpected error occurred during git commit: {type(e).__name__}"

def handle_git_push_request(remote_name: Optional[str], branch_name_to_push: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Handles an AGI's request to push a Git branch to a remote repository, after user confirmation.

    Determines the effective remote (defaults to 'origin') and branch (defaults to current
    checked-out branch if not specified). Constructs and executes the `git push` command.
    Interprets Git's output (often from stderr) for success or failure, as push
    operations can have various outcomes (success, auth failure, remote errors, etc.).
    Force pushing is not permitted by this handler.

    Args:
        remote_name: Optional name of the remote repository (e.g., 'origin').
        branch_name_to_push: Optional name of the branch to push.

    Returns:
        A tuple `(success_message, None)` on successful push (including messages like
        "Everything up-to-date") or if the operation was cancelled by the user.
        A tuple `(None, error_message_string)` if determining defaults fails, the Git command
        execution results in an error, or an unexpected error occurs (e.g., timeout).
    """

    effective_remote = remote_name.strip() if remote_name and remote_name.strip() else "origin"
    effective_branch = branch_name_to_push.strip() if branch_name_to_push and branch_name_to_push.strip() else None

    if not effective_branch:
        try:
            # Get current branch if not specified
            proc = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True)
            effective_branch = proc.stdout.strip()
            if not effective_branch or effective_branch == "HEAD": # Detached HEAD or other issue
                return None, "Could not determine current branch, or in detached HEAD state. Please specify branch to push."
        except FileNotFoundError:
            return None, "Git command not found. Is Git installed and in PATH?"
        except subprocess.CalledProcessError as e:
            return None, f"Failed to determine current branch: {e.stderr.strip()}"
        except Exception as e:
            return None, f"Unexpected error determining current branch: {e}"

    git_command_list = ["git", "push", effective_remote, effective_branch]
    command_str_display = " ".join(git_command_list)

    console.print(Panel(Text(f"AGI suggests the following Git push command:\n\n[bold cyan]{command_str_display}[/bold cyan]", justify="left"),
                        title="[bold blue]Git Push Request[/bold blue]", border_style="blue"))

    if RICH_AVAILABLE:
        from rich.prompt import Confirm
        confirmed = Confirm.ask(f"Proceed with push: {command_str_display}?", default=False, console=console)
    else:
        confirmed = input(f"Proceed with push: {command_str_display}? (yes/NO): ").lower() == "yes"

    if not confirmed:
        return "Push operation cancelled by user.", None

    try:
        # Git push often requires credentials; this subprocess call won't handle interactive prompts for them.
        # It assumes credentials are cached or handled by a credential helper.
        # Output often goes to stderr, even for success.
        console.print(f"[info]Attempting to execute: {command_str_display} (This might take a moment...)[/info]")
        process = subprocess.run(git_command_list, shell=False, capture_output=True, text=True, check=False, timeout=60) # Increased timeout for network op

        # Success/failure for push is complex. Return code 0 is good, but stderr can still have info.
        # Non-zero often means clear failure.
        if process.returncode == 0:
            output = process.stderr.strip() if process.stderr.strip() else process.stdout.strip() # Prefer stderr for push status messages
            if not output: output = "Push command executed, no detailed output from git."
            # Check for common success phrases if output is ambiguous
            if "everything up-to-date" in output.lower() or "->" in output or "branch" in output.lower() and "pushed" in output.lower():
                 return f"Push successful: {output}", None
            # If return code 0 but output looks like an error, treat as error
            elif "error:" in output.lower() or "fatal:" in output.lower() or "rejected" in output.lower():
                 return None, f"Git push error (despite return code 0): {output}"
            return f"Push command processed: {output}", None # General success
        else:
            error_msg = process.stderr.strip() if process.stderr.strip() else process.stdout.strip()
            if not error_msg: error_msg = f"Git push command failed with return code {process.returncode} but no specific error message."
            return None, f"Git push error: {error_msg}"

    except FileNotFoundError:
        return None, "Git command not found. Is Git installed and in PATH?"
    except subprocess.TimeoutExpired:
        return None, "Git push command timed out after 60 seconds."
    except Exception as e:
        console.print(f"[bold red]Unexpected error in handle_git_push_request: {type(e).__name__} - {e}[/bold red]")
        return None, f"An unexpected error occurred during git push: {type(e).__name__}"

import io # For capturing stdout/stderr from exec
import contextlib # For redirect_stdout/stderr
import traceback # For formatting exceptions

def handle_execute_python_code_request(code_snippet: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Handles an AGI's request to execute a Python code snippet in a restricted environment.

    IMPORTANT SECURITY CONSIDERATIONS:
    This function uses `exec()` which is inherently risky if not handled carefully.
    It attempts to mitigate risks by:
    1.  **Mandatory User Confirmation:** The exact code is shown, and the user must explicitly approve.
    2.  **Pre-execution Static Analysis:** Basic regex checks for obviously dangerous patterns
        (e.g., `import os`, `open(`). If found, execution is rejected before user confirmation.
    3.  **Restricted Builtins:** A whitelist of `__builtins__` is provided to `exec()`,
        excluding functions like `open`, `eval`, `exec`, `__import__`, `getattr`, `setattr`.
    4.  **No Direct Module Imports:** The restricted environment aims to prevent arbitrary `import` statements.
        Safe modules (like `math`, `json`) would need to be explicitly added to globals if desired.

    This is NOT a foolproof sandbox. It aims to prevent common accidental or naive malicious attempts.
    Resource limits (CPU, memory, timeout) are not strictly enforced by this basic implementation.

    Args:
        code_snippet: The Python code string provided by the AGI.

    Returns:
        A tuple `(stdout_str, stderr_str, exception_str_or_none)`:
        - `stdout_str`: Captured standard output from the executed code. Contains a
                        cancellation message if the user denied execution.
        - `stderr_str`: Captured standard error output from the executed code.
        - `exception_str`: A string representation of any Python exception that occurred
                           during `exec()`, or the special marker "StaticAnalysisReject"
                           if pre-execution checks failed. None if no exception.
    """
    # --- Pre-execution Static Analysis (Basic Heuristic Checks) ---
    # AGI should be prompted NOT to use these. This is a fallback check.
    dangerous_patterns = [
        r"import\s+(os|sys|subprocess|shutil|socket|requests|pathlib|ctypes|pty|fcntl|resource|select|signal|termios|tty|asyncio|multiprocessing|threading)", # Common dangerous imports
        r"from\s+(os|sys|subprocess|shutil|socket|requests|pathlib|ctypes|pty|fcntl|resource|select|signal|termios|tty|asyncio|multiprocessing|threading)\s+import",
        r"open\s*\(",      # File system access
        r"eval\s*\(",      # Dynamic execution
        r"exec\s*\(",      # Dynamic execution (exec within exec)
        r"getattr\s*\(",   # Potential attribute snooping/calling
        r"setattr\s*\(",   # Potential attribute modification
        r"delattr\s*\(",   # Potential attribute deletion
        r"__\w+__",      # Access to dunder methods/attributes (broad, might have false positives but good for caution)
        r"socket\s*\(",   # Network access
        r"subprocess\s*\.", # Subprocess module usage
        r"ctypes\s*\."      # CTypes usage
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, code_snippet, re.IGNORECASE):
            error_msg = f"Execution rejected by system: Code snippet contains potentially unsafe pattern: '{pattern}'. Please use provided tools for I/O, system, or network operations, or simplify the code."
            console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg, None, "StaticAnalysisReject" # Special marker for exception type

    # If static analysis passes, then proceed to user confirmation
    console.print(Panel(Syntax(code_snippet, "python", theme=APP_CONFIG.get("display",{}).get("syntax_theme","monokai"), line_numbers=True, word_wrap=True),
                        title="[bold yellow]AGI suggests executing this Python code snippet:[/bold yellow]", border_style="yellow"))

    warning_text = Text.assemble(
        ("WARNING:", "bold red"),
        " Executing code suggested by an AGI carries risks. ",
        ("Although efforts are made to restrict its capabilities, review the code carefully.\n", "yellow"),
        "Ensure it does not perform unintended actions, access sensitive data, or harm your system.\n",
        ("Do you want to execute this code?", "bold white")
    )
    console.print(Panel(warning_text, title="[bold red]Security Warning[/bold red]", border_style="red", expand=False))

    if RICH_AVAILABLE:
        from rich.prompt import Confirm
        confirmed = Confirm.ask("Execute this Python code snippet?", default=False, console=console)
    else:
        confirmed = input("Execute this Python code snippet? (yes/NO): ").lower() == "yes"

    if not confirmed:
        return "Code execution cancelled by user.", None, None

    # --- Prepare Restricted Execution Environment ---
    # Whitelist of safe built-ins
    # (Based on https://docs.python.org/3/library/builtins.html and general safety)
    # Default-deny approach: only allow what's explicitly listed.
    # Keep this list as minimal as possible for safety.
    # AGI should be prompted that only these are available.
    _ALLOWED_BUILTINS_DICT = {
        # Safe data types & constructors
        'str': str, 'int': int, 'float': float, 'bool': bool,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'bytes': bytes, 'bytearray': bytearray, 'complex': complex,
        'frozenset': frozenset, 'memoryview': memoryview,
        'object': object, 'slice': slice, 'type': type, # type() is powerful but hard to exploit without more.
        # Safe functions for data manipulation & iteration
        'print': print, 'len': len, 'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
        'range': range, 'zip': zip, 'enumerate': enumerate,
        'map': map, 'filter': filter, 'sorted': sorted, 'reversed': reversed,
        'all': all, 'any': any, 'isinstance': isinstance, 'issubclass': issubclass,
        # Common, relatively safe Exception types (primarily for `isinstance` or `except` clauses)
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'AttributeError': AttributeError, 'NameError': NameError, 'IndexError': IndexError,
        'KeyError': KeyError, 'ZeroDivisionError': ZeroDivisionError, 'StopIteration': StopIteration,
        'ArithmeticError': ArithmeticError, 'AssertionError': AssertionError,
        'NotImplementedError': NotImplementedError,
        # Constants (already part of Python syntax, but explicitly listing for clarity of intent)
        'True': True, 'False': False, 'None': None,
    }
    # Explicitly disallowed (even if some are not in __builtins__ by default, good to document intent)
    # 'open', 'file', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 'vars',
    # 'getattr', 'setattr', 'delattr', 'dir', 'input', 'help', 'breakpoint', 'memoryview.tobytes' (example)
    # 'classmethod', 'staticmethod', 'property', 'super' (less risky, but for simple snippets, likely not needed)

    # --- Pre-execution Static Analysis (Basic Heuristic Checks) ---
    # AGI should be prompted NOT to use these. This is a fallback check.
    dangerous_patterns = [
        r"import\s+(os|sys|subprocess|shutil|socket|requests|pathlib|ctypes|pty|fcntl|resource|select|signal|termios|tty|asyncio|multiprocessing|threading)", # Common dangerous imports
        r"from\s+(os|sys|subprocess|shutil|socket|requests|pathlib|ctypes|pty|fcntl|resource|select|signal|termios|tty|asyncio|multiprocessing|threading)\s+import",
        r"open\s*\(",      # File system access
        r"eval\s*\(",      # Dynamic execution
        r"exec\s*\(",      # Dynamic execution (exec within exec)
        r"getattr\s*\(",   # Potential attribute snooping/calling
        r"setattr\s*\(",   # Potential attribute modification
        r"delattr\s*\(",   # Potential attribute deletion
        r"__\w+__",      # Access to dunder methods/attributes (broad, might have false positives but good for caution)
        r"socket\s*\(",   # Network access
        r"subprocess\s*\.", # Subprocess module usage
        r"ctypes\s*\."      # CTypes usage
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, code_snippet, re.IGNORECASE):
            error_msg = f"Execution rejected: Code snippet contains potentially unsafe pattern: '{pattern}'. Please use provided tools for I/O, system, or network operations."
            console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg, None, "StaticAnalysisReject" # Special marker for exception type

    # User has already confirmed, proceed with restricted execution
    restricted_globals = {"__builtins__": _ALLOWED_BUILTINS_DICT.copy()} # Use a copy
    # Example of adding safe modules (if decided later):
    # import math
    # restricted_globals['math'] = math
    restricted_locals = {}

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    exception_str = None

    console.print("[info]Executing code snippet...[/info]")
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code_snippet, restricted_globals, restricted_locals)
    except Exception:
        exception_str = traceback.format_exc()

    stdout_val = stdout_buffer.getvalue()
    stderr_val = stderr_buffer.getvalue()

    # Log captured output for debugging/visibility by the system admin (not AGI directly yet)
    if stdout_val: console.print(Panel(stdout_val, title="[dim]Captured stdout from snippet[/dim]", border_style="dim cyan", expand=False))
    if stderr_val: console.print(Panel(stderr_val, title="[dim]Captured stderr from snippet[/dim]", border_style="dim yellow", expand=False))
    if exception_str: console.print(Panel(exception_str, title="[dim]Captured exception from snippet[/dim]", border_style="dim red", expand=False))

    return stdout_val, stderr_val, exception_str


# --- Conceptual Design for Secure Code Execution Environment ---
#
# GOAL: Allow the AGI to request the execution of Python code snippets in a way that
# minimizes risk to the underlying system. True sandboxing is complex and
# often OS-dependent. For this project, we will implement a basic restricted
# execution environment, heavily relying on user confirmation and strict limitations
# on what the code can do.
#
# PRINCIPLES & CONSTRAINTS:
#
# 1.  USER CONFIRMATION IS MANDATORY:
#     -   The exact code snippet proposed by the AGI MUST be displayed to the user.
#     -   Explicit user confirmation (e.g., "yes/no") MUST be obtained before any
#         execution attempt.
#     -   The user should be warned about the potential risks of running code, even
#         if it appears simple.
#
# 2.  SUPPORTED LANGUAGES:
#     -   Initially, only Python 3 snippets will be supported.
#     -   The Python interpreter will be the one running this `interactive_agi.py` script.
#
# 3.  EXECUTION MECHANISM (Basic Restriction):
#     -   The primary mechanism will be Python's `exec()` function.
#     -   A heavily restricted `globals` and `locals` dictionary will be provided to `exec()`.
#         -   `globals['__builtins__']` will be a carefully curated dictionary containing
#             only safe built-in functions and types.
#         -   Disallowed builtins: `open`, `eval`, `exec`, `compile`, `getattr`, `setattr`,
#           `delattr`, `importlib`, `__import__` (direct use).
#         -   Allowed builtins (example list, needs careful review): `print`, `len`, `sum`,
#           `min`, `max`, `abs`, `round`, `range`, `zip`, `enumerate`, `map`, `filter`,
#           `sorted`, `reversed`, `all`, `any`, `isinstance`, `issubclass`,
#           `str`, `int`, `float`, `bool`, `list`, `dict`, `set`, `tuple`, `bytes`,
#           `bytearray`, `complex`, `frozenset`, `memoryview`, `object`, `slice`, `type`,
#           `Exception`, `ValueError`, `TypeError`, `AttributeError`, `NameError`, etc.
#           (standard error types are generally safe to allow to be raised).
#         -   No access to modules like `os`, `sys`, `subprocess`, `socket`, `requests`,
#           `pathlib` (for direct manipulation), `shutil`, `ctypes`, etc., unless a
#           specific, safe subset is explicitly exposed via the globals. For now, assume none.
#
# 4.  NO DIRECT FILESYSTEM ACCESS:
#     -   Code snippets executed via this tool SHOULD NOT attempt to read or write files directly.
#     -   If file content is needed as input, the AGI should use the `read_file` tool first.
#     -   If output needs to be saved, the AGI should generate the content and then use
#         the `write_file` tool.
#
# 5.  NO DIRECT NETWORK ACCESS:
#     -   Code snippets SHOULD NOT attempt to make network requests.
#     -   If web content is needed, the AGI should use the `web_search` tool.
#
# 6.  RESOURCE LIMITS (Conceptual - Not Implemented in Basic Version):
#     -   Timeout: Execution should be time-limited (e.g., a few seconds). This is hard
#       to enforce reliably with `exec()` alone in a single thread without more complex
#       subprocess/threading logic, which is out of scope for the basic version.
#       *Initial implementation might not have a strict timeout for simplicity, relying on user to kill if hangs.*
#     -   Memory/CPU: True limits require OS-level sandboxing (e.g., containers), which
#       is out of scope. Code should be assumed to be small and efficient.
#
# 7.  STDOUT/STDERR CAPTURING:
#     -   `sys.stdout` and `sys.stderr` will be redirected during the `exec()` call to
#         capture any output or error messages generated by the snippet.
#     -   This captured output will be returned to the AGI.
#
# 8.  EXCEPTION HANDLING:
#     -   Any Python exceptions raised by the snippet during `exec()` will be caught.
#     -   A string representation of the exception will be returned to the AGI.
#
# 9.  IMPORTS:
#     -   `import` statements within the executed code will be subject to the restricted
#         environment. If `__import__` is removed from builtins and no import-related
#         modules are in globals, imports will fail or be severely limited.
#     -   A small, safe list of standard library modules could potentially be whitelisted
#         and pre-imported into the globals (e.g., `math`, `json`, `datetime`, `random`, `re`).
#         *Initial decision: No `import` statements allowed in the snippet for simplicity and max safety.
#          If specific modules are needed, they must be pre-approved and added to the restricted globals.*
#
# This design prioritizes user awareness and control, and basic restriction of capabilities,
# over foolproof sandboxing, acknowledging the limitations of implementing a true sandbox
# within this project's current scope and architecture. The risk is mitigated by the
# AGI's nature (not inherently malicious) and mandatory user confirmation for all executions.
#

# --- Help Command ---
def display_help_command():
    """Displays a list of available slash commands and their descriptions."""
    help_table = Table(title="[bold green]AGI Terminal Help - Available Commands[/bold green]", show_lines=True)
    help_table.add_column("Command", style="bold cyan", min_width=20)
    help_table.add_column("Parameters", style="yellow", min_width=25)
    help_table.add_column("Description", style="white", max_width=70)

    commands = [
        ("/help", "", "Displays this help message."),
        ("/ls", "[path]", "Lists directory contents. Defaults to current directory."),
        ("/cd", "<path>", "Changes the current working directory."),
        ("/cwd", "", "Shows the current working directory."),
        ("/mkdir", "<directory_name>", "Creates a new directory."),
        ("/rm", "<path>", "Removes a file or directory (confirms before deleting)."),
        ("/cp", "<source> <destination>", "Copies a file or directory."),
        ("/mv", "<source> <destination>", "Moves or renames a file or directory."),
        ("/cat", "<file_path>", "Displays file content with syntax highlighting. Offers AGI summary for large files."),
        ("/edit", "<file_path>", "Opens a file with the system's default editor (or $EDITOR)."),
        ("/read_script", "<script_filename>", "Displays content of whitelisted project scripts (e.g., interactive_agi.py)."),
        ("/config", "show | get <key> | set <key> <value>", "Manages application configuration. Shows current, gets a key, or sets a key's value."),
        ("/git", "status | diff [file] | log [-n count]", "Executes git commands: status, diff (optionally for a file or staged), or log (optionally with count)."),
        ("/search", "<query>", "Performs an internet search using DuckDuckGo (HTML version)."),
        ("/history", "", "Displays the conversation history for the current session."),
        ("/clear", "", "Clears the terminal screen and re-displays the startup banner."),
        ("/sysinfo", "", "Displays system information (OS, Python, CPU, Memory, Disk)."),
        ("/set parameter", "<NAME> <VALUE>", "Sets an AGI generation parameter (e.g., MAX_TOKENS, TEMPERATURE)."),
        ("/show parameters", "", "Shows current AGI generation parameters."),
        ("/analyze_dir", "[path]", "Asks AGI to analyze directory structure and provide a JSON summary. Defaults to current directory."),
        ("/suggest_code_change", "<file_path>", "Asks AGI to suggest code changes for a whitelisted file based on user description."),
        ("exit / quit", "", "Exits the AGI terminal session.")
    ]

    for cmd, params, desc in commands:
        help_table.add_row(cmd, params, desc)

    console.print(help_table)

# --- File Content Interaction Commands ---
MAX_CAT_LINES = 200 # Lines to display for large files before asking to summarize
MAX_CAT_CHARS_FOR_SUMMARY = 5000 # Max chars to send for summary
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


MAX_SEARCH_RESULTS_FOR_AGI = 3 # Max number of search results to format for AGI context
MAX_SEARCH_SNIPPET_LENGTH_FOR_AGI = 250 # Max chars per snippet for AGI

def fetch_and_parse_search_results(query: str) -> tuple[Optional[list[dict]], Optional[str]]:
    """
    Fetches HTML from DuckDuckGo for a query and parses it.

    Returns:
        A tuple (list_of_result_dicts, None) on success,
        or (None, error_message_string) on failure.
        Each result_dict contains 'title', 'snippet', 'url'.
    """
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    html_content = ""

    try:
        import requests # Ensure requests is available
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
    except ImportError:
        return None, "The 'requests' library is required for web search but is not installed."
    except requests.exceptions.RequestException as e:
        return None, f"Network error during web search: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred during web fetch: {e}"

    if not html_content:
        return None, "No content received from search page."

    parsed_results = parse_duckduckgo_html_results(html_content)
    if not parsed_results:
        return None, "Could not parse any search results from the page."

    return parsed_results, None

def perform_internet_search(query: str):
    """User-facing /search command. Fetches, parses, and displays search results."""
    console.print(f"Searching the web for: \"{query}\"...", style="info")

    parsed_results, error = fetch_and_parse_search_results(query)

    if error:
        console.print(f"[bold red]Error performing search:[/bold red] {error}", style="error")
        return

    if not parsed_results: # Should be caught by error, but as a safeguard
        console.print("No search results found or failed to parse.", style="warning")
        return

    console.print(f"\n[bold green]Search Results for \"{query}\":[/bold green]")
    for i, res in enumerate(parsed_results): # Show all results from parser for user command
        panel_content = Text()
        panel_content.append(f"{i+1}. {res['title']}\n", style="bold link " + str(res['url']))
        panel_content.append(f"   {res['snippet']}\n", style="italic")
        panel_content.append(f"   Source: {res['url']}", style="dim")
        console.print(Panel(panel_content, expand=False, border_style="blue"))


def handle_web_search_request(query: str) -> tuple[Optional[str], Optional[str]]:
    """
    Handles an AGI's request to search the web using DuckDuckGo (HTML version).

    This function calls `fetch_and_parse_search_results` to get raw search data,
    then formats a limited number of results (titles, snippets, URLs) into a
    single string suitable for providing as context to the AGI.
    Snippets are truncated to manage context length.

    Args:
        query: The search query string provided by the AGI.

    Returns:
        A tuple `(formatted_results_string, None)` on success. `formatted_results_string`
        will contain the summarized search results, or a message like "No search results found."
        A tuple `(None, error_message_string)` if the search fetch or parsing failed.
    """
    console.print(f"[dim info]AGI requested web search for: \"{query}\"[/dim info]")
    parsed_results, error = fetch_and_parse_search_results(query)

    if error:
        return None, f"Web search failed: {error}"

    if not parsed_results:
        return "No search results found.", None # Not an error, but no results to give

    # Format results for AGI
    formatted_results_str = "Web Search Results:\n\n"
    for i, res_dict in enumerate(parsed_results[:MAX_SEARCH_RESULTS_FOR_AGI]):
        title = res_dict.get('title', 'N/A')
        snippet = res_dict.get('snippet', 'N/A')
        url = res_dict.get('url', '#')

        # Truncate snippet if too long
        if len(snippet) > MAX_SEARCH_SNIPPET_LENGTH_FOR_AGI:
            snippet = snippet[:MAX_SEARCH_SNIPPET_LENGTH_FOR_AGI] + "..."

        formatted_results_str += f"Result {i+1}:\n"
        formatted_results_str += f"  Title: {title}\n"
        formatted_results_str += f"  Snippet: {snippet}\n"
        formatted_results_str += f"  URL: {url}\n\n"

    if not formatted_results_str.strip() or len(parsed_results) == 0 :
        return "No effectively formatted search results to provide.", None

    return formatted_results_str.strip(), None


# --- Command Implementations ---
def list_directory_contents(path_str: str):
