# interactive_agi.py

import subprocess
import os
import sys
from pathlib import Path
import re

from typing import Optional, Union

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
    },
    "model": { # New section for model related configurations
        "merged_model_path": "./merged_model" # Default path for the main model
    }
}
APP_CONFIG = {} # Will be loaded from file or defaults

# Global context analyzer instance
context_analyzer = ContextAnalyzer()

# Global conversation history - maxlen will be set from config
conversation_history = deque()

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

# --- History Loading and Saving ---
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
    def __init__(self): # model_path_str argument removed
        # Determine model path from config, with a fallback to default
        configured_model_path = APP_CONFIG.get("model", {}).get("merged_model_path", "./merged_model")
        self.model_path = Path(configured_model_path).resolve() # Resolve to absolute path early

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
                    "For tasks requiring a sequence of tool operations, you can use the 'execute_plan' action:\n"
                    "{\n"
                    "  \"action\": \"execute_plan\",\n"
                    "  \"plan_reasoning\": \"<Overall rationale for the entire plan>\",\n"
                    "  \"steps\": [\n"
                    "    { \"action\": \"<tool_1>\", ..., \"reasoning\": \"<step_1_reasoning>\" },\n"
                    "    { \"action\": \"<tool_2>\", ..., \"reasoning\": \"<step_2_reasoning>\" },\n"
                    "    ...\n"
                    "  ]\n"
                    "}\n"
                    "Details for 'execute_plan':\n"
                    "- 'plan_reasoning' (optional but recommended) explains the overall goal of the sequence.\n"
                    "- 'steps' is a list, where each item is a complete JSON object for a single tool call (e.g., a full `run_shell` request, a `read_file` request, etc.), including its own 'action' and 'reasoning'.\n"
                    "- The system will attempt to execute these steps sequentially. User confirmation will be sought for each sensitive operation within the plan (like `write_file`, `execute_python_code`, git modifications).\n"
                    "- If any step is cancelled by the user or results in an error, the entire plan execution halts immediately.\n"
                    "- After the plan attempts execution (fully or partially), you will be re-prompted once with a summary of all attempted steps and their outcomes.\n"
                    "- Example: {\"action\": \"execute_plan\", \"plan_reasoning\": \"To create a script, make it executable, and then run it.\", \"steps\": [ \n"
                    "    { \"action\": \"write_file\", \"filepath\": \"temp_script.py\", \"content\": \"print('Hello from plan!')\", \"reasoning\": \"Create the temporary script.\" },\n"
                    "    { \"action\": \"run_shell\", \"command\": \"chmod\", \"args\": [\"+x\", \"temp_script.py\"], \"reasoning\": \"Make the script executable.\" },\n"
                    "    { \"action\": \"run_shell\", \"command\": \"./temp_script.py\", \"args\": [], \"reasoning\": \"Run the script.\" }\n"
                    "  ]}\n"
                    "\n"
                    "General Guidelines for Tool Use and Error Handling:\n"
                    "- If you request a tool action (single or as part of a plan) and the system informs you that it failed, you MUST acknowledge this failure in your response to the user.\n"
                    "- Explain the problem clearly based on the system's feedback.\n"
                    "- If appropriate, ask the user for clarification or suggest alternatives.\n"
                    "- If a user's request is too ambiguous, ask for clarification BEFORE attempting tool use or formulating a plan.\n"
                    "- For complex requests not suitable for a single tool, consider using 'execute_plan'. If using single tools iteratively for a complex task, outline your plan in your reasoning and use each tool's outcome to inform your next step.\n"
                    "\n"
                    "If the query cannot be answered with one of these tools (single actions or `execute_plan`), or if it requires actions not supported, answer directly as a helpful assistant without using the JSON format.\n"
                    "Choose only ONE top-level action per turn (e.g., one `run_shell` OR one `read_file` OR one `execute_plan`).\n"
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
    load_history() # Load history after config is loaded and conversation_history deque is initialized

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

    agi_interface = MergedAGI() # No longer pass model_path_str

    if not agi_interface.is_model_loaded:
        console.print("INFO: MergedAGI could not load the model. Falling back to AGIPPlaceholder.", style="warning")
        # agi_interface.model_path is now the resolved path from config or default
        model_path_for_error_msg = agi_interface.model_path
        if not model_path_for_error_msg.exists() or not any(model_path_for_error_msg.iterdir()):
            msg = Text()
            msg.append(f"\n---\nIMPORTANT: The model directory '{model_path_for_error_msg}',\n", style="bold yellow")
            msg.append("expected to contain the AI model, is missing or empty.\n", style="yellow")
            msg.append("This script will use MOCK responses.\nTo enable actual AI responses:\n", style="yellow")
            msg.append("  1. Ensure models are downloaded (e.g., via `setup_agi_terminal.py`).\n")
            msg.append(f"  2. Ensure 'merge_config.yml' (if used) points to correct downloaded models and mergekit output is at '{model_path_for_error_msg}'.\n")
            msg.append(f"  3. OR, update 'merged_model_path' in '{CONFIG_FILE_PATH.name}' (in {CONFIG_FILE_PATH.parent}) to your model's location.\n---")
            console.print(Panel(msg, title="[bold red]Model Not Found[/bold red]", border_style="red"))

        # AGIPPlaceholder doesn't actually use the path, but pass it for consistency if it ever did
        agi_interface = AGIPPlaceholder(model_path_str=str(model_path_for_error_msg))
        terminal_mode = "[bold yellow]Mock Mode[/bold yellow]"
    else:
        terminal_mode = "[bold green]Merged Model Mode[/bold green]"

    console.print(f"\nAGI Interactive Terminal ({terminal_mode}) - Model: {agi_interface.model_path}")
    console.print("Type '/set parameter <NAME> <VALUE>' to change generation settings (e.g., /set parameter MAX_TOKENS 512).")
    console.print("Type '/show parameters' to see current settings.")
    console.print("Type 'exit', 'quit', or press Ctrl+D to end.")
    console.print("Press Ctrl+C for forceful interruption.")
    console.print("-" * console.width)

    # --- Main Input Processing Function ---

# Helper function to process a single tool call, used by process_agi_tool_request
# This is a new internal helper for Stage 11's execute_plan
def _execute_single_tool_step(
    tool_step_data: dict, # The JSON data for the single tool step
    raw_user_input_for_turn: str, # The original user input for the whole turn
    agi_interface_instance: Union[MergedAGI, AGIPPlaceholder],
    turn_log_data_ref: dict, # The main log data for the entire turn
    plan_step_index: Optional[int] = None # If part of a plan, its index
    ) -> tuple[str, bool, dict]: # Returns (outcome_summary_for_agi, success_bool, tool_interaction_log_entry_for_this_step)
    """
    Processes a single tool call, either standalone or as part of a plan.
    Handles interaction, logging for this step, and preparing outcome for AGI.
    Does NOT perform the AGI re-prompt itself if it's part of a plan.
    """
    action = tool_step_data.get("action")
    # Initialize common fields for the log entry of this specific tool step
    tool_interaction_log_entry = {
        "tool_request_timestamp": datetime.now().isoformat(),
        "action_type": action,
        "action_details": tool_step_data, # Log the full step data as details
        "reasoning": tool_step_data.get("reasoning", f"Reasoning not provided for {action} step."),
        "part_of_plan_step": plan_step_index if plan_step_index is not None else None,
        # user_confirmation, tool_outcome_summary, tool_outcome_timestamp will be added below
    }

    step_outcome_summary_for_agi = ""
    step_succeeded = False # Tracks if this specific tool step was successful (or cancelled cleanly)

    # --- Actual Tool Handling Logic (adapted from existing process_agi_tool_request) ---
    # Note: This section will become very long, mirroring the previous if/elif structure.
    # For brevity in this diff, I'll only show the structure and one example (run_shell).
    # The actual implementation will move all tool handlers here.

    if action == "run_shell":
        command_executable = tool_step_data.get("command")
        command_args_list = tool_step_data.get("args", [])
        # ... (validation for command_executable, command_args_list as before) ...
        if not isinstance(command_args_list, list) or not all(isinstance(arg, str) for arg in command_args_list) or not command_executable:
            error_msg = f"Malformed '{action}' data: {tool_step_data}"
            console.print(f"[warning]{error_msg}[/warning]")
            step_outcome_summary_for_agi = f"Tool call '{action}' was malformed: {error_msg}"
            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
            tool_interaction_log_entry["tool_outcome_summary"] = step_outcome_summary_for_agi
            step_succeeded = False
        elif command_executable not in SHELL_COMMAND_WHITELIST:
            error_msg = f"Command '{command_executable}' is not whitelisted."
            console.print(f"[warning]AGI suggested non-whitelisted command: '{command_executable}'. Denied.[/warning]")
            step_outcome_summary_for_agi = f"Tool call '{action}' denied: {error_msg}"
            tool_interaction_log_entry["user_confirmation"] = "denied_by_system_whitelist"
            tool_interaction_log_entry["tool_outcome_summary"] = step_outcome_summary_for_agi
            step_succeeded = False # System rejection is a form of step failure
        else:
            command_to_display = f"{command_executable} {' '.join(command_args_list)}"
            console.print(Panel(Text(f"AGI tool step: [bold cyan]{command_to_display}[/bold cyan]\nReason: {tool_interaction_log_entry['reasoning']}", style="yellow"), title=f"[bold blue]Plan Step: Shell Command[/bold blue]"))
            confirmed = Confirm.ask("Execute this shell command step?", default=False, console=console) if RICH_AVAILABLE else input("Execute? (yes/NO):").lower() == "yes"
            tool_interaction_log_entry["user_confirmation"] = "confirmed" if confirmed else "cancelled"

            if confirmed:
                stdout_s, stderr_s, retcode = execute_shell_command(command_executable, command_args_list)
                outcome_parts = []
                if stdout_s and stdout_s.strip(): outcome_parts.append(f"Stdout: {stdout_s.strip()}")
                if stderr_s and stderr_s.strip(): outcome_parts.append(f"Stderr: {stderr_s.strip()}")

                log_summary = f"ReturnCode: {retcode}\n" + "\n".join(outcome_parts)
                if not outcome_parts and retcode == 0 : log_summary = f"ReturnCode: {retcode}\nCommand executed with no output."
                elif not outcome_parts and retcode !=0 : log_summary = f"ReturnCode: {retcode}\nCommand failed with no output."

                if len(log_summary) > 500: log_summary = log_summary[:497] + "..."
                tool_interaction_log_entry["tool_outcome_summary"] = log_summary
                step_outcome_summary_for_agi = log_summary # For plan summary
                step_succeeded = (retcode == 0) # Shell command success is usually retcode 0
            else:
                msg = f"Command '{command_to_display}' execution cancelled by user."
                console.print(f"[yellow]{msg}[/yellow]")
                tool_interaction_log_entry["tool_outcome_summary"] = msg
                step_outcome_summary_for_agi = msg
                step_succeeded = False # Cancellation is a form of step failure for plan progression

    elif action == "read_file":
        filepath_str = tool_step_data.get("filepath")
        max_lines = tool_step_data.get("max_lines")
        reasoning = tool_step_data.get("reasoning", "No reasoning for read_file.") # Already in main log entry
        tool_interaction_log_entry["action_details"] = {"filepath": filepath_str, "max_lines": max_lines}

        if not filepath_str:
            error_msg = f"Malformed '{action}' data: missing 'filepath'. Payload: {tool_step_data}"
            console.print(f"[warning]{error_msg}[/warning]")
            step_outcome_summary_for_agi = error_msg
            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
            tool_interaction_log_entry["tool_outcome_summary"] = error_msg
            step_succeeded = False
        else:
            # User confirmation for read_file is implicit (AGI decides to read)
            tool_interaction_log_entry["user_confirmation"] = "not_applicable"
            file_content_for_agi, read_error = handle_read_file_request(filepath_str, max_lines)
            if read_error:
                step_outcome_summary_for_agi = f"Attempt to read file '{filepath_str}' failed. Error: {read_error}"
                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {read_error}"
                step_succeeded = False
            else:
                step_outcome_summary_for_agi = file_content_for_agi # This is the "data" for AGI
                tool_interaction_log_entry["tool_outcome_summary"] = f"File '{filepath_str}' read successfully (snippet provided to AGI)."
                step_succeeded = True

    # ... (Implement similar blocks for write_file, execute_python_code, all git_ actions, web_search)

    # Example for write_file (needs full implementation for all tools)
    elif action == "write_file":
        filepath_str = tool_step_data.get("filepath")
        content_to_write = tool_step_data.get("content")
        tool_interaction_log_entry["action_details"] = {"filepath": filepath_str, "content_length": len(content_to_write or "")}

        if filepath_str is None or content_to_write is None:
            error_msg = f"Malformed '{action}' data: missing fields. Payload: {tool_step_data}"
            console.print(f"[warning]{error_msg}[/warning]")
            step_outcome_summary_for_agi = error_msg
            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
            tool_interaction_log_entry["tool_outcome_summary"] = error_msg
            step_succeeded = False
        else:
            # handle_write_file_request includes user confirmation
            outcome_message, outcome_error = handle_write_file_request(filepath_str, content_to_write)
            if outcome_error:
                step_outcome_summary_for_agi = f"Attempt to write file '{filepath_str}' failed. Error: {outcome_error}"
                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                tool_interaction_log_entry["user_confirmation"] = "n/a_error_occurred" # Or based on outcome_message if it indicates prior cancel
                step_succeeded = False
            else: # outcome_message is not None
                step_outcome_summary_for_agi = outcome_message
                tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                if "cancelled by user" in outcome_message.lower():
                    tool_interaction_log_entry["user_confirmation"] = "cancelled"
                    step_succeeded = False # Plan should halt on user cancel
                else:
                    tool_interaction_log_entry["user_confirmation"] = "confirmed"
                    step_succeeded = True # File written or "no changes"

    elif action == "execute_python_code":
        code_snippet = tool_step_data.get("code")
        tool_interaction_log_entry["action_details"] = {"code_snippet_length": len(code_snippet or "")} # Log length, not full code for brevity here

        if not isinstance(code_snippet, str) or not code_snippet.strip():
            error_msg = f"Malformed '{action}' data: missing or invalid 'code'. Payload: {tool_step_data}"
            console.print(f"[warning]{error_msg}[/warning]")
            step_outcome_summary_for_agi = error_msg
            tool_interaction_log_entry["user_confirmation"] = "n/a_malformed_request"
            tool_interaction_log_entry["tool_outcome_summary"] = error_msg
            step_succeeded = False
        else:
            stdout_s, stderr_s, exception_s = handle_execute_python_code_request(code_snippet)
            # handle_execute_python_code_request includes user confirmation and prints snippet/warning
            # It now also prints the panel title using the prefix
            stdout_s, stderr_s, exception_s = handle_execute_python_code_request(code_snippet, panel_title_prefix)

            tool_outcome_parts = []
            if stdout_s == "Code execution cancelled by user.":
                tool_interaction_log_entry["user_confirmation"] = "cancelled"
                step_succeeded = False
                tool_outcome_parts.append(stdout_s)
            elif exception_s == "StaticAnalysisReject":
                tool_interaction_log_entry["user_confirmation"] = "denied_by_system_static_analysis"
                step_succeeded = False
                tool_outcome_parts.append(stdout_s) # stdout_s contains the rejection message here
            else:
                tool_interaction_log_entry["user_confirmation"] = "confirmed"
                step_succeeded = not bool(exception_s) # Success if no Python exception during exec
                if stdout_s and stdout_s.strip(): tool_outcome_parts.append(f"Stdout:\n{stdout_s.strip()}")
                if stderr_s and stderr_s.strip(): tool_outcome_parts.append(f"Stderr:\n{stderr_s.strip()}")
                if exception_s and exception_s.strip(): tool_outcome_parts.append(f"Exception:\n{exception_s.strip()}")

            tool_outcome_summary_for_log = "\n---\n".join(tool_outcome_parts) if tool_outcome_parts else "No output or exception."
            tool_interaction_log_entry["tool_outcome_summary"] = tool_outcome_summary_for_log
            step_outcome_summary_for_agi = tool_outcome_summary_for_log # Full outcome for AGI

            # If standalone execute_python_code, re-prompt AGI
            if plan_step_index is None:
                subsequent_prompt_for_agi = (
                    f"{context_analyzer.get_full_context_string()}\n\n"
                    f"Outcome of your 'execute_python_code' request:\n```python\n{code_snippet}\n```\nExecution Result:\n{step_outcome_summary_for_agi}\n\n"
                    f"Original User Query: \"{raw_user_input_for_turn}\"\n\n"
                    "Based on this outcome, please formulate your response to the user or decide on the next step."
                )
                tool_interaction_log_entry["context_for_next_agi_step"] = subsequent_prompt_for_agi
                console.print(f"[info]Python code execution outcome processed. Re-prompting AGI...[/info]")
                with console.status("[yellow]AGI is processing code execution outcome...[/yellow]", spinner="dots"):
                    agi_secondary_raw_response = agi_interface_instance.generate_response(subsequent_prompt_for_agi)
                tool_interaction_log_entry["agi_secondary_raw_response"] = agi_secondary_raw_response
                step_outcome_summary_for_agi = agi_secondary_raw_response

    elif action == "git_branch_create":
        branch_name = tool_step_data.get("branch_name")
        base_branch = tool_step_data.get("base_branch")
        tool_interaction_log_entry["action_details"] = {"branch_name": branch_name, "base_branch": base_branch}
        if not branch_name:
            error_msg = f"Malformed '{action}' data: missing 'branch_name'. Payload: {tool_step_data}"
            step_outcome_summary_for_agi = error_msg
            tool_interaction_log_entry.update({"user_confirmation": "n/a_malformed_request", "tool_outcome_summary": error_msg})
            step_succeeded = False
        else:
            # handle_git_branch_create_request includes user confirmation
            outcome_message, outcome_error = handle_git_branch_create_request(branch_name, base_branch, panel_title_prefix)
            tool_interaction_log_entry["user_confirmation"] = "confirmed" if outcome_message and ("successfully" in outcome_message.lower() or "processed" in outcome_message.lower()) else \
                                                           ("cancelled" if outcome_message and "cancelled" in outcome_message.lower() else "n/a_error")
            if outcome_error:
                step_outcome_summary_for_agi = f"Git branch create failed: {outcome_error}"
                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                step_succeeded = False
            else:
                step_outcome_summary_for_agi = outcome_message
                tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                step_succeeded = True

    elif action == "git_checkout":
        branch_name = tool_step_data.get("branch_name")
        create_new = tool_step_data.get("create_new", False)
        tool_interaction_log_entry["action_details"] = {"branch_name": branch_name, "create_new": create_new}
        if not branch_name:
            error_msg = f"Malformed '{action}' data: missing 'branch_name'. Payload: {tool_step_data}"
            step_outcome_summary_for_agi = error_msg
            tool_interaction_log_entry.update({"user_confirmation": "n/a_malformed_request", "tool_outcome_summary": error_msg})
            step_succeeded = False
        else:
            outcome_message, outcome_error = handle_git_checkout_request(branch_name, create_new)
            tool_interaction_log_entry["user_confirmation"] = "confirmed" if outcome_message and "Successfully" in outcome_message else \
                                                           ("cancelled" if outcome_message and "cancelled" in outcome_message else "n/a_error")
            if outcome_error:
                step_outcome_summary_for_agi = f"Git checkout failed: {outcome_error}"
                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                step_succeeded = False
            else:
                step_outcome_summary_for_agi = outcome_message
                tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                step_succeeded = True

    elif action == "git_commit":
        commit_message = tool_step_data.get("commit_message")
        stage_all = tool_step_data.get("stage_all", False)
        tool_interaction_log_entry["action_details"] = {"commit_message_length": len(commit_message or ""), "stage_all": stage_all}
        if not commit_message or not commit_message.strip():
            error_msg = f"Malformed '{action}' data: missing 'commit_message'. Payload: {tool_step_data}"
            step_outcome_summary_for_agi = error_msg
            tool_interaction_log_entry.update({"user_confirmation": "n/a_malformed_request", "tool_outcome_summary": error_msg})
            step_succeeded = False
        else:
            outcome_message, outcome_error = handle_git_commit_request(commit_message, stage_all)
            tool_interaction_log_entry["user_confirmation"] = "confirmed" if outcome_message and ("successfully" in outcome_message.lower() or "nothing to commit" in outcome_message.lower()) else \
                                                           ("cancelled" if outcome_message and "cancelled" in outcome_message.lower() else "n/a_error")
            if outcome_error:
                step_outcome_summary_for_agi = f"Git commit failed: {outcome_error}"
                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
                step_succeeded = False
            else:
                step_outcome_summary_for_agi = outcome_message
                tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
                step_succeeded = True # "nothing to commit" is also a success for the step

    elif action == "git_push":
        remote_name = tool_step_data.get("remote_name")
        branch_name = tool_step_data.get("branch_name")
        tool_interaction_log_entry["action_details"] = {"remote_name": remote_name, "branch_name": branch_name}
        # No specific validation for missing optional fields here, handler defaults them
        outcome_message, outcome_error = handle_git_push_request(remote_name, branch_name)
        tool_interaction_log_entry["user_confirmation"] = "confirmed" if outcome_message and ("successfully" in outcome_message.lower() or "up-to-date" in outcome_message.lower()) else \
                                                       ("cancelled" if outcome_message and "cancelled" in outcome_message.lower() else "n/a_error")
        if outcome_error:
            step_outcome_summary_for_agi = f"Git push failed: {outcome_error}"
            tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {outcome_error}"
            step_succeeded = False
        else:
            step_outcome_summary_for_agi = outcome_message
            tool_interaction_log_entry["tool_outcome_summary"] = outcome_message
            step_succeeded = True

    elif action == "web_search":
        search_query = tool_step_data.get("query")
        tool_interaction_log_entry["action_details"] = {"query": search_query}
        if not search_query:
            error_msg = f"Malformed '{action}' data: missing 'query'. Payload: {tool_step_data}"
            step_outcome_summary_for_agi = error_msg
            tool_interaction_log_entry.update({"user_confirmation": "n/a_malformed_request", "tool_outcome_summary": error_msg})
            step_succeeded = False
        else:
            tool_interaction_log_entry["user_confirmation"] = "not_applicable"
            search_results_str, search_error = handle_web_search_request(search_query)
            if search_error:
                step_outcome_summary_for_agi = f"Web search failed: {search_error}"
                tool_interaction_log_entry["tool_outcome_summary"] = f"Error: {search_error}"
                step_succeeded = False
            else:
                step_outcome_summary_for_agi = search_results_str # This is the data for AGI
                tool_interaction_log_entry["tool_outcome_summary"] = "Search results provided to AGI." if search_results_str and "No search results found" not in search_results_str else search_results_str
                step_succeeded = True
    else: # Unknown action
        error_msg = f"Unknown tool action '{action}' requested in plan step or standalone."
        console.print(f"[red]{error_msg}[/red]")
        step_outcome_summary_for_agi = error_msg
        tool_interaction_log_entry["user_confirmation"] = "n/a_unknown_action"
        tool_interaction_log_entry["tool_outcome_summary"] = error_msg
        step_succeeded = False

    tool_interaction_log_entry["tool_outcome_timestamp"] = datetime.now().isoformat()
    turn_log_data_ref["tool_interactions"].append(tool_interaction_log_entry) # Append this step's log

    return step_outcome_summary_for_agi, step_succeeded, tool_interaction_log_entry


def process_single_input_turn(user_input_str: str,
                              agi_interface_instance: Union[MergedAGI, AGIPPlaceholder],
                              turn_log_data_ref: dict, # This is current_turn_interaction_data from the main loop
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

    if user_input_str.strip().lower() in ["exit", "quit"]:
        console.print("Exiting AGI session.", style="info")
        turn_log_data_ref["agi_final_response_to_user"] = "User initiated exit."
        turn_log_data_ref["timestamp_final_response"] = datetime.now().isoformat()
        log_interaction_to_jsonl(turn_log_data_ref)
        return "EXIT"

    if not user_input_str.strip():
        return "CONTINUE"

    conversation_history.append({"role": "user", "content": user_input_str, "timestamp": datetime.now().isoformat()})
    if session_logger and not is_scripted_input:
        session_logger.log_entry("User", user_input_str)

    action_taken_by_tool_framework = False # This will be true if a tool framework path is taken (single tool or plan)

    # --- User Command Processing ---
    if user_input_str.lower().startswith("/set parameter "):
        parts = user_input_str.strip().split(maxsplit=3)
        if len(parts) == 4:
            _, _, param_name, param_value = parts
            response = agi_interface_instance.set_parameter(param_name, param_value)
            console.print(f"AGI System: {response}")
        else:
            console.print("AGI System: [red]Invalid command.[/red] Usage: /set parameter <NAME> <VALUE>")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/set parameter", "args": user_input_str[len("/set parameter "):].strip()}
        final_text_for_user_display = f"System: Parameter command processed."
    elif user_input_str.lower() == "/show parameters":
        response = agi_interface_instance.show_parameters()
        console.print(Panel(response, title="[bold blue]AGI Parameters[/bold blue]", border_style="blue"))
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/show parameters"}
        final_text_for_user_display = "System: /show parameters command processed."
    elif user_input_str.lower().startswith("/ls"):
        parts = user_input_str.strip().split(maxsplit=1)
        path_to_list = parts[1] if len(parts) > 1 else "."
        list_directory_contents(path_to_list)
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/ls", "path": path_to_list}
        final_text_for_user_display = f"System: /ls command processed for path '{path_to_list}'."
    elif user_input_str.lower() == "/cwd":
        console.print(Panel(os.getcwd(), title="[bold blue]Current Working Directory[/bold blue]", border_style="blue"))
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/cwd"}
        final_text_for_user_display = "System: /cwd command processed."
    elif user_input_str.lower().startswith("/cd "):
        parts = user_input_str.strip().split(maxsplit=1)
        path_to_change = ""
        if len(parts) > 1:
            path_to_change = parts[1]
            change_directory(path_to_change)
        else:
            console.print("[red]Usage: /cd <path>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/cd", "path": path_to_change}
        final_text_for_user_display = f"System: /cd command processed for path '{path_to_change}'."
    elif user_input_str.lower() == "/clear":
        console.clear()
        display_startup_banner()
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/clear"}
        final_text_for_user_display = "System: /clear command processed."
    elif user_input_str.lower() == "/history":
        display_conversation_history()
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/history"}
        final_text_for_user_display = "System: /history command processed."
    elif user_input_str.lower() == "/sysinfo":
        display_system_info()
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/sysinfo"}
        final_text_for_user_display = "System: /sysinfo command processed."
    elif user_input_str.lower().startswith("/search "):
        query = user_input_str[len("/search "):].strip()
        if query:
            perform_internet_search(query)
        else:
            console.print("[red]Usage: /search <your query>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/search", "query": query}
        final_text_for_user_display = f"System: /search command processed for query '{query}'."
    elif user_input_str.lower().startswith("/mkdir "):
        path_to_create = user_input_str[len("/mkdir "):].strip()
        if path_to_create:
            create_directory_command(path_to_create)
        else:
            console.print("[red]Usage: /mkdir <directory_name>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/mkdir", "path": path_to_create}
        final_text_for_user_display = f"System: /mkdir command processed for path '{path_to_create}'."
    elif user_input_str.lower().startswith("/rm "):
        path_to_remove = user_input_str[len("/rm "):].strip()
        if path_to_remove:
            remove_path_command(path_to_remove)
        else:
            console.print("[red]Usage: /rm <file_or_directory_path>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/rm", "path": path_to_remove}
        final_text_for_user_display = f"System: /rm command processed for path '{path_to_remove}'."
    elif user_input_str.lower().startswith("/cp "):
        parts = user_input_str.strip().split()
        source, dest = "", ""
        if len(parts) == 3:
            source, dest = parts[1], parts[2]
            copy_path_command(source, dest)
        else:
            console.print("[red]Usage: /cp <source> <destination>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/cp", "source": source, "destination": dest}
        final_text_for_user_display = f"System: /cp command processed for source '{source}', destination '{dest}'."
    elif user_input_str.lower().startswith("/mv "):
        parts = user_input_str.strip().split()
        source, dest = "", ""
        if len(parts) == 3:
            source, dest = parts[1], parts[2]
            move_path_command(source, dest)
        else:
            console.print("[red]Usage: /mv <source> <destination>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/mv", "source": source, "destination": dest}
        final_text_for_user_display = f"System: /mv command processed for source '{source}', destination '{dest}'."
    elif user_input_str.lower().startswith("/cat "):
        file_path_to_cat = user_input_str[len("/cat "):].strip()
        if file_path_to_cat:
            cat_file_command(file_path_to_cat, agi_interface_instance)
        else:
            console.print("[red]Usage: /cat <file_path>[/red]")
        turn_log_data_ref["is_user_command"] = True # Still a user command, even if AGI is used for summary
        turn_log_data_ref["command_details"] = {"command": "/cat", "path": file_path_to_cat}
        final_text_for_user_display = f"System: /cat command processed for '{file_path_to_cat}'."
    elif user_input_str.lower().startswith("/edit "):
        file_path_to_edit = user_input_str[len("/edit "):].strip()
        if file_path_to_edit:
            edit_file_command(file_path_to_edit)
        else:
            console.print("[red]Usage: /edit <file_path>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/edit", "path": file_path_to_edit}
        final_text_for_user_display = f"System: /edit command processed for '{file_path_to_edit}'."
    elif user_input_str.lower().startswith("/read_script "):
        script_name_to_read = user_input_str[len("/read_script "):].strip()
        allowed_scripts = ["interactive_agi.py", "setup_agi_terminal.py", "adaptive_train.py", "download_models.sh", "train_on_interaction.sh", "merge_config.yml"]
        if script_name_to_read and script_name_to_read in allowed_scripts:
            script_path = Path(script_name_to_read)
            if not script_path.exists(): script_path = Path("..") / script_name_to_read
            if script_path.exists() and script_path.is_file():
                 cat_file_command(str(script_path), agi_interface_instance)
            else: console.print(f"[red]Error: Script '{script_name_to_read}' not found.[/red]")
        elif script_name_to_read: console.print(f"[red]Error: Reading script '{script_name_to_read}' is not allowed.[/red]")
        else: console.print("[red]Usage: /read_script <script_filename>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/read_script", "script_name": script_name_to_read}
        final_text_for_user_display = f"System: /read_script command processed for '{script_name_to_read}'."
    elif user_input_str.lower().startswith("/config"):
        config_args = user_input_str.strip()[len("/config"):].strip()
        config_command_handler(config_args)
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/config", "args": config_args}
        final_text_for_user_display = f"System: /config command processed with args '{config_args}'."
    elif user_input_str.lower().startswith("/git "):
        git_command_parts = user_input_str.strip().split(maxsplit=2)
        git_subcommand = git_command_parts[1].lower() if len(git_command_parts) > 1 else None
        git_args = git_command_parts[2] if len(git_command_parts) > 2 else None
        if git_subcommand == "status": git_status_command()
        elif git_subcommand == "diff": git_diff_command(git_args)
        elif git_subcommand == "log": git_log_command(git_args)
        else: console.print(f"[red]Unknown git subcommand: {git_subcommand}[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/git", "args": user_input_str[len("/git "):].strip()}
        final_text_for_user_display = f"System: /git command processed."
    elif user_input_str.lower() == "/help":
        display_help_command()
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/help"}
        final_text_for_user_display = "System: /help command processed."
    elif user_input_str.lower().startswith("/save_script "):
        parts = user_input_str.strip().split(maxsplit=2)
        script_name, commands_str = "", ""
        if len(parts) < 3: console.print("[red]Usage: /save_script <script_name> <command1> ; ...[/red]")
        else:
            script_name = parts[1]
            commands_str = parts[2]
            if not re.match(r"^[a-zA-Z0-9_-]+$", script_name): console.print(f"[red]Invalid script name: '{script_name}'.[/red]")
            elif not commands_str.strip(): console.print(f"[red]Cannot save an empty script for '{script_name}'.[/red]")
            else:
                commands_list = [cmd.strip() for cmd in commands_str.split(';') if cmd.strip()]
                if not commands_list: console.print(f"[red]No valid commands in script for '{script_name}'.[/red]")
                else:
                    scripts = load_user_scripts()
                    scripts[script_name] = commands_list
                    save_user_scripts(scripts)
                    console.print(f"[green]Script '{script_name}' saved.[/green]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/save_script", "name": script_name, "commands_str_len": len(commands_str)}
        final_text_for_user_display = "System: /save_script command processed."
    elif user_input_str.lower() == "/list_models":
        display_list_models_command()
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/list_models"}
        final_text_for_user_display = "System: /list_models command processed."
    elif user_input_str.lower().startswith("/run_script "):
        parts = user_input_str.strip().split(maxsplit=1)
        script_name_to_run = ""
        if len(parts) < 2 or not parts[1].strip(): console.print("[red]Usage: /run_script <script_name>[/red]")
        else:
            script_name_to_run = parts[1].strip()
            scripts = load_user_scripts()
            if script_name_to_run not in scripts: console.print(f"[yellow]Script '{script_name_to_run}' not found.[/yellow]")
            else:
                console.print(f"[info]Running script: '{script_name_to_run}'...[/info]")
                script_commands = scripts[script_name_to_run]
                for i, command_str in enumerate(script_commands):
                    console.print(f"\n[magenta]--- Script '{script_name_to_run}' Command {i+1}/{len(script_commands)} ---[/magenta]")
                    console.print(f"[dim]Executing: [/dim][bold cyan]{command_str}[/bold cyan]")
                    action_prompt = Text.assemble(("Press ", "magenta"), ("ENTER", "bold white"), (" to execute, ", "magenta"), ("S", "bold white"), (" to skip, ", "magenta"), ("A", "bold white"), (" to abort script: ", "magenta"))
                    user_choice = console.input(action_prompt).lower().strip() if RICH_AVAILABLE else input("Press ENTER to execute, S to skip, A to abort script: ").lower().strip()
                    if user_choice == 's': console.print("[yellow]Skipped.[/yellow]"); continue
                    if user_choice == 'a': console.print("[yellow]Script aborted by user.[/yellow]"); break
                    if user_choice == "":
                        global TURN_ID_COUNTER
                        TURN_ID_COUNTER +=1
                        script_turn_data = {"session_id": SESSION_ID, "turn_id": TURN_ID_COUNTER, "timestamp_user_query": datetime.now().isoformat(), "user_query": command_str, "tool_interactions": [], "script_info": {"parent_script": script_name_to_run, "command_index": i}}
                        status = process_single_input_turn(command_str, agi_interface_instance, script_turn_data, is_scripted_input=True)
                        if status == "EXIT": console.print(f"[yellow]Script '{script_name_to_run}' encountered exit. Script terminated.[/yellow]"); break
                    else: console.print("[yellow]Invalid choice. Skipping command.[/yellow]"); continue
                else: console.print(f"[info]Script '{script_name_to_run}' finished.[/info]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/run_script", "script_name": script_name_to_run}
        final_text_for_user_display = f"System: /run_script '{script_name_to_run}' execution sequence initiated/completed."
    elif user_input_str.lower() == "/list_scripts":
        scripts = load_user_scripts()
        if not scripts: console.print("[yellow]No user scripts saved.[/yellow]")
        else:
            table = Table(title="[bold blue]Saved User Scripts[/bold blue]")
            table.add_column("Script Name", style="cyan", no_wrap=True); table.add_column("# Commands", style="magenta", justify="right"); table.add_column("Commands (first few shown)", style="white")
            for name, cmds in sorted(scripts.items()):
                cmds_preview = " ; ".join(cmds[:3]);
                if len(cmds) > 3: cmds_preview += " ; ..."
                table.add_row(name, str(len(cmds)), cmds_preview)
            console.print(table)
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/list_scripts"}
        final_text_for_user_display = "System: /list_scripts command processed."
    elif user_input_str.lower().startswith("/delete_script "):
        parts = user_input_str.strip().split(maxsplit=1)
        script_name_to_delete = ""
        if len(parts) < 2 or not parts[1].strip(): console.print("[red]Usage: /delete_script <script_name>[/red]")
        else:
            script_name_to_delete = parts[1].strip()
            scripts = load_user_scripts()
            if script_name_to_delete in scripts: del scripts[script_name_to_delete]; save_user_scripts(scripts); console.print(f"[green]Script '{script_name_to_delete}' deleted.[/green]")
            else: console.print(f"[yellow]Script '{script_name_to_delete}' not found.[/yellow]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/delete_script", "script_name": script_name_to_delete}
        final_text_for_user_display = f"System: /delete_script command processed for '{script_name_to_delete}'."
    elif user_input_str.lower().startswith("/suggest_code_change "):
        file_path_to_suggest = user_input_str[len("/suggest_code_change "):].strip()
        if file_path_to_suggest:
            suggest_code_change_command(file_path_to_suggest, agi_interface_instance)
        else:
            console.print("[red]Usage: /suggest_code_change <file_path>[/red]")
        turn_log_data_ref["is_user_command"] = True
        turn_log_data_ref["command_details"] = {"command": "/suggest_code_change", "path": file_path_to_suggest}
        final_text_for_user_display = f"System: /suggest_code_change processed for '{file_path_to_suggest}' (AGI interaction handled within)."
    # --- End of User Command Block ---

    else: # Default: Input goes to AGI
        turn_log_data_ref["is_user_command"] = False # This is an AGI query
        turn_log_data_ref["context_at_query_time"] = context_analyzer.get_full_context_dict()
        turn_log_data_ref["agi_initial_processing_details"] = {
            "generation_params_used": agi_interface_instance.generation_params.copy(),
            "detected_task_type": agi_interface_instance.last_detected_task_type
        }

        with console.status("[yellow]AGI is thinking...[/yellow]", spinner="dots"):
            agi_initial_raw_response = agi_interface_instance.generate_response(user_input_str)

        turn_log_data_ref["agi_initial_raw_response"] = agi_initial_raw_response
        final_text_for_user_display = agi_initial_raw_response # Default if no tool or tool fails early

        # --- Try to process AGI response as a potential tool request (single or plan) ---
        try:
            json_match = re.search(r"```json\n(.*?)\n```", agi_initial_raw_response, re.DOTALL)
            json_str_to_parse = json_match.group(1) if json_match else agi_initial_raw_response
            first_brace = json_str_to_parse.find('{')
            last_brace = json_str_to_parse.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str_to_parse = json_str_to_parse[first_brace : last_brace+1]

            parsed_tool_request_data = json.loads(json_str_to_parse)
            action_type_from_agi = parsed_tool_request_data.get("action")

            if isinstance(parsed_tool_request_data, dict) and action_type_from_agi:
                action_taken_by_tool_framework = True # A tool path is being taken

                if action_type_from_agi == "execute_plan":
                    plan_steps_data = parsed_tool_request_data.get("steps", [])
                    plan_reasoning = parsed_tool_request_data.get("plan_reasoning", "No overall plan reasoning provided.")

                    # Initialize plan_execution_details in the main turn log
                    turn_log_data_ref["plan_execution_details"] = {
                        "action_type": "execute_plan", # Explicitly log that this was a plan
                        "plan_reasoning": plan_reasoning,
                        "requested_steps_count": len(plan_steps_data),
                        "executed_steps_count": 0, # Will be incremented
                        "plan_outcome": "Pending", # Will be updated: "Completed", "Halted", "Malformed"
                        "context_for_final_agi_step": None, # Will be populated before re-prompt
                        "agi_final_response_after_plan": None # Will be populated after re-prompt
                    }

                    if not isinstance(plan_steps_data, list) or not plan_steps_data:
                        final_text_for_user_display = "AGI suggested a plan with no steps or invalid format. Please try again."
                        turn_log_data_ref["plan_execution_details"]["plan_outcome"] = "Malformed Plan (no steps)"
                        action_taken_by_tool_framework = False
                    else:
                        console.print(Panel(Text(f"AGI proposes a multi-step plan:\n[italic]{plan_reasoning}[/italic]", style="yellow"),
                                            title="[bold blue]AGI Execution Plan[/bold blue]", border_style="blue"))

                        plan_outcome_summaries_for_agi = []
                        plan_halted = False
                        executed_step_count = 0

                        for i, step_data in enumerate(plan_steps_data):
                            console.print(f"\n[magenta]--- Executing Plan Step {i+1}/{len(plan_steps_data)} ---[/magenta]")
                            step_outcome_summary, step_success, step_tool_log_entry = _execute_single_tool_step(
                                tool_step_data=step_data,
                                raw_user_input_for_turn=user_input_str,
                                agi_interface_instance=agi_interface_instance,
                                turn_log_data_ref=turn_log_data_ref,
                                plan_step_index=i
                            )
                            # step_tool_log_entry is already appended to turn_log_data_ref["tool_interactions"]
                            plan_outcome_summaries_for_agi.append(f"Step {i+1} ({step_tool_log_entry.get('action_type', 'unknown_action')} - Status: {'Success' if step_success else 'Failed/Cancelled'}): {step_outcome_summary}")
                            executed_step_count += 1

                            if not step_success:
                                console.print(f"[bold red]Plan step {i+1} failed or was cancelled. Halting plan execution.[/bold red]")
                                plan_halted = True
                                turn_log_data_ref["plan_execution_details"]["plan_outcome"] = f"Halted at step {i+1} ({step_tool_log_entry.get('action_type')})"
                                break

                        turn_log_data_ref["plan_execution_details"]["executed_steps_count"] = executed_step_count
                        if not plan_halted:
                             turn_log_data_ref["plan_execution_details"]["plan_outcome"] = "All steps attempted."

                        full_plan_outcome_summary_for_agi = "\n".join(plan_outcome_summaries_for_agi)
                        prompt_after_plan = (
                            f"{context_analyzer.get_full_context_string()}\n\n"
                            f"The following multi-step plan was attempted based on your previous suggestion:\n"
                            f"Overall Plan Reasoning: {plan_reasoning}\n\n"
                            f"Execution Summary of all steps:\n{full_plan_outcome_summary_for_agi}\n\n"
                            f"Original User Query: \"{user_input_str}\"\n\n"
                            "Based on the outcome of this plan, please formulate your final response to the user."
                        )
                        turn_log_data_ref["plan_execution_details"]["context_for_final_agi_step"] = prompt_after_plan

                        console.print(f"\n[info]Plan execution finished. Re-prompting AGI with summary of all steps...[/info]")
                        with console.status("[yellow]AGI is processing plan execution summary...[/yellow]", spinner="dots"):
                            agi_final_response_after_plan = agi_interface_instance.generate_response(prompt_after_plan)

                        turn_log_data_ref["plan_execution_details"]["agi_final_response_after_plan"] = agi_final_response_after_plan
                        final_text_for_user_display = agi_final_response_after_plan

                elif action_type_from_agi: # Single tool call (not a plan)
                    # _execute_single_tool_step will handle the single tool call.
                    # If it needs to re-prompt AGI (e.g., read_file), it will do so, and its returned
                    # `outcome_summary_for_agi` will be the AGI's secondary response.
                    # This secondary response then becomes `final_text_for_user_display`.
                    # The helper also appends its specific tool interaction log to turn_log_data_ref.

                    outcome_summary_from_tool_step, step_succeeded, _ = _execute_single_tool_step(
                        tool_step_data=parsed_tool_request_data,
                        raw_user_input_for_turn=user_input_str,
                        agi_interface_instance=agi_interface_instance,
                        turn_log_data_ref=turn_log_data_ref,
                        plan_step_index=None
                    )
                    final_text_for_user_display = outcome_summary_from_tool_step
                    # action_taken_by_tool_framework remains True

                else: # Should not happen if action_type_from_agi was truthy
                    action_taken_by_tool_framework = False # No valid tool action

            else: # Not a dict or no "action" key
                action_taken_by_tool_framework = False

        except json.JSONDecodeError:
            action_taken_by_tool_framework = False # Not a JSON response for tool use, treat as normal chat
        except Exception as e:
            console.print(f"[warning]Could not fully process AGI response for potential tool use: {type(e).__name__} - {e}[/warning]")
            action_taken_by_tool_framework = False # Tool processing failed

        # --- Display Final AGI Response / Tool Outcome ---
        conversation_history.append({"role": "assistant", "content": final_text_for_user_display, "timestamp": datetime.now().isoformat()})
        if session_logger and getattr(session_logger, 'enabled', True):
            session_logger.log_entry("AGI", final_text_for_user_display)

        response_parts = detect_code_blocks(final_text_for_user_display)
        panel_title_text = "[agiprompt]AGI Output[/agiprompt]"
        panel_border_style_color = "blue"
        task_type_for_style = agi_interface_instance.last_detected_task_type

        if task_type_for_style == "code_generation":
            panel_title_text = "[agiprompt]AGI Code Generation[/agiprompt]"
            panel_border_style_color = "green"
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

        turn_log_data_ref["timestamp_final_response"] = datetime.now().isoformat()
        turn_log_data_ref["agi_final_response_to_user"] = final_text_for_user_display
        turn_log_data_ref["final_response_formatting_details"] = {
            "panel_title": panel_title_text,
            "panel_border_style": panel_border_style_color,
            "contains_code_blocks": contains_code_blocks_in_final_output
        }

        if isinstance(agi_interface_instance, MergedAGI) and agi_interface_instance.is_model_loaded:
            call_training_script(user_input_str, final_text_for_user_display)

    # Log the completed turn data (this happens regardless of whether it was a user command or AGI query)
    log_interaction_to_jsonl(turn_log_data_ref)
    return "CONTINUE"


def main():
    global session_logger, APP_CONFIG, conversation_history, DEFAULT_GENERATION_PARAMS, PROJECT_ROOT_PATH, SESSION_ID, JSONL_LOGGING_ENABLED, TURN_ID_COUNTER

    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S_") + Path(sys.argv[0]).stem
    TURN_ID_COUNTER = 0 # Initialize turn counter for the session

    PROJECT_ROOT_PATH = find_project_root(Path.cwd())
    if PROJECT_ROOT_PATH:
        console.print(f"INFO: Project root detected at: [cyan]{PROJECT_ROOT_PATH}[/cyan]", style="info")
    else:
        PROJECT_ROOT_PATH = Path.cwd().resolve()
        console.print(f"WARNING: No project root marker (.git or .agi_project_root) found. "
                      f"Using current working directory as project scope: [cyan]{PROJECT_ROOT_PATH}[/cyan]", style="warning")

    load_config()

    desktop_path = get_desktop_path()
    session_logger = SessionLogger(desktop_path)
    if APP_CONFIG.get("logging", {}).get("desktop_logging_enabled", True) == False:
        session_logger.enabled = False
        console.print("[info]Desktop session logging is disabled via config.[/info]")

    display_startup_banner()
    console.print("Initializing AGI System (Interactive Mode)...", style="info")

    agi_interface = MergedAGI()

    if not agi_interface.is_model_loaded:
        console.print("INFO: MergedAGI could not load the model. Falling back to AGIPPlaceholder.", style="warning")
        model_path_for_error_msg = agi_interface.model_path
        if not model_path_for_error_msg.exists() or not any(model_path_for_error_msg.iterdir()):
            msg = Text()
            msg.append(f"\n---\nIMPORTANT: The model directory '{model_path_for_error_msg}',\n", style="bold yellow")
            msg.append("expected to contain the AI model, is missing or empty.\n", style="yellow")
            msg.append("This script will use MOCK responses.\nTo enable actual AI responses:\n", style="yellow")
            msg.append("  1. Ensure models are downloaded (e.g., via `setup_agi_terminal.py`).\n")
            msg.append(f"  2. Ensure 'merge_config.yml' (if used) points to correct downloaded models and mergekit output is at '{model_path_for_error_msg}'.\n")
            msg.append(f"  3. OR, update 'merged_model_path' in '{CONFIG_FILE_PATH.name}' (in {CONFIG_FILE_PATH.parent}) to your model's location.\n---")
            console.print(Panel(msg, title="[bold red]Model Not Found[/bold red]", border_style="red"))

        agi_interface = AGIPPlaceholder(model_path_str=str(model_path_for_error_msg))
        terminal_mode = "[bold yellow]Mock Mode[/bold yellow]"
    else:
        terminal_mode = "[bold green]Merged Model Mode[/bold green]"

    console.print(f"\nAGI Interactive Terminal ({terminal_mode}) - Model: {agi_interface.model_path}")
    console.print("Type '/set parameter <NAME> <VALUE>' to change generation settings (e.g., /set parameter MAX_TOKENS 512).")
    console.print("Type '/show parameters' to see current settings.")
    console.print("Type 'exit', 'quit', or press Ctrl+D to end.")
    console.print("Type '/help' for a list of user commands.")
    console.print("-" * console.width)

    try:
        while True:
            user_input = console.input("[bold prompt]You> [/bold prompt]")

            TURN_ID_COUNTER += 1
            current_turn_interaction_data = {
                "session_id": SESSION_ID,
                "turn_id": TURN_ID_COUNTER,
                "timestamp_user_query": datetime.now().isoformat(),
                "user_query": user_input,
                "tool_interactions": [],
            }

            status = process_single_input_turn(user_input, agi_interface, current_turn_interaction_data)
            if status == "EXIT":
                break

            console.print("-" * console.width)

    except KeyboardInterrupt:
        console.print("\nExiting due to KeyboardInterrupt...", style="info")
    except EOFError: # Handle Ctrl+D
        console.print("\nExiting due to EOF (Ctrl+D)...", style="info")
    finally:
        save_history()
        console.print("AGI session terminated. History saved.", style="info")


if __name__ == "__main__":
    # load_history() # Moved into main()
    try:
        main()
    except SystemExit:
        pass
    finally:
        save_history()
        # The message below might be redundant if main's finally also prints it.
        # Consider if only one "session terminated" message is desired.
        # For now, it's fine.
        # console.print("AGI session terminated. History saved.", style="info")
    # The main __main__ block's finally clause handles saving history and the final "AGI session terminated" message.


# --- History Loading and Saving ---
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
        console.print(f"[red]Error running git status: {e.stderr}[/red]")
    except ValueError:
        console.print("[red]Invalid value encountered during git status processing.[/red]")
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
    # load_history() # Moved into main()
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
    pass
