#!/usr/bin/env python3
# adaptive_train.py

import argparse
import os
import glob
import random
from pathlib import Path
import json
import re
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.text import Text
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()
    console.print("[yellow]WARNING: Rich library not found. Install with `pip install rich`[/yellow]")

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

DEFAULT_MODEL_PATH = "./merged_model"
DEFAULT_OUTPUT_DIR = "./merged_model_adapters"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive fine-tuning script for AGI model using JSONL interaction logs.")
    parser.add_argument("--analyze_jsonl_logs", nargs='?', const=-1, type=int, metavar='N', help="Analyze JSONL logs, print stats, and optionally N random formatted examples.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to base model.")
    parser.add_argument("--jsonl_log_path", type=str, default=str(Path(__file__).resolve().parent / ".agi_terminal_cache" / "interaction_logs.jsonl"), help="Path to JSONL interaction log file.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for PEFT adapters.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA (4-bit quantization).")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated LoRA target modules.")
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # New CLI arguments for selective training data strategies
    parser.add_argument("--train-on-successful-tool-use", type=str, default=None, metavar="TOOL_NAME|all", help="Filter for successful tool use (specific tool or 'all').")
    parser.add_argument("--train-on-failed-tool-use", type=str, default=None, metavar="TOOL_NAME|all", help="Filter for failed/cancelled tool use (specific tool or 'all').")
    parser.add_argument("--train-on-successful-plans", action="store_true", help="Filter for successfully completed multi-step plans.")
    parser.add_argument("--train-on-halted-plans", action="store_true", help="Filter for halted multi-step plans.")
    parser.add_argument("--min-tool-interactions", type=int, default=0, metavar="N", help="Filter for turns with at least N tool interactions.")

    return parser.parse_args()

def format_context_for_prompt(context_dict: dict) -> str:
    if not context_dict: return "SYSTEM_CONTEXT: None"
    parts = []
    if "cwd" in context_dict: parts.append(f"CWD: {context_dict['cwd']}")
    if "project_root" in context_dict: parts.append(f"ProjectRoot: {context_dict['project_root']}")
    git_info = context_dict.get("git_info", {})
    if git_info.get("in_git_repo"):
        branch = git_info.get('branch', 'N/A')
        modified_files = git_info.get('modified_files', 0)
        status_str = f"{modified_files} modified" if modified_files > 0 else "Clean"
        parts.append(f"Git: Branch='{branch}', Status='{status_str}'")
    file_counts = context_dict.get("file_type_counts", {})
    if file_counts:
        top_files = sorted(file_counts.items(), key=lambda item: item[1], reverse=True)[:3]
        files_str = ", ".join([f"{lang}:{count}" for lang, count in top_files])
        parts.append(f"FileTypes: {files_str}{'...' if len(file_counts) > 3 else ''}")
    key_snippets_headers = context_dict.get("key_file_snippets", [])
    if key_snippets_headers:
        filenames = []
        for header_text in key_snippets_headers:
            match = re.search(r"--- Snippet from ([\w\.\-]+)", header_text)
            if match: filenames.append(match.group(1))
        if filenames: parts.append(f"KeyFilesProvided: [{', '.join(filenames)}]")
    return "SYSTEM_CONTEXT:\n" + "\n".join(f"  {p}" for p in parts) if parts else "SYSTEM_CONTEXT: (No specific context details available)"

def format_tool_interactions_for_prompt(tool_interactions: list, max_outcome_chars: int = 200) -> str:
    if not tool_interactions: return ""
    dialogue = ["\nINTERMEDIATE_STEPS:"]
    for i, tool_call in enumerate(tool_interactions):
        action_type = tool_call.get('action_type', 'unknown_action')
        action_details = tool_call.get('action_details', {})
        reasoning = tool_call.get('reasoning', 'No reasoning.')
        outcome_summary = tool_call.get('tool_outcome_summary', 'No outcome summary.')
        agi_secondary_response = tool_call.get('agi_secondary_raw_response')
        dialogue.append(f"  --- TOOL_CALL_ATTEMPT_{i+1} ---")
        dialogue.append(f"    AGI_REQUESTED_ACTION: {action_type}")
        details_str = json.dumps(action_details);
        if len(details_str) > 150: details_str = details_str[:150] + "..."
        dialogue.append(f"    ACTION_DETAILS: {details_str}")
        dialogue.append(f"    AGI_REASONING: {reasoning}")
        outcome_display = outcome_summary
        if len(outcome_display) > max_outcome_chars: outcome_display = outcome_display[:max_outcome_chars] + f"..."
        dialogue.append(f"    SYSTEM_OUTCOME: {outcome_display}")
        if agi_secondary_response:
            secondary_resp_display = agi_secondary_response
            if len(secondary_resp_display) > max_outcome_chars * 2: secondary_resp_display = secondary_resp_display[:max_outcome_chars*2] + f"..."
            dialogue.append(f"    AGI_RESPONSE_TO_OUTCOME: {secondary_resp_display}")
    return "\n".join(dialogue)

def _format_for_direct_answer(entry: dict, eos_token: str) -> Optional[str]:
    user_query = entry.get("user_query", ""); agi_final_response = entry.get("agi_final_response_to_user", ""); context_dict = entry.get("context_at_query_time", {})
    if not user_query or not agi_final_response: return None
    context_str = format_context_for_prompt(context_dict)
    return f"{context_str}\nUSER: {user_query}\nASSISTANT: {agi_final_response}{eos_token}"

def _format_for_single_tool_call_generation(entry: dict, eos_token: str) -> Optional[str]:
    user_query = entry.get("user_query", ""); agi_initial_raw_response = entry.get("agi_initial_raw_response", ""); context_dict = entry.get("context_at_query_time", {})
    if not user_query or not agi_initial_raw_response or not (agi_initial_raw_response.strip().startswith("{") and agi_initial_raw_response.strip().endswith("}")): return None
    try:
        parsed_json = json.loads(agi_initial_raw_response)
        if not isinstance(parsed_json, dict) or not parsed_json.get("action") or parsed_json.get("action") == "execute_plan": return None
    except json.JSONDecodeError: return None
    context_str = format_context_for_prompt(context_dict)
    system_instruction = "SYSTEM_INSTRUCTION: You have access to tools. Respond with a JSON action or a direct answer."
    return f"{context_str}\nUSER: {user_query}\n{system_instruction}\nASSISTANT: {agi_initial_raw_response.strip()}{eos_token}"

def _format_for_tool_outcome_processing(entry: dict, tool_interaction: dict, eos_token: str) -> Optional[str]:
    user_query = entry.get("user_query", ""); context_dict = entry.get("context_at_query_time", {}); agi_secondary_response = tool_interaction.get("agi_secondary_raw_response")
    if not user_query or not agi_secondary_response or not context_dict: return None
    context_str = format_context_for_prompt(context_dict); tool_summary_str = format_tool_interactions_for_prompt([tool_interaction])
    system_instruction = "SYSTEM_INSTRUCTION: Based on the tool outcome, respond to the user's original query."
    return f"{context_str}\nUSER: {user_query}{tool_summary_str}\n{system_instruction}\nASSISTANT: {agi_secondary_response}{eos_token}"

def _format_for_plan_generation(entry: dict, eos_token: str) -> Optional[str]:
    user_query = entry.get("user_query", ""); agi_initial_raw_response = entry.get("agi_initial_raw_response", ""); context_dict = entry.get("context_at_query_time", {})
    if not user_query or not agi_initial_raw_response: return None
    try:
        parsed_json = json.loads(agi_initial_raw_response)
        if not isinstance(parsed_json, dict) or parsed_json.get("action") != "execute_plan": return None
    except json.JSONDecodeError: return None
    context_str = format_context_for_prompt(context_dict)
    system_instruction = "SYSTEM_INSTRUCTION: For multi-step tasks, use 'execute_plan'."
    return f"{context_str}\nUSER: {user_query}\n{system_instruction}\nASSISTANT: {agi_initial_raw_response.strip()}{eos_token}"

def _format_for_plan_summary(entry: dict, eos_token: str) -> Optional[str]:
    user_query = entry.get("user_query", ""); context_dict = entry.get("context_at_query_time", {}); plan_details = entry.get("plan_execution_details", {})
    agi_initial_plan_json = entry.get("agi_initial_raw_response", ""); final_response_after_plan = plan_details.get("agi_final_response_after_plan")
    if not all([user_query, context_dict, plan_details, final_response_after_plan, agi_initial_plan_json]): return None
    context_str = format_context_for_prompt(context_dict); plan_step_outcomes = []
    for ti in entry.get("tool_interactions", []):
        if ti.get("part_of_plan_step") is not None:
            action = ti.get('action_type', 'unknown'); outcome = ti.get('tool_outcome_summary', 'No outcome.')[:150] + ("..." if len(ti.get('tool_outcome_summary', '')) > 150 else "")
            plan_step_outcomes.append(f"  - Step {ti['part_of_plan_step']+1} ({action}): {outcome}")
    plan_execution_summary_str = "PLAN_EXECUTION_SUMMARY:\n" + "\n".join(plan_step_outcomes) if plan_step_outcomes else "PLAN_EXECUTION_SUMMARY: No step outcomes."
    proposed_plan_str = f"PROPOSED_PLAN_JSON:\n{agi_initial_plan_json.strip()}"
    system_instruction = "SYSTEM_INSTRUCTION: Summarize plan execution and respond to user."
    return f"{context_str}\nUSER: {user_query}\n{proposed_plan_str}\n{plan_execution_summary_str}\n{system_instruction}\nASSISTANT: {final_response_after_plan}{eos_token}"

def format_interaction_for_training(interaction_entry: dict, tokenizer_eos_token: str) -> list[str]:
    training_examples = []; user_query = interaction_entry.get("user_query"); agi_final_response = interaction_entry.get("agi_final_response_to_user")
    if not user_query or not agi_final_response: return []
    is_plan = "plan_execution_details" in interaction_entry and interaction_entry["plan_execution_details"].get("action_type") == "execute_plan"
    tool_interactions = interaction_entry.get("tool_interactions", [])
    agi_initial_raw_is_json = interaction_entry.get("agi_initial_raw_response", "").strip().startswith("{")
    if is_plan:
        ex_plan_gen = _format_for_plan_generation(interaction_entry, tokenizer_eos_token)
        if ex_plan_gen: training_examples.append(ex_plan_gen)
        ex_plan_sum = _format_for_plan_summary(interaction_entry, tokenizer_eos_token)
        if ex_plan_sum: training_examples.append(ex_plan_sum)
    elif tool_interactions:
        if agi_initial_raw_is_json:
            ex_single_tool_gen = _format_for_single_tool_call_generation(interaction_entry, tokenizer_eos_token)
            if ex_single_tool_gen: training_examples.append(ex_single_tool_gen)
        if tool_interactions[0].get("agi_secondary_raw_response"):
            ex_tool_outcome_proc = _format_for_tool_outcome_processing(interaction_entry, tool_interactions[0], tokenizer_eos_token)
            if ex_tool_outcome_proc: training_examples.append(ex_tool_outcome_proc)
        elif not training_examples:
            ex_direct = _format_for_direct_answer(interaction_entry, tokenizer_eos_token)
            if ex_direct: training_examples.append(ex_direct)
    else:
        ex_direct = _format_for_direct_answer(interaction_entry, tokenizer_eos_token)
        if ex_direct: training_examples.append(ex_direct)
    return training_examples

def load_raw_interaction_logs(jsonl_log_path: str) -> list[dict]:
    raw_interactions = []; log_file = Path(jsonl_log_path)
    if not log_file.exists(): console.print(f"[yellow]JSONL log file not found: {jsonl_log_path}.[/yellow]"); return []
    console.print(f"[info]Processing JSONL: {jsonl_log_path}...[/info]")
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try: raw_interactions.append(json.loads(line))
                except json.JSONDecodeError as e: console.print(f"[yellow]Skipping malformed JSON line {line_num + 1}: {e}[/yellow]")
    except Exception as e: console.print(f"[error]Error reading {jsonl_log_path}: {e}[/error]"); return []
    if not raw_interactions: console.print(f"[yellow]No valid interactions in {jsonl_log_path}.[/yellow]")
    return raw_interactions

def _is_tool_call_successful(tool_call: dict) -> bool:
    user_confirmation = tool_call.get("user_confirmation", "").lower()
    if user_confirmation in ["cancelled", "denied_by_system_whitelist", "n/a_malformed_request", "denied_by_system_static_analysis"]: return False
    outcome_summary = tool_call.get("tool_outcome_summary", "").lower()
    if any(err_kw in outcome_summary for err_kw in ["error:", "failed", "malformed", "exception:"]): return False
    if any(succ_kw in outcome_summary for succ_kw in ["success", "processed", "executed", "identical", "up-to-date", "no output", "nothing to commit", "provided to agi"]): return True
    if tool_call.get("action_type") == "execute_python_code" and "exception:" not in outcome_summary: return True # Assumed success if no exception & not cancelled
    return False

def filter_raw_interactions(raw_interactions: list[dict], args: argparse.Namespace) -> list[dict]:
    if not any([args.train_on_successful_tool_use, args.train_on_failed_tool_use, args.train_on_successful_plans, args.train_on_halted_plans, args.min_tool_interactions > 0]):
        return raw_interactions
    filtered_interactions = []
    console.print("[info]Applying training data filters...")
    for entry in raw_interactions:
        passes_filter = True
        is_plan_turn = "plan_execution_details" in entry
        plan_details = entry.get("plan_execution_details", {})

        if args.train_on_successful_plans:
            if not (is_plan_turn and "Halted" not in plan_details.get("plan_outcome", "")): passes_filter = False
        if args.train_on_halted_plans and passes_filter:
            if not (is_plan_turn and "Halted" in plan_details.get("plan_outcome", "")): passes_filter = False

        num_tool_interactions = len(entry.get("tool_interactions", []))
        if args.min_tool_interactions > 0 and num_tool_interactions < args.min_tool_interactions: passes_filter = False

        if passes_filter and (args.train_on_successful_tool_use or args.train_on_failed_tool_use):
            tool_match_found = False
            for tool_call in entry.get("tool_interactions", []):
                tool_action = tool_call.get("action_type")
                is_successful = _is_tool_call_successful(tool_call)
                if args.train_on_successful_tool_use and is_successful and \
                   (args.train_on_successful_tool_use == "all" or args.train_on_successful_tool_use == tool_action):
                    tool_match_found = True; break
                if args.train_on_failed_tool_use and not is_successful and \
                   (args.train_on_failed_tool_use == "all" or args.train_on_failed_tool_use == tool_action):
                    tool_match_found = True; break
            if not tool_match_found: passes_filter = False

        if passes_filter: filtered_interactions.append(entry)
    console.print(f"[info]Retained {len(filtered_interactions)} interactions after filtering (out of {len(raw_interactions)}).")
    return filtered_interactions

def prepare_training_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> list[dict]:
    raw_interactions = load_raw_interaction_logs(args.jsonl_log_path)
    if not raw_interactions: return []
    filtered_interactions = filter_raw_interactions(raw_interactions, args)
    if not filtered_interactions: console.print("[yellow]No interactions after filtering.[/yellow]"); return []
    all_formatted_texts = []
    console.print(f"[info]Formatting {len(filtered_interactions)} filtered interactions...[/info]")
    # ... (progress bar logic as before) ...
    for entry in filtered_interactions:
        formatted_texts_for_entry = format_interaction_for_training(entry, tokenizer.eos_token)
        if formatted_texts_for_entry: all_formatted_texts.extend(formatted_texts_for_entry)
    # ... (progress bar stop, checks for min_dataset_size, tokenization as before) ...
    if not all_formatted_texts: console.print("[yellow]No examples generated from filtered logs.[/yellow]"); return []
    min_examples = 5
    if len(all_formatted_texts) < min_examples:
        console.print(f"[warning]Number of examples ({len(all_formatted_texts)}) is < {min_examples}. Training might be ineffective.[/warning]")
        if input(f"Continue with {len(all_formatted_texts)} examples? (yes/NO): ").lower() != "yes":
            console.print("[info]Training aborted.[/info]"); return []
    console.print(f"[success]Generated {len(all_formatted_texts)} training examples.[/success]")
    tokenized_dataset = []
    # ... (tokenization loop as before) ...
    for text in all_formatted_texts:
        tokenized_input = tokenizer(text, truncation=True, max_length=args.max_seq_length, padding="max_length")
        tokenized_dataset.append({"input_ids": tokenized_input["input_ids"], "attention_mask": tokenized_input["attention_mask"], "labels": tokenized_input["input_ids"].copy()})
    console.print(f"[success]Tokenized {len(tokenized_dataset)} examples.[/success]")
    return tokenized_dataset

def analyze_and_print_stats(raw_interactions: list[dict], num_examples_to_print: int, tokenizer_eos_for_examples: str, args_for_filtering: Optional[argparse.Namespace] = None):
    if not raw_interactions: console.print("[yellow]No interactions to analyze.[/yellow]"); return

    interactions_to_analyze = raw_interactions
    if args_for_filtering: # If CLI args provided, filter before analyzing (to see effect of filters)
        console.print("[info]Analyzing logs based on provided filter arguments...")
        interactions_to_analyze = filter_raw_interactions(raw_interactions, args_for_filtering)
        if not interactions_to_analyze:
            console.print("[yellow]No interactions remain after filtering for analysis. Original stats will be shown if different.[/yellow]")
            # Optionally, one could fall back to showing stats for raw_interactions or just stop.
            # For now, let's indicate that filtering resulted in zero.
            # To show original stats if filtered is empty:
            # if not interactions_to_analyze and raw_interactions is not interactions_to_analyze:
            #     console.print("[info]Showing stats for UNFILTERED logs as filtered set is empty.")
            #     interactions_to_analyze = raw_interactions # Fallback
            # else:
            #     return # If already analyzing raw and it's empty, or filtered is empty and we don't want to fallback
            if not interactions_to_analyze: return


    total_turns = len(interactions_to_analyze); turns_with_tools = 0; tool_type_counts = {}; tool_outcomes = {}
    total_user_query_len = 0; total_agi_response_len = 0; plan_turns = 0; successful_plans = 0; halted_plans = 0

    for entry in interactions_to_analyze: # Use the (potentially filtered) list
        total_user_query_len += len(entry.get("user_query", ""))
        total_agi_response_len += len(entry.get("agi_final_response_to_user", ""))
        if entry.get("plan_execution_details"):
            plan_turns +=1
            if "Halted" not in entry["plan_execution_details"].get("plan_outcome", "") and \
               "Malformed" not in entry["plan_execution_details"].get("plan_outcome", ""): # Count successful/completed
                successful_plans +=1
            elif "Halted" in entry["plan_execution_details"].get("plan_outcome", ""):
                halted_plans +=1
        tool_interactions_list = entry.get("tool_interactions", [])
        if tool_interactions_list:
            turns_with_tools += 1
            for tool_call in tool_interactions_list:
                action = tool_call.get("action_type", "unknown_action")
                tool_type_counts[action] = tool_type_counts.get(action, 0) + 1
                if action not in tool_outcomes: tool_outcomes[action] = {"success": 0, "error": 0, "cancelled": 0, "other": 0}

                # Use _is_tool_call_successful for more robust outcome checking
                if _is_tool_call_successful(tool_call):
                    tool_outcomes[action]["success"] += 1
                elif tool_call.get("user_confirmation", "").lower() == "cancelled":
                    tool_outcomes[action]["cancelled"] +=1
                elif any(err_kw in tool_call.get("tool_outcome_summary", "").lower() for err_kw in ["error:", "failed", "malformed", "exception:"]) or \
                     tool_call.get("user_confirmation", "").lower() in ["denied_by_system_whitelist", "n/a_malformed_request", "denied_by_system_static_analysis"]:
                    tool_outcomes[action]["error"] += 1
                else: tool_outcomes[action]["other"] += 1

    console.print(f"\n[bold underline]Interaction Log Analysis (Analyzed {total_turns} entries)[/bold underline]")
    # ... (rest of the stats printing as before, using the potentially filtered counts) ...
    console.print(f"Total Interaction Turns Analyzed: {total_turns}")
    console.print(f"Turns with any Tool Use: {turns_with_tools} ({turns_with_tools/total_turns:.1%} if total_turns > 0 else 0.0)%)")
    console.print(f"Turns involving a Plan: {plan_turns} ({plan_turns/total_turns:.1%} if total_turns > 0 else 0.0)%)")
    if plan_turns > 0 : console.print(f"  Successful/Completed Plans: {successful_plans}")
    if plan_turns > 0 : console.print(f"  Halted Plans: {halted_plans}")
    avg_user_query_len = total_user_query_len / total_turns if total_turns > 0 else 0
    avg_agi_response_len = total_agi_response_len / total_turns if total_turns > 0 else 0
    console.print(f"Average User Query Length: {avg_user_query_len:.0f} chars")
    console.print(f"Average AGI Final Response Length: {avg_agi_response_len:.0f} chars")
    if tool_type_counts:
        console.print("\n[bold]Tool Usage Breakdown (individual calls):[/bold]")
        # ... (tool_table printing as before) ...
        tool_table = Table(title="Tool Call Counts"); tool_table.add_column("Tool Action", style="cyan"); tool_table.add_column("Count", style="magenta", justify="right")
        for tool, count in sorted(tool_type_counts.items()): tool_table.add_row(tool, str(count))
        console.print(tool_table)
        console.print("\n[bold]Tool Outcome Summary (individual calls):[/bold]")
        outcome_table = Table(title="Tool Call Outcome Details"); outcome_table.add_column("Tool Action", style="cyan"); outcome_table.add_column("Success", style="green", justify="right"); outcome_table.add_column("Error/Reject", style="red", justify="right"); outcome_table.add_column("Cancelled", style="yellow", justify="right"); outcome_table.add_column("Other/Unknown", style="dim", justify="right")
        for tool, outcomes_map in sorted(tool_outcomes.items()): outcome_table.add_row(tool, str(outcomes_map.get("success",0)), str(outcomes_map.get("error",0)), str(outcomes_map.get("cancelled",0)), str(outcomes_map.get("other",0)))
        console.print(outcome_table)

    else: console.print("No individual tool calls found in analyzed logs.")

    if num_examples_to_print > 0 and interactions_to_analyze:
        console.print(f"\n[bold underline]Random Formatted Training Examples (N={num_examples_to_print} original log entries from analyzed set):[/bold underline]")
        actual_num_to_print = min(num_examples_to_print, len(interactions_to_analyze))
        selected_log_entries = random.sample(interactions_to_analyze, actual_num_to_print)
        example_count = 0
        for i, entry in enumerate(selected_log_entries):
            console.print(f"\n--- Log Entry {i+1} (Original Turn ID: {entry.get('turn_id', 'N/A')}) ---")
            formatted_prompts_for_entry = format_interaction_for_training(entry, tokenizer_eos_for_examples)
            if formatted_prompts_for_entry:
                for j, formatted_prompt in enumerate(formatted_prompts_for_entry):
                    example_count += 1
                    console.print(f"  -- Example {example_count} (from log entry {i+1}, sub-example {j+1}) --")
                    console.print(Text(formatted_prompt, overflow="fold"))
            else: console.print(f"  (No training examples generated for this log entry)")
        console.print("\n--- End of Examples ---")

def main():
    args = parse_arguments()
    if args.analyze_jsonl_logs is not None:
        console.print(f"[bold cyan]--- JSONL Log Analysis Mode ---[/bold cyan]")
        raw_logs = load_raw_interaction_logs(args.jsonl_log_path)
        if not raw_logs: console.print("[error]No logs loaded. Exiting.[/error]"); return
        num_examples_to_show = args.analyze_jsonl_logs if args.analyze_jsonl_logs != -1 else 5
        example_eos_token = "<|eos|>"
        if num_examples_to_show > 0:
            try:
                tokenizer_for_examples = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                if tokenizer_for_examples.eos_token: example_eos_token = tokenizer_for_examples.eos_token
            except Exception as e: console.print(f"[warning]Could not load tokenizer from '{args.model_path}' for examples: {e}.[/warning]")
        analyze_and_print_stats(raw_logs, num_examples_to_show, example_eos_token, args) # Pass args for filtering in analysis
        console.print("[info]Log analysis complete.[/info]"); return

    console.print("[bold blue]Starting Adaptive Training Script...[/bold blue]")
    # ... (rest of main training logic as before, but prepare_training_dataset now takes args) ...
    console.print(f"[info]Configuration: {args}[/info]")
    console.print(f"[info]Loading base model from: {args.model_path}[/info]")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if args.use_qlora else None
    if args.use_qlora: console.print("[info]QLoRA enabled (4-bit).[/info]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; console.print(f"[info]Set pad_token to eos_token ({tokenizer.eos_token})[/info]")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
        console.print("[success]Tokenizer and model loaded.[/success]")
        if args.use_qlora: model = prepare_model_for_kbit_training(model); console.print("[info]Prepared model for k-bit training.[/info]")
    except Exception as e: console.print(f"[error]Error loading model/tokenizer: {e}[/error]"); return

    parsed_target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()] if args.lora_target_modules else ["q_proj", "v_proj"]
    console.print(f"[info]LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}) for modules: {parsed_target_modules}[/info]")
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=parsed_target_modules, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    try:
        model = get_peft_model(model, lora_config); console.print("[success]PEFT model configured.[/success]")
        trainable_params, all_param = model.get_nb_trainable_parameters(); console.print(f"[info]Trainable LoRA params: {trainable_params} ({100 * trainable_params / all_param:.2f}% of total)[/info]")
    except Exception as e: console.print(f"[error]Error configuring PEFT: {e}[/error]"); return

    console.print("[info]Loading and preparing dataset...[/info]")
    train_dataset = prepare_training_dataset(args, tokenizer) # Pass full args object
    if not train_dataset: console.print("[info]No training data. Exiting.[/info]"); return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    console.print(f"[info]Training arguments. Output dir: {args.output_dir}[/info]")
    os.makedirs(args.output_dir, exist_ok=True)

    class RichProgressCallback(transformers.TrainerCallback):
        def __init__(self): super().__init__(); self.progress = None; self.train_task_id = None
        def on_train_begin(self, args, state, control, **kwargs):
            if RICH_AVAILABLE and state.is_world_process_zero:
                self.progress = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.1f}%"), TextColumn("Steps: {task.completed}/{task.total}"), TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=True)
                self.progress.start(); self.train_task_id = self.progress.add_task("Training", total=state.max_steps)
        def on_step_end(self, args, state, control, **kwargs):
            if RICH_AVAILABLE and state.is_world_process_zero and self.progress: self.progress.update(self.train_task_id, advance=1)
        def on_train_end(self, args, state, control, **kwargs):
            if RICH_AVAILABLE and state.is_world_process_zero and self.progress: self.progress.stop()
    callbacks = [RichProgressCallback()] if RICH_AVAILABLE else []

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, logging_steps=10, save_strategy="epoch",
        fp16=not args.use_qlora, bf16=args.use_qlora and torch.cuda.is_bf16_supported(), optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        disable_tqdm=RICH_AVAILABLE, logging_dir=f"{args.output_dir}/logs", report_to="none"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=data_collator, callbacks=callbacks)
    console.print("[bold green]Starting training...[/bold green]")
    try:
        train_result = trainer.train(); console.print(f"[success]Training completed. Metrics: {train_result.metrics}[/success]")
    except Exception as e: console.print(f"[error]Error during trainer.train(): {e}[/error]")
    console.print(f"[info]Saving PEFT adapters to {args.output_dir}...[/info]")
    try:
        trainer.save_model(args.output_dir); tokenizer.save_pretrained(args.output_dir)
        console.print(f"[success]Adapters and tokenizer saved to {args.output_dir}.[/success]")
    except Exception as e: console.print(f"[error]Error saving model/adapters: {e}[/error]")
    console.print("[bold blue]Adaptive training script finished.[/bold blue]")

if __name__ == "__main__":
    main()
