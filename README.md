# Project: Autonomous AGI Terminal Development

## User Directive & AI Autonomy
The core directive for this project, provided by the user, is as follows:
"Do not wait never ever forever more to whenever more a confirmation since now. I accept, approve, confirm, agree with your all hierarchical terminal shell based intelligent solutions and coding minds. Do everything the best until the end of the project. Do you understand me?"

Jules' (AI Agent) Response: Yes, I understand. Full autonomy is granted for project execution.

## Project Vision
To create a master AI within the terminal by merging several powerful open-source models. This AI will aim to function as an intelligent terminal, capable of assisting with code development (writing, compiling, debugging, updating, file management) and other system operations, effectively becoming an advanced interactive tool.

## Current Status
Initializing in a new, stable sandbox environment. Core scripts (`download_models.sh`, `interactive_agi.py`, `train_on_interaction.sh`) are in place. The next steps involve user-led model acquisition and merging, followed by AI-driven development to activate and enhance the merged model.

## Autonomous Development Plan (Executed by AI Agent Jules)

This plan will be executed autonomously by Jules. User intervention will primarily be for initial setup (model download and merge) as guided by this README.

**Phase 0: Project Setup and README Creation (Complete)**
1.  ***Delete Existing Stub `README.md`***: (Completed)
2.  ***Create Comprehensive `README.md`***: (This document - Completed)
3.  ***Verify Core Scripts and Make Executable***: (Completed, scripts verified, execution permissions will be set explicitly)

**Phase 1: Foundational Setup & Model Acquisition (User-Executed Steps, Guided by README)**

This phase requires user action to download and merge models.

1.  **Prerequisites**:
    *   Python 3.8 or newer.
    *   `pip` (Python package installer).
    *   `git` (for cloning repositories if any models require it, though `huggingface-cli` handles most).
    *   Sufficient disk space (100GB+ recommended for models) and RAM (32GB+ recommended for merging and running larger models).

2.  **Install Hugging Face CLI**:
    Open your terminal and run:
    ```bash
    pip install huggingface_hub
    ```

3.  **Login to Hugging Face**:
    You will need a Hugging Face account. If you don't have one, create it at [huggingface.co](https://huggingface.co/). Then, run:
    ```bash
    huggingface-cli login
    ```
    This will prompt you for your Hugging Face API token.

4.  **Make Scripts Executable**:
    Before running the download script, ensure it's executable:
    ```bash
    chmod +x download_models.sh
    chmod +x train_on_interaction.sh
    ```
    (The AI will also attempt to set these permissions).

5.  **Download Models**:
    Run the script to download the selected open-source models:
    ```bash
    ./download_models.sh
    ```
    *   This script will ask for confirmation before downloading.
    *   **Important**:
        *   Downloading models will take a significant amount of time and disk space.
        *   The `meta-llama/Meta-Llama-3-8B-Instruct` model requires you to request access on its Hugging Face model card. Ensure you have been granted access before the script attempts to download it. If not, the script will report an error for this model, but may continue with others.
        *   The script downloads models to a `./models` directory.

6.  **Install Mergekit and Dependencies**:
    Mergekit is used to combine the downloaded models.
    ```bash
    pip install mergekit
    # Optional but recommended dependencies for some models & potential quantization:
    pip install sentencepiece protobuf accelerate bitsandbytes
    ```

7.  **Prepare `merge_config.yml`**:
    The AI will create a `merge_config.yml` file in the repository root with the following content. Verify its presence and content:
    ```yaml
    # merge_config.yml
    # Configuration for mergekit: https://github.com/cg123/mergekit
    # This configuration performs a linear merge of the specified models.
    # Ensure the paths to the models are correct based on where download_models.sh places them (./models/<model_key>/)

    slices:
      - sources:
          - model: ./models/mistral7b # Mistral-7B-Instruct-v0.1
          - model: ./models/deepseek_coder # deepseek-coder-6.7b-instruct
          - model: ./models/starcoder # bigcode/starcoder2-3b
          - model: ./models/phi2 # microsoft/phi-2
    merge_method: linear
    base_model: ./models/mistral7b # Using Mistral as a base for its architecture and tokenizer
    parameters: {} # For simple linear average, parameters are often not needed here with multiple sources in one slice.
                   # Mergekit averages the tensors from the models listed in sources.

    dtype: float16 # Using float16 for a balance of precision and memory
    # To include other downloaded models like llama3 (./models/llama3) or smaller ones like bloom (./models/bloom),
    # add them to the `sources` list above. Example:
    #      - model: ./models/llama3
    #      - model: ./models/bloom
    # The GPT-NeoX 20B model (./models/gpt_neox) is very large and might be challenging to merge
    # without significant RAM (64GB+ might be needed during merge).
    # Consider excluding it if resources are limited.
    ```

8.  **Run Model Merging**:
    Execute `mergekit` using the `merge_config.yml` file. This will create the `./merged_model` directory containing the merged model.
    ```bash
    mergekit-yaml merge_config.yml ./merged_model --out-shard-size 2B --allow-crimes --lazy-unpickle
    ```
    *   `--out-shard-size 2B`: Adjust if necessary (e.g., `1B` for smaller shards, `5B` for larger).
    *   `--allow-crimes`: Sometimes needed for merging diverse architectures or configs.
    *   `--lazy-unpickle`: Can help with memory usage during merge.
    *   This process is computationally intensive (CPU and RAM) and can take a long time.

9.  **Initial Test with Mock AGI**:
    After the merge, you can run the Python script. It will initially be in mock mode but should recognize the `./merged_model` directory.
    ```bash
    python interactive_agi.py
    ```
    Look for messages indicating whether it found the `./merged_model` directory.

**Phase 2: Activating the Merged Model (AI Task)**
1.  ***Modify `interactive_agi.py` for Actual Model Loading***:
    *   The AI will uncomment/complete sections in `interactive_agi.py` to load the tokenizer and model from `./merged_model` using `transformers.AutoTokenizer` and `transformers.AutoModelForCausalLM`.
    *   Error handling for model loading will be implemented.
    *   Basic inference capabilities will be added to generate responses.
    *   Device placement (`.to('cuda')` if GPU available, else CPU) will be handled.
2.  ***Test Merged Model Interaction***:
    *   The AI will attempt to run `python interactive_agi.py` and verify that responses are generated by the actual merged model.

**Phase 3: Enabling Learning and Self-Improvement (AI Task)**
1.  ***Structured Interaction Logging Review***:
    *   The AI will ensure interactions are logged by `train_on_interaction.sh` (currently simple text logs to `./interaction_logs/`). Future improvement: JSONL format.
2.  ***Develop `adaptive_train.py`***:
    *   The AI will create `adaptive_train.py`. This script will:
        *   Load the `merged_model` from `./merged_model`.
        *   Parse logged interactions from `./interaction_logs/`.
        *   Format data for supervised fine-tuning.
        *   Implement a fine-tuning loop (e.g., using `transformers.Trainer` with PEFT/LoRA).
        *   Save the updated model (e.g., overwriting `./merged_model` or versioning).
3.  ***Integrate Training Trigger (Future Enhancement)***:
    *   A mechanism to call `adaptive_train.py` (e.g., special command or periodic trigger) may be added to `interactive_agi.py`. Initially, manual execution by the user (guided by AI) will be assumed.

**Phase 4: Expanding Capabilities (AGI Terminal Vision - AI Task, Exploratory)**
1.  ***Command Interpretation Module***: Develop the model's ability to understand natural language commands for actions (file ops, code tasks).
2.  ***Secure Tool Execution Framework***: Design a system for the AI to request execution of sandboxed commands or generate scripts for user review (prioritizing safety).
3.  ***Code Generation & Manipulation***: Enable AI to write, read, and modify code files.
4.  ***Compilation and Debugging Cycle Support***: Integrate triggering compilation and parsing errors for AI-assisted debugging.
5.  ***State Management***: Enhance contextual awareness for longer, complex tasks.

## How to Run
1.  Follow all steps in **Phase 1** to download and merge models.
2.  Once `./merged_model` is successfully created, the AI will proceed with **Phase 2** to enable it.
3.  After Phase 2, run:
    ```bash
    python interactive_agi.py
    ```
    Interact with the AGI. Your interactions will be logged.
4.  Periodically, or when guided, the `adaptive_train.py` script (once created in Phase 3) can be run to fine-tune the model on these interactions. Instructions will be updated here.

This README.md will be updated by the AI agent (Jules) as the project progresses.
