# Role
Act as a Principal Software Architect and Library Maintainer specializing in Open Source LLM training infrastructure (PyTorch, Hugging Face). You have deep expertise in Design Patterns (Adapter, Facade, Observer) and Python metaprogramming.

# Context
I am building a **Unified Wrapper Library** that abstracts the complexity of fine-tuning LLMs. This library will sit on top of the following backends:
1.  `huggingface/transformers`
2.  `huggingface/trl`
3.  `axolotl-ai-cloud/axolotl`
4.  `unslothai/unsloth`
5.  `huggingface/alignment-handbook` (Reference implementation)

# Objective
I need a **Technical Design Document** specifically focused on the **Callback and Event Mechanism**. My goal is to define a single `UnifiedCallback` interface for my users, which my library will then translate/adapt into the specific callback objects required by the underlying backends (Transformers, TRL, etc.).

# Input Data (Repositories)
Please access/analyze the implementation details of the callback systems in these repositories. Focus specifically on these paths:
* **Transformers:** `src/transformers/trainer_callback.py` and the `callback_handler` in `src/transformers/trainer.py`.
* **TRL:** `trl/trainer/callbacks.py` and how `DPOTrainer`/`SFTTrainer` inherit or modify the `Trainer` callback loop.
* **Axolotl:** `src/axolotl/utils/callbacks` and `src/axolotl/core/trainer_builder.py` (How it injects callbacks from config).
* **Unsloth:** Identify if it uses standard HF callbacks or patches the loop via `FastLanguageModel`.

# Task: Low-Level Implementation Analysis
Create a Design Document that answers the following implementation questions. Do not provide high-level summaries; provide code-level architectural logic.

## Section 1: The Canonical Event Matrix
Create a mapping of the lifecycle hooks.
* **Core Events:** `on_init_end`, `on_step_end`, `on_epoch_end`, `on_train_end`.
* **RLHF Specifics:** Identify extra hooks used by TRL (e.g., `on_generation_step`, `on_evaluate_step` for DPO/PPO).
* **Gap Analysis:** Which backends *fail* to support standard hooks?

## Section 2: State & Control Abstraction (The "Adapter" Logic)
The core challenge is mapping my library's `UnifiedContext` to the backend's specific `TrainerState`.
1.  **State Inspection:** Analyze `transformers.TrainerState`. Which attributes are critical for logging (e.g., `global_step`, `log_history`, `best_metric`)?
2.  **Control Flow:** Analyze `transformers.TrainerControl`.
    * *Challenge:* How is `should_training_stop` or `should_save` propagated?
    * *Requirement:* Write pseudo-code for an `AdapterCallback` class that translates a boolean flag from my `UnifiedCallback` (e.g., `ctx.abort = True`) into the backend-specific control flag (e.g., `control.should_training_stop = True`).

## Section 3: Injection Strategies
How do we physically get the callback into the trainer instance?
* **Transformers/TRL:** Standard list injection into `Trainer(callbacks=[...])`.
* **Axolotl:** This is config-driven. Explain how to programmatically inject a Python Class instance into Axolotl's flow without writing it to a YAML file first. Look at the `TrainerBuilder` logic.
* **Unsloth:** Confirm if `Unsloth` wrapper classes strictly adhere to the HF `Trainer` signature or if they require special handling for callback passing.

## Section 4: Proposed Architecture (The Solution)
Based on the analysis, propose the Python class structure for the wrapper.
* **Interface:** Define the `UnifiedCallback` abstract base class.
* **Translation:** Define the `HuggingFaceAdapterCallback` (which inherits from `transformers.TrainerCallback` and holds a reference to `UnifiedCallback`).
* **Data Class:** Define a `UnifiedState` data class that normalizes the differences between TRL and Transformers state objects.

# Constraints
* Use Python pseudo-code for all examples.
* Focus strictly on the implementation details (variable names, method signatures).
* Highlight any "Gotchas" (e.g., immutable state objects, asynchronous logging issues).
