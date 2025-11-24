# Technical Design Document: Unified Callback and Event Mechanism

**Version:** 1.0
**Date:** 2025-01-24
**Author:** Principal Software Architect
**Status:** Design Specification

---

## Executive Summary

This document defines the technical architecture for a **Unified Callback System** that abstracts callback mechanisms across multiple LLM training backends (HuggingFace Transformers, TRL, Axolotl, Unsloth). The design employs the **Adapter Pattern** to translate a single `UnifiedCallback` interface into backend-specific implementations, enabling users to write training callbacks once and deploy across any supported backend.

**Key Finding:** All analyzed backends use the **same core event set** from HuggingFace Transformers. There are NO custom RLHF-specific events in TRL. The primary challenges are:
1. State object normalization (TrainerState variations)
2. Control flow translation (boolean flags to TrainerControl)
3. Backend-specific injection mechanisms (especially Axolotl's two-phase system)

---

## Table of Contents

1. [Section 1: Canonical Event Matrix](#section-1-canonical-event-matrix)
2. [Section 2: State & Control Abstraction](#section-2-state--control-abstraction)
3. [Section 3: Injection Strategies](#section-3-injection-strategies)
4. [Section 4: Proposed Architecture](#section-4-proposed-architecture)
5. [Section 5: Implementation Roadmap](#section-5-implementation-roadmap)
6. [Appendix A: Backend Analysis Summary](#appendix-a-backend-analysis-summary)

---

## Section 1: Canonical Event Matrix

### 1.1 Core Event Lifecycle

The following table maps all lifecycle hooks across backends. **Source:** `transformers/trainer_callback.py`

| Event Hook | Signature | Timing | Supported Backends | Notes |
|-----------|-----------|--------|-------------------|-------|
| `on_init_end` | `(args, state, control, **kwargs)` | After Trainer.__init__ completes | ALL | kwargs includes: `model`, `tokenizer`, `train_dataloader`, `eval_dataloader`, `optimizer`, `lr_scheduler` |
| `on_train_begin` | `(args, state, control, **kwargs)` | Before first training step | ALL | State initialized with max_steps, num_train_epochs |
| `on_train_end` | `(args, state, control, **kwargs)` | After final training step | ALL | State contains final metrics |
| `on_epoch_begin` | `(args, state, control, **kwargs)` | Start of each epoch | ALL | state.epoch is current epoch (float) |
| `on_epoch_end` | `(args, state, control, **kwargs)` | End of each epoch | ALL | Fires even if epoch interrupted |
| `on_step_begin` | `(args, state, control, **kwargs)` | Before optimizer step | ALL | After gradient accumulation completes |
| `on_step_end` | `(args, state, control, **kwargs)` | After optimizer step | ALL | state.global_step incremented |
| `on_substep_end` | `(args, state, control, **kwargs)` | After each forward/backward pass | ALL | During gradient accumulation (before optimizer step) |
| `on_pre_optimizer_step` | `(args, state, control, **kwargs)` | After gradient clipping, before optimizer | ALL | Added in Transformers v4.52+ |
| `on_optimizer_step` | `(args, state, control, **kwargs)` | After optimizer.step(), before zero_grad() | ALL | Gradients still available |
| `on_log` | `(args, state, control, logs, **kwargs)` | When metrics logged | ALL | `logs` dict contains current metrics |
| `on_save` | `(args, state, control, **kwargs)` | Before checkpoint saved | ALL | kwargs includes `metrics` |
| `on_evaluate` | `(args, state, control, metrics, **kwargs)` | After evaluation completes | ALL | `metrics` dict contains eval results |
| `on_predict` | `(args, state, control, metrics, **kwargs)` | After prediction completes | ALL | `metrics` dict contains prediction results |
| `on_prediction_step` | `(args, state, control, **kwargs)` | Before each prediction batch | ALL | kwargs includes `inputs` |

**Total Events:** 15 standard hooks

### 1.2 RLHF-Specific Events Analysis

**CRITICAL FINDING:** TRL does **NOT** introduce custom callback events.

**Investigation Results:**
- **Searched for:** `on_generation_step`, `on_reward_step`, `on_evaluate_step` (DPO/PPO specific)
- **Result:** No such events exist in TRL codebase
- **Explanation:** TRL implements RLHF-specific logic via:
  1. Custom callback classes (e.g., `SyncRefModelCallback`, `WinRateCallback`)
  2. Standard events (`on_step_end`, `on_evaluate`, `on_train_begin`)
  3. Trainer-specific state (e.g., `ref_model` stored in trainer, not in state)

**TRL Custom Callbacks** (use standard events):
| Callback | Primary Event | Purpose |
|----------|---------------|---------|
| `SyncRefModelCallback` | `on_step_end` | Synchronize reference model with training model |
| `WinRateCallback` | `on_evaluate` | LLM-as-judge win rate evaluation |
| `LogCompletionsCallback` | `on_step_end`, `on_evaluate` | Log model generations to W&B/MLflow |
| `WeaveCallback` | `on_train_begin`, `on_evaluate` | W&B Weave integration |

### 1.3 Gap Analysis

**Backend Compatibility Matrix:**

| Backend | Transformers Events (15) | Custom Events | Deviations |
|---------|-------------------------|---------------|------------|
| **Transformers** | ✅ All 15 | None | Reference implementation |
| **TRL** | ✅ All 15 | None | 100% compatible |
| **Axolotl** | ✅ All 15 | None | Uses factory pattern for trainer-dependent callbacks |
| **Unsloth** | ✅ All 15 | None | 100% compatible (thin wrapper) |

**Conclusion:** ✅ **NO GAPS** - All backends support the full Transformers event set.

### 1.4 Event Timing Diagram

```
Trainer.__init__()
    └─> on_init_end

Trainer.train()
    └─> on_train_begin

        FOR each epoch:
            └─> on_epoch_begin

                FOR each batch (gradient accumulation):
                    └─> on_substep_end (forward + backward)

                └─> on_step_begin (after accumulation)
                └─> on_pre_optimizer_step (after grad clipping)
                └─> optimizer.step()
                └─> on_optimizer_step (before zero_grad)
                └─> on_step_end

                IF global_step % logging_steps == 0:
                    └─> on_log

                IF global_step % eval_steps == 0:
                    └─> evaluation_loop()
                        └─> on_prediction_step (per batch)
                        └─> on_evaluate

                IF global_step % save_steps == 0:
                    └─> on_save

            └─> on_epoch_end

        └─> on_train_end
```

---

## Section 2: State & Control Abstraction

### 2.1 TrainerState Analysis

**Source:** `transformers/trainer_callback.py` lines 37-162

#### Complete Attribute List (20 fields):

```python
@dataclass
class TrainerState:
    # Training Progress
    epoch: Optional[float] = None              # Current epoch (can be fractional)
    global_step: int = 0                       # Total optimizer steps across all epochs
    max_steps: int = 0                         # Total steps to train (computed from epochs * steps_per_epoch)

    # Interval Configuration
    logging_steps: int = 500                   # Steps between log events
    eval_steps: int = 500                      # Steps between evaluations
    save_steps: int = 500                      # Steps between checkpoints

    # Metrics and Logging
    log_history: List[Dict[str, float]] = field(default_factory=list)  # All logged metrics
    best_metric: Optional[float] = None        # Best metric value seen
    best_model_checkpoint: Optional[str] = None  # Path to best checkpoint

    # Resource Tracking
    num_input_tokens_seen: int = 0             # Total tokens processed
    total_flos: float = 0                      # Floating point operations (for cost estimation)

    # Distributed Training
    is_local_process_zero: bool = True         # Is this the local rank 0 process?
    is_world_process_zero: bool = True         # Is this the global rank 0 process?

    # State Management
    stateful_callbacks: Dict[str, dict] = field(default_factory=dict)  # Callback state persistence

    # Hyperparameter Search
    trial_name: Optional[str] = None           # HPO trial identifier
    trial_params: Optional[Dict[str, Any]] = None  # Current trial hyperparameters

    # Training Control
    is_hyper_param_search: bool = False        # Is this part of HPO?

    # Legacy (kept for backwards compatibility)
    num_train_epochs: int = 0                  # Total epochs (deprecated, use max_steps)
    train_batch_size: int = 0                  # Effective batch size (deprecated)
```

#### Critical Attributes for Logging:

| Attribute | Type | Use Case | Availability |
|-----------|------|----------|--------------|
| `global_step` | int | Step counter for metrics | Always |
| `epoch` | float | Epoch counter for metrics | Always (can be fractional) |
| `log_history` | List[Dict] | All historical metrics | Always (appended to) |
| `best_metric` | float | Early stopping, model selection | After first evaluation |
| `best_model_checkpoint` | str | Resume from best | After first checkpoint |
| `num_input_tokens_seen` | int | Token-based tracking | Transformers 4.30+ |
| `total_flos` | float | Cost estimation | When model.num_parameters() available |

#### State Immutability Contract:

**GOTCHA #1:** While `TrainerState` is a mutable dataclass, callbacks should **treat it as read-only**.

```python
# ❌ WRONG - Modifying state directly
def on_step_end(self, args, state, control, **kwargs):
    state.global_step += 100  # DON'T DO THIS
    return control

# ✅ CORRECT - Read state, write to control
def on_step_end(self, args, state, control, **kwargs):
    if state.global_step > 1000:
        control.should_training_stop = True
    return control
```

**Why?** The trainer may cache or copy state objects. Direct mutations won't propagate correctly.

### 2.2 TrainerControl Analysis

**Source:** `transformers/trainer_callback.py` lines 165-237

#### Complete Control Flags (5 fields):

```python
@dataclass
class TrainerControl:
    # Persistent Flags (don't auto-reset)
    should_training_stop: bool = False         # Permanently halt training
    should_epoch_stop: bool = False            # Stop current epoch (resets next epoch)

    # Transient Flags (auto-reset each step)
    should_save: bool = False                  # Trigger checkpoint save (resets after save)
    should_evaluate: bool = False              # Trigger evaluation (resets after eval)
    should_log: bool = False                   # Trigger logging (resets after log)
```

#### Control Flag Lifecycle:

| Flag | Checked At | Auto-Reset Point | Effect |
|------|-----------|------------------|--------|
| `should_training_stop` | End of each step | Never (persistent) | Break training loop immediately |
| `should_epoch_stop` | End of each step | Start of next epoch | Break epoch loop, proceed to next epoch |
| `should_save` | After step_end callbacks | After checkpoint saved | Save checkpoint regardless of save_steps |
| `should_evaluate` | After step_end callbacks | After evaluation completes | Run evaluation regardless of eval_steps |
| `should_log` | After step_end callbacks | After logging completes | Log metrics regardless of logging_steps |

#### Control Flow Propagation:

```python
# From transformers/trainer.py (simplified)
def train(self):
    control = self.callback_handler.on_train_begin(args, state, control)

    for epoch in range(num_epochs):
        if control.should_training_stop:
            break

        control = self.callback_handler.on_epoch_begin(args, state, control)

        for step, inputs in enumerate(dataloader):
            # ... forward/backward ...

            control = self.callback_handler.on_step_end(args, state, control)

            # Check persistent flags
            if control.should_training_stop or control.should_epoch_stop:
                break

            # Check transient flags
            if control.should_log or (step % logging_steps == 0):
                self.log(state.log_history[-1])
                control.should_log = False  # Auto-reset

            if control.should_evaluate or (step % eval_steps == 0):
                self.evaluate()
                control.should_evaluate = False  # Auto-reset

            if control.should_save or (step % save_steps == 0):
                self.save_checkpoint()
                control.should_save = False  # Auto-reset

        control.should_epoch_stop = False  # Reset for next epoch

    control = self.callback_handler.on_train_end(args, state, control)
```

**GOTCHA #2:** Multiple callbacks can modify the same control object sequentially.

```python
# Callback execution order matters!
callbacks = [CallbackA(), CallbackB(), CallbackC()]

# Execution sequence:
control = TrainerControl()
for callback in callbacks:
    control = callback.on_step_end(args, state, control)  # Each sees previous modifications
```

**Best Practice:** Always return the control object, even if you didn't modify it:

```python
# ✅ CORRECT - Always return control
def on_step_end(self, args, state, control, **kwargs):
    if state.global_step == 100:
        control.should_save = True
    return control  # Even if not modified

# ⚠️ ACCEPTABLE (per docs) - Return None if not modified
def on_step_end(self, args, state, control, **kwargs):
    if state.global_step == 100:
        control.should_save = True
        return control
    # Implicit return None (control unchanged)
```

### 2.3 Adapter Translation Logic (Pseudo-Code)

#### Challenge: Mapping `UnifiedContext` ↔ `TrainerState` + `TrainerControl`

**Design Goal:** Users write:
```python
def on_step_end(self, ctx: UnifiedContext):
    if ctx.global_step > 1000:
        ctx.abort_training()
```

**Backend sees:**
```python
def on_step_end(self, args, state, control, **kwargs):
    # ... adapter magic ...
    control.should_training_stop = True
    return control
```

#### Adapter Implementation:

```python
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


# ========================================
# UNIFIED CONTEXT (User-Facing)
# ========================================

@dataclass
class UnifiedContext:
    """
    Normalized training context that abstracts backend differences.
    Users interact only with this object.
    """
    # Training Progress
    global_step: int
    epoch: float
    max_steps: int

    # Metrics
    current_metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: list[Dict[str, float]] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_checkpoint_path: Optional[str] = None

    # Resource Tracking
    tokens_seen: int = 0
    total_flops: float = 0.0

    # Distributed Training
    is_main_process: bool = True
    is_local_main_process: bool = True

    # Model References (populated by adapter)
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    optimizer: Optional[Any] = None

    # Control Flags (user-facing)
    _should_stop_training: bool = False
    _should_stop_epoch: bool = False
    _should_save_now: bool = False
    _should_evaluate_now: bool = False
    _should_log_now: bool = False

    # Backend-Specific (read-only for users)
    backend_name: str = "unknown"  # "transformers", "trl", "axolotl", "unsloth"
    backend_args: Optional[Any] = None  # Original TrainingArguments

    # User Methods
    def abort_training(self):
        """Stop training immediately."""
        self._should_stop_training = True

    def stop_current_epoch(self):
        """Finish current epoch and stop."""
        self._should_stop_epoch = True

    def trigger_checkpoint(self):
        """Save checkpoint at end of current step."""
        self._should_save_now = True

    def trigger_evaluation(self):
        """Run evaluation at end of current step."""
        self._should_evaluate_now = True

    def trigger_logging(self):
        """Log metrics at end of current step."""
        self._should_log_now = True

    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get current metric value."""
        return self.current_metrics.get(key, default)

    def is_evaluation_step(self) -> bool:
        """Check if current step is scheduled for evaluation."""
        return self._should_evaluate_now

    def is_logging_step(self) -> bool:
        """Check if current step is scheduled for logging."""
        return self._should_log_now


# ========================================
# ADAPTER CALLBACK (Backend-Facing)
# ========================================

class HuggingFaceAdapterCallback(TrainerCallback):
    """
    Adapter that translates between UnifiedCallback and HuggingFace TrainerCallback.

    This class:
    1. Converts TrainerState → UnifiedContext (before user callback)
    2. Invokes user's UnifiedCallback methods
    3. Converts UnifiedContext → TrainerControl (after user callback)
    """

    def __init__(self, unified_callback: 'UnifiedCallback'):
        """
        Args:
            unified_callback: User-defined callback implementing UnifiedCallback interface
        """
        self.unified_callback = unified_callback
        self._context_cache: Optional[UnifiedContext] = None

    def _state_to_context(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> UnifiedContext:
        """
        Translate HuggingFace objects to UnifiedContext.

        GOTCHA: This creates a NEW context object each time. If you need
        stateful behavior across events, store state in your UnifiedCallback.
        """
        ctx = UnifiedContext(
            # Core progress
            global_step=state.global_step,
            epoch=state.epoch or 0.0,
            max_steps=state.max_steps,

            # Metrics
            current_metrics=state.log_history[-1] if state.log_history else {},
            metric_history=state.log_history.copy(),
            best_metric=state.best_metric,
            best_checkpoint_path=state.best_model_checkpoint,

            # Resources
            tokens_seen=state.num_input_tokens_seen,
            total_flops=state.total_flos,

            # Distributed
            is_main_process=state.is_world_process_zero,
            is_local_main_process=state.is_local_process_zero,

            # References from kwargs
            model=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
            optimizer=kwargs.get("optimizer"),

            # Backend info
            backend_name="transformers",
            backend_args=args,
        )

        return ctx

    def _context_to_control(
        self,
        ctx: UnifiedContext,
        control: TrainerControl
    ) -> TrainerControl:
        """
        Apply UnifiedContext control flags to TrainerControl.

        IMPORTANT: This modifies the control object in-place and returns it.
        This allows multiple callbacks to compose their control decisions.
        """
        if ctx._should_stop_training:
            control.should_training_stop = True

        if ctx._should_stop_epoch:
            control.should_epoch_stop = True

        if ctx._should_save_now:
            control.should_save = True

        if ctx._should_evaluate_now:
            control.should_evaluate = True

        if ctx._should_log_now:
            control.should_log = True

        return control

    # ========================================
    # Event Forwarding (15 methods)
    # ========================================

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_init_end(ctx)
        return self._context_to_control(ctx, control)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_train_begin(ctx)
        return self._context_to_control(ctx, control)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_train_end(ctx)
        return self._context_to_control(ctx, control)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_epoch_begin(ctx)
        return self._context_to_control(ctx, control)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_epoch_end(ctx)
        return self._context_to_control(ctx, control)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_step_begin(ctx)
        return self._context_to_control(ctx, control)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_step_end(ctx)
        return self._context_to_control(ctx, control)

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_substep_end(ctx)
        return self._context_to_control(ctx, control)

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_pre_optimizer_step(ctx)
        return self._context_to_control(ctx, control)

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_optimizer_step(ctx)
        return self._context_to_control(ctx, control)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        if logs:
            ctx.current_metrics = logs
        self.unified_callback.on_log(ctx, logs or {})
        return self._context_to_control(ctx, control)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        metrics = kwargs.get("metrics", {})
        self.unified_callback.on_save(ctx, metrics)
        return self._context_to_control(ctx, control)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_evaluate(ctx, metrics or {})
        return self._context_to_control(ctx, control)

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        self.unified_callback.on_predict(ctx, metrics or {})
        return self._context_to_control(ctx, control)

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        ctx = self._state_to_context(args, state, control, **kwargs)
        inputs = kwargs.get("inputs", {})
        self.unified_callback.on_prediction_step(ctx, inputs)
        return self._context_to_control(ctx, control)


# ========================================
# EXAMPLE USAGE
# ========================================

class MyCustomCallback(UnifiedCallback):
    """User-defined callback using unified interface."""

    def on_step_end(self, ctx: UnifiedContext):
        # Early stopping based on step count
        if ctx.global_step > 1000:
            print(f"Stopping at step {ctx.global_step}")
            ctx.abort_training()

        # Trigger checkpoint every 100 steps
        if ctx.global_step % 100 == 0:
            ctx.trigger_checkpoint()

    def on_evaluate(self, ctx: UnifiedContext, metrics: Dict[str, float]):
        # Early stopping based on metric
        eval_loss = metrics.get("eval_loss", float('inf'))
        if eval_loss < 0.1:
            print(f"Target loss reached: {eval_loss}")
            ctx.abort_training()


# Usage with HuggingFace Trainer
from transformers import Trainer

user_callback = MyCustomCallback()
adapter = HuggingFaceAdapterCallback(user_callback)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[adapter],  # Pass adapter, not user callback
)

trainer.train()
```

### 2.4 State Normalization Gotchas

**GOTCHA #3:** `TrainerState.log_history` structure varies by event

```python
# During on_log
state.log_history[-1] == {
    "loss": 0.5,
    "learning_rate": 2e-5,
    "epoch": 1.0,
    "step": 100,
}

# During on_evaluate
state.log_history[-1] == {
    "eval_loss": 0.3,
    "eval_accuracy": 0.85,
    "epoch": 1.0,
    "step": 100,
}

# Solution: Check keys before accessing
def on_log(self, ctx, metrics):
    loss = metrics.get("loss") or metrics.get("eval_loss")
    if loss is not None:
        print(f"Loss: {loss}")
```

**GOTCHA #4:** `best_metric` is None until first evaluation

```python
# ❌ WRONG
def on_step_end(self, ctx):
    if ctx.best_metric < 0.5:  # Crashes if None!
        ctx.trigger_checkpoint()

# ✅ CORRECT
def on_step_end(self, ctx):
    if ctx.best_metric is not None and ctx.best_metric < 0.5:
        ctx.trigger_checkpoint()
```

**GOTCHA #5:** Step vs. Substep confusion with gradient accumulation

```python
# With gradient_accumulation_steps=4:
# - on_substep_end fires 4 times per optimizer step
# - on_step_end fires 1 time per optimizer step
# - global_step increments only in on_step_end

# If you want per-batch metrics (not per-optimizer-step):
def on_substep_end(self, ctx):
    print(f"Processed batch (substep), global_step still {ctx.global_step}")

def on_step_end(self, ctx):
    print(f"Optimizer step complete, global_step now {ctx.global_step}")
```

---

## Section 3: Injection Strategies

### 3.1 Transformers/TRL/Unsloth (Standard Injection)

**Mechanism:** Direct constructor parameter

```python
from transformers import Trainer, TrainingArguments

# Standard HuggingFace injection
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[
        adapter_callback_1,
        adapter_callback_2,
    ],
)
```

**Characteristics:**
- ✅ Simple, documented API
- ✅ Works identically across Transformers/TRL/Unsloth
- ✅ Callbacks passed as list
- ✅ Can add more via `trainer.add_callback()` after construction

**Example with TRL:**
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=dataset,
    callbacks=[HuggingFaceAdapterCallback(my_callback)],  # Same API!
)
```

**Example with Unsloth:**
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(...)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    callbacks=[HuggingFaceAdapterCallback(my_callback)],  # Same API!
)
```

### 3.2 Axolotl (Two-Phase Injection)

**Challenge:** Axolotl uses a config-driven `TrainerBuilder` pattern with two callback injection phases.

**Phase 1:** Pre-Trainer Callbacks (before `Trainer.__init__`)
**Phase 2:** Post-Trainer Callbacks (after `Trainer.__init__`, requires trainer reference)

#### 3.2.1 Config-Driven Injection (Standard Axolotl Usage)

**YAML Config:**
```yaml
# axolotl_config.yaml
base_model: meta-llama/Llama-2-7b-hf

# Built-in callbacks (registered automatically)
use_wandb: true                    # Triggers SaveAxolotlConfigtoWandBCallback
gc_steps: 100                      # Triggers GCCallback
save_first_step: true              # Triggers SaveModelOnFirstStepCallback
include_tkps: true                 # Triggers TokensPerSecondCallback

# LISA
lisa_n_layers: 8
lisa_step_interval: 20

# Dynamic checkpointing
dynamic_checkpoint:
  enabled: true
  check_interval: 100

# Evaluation
do_bench_eval: true                # Triggers BenchEvalCallback (post-trainer)
eval_table_size: 10                # Triggers LogPredictionCallback (post-trainer)
```

**Python Usage:**
```python
from axolotl.cli.train import do_train

# Callbacks auto-injected based on config
do_train("axolotl_config.yaml")
```

#### 3.2.2 Programmatic Injection (Plugin System)

**Recommended Approach:** Use Axolotl's plugin system

**Step 1: Create Plugin**

```python
# File: my_project/axolotl_plugin.py
from axolotl.integrations.base import BasePlugin
from typing import List, Callable

class UnifiedCallbackPlugin(BasePlugin):
    """
    Plugin to inject UnifiedCallback into Axolotl training.
    """

    def __init__(self, unified_callback):
        self.unified_callback = unified_callback

    def add_callbacks_pre_trainer(self, cfg, model) -> List[Callable]:
        """
        Called before Trainer creation.

        Args:
            cfg: Axolotl config object
            model: Pre-initialized model

        Returns:
            List of TrainerCallback instances
        """
        from your_library.adapters import HuggingFaceAdapterCallback

        adapter = HuggingFaceAdapterCallback(self.unified_callback)
        return [adapter]

    def add_callbacks_post_trainer(self, cfg, trainer) -> List[Callable]:
        """
        Called after Trainer creation.
        Use this if your callback needs access to the trainer instance.

        Args:
            cfg: Axolotl config object
            trainer: Initialized Trainer instance

        Returns:
            List of TrainerCallback instances
        """
        # For callbacks that need trainer access:
        # return [TrainerDependentCallback(trainer)]
        return []
```

**Step 2: Register Plugin Programmatically**

```python
from axolotl.cli import load_cfg
from axolotl.core.builders import causal
from axolotl.integrations.base import PluginManager
from my_project.axolotl_plugin import UnifiedCallbackPlugin

# Load config
cfg = load_cfg("axolotl_config.yaml")

# Create and register plugin
my_callback = MyUnifiedCallback()
plugin = UnifiedCallbackPlugin(my_callback)

plugin_manager = PluginManager.get_instance()
plugin_manager.register_plugin(plugin)

# Build and train (plugin callbacks auto-injected)
trainer_builder = causal.HFCausalTrainerBuilder(cfg, ...)
trainer = trainer_builder.build()
trainer.train()
```

**Step 3: Register Plugin via Config** (Alternative)

```yaml
# axolotl_config.yaml
plugins:
  - "my_project.axolotl_plugin.UnifiedCallbackPlugin"
```

```python
# Plugin must have zero-arg constructor or factory
class UnifiedCallbackPlugin(BasePlugin):
    def __init__(self):
        # Load callback from config or singleton
        self.unified_callback = get_global_callback()
```

#### 3.2.3 Direct TrainerBuilder Modification (Not Recommended)

**Alternative:** Subclass `HFCausalTrainerBuilder`

```python
from axolotl.core.builders.causal import HFCausalTrainerBuilder

class CustomTrainerBuilder(HFCausalTrainerBuilder):
    def __init__(self, cfg, unified_callback, **kwargs):
        super().__init__(cfg, **kwargs)
        self.unified_callback = unified_callback

    def get_callbacks(self):
        """Override pre-trainer callback injection."""
        callbacks = super().get_callbacks()

        # Add your adapter
        from your_library.adapters import HuggingFaceAdapterCallback
        adapter = HuggingFaceAdapterCallback(self.unified_callback)
        callbacks.append(adapter)

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        """Override post-trainer callback injection."""
        callbacks = super().get_post_trainer_create_callbacks(trainer)

        # Add trainer-dependent callbacks if needed
        # callbacks.append(MyTrainerDependentCallback(trainer))

        return callbacks

# Usage
cfg = load_cfg("axolotl_config.yaml")
builder = CustomTrainerBuilder(cfg, my_unified_callback, ...)
trainer = builder.build()
```

**Why Not Recommended:**
- ❌ Requires subclassing internal Axolotl classes
- ❌ May break with Axolotl updates
- ❌ Plugin system is the official extension mechanism

### 3.3 Backend-Specific Injection Summary

| Backend | Mechanism | Code Location | API |
|---------|-----------|---------------|-----|
| **Transformers** | Constructor param | `transformers.Trainer.__init__` | `Trainer(callbacks=[...])` |
| **TRL** | Constructor param (inherited) | `trl.trainer.BaseTrainer.__init__` | `SFTTrainer(callbacks=[...])` |
| **Unsloth** | Constructor param (inherited) | `unsloth.UnslothTrainer.__init__` | `SFTTrainer(callbacks=[...])` |
| **Axolotl (Config)** | Config-driven auto-injection | `axolotl.core.builders.base.TrainerBuilderBase.get_callbacks()` | YAML config flags |
| **Axolotl (Plugin)** | Plugin lifecycle hooks | `axolotl.integrations.base.BasePlugin` | `plugins: ["module.PluginClass"]` |
| **Axolotl (Direct)** | TrainerBuilder override | `axolotl.core.builders.causal.HFCausalTrainerBuilder` | Subclass `get_callbacks()` |

### 3.4 Unified Injection Factory (Recommended Design)

**Goal:** Single API for users across all backends

```python
from typing import Union, List
from enum import Enum

class Backend(Enum):
    TRANSFORMERS = "transformers"
    TRL = "trl"
    UNSLOTH = "unsloth"
    AXOLOTL = "axolotl"


class CallbackInjector:
    """
    Utility to inject UnifiedCallbacks into any supported backend.
    """

    @staticmethod
    def inject(
        trainer_or_config,
        callbacks: List['UnifiedCallback'],
        backend: Backend,
    ):
        """
        Inject callbacks into trainer based on backend type.

        Args:
            trainer_or_config:
                - For Transformers/TRL/Unsloth: Trainer instance or constructor kwargs
                - For Axolotl: Config dict or path
            callbacks: List of UnifiedCallback instances
            backend: Backend type

        Returns:
            - For Transformers/TRL/Unsloth: Modified kwargs dict
            - For Axolotl: Plugin instance to register
        """
        from your_library.adapters import HuggingFaceAdapterCallback

        adapters = [HuggingFaceAdapterCallback(cb) for cb in callbacks]

        if backend in (Backend.TRANSFORMERS, Backend.TRL, Backend.UNSLOTH):
            # Simple: just add to kwargs
            if isinstance(trainer_or_config, dict):
                # Constructor kwargs
                trainer_or_config.setdefault("callbacks", []).extend(adapters)
                return trainer_or_config
            else:
                # Existing trainer instance
                for adapter in adapters:
                    trainer_or_config.add_callback(adapter)
                return trainer_or_config

        elif backend == Backend.AXOLOTL:
            # Return plugin for user to register
            from your_library.adapters.axolotl import UnifiedCallbackPlugin
            return UnifiedCallbackPlugin(callbacks)

        else:
            raise ValueError(f"Unsupported backend: {backend}")


# ========================================
# USAGE EXAMPLES
# ========================================

# Example 1: Transformers
from transformers import Trainer, TrainingArguments

my_callback = MyUnifiedCallback()

trainer_kwargs = {
    "model": model,
    "args": TrainingArguments(...),
    "train_dataset": dataset,
}

# Inject callback
CallbackInjector.inject(trainer_kwargs, [my_callback], Backend.TRANSFORMERS)

trainer = Trainer(**trainer_kwargs)


# Example 2: TRL
from trl import SFTTrainer, SFTConfig

trainer_kwargs = {
    "model": model,
    "args": SFTConfig(...),
    "train_dataset": dataset,
}

CallbackInjector.inject(trainer_kwargs, [my_callback], Backend.TRL)

trainer = SFTTrainer(**trainer_kwargs)


# Example 3: Axolotl
from axolotl.integrations.base import PluginManager

plugin = CallbackInjector.inject(
    "axolotl_config.yaml",
    [my_callback],
    Backend.AXOLOTL
)

PluginManager.get_instance().register_plugin(plugin)
```

---

## Section 4: Proposed Architecture

### 4.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         USER CODE                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  class MyCallback(UnifiedCallback):                   │  │
│  │      def on_step_end(self, ctx: UnifiedContext):     │  │
│  │          if ctx.global_step > 1000:                   │  │
│  │              ctx.abort_training()                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ (unified interface)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED LIBRARY                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  UnifiedCallback (ABC)                                │  │
│  │  - on_step_end(ctx: UnifiedContext)                   │  │
│  │  - on_train_begin(ctx: UnifiedContext)                │  │
│  │  - ... (15 methods)                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  UnifiedContext (dataclass)                           │  │
│  │  - global_step: int                                   │  │
│  │  - epoch: float                                       │  │
│  │  - abort_training()                                   │  │
│  │  - trigger_checkpoint()                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ (adapter layer)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ADAPTER LAYER                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HuggingFaceAdapterCallback(TrainerCallback)          │  │
│  │  - Wraps UnifiedCallback                              │  │
│  │  - Translates TrainerState → UnifiedContext           │  │
│  │  - Translates UnifiedContext → TrainerControl         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AxolotlPlugin(BasePlugin)                            │  │
│  │  - Wraps UnifiedCallback for Axolotl                  │  │
│  │  - Implements plugin lifecycle hooks                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ (backend-specific)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND FRAMEWORKS                        │
│  ┌────────────┐  ┌─────┐  ┌────────┐  ┌─────────┐          │
│  │Transformers│  │ TRL │  │Axolotl │  │ Unsloth │          │
│  └────────────┘  └─────┘  └────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Core Interfaces

#### 4.2.1 UnifiedCallback (User-Facing ABC)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class UnifiedCallback(ABC):
    """
    Abstract base class for unified training callbacks.

    Users subclass this to define custom training logic that works
    across all supported backends (Transformers, TRL, Axolotl, Unsloth).

    All methods receive a UnifiedContext object and optionally
    additional parameters (metrics, logs, etc.).

    Methods are optional - override only what you need.
    """

    # ========================================
    # Training Lifecycle
    # ========================================

    def on_init_end(self, ctx: UnifiedContext):
        """
        Called after trainer initialization completes.

        Args:
            ctx: Training context with model, tokenizer, optimizer, etc.
        """
        pass

    def on_train_begin(self, ctx: UnifiedContext):
        """
        Called at the start of training, before first step.

        Args:
            ctx: Training context with max_steps initialized
        """
        pass

    def on_train_end(self, ctx: UnifiedContext):
        """
        Called after training completes (or is stopped early).

        Args:
            ctx: Final training context with all metrics
        """
        pass

    # ========================================
    # Epoch Lifecycle
    # ========================================

    def on_epoch_begin(self, ctx: UnifiedContext):
        """
        Called at the start of each epoch.

        Args:
            ctx: Context with current epoch number
        """
        pass

    def on_epoch_end(self, ctx: UnifiedContext):
        """
        Called at the end of each epoch.

        Args:
            ctx: Context with epoch metrics
        """
        pass

    # ========================================
    # Step Lifecycle
    # ========================================

    def on_step_begin(self, ctx: UnifiedContext):
        """
        Called before each optimizer step (after gradient accumulation).

        Args:
            ctx: Context with current global_step
        """
        pass

    def on_step_end(self, ctx: UnifiedContext):
        """
        Called after each optimizer step.

        Args:
            ctx: Context with updated global_step
        """
        pass

    def on_substep_end(self, ctx: UnifiedContext):
        """
        Called after each forward/backward pass during gradient accumulation.

        Note: This fires multiple times per optimizer step if
        gradient_accumulation_steps > 1.

        Args:
            ctx: Context (global_step NOT incremented until on_step_end)
        """
        pass

    # ========================================
    # Optimization Lifecycle
    # ========================================

    def on_pre_optimizer_step(self, ctx: UnifiedContext):
        """
        Called after gradient clipping, before optimizer.step().

        Args:
            ctx: Context with optimizer available
        """
        pass

    def on_optimizer_step(self, ctx: UnifiedContext):
        """
        Called after optimizer.step(), before zero_grad().

        Args:
            ctx: Context with optimizer available (gradients still present)
        """
        pass

    # ========================================
    # Logging & Checkpointing
    # ========================================

    def on_log(self, ctx: UnifiedContext, metrics: Dict[str, float]):
        """
        Called when metrics are logged.

        Args:
            ctx: Training context
            metrics: Dict of logged metrics (loss, learning_rate, etc.)
        """
        pass

    def on_save(self, ctx: UnifiedContext, checkpoint_metrics: Dict[str, float]):
        """
        Called before saving a checkpoint.

        Args:
            ctx: Training context
            checkpoint_metrics: Metrics at checkpoint time
        """
        pass

    # ========================================
    # Evaluation & Prediction
    # ========================================

    def on_evaluate(self, ctx: UnifiedContext, eval_metrics: Dict[str, float]):
        """
        Called after evaluation completes.

        Args:
            ctx: Training context
            eval_metrics: Evaluation results (eval_loss, eval_accuracy, etc.)
        """
        pass

    def on_predict(self, ctx: UnifiedContext, predict_metrics: Dict[str, float]):
        """
        Called after prediction completes.

        Args:
            ctx: Training context
            predict_metrics: Prediction results
        """
        pass

    def on_prediction_step(self, ctx: UnifiedContext, inputs: Dict[str, Any]):
        """
        Called before each prediction batch.

        Args:
            ctx: Training context
            inputs: Input batch dict
        """
        pass


# ========================================
# USAGE EXAMPLE
# ========================================

class EarlyStoppingCallback(UnifiedCallback):
    """Early stopping based on metric threshold."""

    def __init__(self, metric_name: str = "eval_loss", threshold: float = 0.1):
        self.metric_name = metric_name
        self.threshold = threshold

    def on_evaluate(self, ctx: UnifiedContext, eval_metrics: Dict[str, float]):
        metric_value = eval_metrics.get(self.metric_name)

        if metric_value is not None and metric_value < self.threshold:
            print(f"Early stopping: {self.metric_name}={metric_value} < {self.threshold}")
            ctx.abort_training()


class CheckpointEveryNSteps(UnifiedCallback):
    """Checkpoint at fixed step intervals."""

    def __init__(self, interval: int = 100):
        self.interval = interval

    def on_step_end(self, ctx: UnifiedContext):
        if ctx.global_step % self.interval == 0:
            print(f"Triggering checkpoint at step {ctx.global_step}")
            ctx.trigger_checkpoint()


class LogTokenThroughput(UnifiedCallback):
    """Log tokens per second."""

    def __init__(self):
        self.last_step = 0
        self.last_tokens = 0
        self.last_time = None

    def on_log(self, ctx: UnifiedContext, metrics: Dict[str, float]):
        import time

        current_time = time.time()

        if self.last_time is not None:
            elapsed = current_time - self.last_time
            steps_delta = ctx.global_step - self.last_step
            tokens_delta = ctx.tokens_seen - self.last_tokens

            tokens_per_sec = tokens_delta / elapsed if elapsed > 0 else 0
            print(f"Step {ctx.global_step}: {tokens_per_sec:.0f} tokens/sec")

        self.last_step = ctx.global_step
        self.last_tokens = ctx.tokens_seen
        self.last_time = current_time
```

#### 4.2.2 UnifiedContext (Data Class)

**Already defined in Section 2.3** - reproduced here for completeness:

```python
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class UnifiedContext:
    """Normalized training context that abstracts backend differences."""

    # Training Progress
    global_step: int
    epoch: float
    max_steps: int

    # Metrics
    current_metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: list[Dict[str, float]] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_checkpoint_path: Optional[str] = None

    # Resource Tracking
    tokens_seen: int = 0
    total_flops: float = 0.0

    # Distributed Training
    is_main_process: bool = True
    is_local_main_process: bool = True

    # Model References
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    optimizer: Optional[Any] = None

    # Control Flags (internal)
    _should_stop_training: bool = False
    _should_stop_epoch: bool = False
    _should_save_now: bool = False
    _should_evaluate_now: bool = False
    _should_log_now: bool = False

    # Backend Info
    backend_name: str = "unknown"
    backend_args: Optional[Any] = None

    # User Methods
    def abort_training(self):
        """Stop training immediately."""
        self._should_stop_training = True

    def stop_current_epoch(self):
        """Finish current epoch and stop."""
        self._should_stop_epoch = True

    def trigger_checkpoint(self):
        """Save checkpoint at end of current step."""
        self._should_save_now = True

    def trigger_evaluation(self):
        """Run evaluation at end of current step."""
        self._should_evaluate_now = True

    def trigger_logging(self):
        """Log metrics at end of current step."""
        self._should_log_now = True

    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get current metric value."""
        return self.current_metrics.get(key, default)
```

#### 4.2.3 HuggingFaceAdapterCallback (Adapter)

**Already defined in Section 2.3** - key methods:

```python
class HuggingFaceAdapterCallback(TrainerCallback):
    """Adapter that translates UnifiedCallback ↔ HuggingFace TrainerCallback."""

    def __init__(self, unified_callback: UnifiedCallback):
        self.unified_callback = unified_callback

    def _state_to_context(self, args, state, control, **kwargs) -> UnifiedContext:
        """Convert TrainerState → UnifiedContext."""
        # ... (see Section 2.3)

    def _context_to_control(self, ctx, control) -> TrainerControl:
        """Apply UnifiedContext control flags → TrainerControl."""
        # ... (see Section 2.3)

    # 15 event forwarding methods (on_init_end, on_train_begin, etc.)
    # ... (see Section 2.3)
```

### 4.3 Backend-Specific Adapters

#### 4.3.1 Axolotl Plugin Adapter

```python
from axolotl.integrations.base import BasePlugin
from typing import List, Callable


class UnifiedCallbackPlugin(BasePlugin):
    """
    Axolotl plugin that injects UnifiedCallback instances.

    Usage:
        plugin = UnifiedCallbackPlugin([callback1, callback2])
        PluginManager.get_instance().register_plugin(plugin)
    """

    def __init__(self, callbacks: List[UnifiedCallback]):
        """
        Args:
            callbacks: List of UnifiedCallback instances to inject
        """
        self.callbacks = callbacks

    def add_callbacks_pre_trainer(self, cfg, model) -> List[Callable]:
        """
        Inject callbacks before trainer creation.

        Returns:
            List of HuggingFaceAdapterCallback instances
        """
        from your_library.adapters import HuggingFaceAdapterCallback

        adapters = [
            HuggingFaceAdapterCallback(cb) for cb in self.callbacks
        ]
        return adapters

    def add_callbacks_post_trainer(self, cfg, trainer) -> List[Callable]:
        """
        Inject callbacks after trainer creation.

        Use this if callbacks need trainer reference.
        """
        return []
```

### 4.4 Complete Usage Example (All Backends)

```python
# ========================================
# 1. DEFINE CALLBACK (ONCE)
# ========================================

from your_library import UnifiedCallback, UnifiedContext

class MyCallback(UnifiedCallback):
    def on_step_end(self, ctx: UnifiedContext):
        if ctx.global_step % 100 == 0:
            print(f"Step {ctx.global_step}: loss={ctx.get_metric('loss')}")


# ========================================
# 2. USE WITH TRANSFORMERS
# ========================================

from transformers import Trainer, TrainingArguments
from your_library.adapters import HuggingFaceAdapterCallback

callback = MyCallback()
adapter = HuggingFaceAdapterCallback(callback)

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=dataset,
    callbacks=[adapter],
)

trainer.train()


# ========================================
# 3. USE WITH TRL
# ========================================

from trl import SFTTrainer, SFTConfig
from your_library.adapters import HuggingFaceAdapterCallback

callback = MyCallback()
adapter = HuggingFaceAdapterCallback(callback)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=dataset,
    callbacks=[adapter],
)

trainer.train()


# ========================================
# 4. USE WITH UNSLOTH
# ========================================

from unsloth import FastLanguageModel
from trl import SFTTrainer
from your_library.adapters import HuggingFaceAdapterCallback

model, tokenizer = FastLanguageModel.from_pretrained(...)

callback = MyCallback()
adapter = HuggingFaceAdapterCallback(callback)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    callbacks=[adapter],
)

trainer.train()


# ========================================
# 5. USE WITH AXOLOTL
# ========================================

from axolotl.integrations.base import PluginManager
from axolotl.cli.train import do_train
from your_library.adapters.axolotl import UnifiedCallbackPlugin

callback = MyCallback()
plugin = UnifiedCallbackPlugin([callback])

# Register plugin
PluginManager.get_instance().register_plugin(plugin)

# Train (plugin auto-injects callback)
do_train("axolotl_config.yaml")
```

### 4.5 Class Diagram

```
┌─────────────────────────────────────┐
│     UnifiedCallback (ABC)           │
├─────────────────────────────────────┤
│ + on_init_end(ctx)                  │
│ + on_train_begin(ctx)               │
│ + on_train_end(ctx)                 │
│ + on_epoch_begin(ctx)               │
│ + on_epoch_end(ctx)                 │
│ + on_step_begin(ctx)                │
│ + on_step_end(ctx)                  │
│ + on_substep_end(ctx)               │
│ + on_pre_optimizer_step(ctx)        │
│ + on_optimizer_step(ctx)            │
│ + on_log(ctx, metrics)              │
│ + on_save(ctx, metrics)             │
│ + on_evaluate(ctx, metrics)         │
│ + on_predict(ctx, metrics)          │
│ + on_prediction_step(ctx, inputs)   │
└─────────────────────────────────────┘
             △
             │ (implements)
             │
┌────────────────────────────────────────┐
│   MyCustomCallback                     │
├────────────────────────────────────────┤
│ - threshold: float                     │
│ + on_step_end(ctx)                     │
│ + on_evaluate(ctx, metrics)            │
└────────────────────────────────────────┘


┌─────────────────────────────────────┐
│     UnifiedContext (dataclass)      │
├─────────────────────────────────────┤
│ + global_step: int                  │
│ + epoch: float                      │
│ + max_steps: int                    │
│ + current_metrics: Dict             │
│ + metric_history: List              │
│ + best_metric: float                │
│ + best_checkpoint_path: str         │
│ + tokens_seen: int                  │
│ + total_flops: float                │
│ + is_main_process: bool             │
│ + model: Any                        │
│ + tokenizer: Any                    │
│ + optimizer: Any                    │
│ + backend_name: str                 │
│ + backend_args: Any                 │
├─────────────────────────────────────┤
│ + abort_training()                  │
│ + stop_current_epoch()              │
│ + trigger_checkpoint()              │
│ + trigger_evaluation()              │
│ + trigger_logging()                 │
│ + get_metric(key) → Any             │
└─────────────────────────────────────┘
             △
             │ (uses)
             │
┌─────────────────────────────────────┐
│     UnifiedCallback                 │
└─────────────────────────────────────┘


┌──────────────────────────────────────────────┐
│  HuggingFaceAdapterCallback                  │
│  (TrainerCallback)                           │
├──────────────────────────────────────────────┤
│ - unified_callback: UnifiedCallback          │
├──────────────────────────────────────────────┤
│ - _state_to_context(args, state, control)    │
│     → UnifiedContext                         │
│ - _context_to_control(ctx, control)          │
│     → TrainerControl                         │
│ + on_init_end(args, state, control, **kw)    │
│ + on_train_begin(args, state, control, **kw) │
│ + ... (15 methods total)                     │
└──────────────────────────────────────────────┘
             │
             │ (wraps)
             ▼
┌─────────────────────────────────────┐
│     UnifiedCallback                 │
└─────────────────────────────────────┘


┌──────────────────────────────────────────────┐
│  UnifiedCallbackPlugin                       │
│  (BasePlugin) [Axolotl]                      │
├──────────────────────────────────────────────┤
│ - callbacks: List[UnifiedCallback]           │
├──────────────────────────────────────────────┤
│ + add_callbacks_pre_trainer(cfg, model)      │
│     → List[TrainerCallback]                  │
│ + add_callbacks_post_trainer(cfg, trainer)   │
│     → List[TrainerCallback]                  │
└──────────────────────────────────────────────┘
             │
             │ (creates)
             ▼
┌──────────────────────────────────────────────┐
│  HuggingFaceAdapterCallback                  │
└──────────────────────────────────────────────┘
```

---

## Section 5: Implementation Roadmap

### Phase 1: Core Abstractions (Week 1-2)

**Deliverables:**
- [ ] `UnifiedContext` dataclass
- [ ] `UnifiedCallback` ABC
- [ ] `HuggingFaceAdapterCallback` implementation
- [ ] Unit tests for adapter translation logic

**Validation:**
- Verify all 15 events forward correctly
- Test control flag translation (abort, checkpoint, etc.)
- Test state normalization (metrics, progress tracking)

### Phase 2: Backend Integration (Week 3-4)

**Deliverables:**
- [ ] Transformers integration tests
- [ ] TRL integration tests
- [ ] Unsloth integration tests
- [ ] `UnifiedCallbackPlugin` for Axolotl
- [ ] Axolotl integration tests

**Validation:**
- Same callback works across all 4 backends
- Control flags behave identically
- Metrics logged correctly on all backends

### Phase 3: Advanced Features (Week 5-6)

**Deliverables:**
- [ ] `CallbackInjector` utility
- [ ] Example callbacks (early stopping, custom logging, etc.)
- [ ] Documentation and tutorials
- [ ] Performance benchmarks (callback overhead)

**Validation:**
- < 1% training time overhead from adapter layer
- All example callbacks work across backends

### Phase 4: Production Hardening (Week 7-8)

**Deliverables:**
- [ ] Error handling (graceful degradation)
- [ ] Distributed training validation
- [ ] Checkpoint resumption with callbacks
- [ ] CI/CD integration

**Validation:**
- No crashes with missing metrics
- Callbacks work correctly on rank 0 vs non-zero processes
- State persists across checkpoint resume

---

## Appendix A: Backend Analysis Summary

### A.1 Transformers

**Repository:** https://github.com/huggingface/transformers
**Files Analyzed:**
- `src/transformers/trainer_callback.py` (240 lines)
- `src/transformers/trainer.py` (4000+ lines)

**Key Findings:**
- 15 standard callback events
- TrainerState: 20 attributes (read-only by convention)
- TrainerControl: 5 boolean flags (2 persistent, 3 transient)
- CallbackHandler: Sequential execution, control threading
- Default callbacks: DefaultFlowCallback (required), ProgressCallback, PrinterCallback

**Documentation:** `/Users/szaher/go/src/github.com/szaher/designs/training-hub-callbacks/huggingface_transformers_callback_analysis.md`

### A.2 TRL

**Repository:** https://github.com/huggingface/trl
**Files Analyzed:**
- `trl/trainer/callbacks.py`
- `trl/trainer/base_trainer.py`
- `trl/trainer/dpo_trainer.py`
- `trl/trainer/sft_trainer.py`

**Key Findings:**
- ✅ 100% Transformers-compatible (NO custom events)
- 7 custom callbacks (SyncRefModelCallback, WinRateCallback, etc.)
- All trainers inherit from Transformers.Trainer
- Callbacks use standard events for RLHF logic

**Documentation:** 7 markdown files in `/Users/szaher/go/src/github.com/szaher/designs/training-hub-callbacks/`

### A.3 Axolotl

**Repository:** https://github.com/axolotl-ai-cloud/axolotl
**Files Analyzed:**
- `src/axolotl/utils/callbacks/` (17 callbacks)
- `src/axolotl/core/builders/base.py`
- `src/axolotl/core/builders/causal.py`

**Key Findings:**
- Two-phase callback injection (pre-trainer + post-trainer)
- Plugin system for extensibility
- 17 custom callbacks (LISA, DynamicCheckpoint, etc.)
- Config-driven callback registration

**Special Requirements:**
- Use `BasePlugin` for programmatic injection
- Factory pattern for trainer-dependent callbacks

### A.4 Unsloth

**Repository:** https://github.com/unslothai/unsloth
**Files Analyzed:**
- `unsloth/trainer.py`
- `unsloth/models/_utils.py`

**Key Findings:**
- ✅ 100% Transformers-compatible
- Thin wrapper around TRL's SFTTrainer
- No callback mechanism modifications
- Callbacks passed through normally

**Special Requirements:** None (use standard HF callback API)

---

## Appendix B: Gotchas and Best Practices

### B.1 Critical Gotchas

1. **State Immutability:** TrainerState is mutable but should be treated as read-only
2. **Control Order:** Callbacks execute sequentially; later callbacks see earlier modifications
3. **Auto-Reset Flags:** `should_save`, `should_evaluate`, `should_log` reset automatically
4. **Step vs Substep:** `on_substep_end` fires multiple times per optimizer step with gradient accumulation
5. **Metric Availability:** `best_metric` is None until first evaluation
6. **Process Checks:** Logging/checkpointing should often be restricted to `is_main_process`
7. **Return Values:** Return control object (or None if unchanged)
8. **Gradient Accumulation:** `global_step` only increments in `on_step_end`, not `on_substep_end`

### B.2 Best Practices

1. **Always check for None:** Metrics may not be available in early events
2. **Use context managers:** For resource-heavy operations in callbacks
3. **Minimize overhead:** Callbacks fire frequently; avoid expensive operations
4. **Respect distributed training:** Check `is_main_process` before I/O
5. **Test across backends:** Verify callback works on all 4 supported backends
6. **Document assumptions:** Specify which metrics your callback requires
7. **Handle missing metrics gracefully:** Use `.get()` with defaults
8. **Avoid side effects in state_to_context:** Keep translation logic pure

---

## Appendix C: Implementation Checklist

### Core Components
- [ ] `UnifiedContext` dataclass with all 20 state fields
- [ ] `UnifiedCallback` ABC with all 15 event methods
- [ ] `HuggingFaceAdapterCallback` with complete translation logic
- [ ] `UnifiedCallbackPlugin` for Axolotl
- [ ] `CallbackInjector` utility

### Translation Logic
- [ ] `_state_to_context()` handles all TrainerState fields
- [ ] `_context_to_control()` handles all 5 control flags
- [ ] Metrics dict properly extracted from log_history
- [ ] Model/tokenizer/optimizer references populated from kwargs
- [ ] Backend name detected correctly

### Testing
- [ ] Unit tests for state translation
- [ ] Unit tests for control translation
- [ ] Integration test: Transformers
- [ ] Integration test: TRL (SFT + DPO)
- [ ] Integration test: Unsloth
- [ ] Integration test: Axolotl
- [ ] Distributed training test (multi-GPU)
- [ ] Checkpoint resumption test

### Documentation
- [ ] API reference for UnifiedCallback
- [ ] API reference for UnifiedContext
- [ ] Usage tutorial (all 4 backends)
- [ ] Migration guide (from raw TrainerCallback)
- [ ] Example callbacks library
- [ ] Troubleshooting guide

### Performance
- [ ] Benchmark: Callback overhead < 1%
- [ ] Benchmark: Memory overhead < 10MB
- [ ] Profile: No memory leaks in long training runs

---

**End of Technical Design Document**
