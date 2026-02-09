# RFE: Unified Callback Abstraction System for Training-Hub

## JIRA RFE Template

---

## Summary

Add a unified callback abstraction layer to the Training-Hub library that enables data scientists to write training callbacks once and deploy them across all Training-Hub backends (InstructLab Training, Mini-Trainer, Unsloth, and future backends). This RFE builds on the callback mechanisms implemented in individual trainers (separate RFE) to provide a framework-agnostic callback interface that eliminates backend vendor lock-in and enables enterprise-wide standardization of monitoring, logging, and governance in Red Hat OpenShift AI.

---

## Component

**Product**: Red Hat OpenShift AI (RHOAI)
**Component**: Training-Hub Library
**Repository**: https://github.com/Red-Hat-AI-Innovation-Team/training_hub
**Documentation**: https://deepwiki.com/Red-Hat-AI-Innovation-Team/training_hub
**Delivery**: Feature enhancement to existing training_hub Python library
**Version Target**: training_hub 1.0.0

---

## Description

### Problem Statement

Training-Hub provides a unified API for multiple LLM training backends, but each backend has its own callback mechanism (or now has one via separate RFE). This creates challenges for enterprise AI/ML teams:

**Current State After Callback Implementation RFE:**

| Backend | Callback Support | Callback API | Code Portability |
|---------|-----------------|--------------|------------------|
| **InstructLab Training** | ✅ Custom callbacks | Custom event hooks | ❌ Backend-specific |
| **Mini-Trainer** | ✅ Custom callbacks | Custom event hooks | ❌ Backend-specific |
| **Unsloth** | ✅ Via TRL/Transformers | HuggingFace TrainerCallback | ❌ Backend-specific |

**Pain Points:**

1. **Backend Inconsistency**: Each backend has different callback interfaces
   - InstructLab callbacks: Custom event hooks
   - Mini-Trainer callbacks: Custom event hooks (different from InstructLab)
   - Unsloth callbacks: HuggingFace TrainerCallback interface
   - Users must learn 3+ different callback APIs

2. **Code Duplication**: Teams must implement backend-specific monitoring solutions
   - Audit logging implemented 3 different ways
   - Cost tracking requires backend-specific code
   - Quality monitoring varies by backend

3. **Migration Friction**: Switching backends within Training-Hub loses all custom instrumentation
   - Custom logging breaks when moving from Unsloth to InstructLab
   - Early stopping logic must be rewritten per backend
   - Enterprise governance becomes backend-specific

4. **Training-Hub Value Dilution**: Training-Hub's unified API doesn't extend to callbacks
   - Users can easily switch backends for training
   - But cannot easily switch backends with callbacks
   - Callbacks negate the "write once, run anywhere" value proposition

**Customer Impact:**

- **Fortune 500 Financial Services**: Need consistent compliance logging across ALL Training-Hub backends
- **Healthcare/Life Sciences**: Require unified audit trails for regulatory validation across SFT/OSFT/LoRA workflows
- **Government/Defense**: Need framework-agnostic security hooks that work with InstructLab and Mini-Trainer
- **Research Teams**: Want custom metric tracking that persists across backend changes

### Why Unified Callback Abstraction is Essential

Unified callback abstraction is critical for multi-backend platforms like Training-Hub:

1. **Preserves Training-Hub's Value Proposition**: "Write once, run anywhere"
   - Without unified callbacks: Users write backend-specific instrumentation
   - With unified callbacks: Same callback code works across all backends

2. **Enables True Backend Flexibility**: Users can switch backends without losing tooling
   - Change from InstructLab → Mini-Trainer without rewriting monitoring
   - Experiment with Unsloth while keeping compliance logging

3. **Enterprise Standardization**: Single callback implementation for organization
   - Audit logging written once, used across all training workflows
   - Cost tracking consistent regardless of backend choice
   - Governance policies enforced uniformly

4. **Reduces Learning Curve**: One callback API to learn
   - Data scientists learn TrainingHubCallback interface once
   - No need to learn HuggingFace, InstructLab, and Mini-Trainer callback APIs
   - Faster onboarding and productivity

### Proposed Solution

Implement a **Unified Callback System** as a core feature of Training-Hub that:

1. **Unified Callback Interface**: Single `TrainingHubCallback` abstract base class
   - Works consistently across ALL Training-Hub backends
   - User writes callback once, Training-Hub handles backend translation

2. **Normalized Training Context**: `TrainingHubContext` provides consistent state
   - Abstracts backend-specific state (InstructLab's `AcceleratorWrapper`, Mini-Trainer's metrics, etc.)
   - Unified access to: model, optimizer, metrics, checkpoints, distributed training state

3. **Backend Adapter Layer**: Automatic translation between unified and backend-specific callbacks
   - **Unsloth**: Adapter translates TrainingHubCallback ↔ HuggingFace TrainerCallback
   - **InstructLab Training**: Adapter translates TrainingHubCallback ↔ InstructLab custom hooks
   - **Mini-Trainer**: Adapter translates TrainingHubCallback ↔ Mini-Trainer custom hooks

4. **Enterprise Callback Library**: Pre-built callbacks for common enterprise needs
   - Works with ALL backends without modification
   - Compliance, cost tracking, quality monitoring, bias detection
   - Users can write custom callbacks following the same interface

### Technical Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   TRAINING-HUB PUBLIC API                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  from training_hub import sft, osft, lora_sft           │  │
│  │  from training_hub.callbacks import (                   │  │
│  │      AuditLoggingCallback,                              │  │
│  │      CostTrackingCallback,                              │  │
│  │      EarlyStoppingCallback,                             │  │
│  │  )                                                       │  │
│  │                                                          │  │
│  │  # Same callbacks work with ANY backend!                │  │
│  │  sft(model, data, callbacks=[...], backend='...')       │  │
│  │  osft(model, data, callbacks=[...])                     │  │
│  │  lora_sft(model, data, callbacks=[...])                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ inject callbacks
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│            TRAINING-HUB UNIFIED CALLBACK SYSTEM                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TrainingHubCallback (ABC)                               │  │
│  │    - on_train_begin(ctx: TrainingHubContext)             │  │
│  │    - on_step_end(ctx: TrainingHubContext)                │  │
│  │    - on_evaluate(ctx: TrainingHubContext, metrics)       │  │
│  │    - 15 standard lifecycle events                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TrainingHubContext (State Abstraction)                  │  │
│  │    - global_step, epoch, metrics, model, tokenizer       │  │
│  │    - backend_name, backend_args                          │  │
│  │    - abort_training(), trigger_checkpoint()              │  │
│  │    - is_main_process (distributed training)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Enterprise Callback Library                             │  │
│  │    - AuditLoggingCallback (SOC2, HIPAA)                  │  │
│  │    - CostTrackingCallback (FinOps)                       │  │
│  │    - QualityMonitorCallback (LLM-as-judge)               │  │
│  │    - BiasDetectionCallback (fairness)                    │  │
│  │    - EarlyStoppingCallback (validation-based)            │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ adapt to backend
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│                   BACKEND ADAPTER LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Unsloth     │  │  InstructLab │  │  Mini-Trainer        │ │
│  │  Adapter     │  │  Adapter     │  │  Adapter             │ │
│  │              │  │              │  │                      │ │
│  │ Translates:  │  │ Translates:  │  │ Translates:          │ │
│  │ TrainingHub  │  │ TrainingHub  │  │ TrainingHub          │ │
│  │ Callback     │  │ Callback     │  │ Callback             │ │
│  │ ↕            │  │ ↕            │  │ ↕                    │ │
│  │ HuggingFace  │  │ InstructLab  │  │ Mini-Trainer         │ │
│  │ Callback     │  │ Hooks        │  │ Hooks                │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ backend-specific
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│              TRAINING-HUB BACKEND IMPLEMENTATIONS               │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  InstructLab   │  │  Mini-Trainer  │  │    Unsloth      │  │
│  │   Training     │  │   (OSFT)       │  │  (LoRA + SFT)   │  │
│  │  - Has native  │  │  - Has native  │  │  - Native       │  │
│  │    callbacks   │  │    callbacks   │  │    HuggingFace  │  │
│  │    (via RFE)   │  │    (via RFE)   │  │    callbacks    │  │
│  └────────────────┘  └────────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Core Components

**1. TrainingHubCallback (Abstract Base Class)**

```python
# File: training_hub/callbacks/base.py

from abc import ABC
from typing import Optional, Dict, Any

class TrainingHubCallback(ABC):
    """
    Base class for Training-Hub callbacks.

    Works across all backends: InstructLab, Mini-Trainer, Unsloth.

    All callback methods receive TrainingHubContext which provides
    normalized access to training state regardless of backend.
    """

    def on_init_end(self, ctx: 'TrainingHubContext'):
        """Called after backend initialization, before training."""
        pass

    def on_train_begin(self, ctx: 'TrainingHubContext'):
        """Called at the start of training."""
        pass

    def on_epoch_begin(self, ctx: 'TrainingHubContext'):
        """Called at the beginning of an epoch."""
        pass

    def on_step_begin(self, ctx: 'TrainingHubContext'):
        """Called at the beginning of a training step."""
        pass

    def on_substep_end(self, ctx: 'TrainingHubContext'):
        """Called after each gradient accumulation substep."""
        pass

    def on_pre_optimizer_step(self, ctx: 'TrainingHubContext'):
        """Called before optimizer.step() (after gradient clipping)."""
        pass

    def on_optimizer_step(self, ctx: 'TrainingHubContext'):
        """Called after optimizer.step()."""
        pass

    def on_log(self, ctx: 'TrainingHubContext', metrics: Dict[str, Any]):
        """Called when metrics are logged."""
        pass

    def on_save(self, ctx: 'TrainingHubContext'):
        """Called when a checkpoint is saved."""
        pass

    def on_evaluate(self, ctx: 'TrainingHubContext', eval_metrics: Dict[str, Any]):
        """Called after evaluation."""
        pass

    def on_step_end(self, ctx: 'TrainingHubContext'):
        """Called at the end of a training step."""
        pass

    def on_epoch_end(self, ctx: 'TrainingHubContext'):
        """Called at the end of an epoch."""
        pass

    def on_train_end(self, ctx: 'TrainingHubContext'):
        """Called at the end of training."""
        pass

    def on_predict(self, ctx: 'TrainingHubContext'):
        """Called before prediction/inference."""
        pass

    def on_prediction_step(self, ctx: 'TrainingHubContext'):
        """Called during each prediction step."""
        pass
```

**2. TrainingHubContext (State Container)**

```python
# File: training_hub/callbacks/context.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class TrainingHubContext:
    """
    Normalized training context that abstracts backend differences.

    Provides consistent API across InstructLab, Mini-Trainer, Unsloth, etc.
    """

    # Core Training State
    global_step: int = 0
    epoch: int = 0
    max_steps: int = 0
    max_epochs: int = 0

    # Current Metrics
    current_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None

    # Metric History
    log_history: List[Dict[str, Any]] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_checkpoint_path: Optional[str] = None

    # Model/Optimizer References
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    optimizer: Optional[Any] = None
    lr_scheduler: Optional[Any] = None

    # Distributed Training
    is_main_process: bool = True
    world_size: int = 1
    process_index: int = 0

    # Backend-Specific
    backend_name: str = "unknown"
    backend_args: Optional[Any] = None
    backend_state: Optional[Any] = None

    # Control Flags (modifiable by callbacks)
    _should_training_stop: bool = False
    _should_epoch_stop: bool = False
    _should_save_now: bool = False
    _should_evaluate_now: bool = False
    _should_log_now: bool = False

    # Control Methods
    def abort_training(self):
        """Stop training immediately."""
        self._should_training_stop = True

    def abort_epoch(self):
        """Stop current epoch and move to next."""
        self._should_epoch_stop = True

    def trigger_checkpoint(self):
        """Request a checkpoint save."""
        self._should_save_now = True

    def trigger_evaluation(self):
        """Request an evaluation run."""
        self._should_evaluate_now = True

    def trigger_logging(self):
        """Request metrics logging."""
        self._should_log_now = True

    def get_metric(self, name: str, default=None):
        """Get current metric value."""
        if self.log_history:
            return self.log_history[-1].get(name, default)
        return default
```

**3. Backend Adapters**

```python
# File: training_hub/callbacks/adapters.py

from transformers import TrainerCallback, TrainerControl, TrainerState

class HuggingFaceCallbackAdapter(TrainerCallback):
    """
    Adapts TrainingHubCallback to HuggingFace TrainerCallback interface.

    Used for backends that use HuggingFace Transformers/TRL (Unsloth).
    """

    def __init__(self, training_hub_callback: TrainingHubCallback):
        self.callback = training_hub_callback

    def _build_context(self, args, state: TrainerState, control: TrainerControl) -> TrainingHubContext:
        """Translate HuggingFace state to TrainingHubContext."""
        ctx = TrainingHubContext(
            global_step=state.global_step,
            epoch=int(state.epoch) if state.epoch else 0,
            max_steps=state.max_steps,
            current_loss=state.log_history[-1].get("loss") if state.log_history else None,
            learning_rate=state.log_history[-1].get("learning_rate") if state.log_history else None,
            log_history=state.log_history,
            best_metric=state.best_metric,
            backend_name="unsloth",
            is_main_process=state.is_world_process_zero,
            world_size=state.world_size,
            process_index=state.process_index,
        )
        return ctx

    def _update_control(self, ctx: TrainingHubContext, control: TrainerControl):
        """Translate TrainingHubContext control flags back to TrainerControl."""
        if ctx._should_training_stop:
            control.should_training_stop = True
        if ctx._should_epoch_stop:
            control.should_epoch_stop = True
        if ctx._should_save_now:
            control.should_save = True
        if ctx._should_evaluate_now:
            control.should_evaluate = True
        if ctx._should_log_now:
            control.should_log = True

    def on_train_begin(self, args, state, control, **kwargs):
        ctx = self._build_context(args, state, control)
        self.callback.on_train_begin(ctx)
        self._update_control(ctx, control)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        ctx = self._build_context(args, state, control)
        self.callback.on_step_end(ctx)
        self._update_control(ctx, control)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        ctx = self._build_context(args, state, control)
        self.callback.on_log(ctx, logs or {})
        self._update_control(ctx, control)
        return control

    # ... implement all 15 event methods ...


class InstructLabCallbackAdapter:
    """
    Adapts TrainingHubCallback to InstructLab custom hooks.
    """

    def __init__(self, training_hub_callback: TrainingHubCallback):
        self.callback = training_hub_callback
        self.context = TrainingHubContext(backend_name="instructlab")

    def on_train_begin(self, model, optimizer, **kwargs):
        """Translate InstructLab hook to TrainingHubCallback."""
        self.context.model = model
        self.context.optimizer = optimizer
        self.callback.on_train_begin(self.context)

    def on_step_end(self, step, loss, **kwargs):
        """Translate InstructLab hook to TrainingHubCallback."""
        self.context.global_step = step
        self.context.current_loss = loss
        self.callback.on_step_end(self.context)

    # ... implement all event translations ...


class MiniTrainerCallbackAdapter:
    """
    Adapts TrainingHubCallback to Mini-Trainer custom hooks.
    """

    def __init__(self, training_hub_callback: TrainingHubCallback):
        self.callback = training_hub_callback
        self.context = TrainingHubContext(backend_name="mini_trainer")

    # Similar to InstructLabCallbackAdapter
    # ... implement all event translations ...
```

**4. Enterprise Callback Library**

```python
# File: training_hub/callbacks/enterprise/audit.py

from training_hub.callbacks.base import TrainingHubCallback
from training_hub.callbacks.context import TrainingHubContext
import json
import logging

class AuditLoggingCallback(TrainingHubCallback):
    """
    Enterprise audit logging for compliance (SOC2, ISO27001, HIPAA).

    Works with ALL Training-Hub backends.
    """

    def __init__(self, logger_name="training_audit"):
        self.logger = logging.getLogger(logger_name)

    def on_train_begin(self, ctx: TrainingHubContext):
        self._log_event("training_started", {
            "backend": ctx.backend_name,
            "global_step": ctx.global_step,
            "max_steps": ctx.max_steps,
            "distributed": {
                "world_size": ctx.world_size,
                "process_index": ctx.process_index,
            }
        })

    def on_step_end(self, ctx: TrainingHubContext):
        if ctx.global_step % 100 == 0:
            self._log_event("training_progress", {
                "step": ctx.global_step,
                "loss": ctx.current_loss,
                "lr": ctx.learning_rate,
            })

    def on_save(self, ctx: TrainingHubContext):
        self._log_event("checkpoint_saved", {
            "step": ctx.global_step,
            "checkpoint_path": ctx.best_checkpoint_path,
        })

    def on_train_end(self, ctx: TrainingHubContext):
        self._log_event("training_completed", {
            "total_steps": ctx.global_step,
            "final_loss": ctx.current_loss,
            "backend": ctx.backend_name,
        })

    def _log_event(self, event_type: str, data: dict):
        log_entry = {
            "event": event_type,
            "timestamp": time.time(),
            **data
        }
        self.logger.info(json.dumps(log_entry))


# File: training_hub/callbacks/enterprise/cost.py

class CostTrackingCallback(TrainingHubCallback):
    """
    Track GPU costs, token counts, and FLOP estimates.

    Works across InstructLab, Mini-Trainer, Unsloth backends.
    """

    def __init__(self, gpu_cost_per_hour: float = 2.50):
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.start_time = None
        self.total_tokens = 0

    def on_train_begin(self, ctx: TrainingHubContext):
        self.start_time = time.time()

    def on_train_end(self, ctx: TrainingHubContext):
        duration_hours = (time.time() - self.start_time) / 3600
        total_cost = duration_hours * self.gpu_cost_per_hour * ctx.world_size

        print(f"Training Cost Summary:")
        print(f"  Duration: {duration_hours:.2f} hours")
        print(f"  GPUs: {ctx.world_size}")
        print(f"  Total Cost: ${total_cost:.2f}")


# File: training_hub/callbacks/enterprise/early_stopping.py

class EarlyStoppingCallback(TrainingHubCallback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

    def on_evaluate(self, ctx: TrainingHubContext, eval_metrics: dict):
        val_loss = eval_metrics.get('val_loss', eval_metrics.get('eval_loss', float('inf')))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            print(f"Early stopping: no improvement for {self.patience} evaluations")
            ctx.abort_training()
```

**5. Backend Integration**

```python
# File: training_hub/backends/unsloth.py

from training_hub.callbacks.adapters import HuggingFaceCallbackAdapter

class UnslothBackend(Backend):
    def execute_training(self, ..., callbacks=None, **kwargs):
        # Adapt TrainingHubCallback -> HuggingFace TrainerCallback
        adapted_callbacks = []
        if callbacks:
            for cb in callbacks:
                adapted_callbacks.append(HuggingFaceCallbackAdapter(cb))

        trainer = SFTTrainer(
            model=model,
            callbacks=adapted_callbacks,
            **kwargs
        )
        trainer.train()


# File: training_hub/backends/instructlab.py

from training_hub.callbacks.adapters import InstructLabCallbackAdapter
from training_hub.callbacks.handler import CallbackHandler

class InstructLabBackend(Backend):
    def execute_training(self, ..., callbacks=None, **kwargs):
        # Adapt TrainingHubCallback -> InstructLab hooks
        adapted_callbacks = []
        if callbacks:
            for cb in callbacks:
                adapted_callbacks.append(InstructLabCallbackAdapter(cb))

        # Use CallbackHandler from callback implementation RFE
        handler = CallbackHandler(adapted_callbacks)

        train_with_callbacks(
            model=model,
            train_loader=train_loader,
            callback_handler=handler,
            **kwargs
        )


# File: training_hub/backends/mini_trainer.py

from training_hub.callbacks.adapters import MiniTrainerCallbackAdapter
from training_hub.callbacks.handler import CallbackHandler

class MiniTrainerBackend(Backend):
    def execute_training(self, ..., callbacks=None, **kwargs):
        # Adapt TrainingHubCallback -> Mini-Trainer hooks
        adapted_callbacks = []
        if callbacks:
            for cb in callbacks:
                adapted_callbacks.append(MiniTrainerCallbackAdapter(cb))

        handler = CallbackHandler(adapted_callbacks)

        train_with_callbacks(
            model=model,
            data=data,
            callback_handler=handler,
            **kwargs
        )
```

---

## Business Justification

### Value Proposition

**For Data Scientists:**
- **Write once, run anywhere**: Callbacks work across ALL Training-Hub backends
- **No backend lock-in**: Switch from InstructLab → Mini-Trainer → Unsloth without rewriting code
- **Simplified learning**: Single callback interface instead of 3+ different systems
- **Algorithm independence**: Same callbacks work for SFT, OSFT, LoRA workflows

**For ML Platform Teams:**
- **Standardized governance**: Single audit logging implementation for entire organization
- **Consistent cost tracking**: Unified FinOps integration regardless of backend choice
- **Reduced maintenance**: 1 callback library vs N backend-specific solutions
- **Backend flexibility**: Can switch backends based on performance without breaking tooling

**For Enterprise Organizations:**
- **Regulatory compliance**: Unified audit trails across ALL Training-Hub training runs
- **Cost optimization**: Consistent GPU cost tracking across all backends
- **Risk reduction**: Standardized security event logging, bias detection, model governance
- **Faster innovation**: Teams can experiment with different backends without losing compliance tooling

### Competitive Differentiation

**vs. Standalone Training Frameworks:**
- HuggingFace, TRL, Axolotl have callbacks but are single-framework
- Training-Hub provides **multi-backend abstraction** with **unified callbacks**
- Only solution that spans InstructLab (enterprise), Mini-Trainer (continual learning), and Unsloth (efficiency)

**vs. Cloud ML Platforms:**
- AWS SageMaker: Multiple frameworks but NO unified callback abstraction
- Azure ML: No multi-framework callback abstraction
- GCP Vertex AI: No multi-framework callback abstraction
- **Training-Hub**: First to provide backend-agnostic callbacks

**Red Hat OpenShift AI Advantage:**
- First AI platform with multi-backend unified callbacks
- Spans enterprise (InstructLab), research (Mini-Trainer), and efficiency (Unsloth)
- Open source foundation enables community contributions

---

## Acceptance Criteria

### Functional Requirements

**FR1: Unified Callback Interface**
- [ ] `TrainingHubCallback` ABC with 15 standard lifecycle events
- [ ] `TrainingHubContext` with normalized state (20+ fields)
- [ ] Type hints and comprehensive docstrings
- [ ] Works consistently across all backends

**FR2: Backend Adapters**
- [ ] `HuggingFaceCallbackAdapter` for Unsloth
- [ ] `InstructLabCallbackAdapter` for InstructLab Training
- [ ] `MiniTrainerCallbackAdapter` for Mini-Trainer
- [ ] All adapters translate all 15 events correctly

**FR3: Training-Hub API**
- [ ] `sft()` function accepts `callbacks` parameter (TrainingHubCallback instances)
- [ ] `osft()` function accepts `callbacks` parameter (TrainingHubCallback instances)
- [ ] `lora_sft()` function accepts `callbacks` parameter (TrainingHubCallback instances)
- [ ] Automatic adapter selection based on backend

**FR4: Enterprise Callback Library**
- [ ] `AuditLoggingCallback`: JSON-formatted compliance logs
- [ ] `CostTrackingCallback`: GPU hours, token counts, cost estimates
- [ ] `QualityMonitorCallback`: LLM-as-judge evaluation
- [ ] `BiasDetectionCallback`: Fairness metrics
- [ ] `EarlyStoppingCallback`: Validation-based early stopping
- [ ] All callbacks tested across all backends

**FR5: Documentation & Examples**
- [ ] API documentation for TrainingHubCallback, TrainingHubContext
- [ ] Enterprise callback reference documentation
- [ ] Example notebooks for each backend
- [ ] Migration guide from backend-specific callbacks
- [ ] Architecture documentation

### Non-Functional Requirements

**NFR1: Performance**
- [ ] Adapter overhead < 1% (on top of callback overhead from implementation RFE)
- [ ] Memory overhead < 5MB per adapter
- [ ] No impact on distributed training scalability

**NFR2: Compatibility**
- [ ] Python 3.10+ support
- [ ] Works with all supported Training-Hub backends
- [ ] Does not break existing Training-Hub API
- [ ] Backward compatible with backend-specific callbacks

**NFR3: Reliability**
- [ ] 90%+ unit test coverage for unified callback system
- [ ] Integration tests for all backends with unified callbacks
- [ ] Cross-backend compatibility tests
- [ ] Adapter exception handling

**NFR4: Usability**
- [ ] Clear error messages when backend doesn't support specific features
- [ ] Helpful warnings for backend-specific limitations
- [ ] Easy-to-understand examples for each use case

### Success Metrics

**Adoption Metrics:**
- 50%+ of Training-Hub users adopt unified callbacks within 6 months
- 80%+ of callback users prefer unified callbacks over backend-specific
- 10+ community-contributed unified callbacks

**Technical Metrics:**
- Zero reported bugs in adapter layer
- < 1% adapter overhead across all backends
- 100% backward compatibility

**Business Metrics:**
- 20%+ increase in RHOAI Training-Hub usage
- 3+ customer case studies published
- 2+ conference talks featuring unified callbacks

---

## Dependencies

### Technical Dependencies

**Required:**
- Python >= 3.10
- Training-Hub (this repository)
- Callback implementation RFE completed (InstructLab, Mini-Trainer have callbacks)
- Unsloth >= 2024.11 (for Unsloth backend)
- HuggingFace Transformers >= 4.30

**Development:**
- pytest >= 7.0
- pytest-cov >= 4.0
- black, ruff (code formatting)

### Product Dependencies

**Blocking Dependencies:**
- **Callback Implementation RFE**: This RFE assumes InstructLab and Mini-Trainer have callback mechanisms (separate RFE)

**Optional RHOAI Integration (Future):**
- RHOAI Model Registry (for `ModelRegistryCallback`)
- RHOAI Data Science Pipelines (for callback-driven orchestration)

---

## Implementation Plan

### Phase 1: Core Unified Callback System (Weeks 1-2)

**Prerequisites:** Callback Implementation RFE must be complete

**Deliverables:**
- `TrainingHubCallback` ABC (15 events)
- `TrainingHubContext` state container
- Unit tests (90%+ coverage)

**Critical Files:**
```
training_hub/
├── callbacks/
│   ├── __init__.py
│   ├── base.py              # TrainingHubCallback ABC
│   └── context.py           # TrainingHubContext
```

**Milestones:**
- [ ] Unified callback interface defined and documented
- [ ] Context normalization designed
- [ ] Unit tests passing

### Phase 2: Backend Adapters (Weeks 3-5)

**Deliverables:**
- `HuggingFaceCallbackAdapter` (for Unsloth)
- `InstructLabCallbackAdapter` (for InstructLab)
- `MiniTrainerCallbackAdapter` (for Mini-Trainer)
- Integration tests with all backends

**Critical Files:**
```
training_hub/
├── callbacks/
│   └── adapters.py          # All adapters
├── backends/
│   ├── unsloth.py           # Updated for unified callbacks
│   ├── instructlab.py       # Updated for unified callbacks
│   └── mini_trainer.py      # Updated for unified callbacks
└── tests/
    ├── test_unified_callbacks_unsloth.py
    ├── test_unified_callbacks_instructlab.py
    └── test_unified_callbacks_mini_trainer.py
```

**Milestones:**
- [ ] Week 3: Unsloth adapter complete
- [ ] Week 4: InstructLab adapter complete
- [ ] Week 5: Mini-Trainer adapter complete
- [ ] All backends work with unified callbacks

### Phase 3: Enterprise Callback Library (Week 6-7)

**Deliverables:**
- 5 enterprise callbacks (audit, cost, quality, bias, early stopping)
- Cross-backend testing
- Performance benchmarks

**Critical Files:**
```
training_hub/
├── callbacks/
│   └── enterprise/
│       ├── audit.py
│       ├── cost.py
│       ├── quality.py
│       ├── bias.py
│       └── early_stopping.py
└── tests/
    └── test_enterprise_callbacks_all_backends.py
```

**Milestones:**
- [ ] All enterprise callbacks implemented
- [ ] Tested across all backends
- [ ] Performance benchmarks show < 1% adapter overhead

### Phase 4: Documentation & Examples (Week 8)

**Deliverables:**
- Comprehensive documentation
- Example notebooks for all backends
- Migration guide
- Architecture documentation

**Critical Files:**
```
training_hub/
├── examples/
│   ├── unified_callbacks_instructlab.ipynb
│   ├── unified_callbacks_mini_trainer.ipynb
│   ├── unified_callbacks_unsloth.ipynb
│   └── multi_backend_callbacks.ipynb
└── docs/
    ├── unified_callbacks_user_guide.md
    ├── unified_callback_api_reference.md
    ├── migration_to_unified_callbacks.md
    └── unified_callbacks_architecture.md
```

**Milestones:**
- [ ] Documentation complete and reviewed
- [ ] Example notebooks runnable on RHOAI
- [ ] Migration guide helps users transition

---

## Testing Requirements

### Unit Tests

**Coverage Target: 90%+**

**Unified Callback System Tests:**
- TrainingHubCallback interface (all 15 methods)
- TrainingHubContext state management
- Control method behavior (abort, checkpoint, evaluate)

**Adapter Tests:**
- HuggingFaceCallbackAdapter translation
- InstructLabCallbackAdapter translation
- MiniTrainerCallbackAdapter translation
- State normalization across adapters

### Integration Tests

**Cross-Backend Tests:**
1. Train with same unified callback on all backends
2. Verify all 15 events fire correctly on each backend
3. Test early stopping via `ctx.abort_training()` on all backends
4. Test checkpoint triggering on all backends

**Enterprise Callback Tests:**
1. AuditLoggingCallback works on all backends
2. CostTrackingCallback produces consistent results
3. EarlyStoppingCallback stops training correctly
4. All callbacks tested in distributed training (InstructLab)

### Performance Tests

**Benchmarking:**
- Measure adapter overhead (should be < 1%)
- Compare unified vs backend-specific callbacks
- Test with multiple callbacks (5+)

---

## Documentation Requirements

### User-Facing Documentation

**1. Unified Callbacks User Guide** (`docs/unified_callbacks_user_guide.md`)
- Why use unified callbacks?
- Writing your first TrainingHubCallback
- Using enterprise callbacks
- Backend-specific considerations
- Best practices

**2. Unified Callback API Reference** (`docs/unified_callback_api_reference.md`)
- `TrainingHubCallback` class documentation
- `TrainingHubContext` field reference
- Enterprise callback reference
- Adapter documentation

**3. Example Notebooks**
- `examples/unified_callbacks_instructlab.ipynb`
- `examples/unified_callbacks_mini_trainer.ipynb`
- `examples/unified_callbacks_unsloth.ipynb`
- `examples/multi_backend_callbacks.ipynb`: Same callback across all backends

**4. Migration Guide** (`docs/migration_to_unified_callbacks.md`)
- Converting backend-specific callbacks to unified callbacks
- Benefits of migration
- Step-by-step examples

### Developer Documentation

**1. Architecture Document** (`docs/unified_callbacks_architecture.md`)
- Why unified callbacks for Training-Hub?
- Adapter pattern explanation
- Design decisions and trade-offs
- Adding new backend support

---

## Risks and Mitigation

### Technical Risks

**Risk 1: Adapter Complexity**
- **Description**: Adapters become complex and hard to maintain
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**:
  - Keep adapter logic simple and focused
  - Comprehensive tests for each adapter
  - Clear documentation of translation logic

**Risk 2: Backend Limitations**
- **Description**: Some backends can't support all unified callback features
- **Likelihood**: Medium
- **Impact**: Low
- **Mitigation**:
  - Document backend-specific limitations
  - Graceful degradation for unsupported features
  - Clear error messages

**Risk 3: Performance Overhead**
- **Description**: Adapter layer adds > 1% overhead
- **Likelihood**: Low
- **Impact**: Medium
- **Mitigation**:
  - Profile adapter performance
  - Optimize hot paths
  - Lazy evaluation where possible

### Product Risks

**Risk 4: Low Adoption**
- **Description**: Users prefer backend-specific callbacks
- **Likelihood**: Low
- **Impact**: Medium
- **Mitigation**:
  - Ship compelling enterprise callbacks
  - Demonstrate value in examples
  - Include in Training-Hub getting started
  - Outreach to users

---

## Future Enhancements (Post-MVP)

### Phase 2 Enhancements

**1. Additional Backends**
- VERL integration
- Additional HuggingFace-based backends
- Custom backends

**2. Advanced Enterprise Callbacks**
- `DriftMonitorCallback`
- `FairnessDashboardCallback`
- `CarbonFootprintCallback`
- `ExplainabilityCallback`

**3. RHOAI Platform Integration**
- `ModelRegistryCallback`
- `PipelineCallback`
- `TrainingOperatorCallback`

**4. Callback Composition**
- Callback pipelines
- Conditional callbacks
- Callback groups

---

## References

### Design Documents

- **Callback Implementation RFE**: `./RFE_Callback_Implementation_Trainers.md` (prerequisite)
- **Original Unified Callback RFE**: `./RFE_Training_Hub_Unified_Callbacks.md` (comprehensive version)
- **Architecture Specification**: `./architecture.md`
- **TRL Callback Analysis**: `./trl_callback_analysis.md`

### Repositories

- **Training-Hub**: https://github.com/Red-Hat-AI-Innovation-Team/training_hub
- **InstructLab Training**: https://github.com/instructlab/training
- **Mini-Trainer**: https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer
- **Unsloth**: https://github.com/unslothai/unsloth

---

## Contact / Stakeholders

**Product Owner**: [TBD - RHOAI Product Manager]
**Engineering Lead**: [TBD - Red Hat AI Innovation Team Lead]
**Technical Lead**: Sameh Zaher (szaher@redhat.com)
**Documentation Lead**: [TBD - RHOAI Technical Writer]
**QE Lead**: [TBD - RHOAI QE Manager]

**Reviewers:**
- Red Hat AI Innovation Team (Training-Hub maintainers)
- Training-Hub Users

---

## Appendix A: Usage Examples

### Example 1: Multi-Backend Usage

```python
from training_hub import sft, osft, lora_sft
from training_hub.callbacks import AuditLoggingCallback, CostTrackingCallback

# Define callbacks once
audit = AuditLoggingCallback()
cost = CostTrackingCallback(gpu_cost_per_hour=2.50)

# Use with InstructLab Training (SFT)
sft(
    base_model="meta-llama/Llama-2-7b-hf",
    data_path="./data.jsonl",
    backend="instructlab-training",
    callbacks=[audit, cost],  # Works!
)

# Use with Mini-Trainer (OSFT)
osft(
    base_model="meta-llama/Llama-2-7b-hf",
    data_path="./data.jsonl",
    backend="mini-trainer",
    callbacks=[audit, cost],  # Same callbacks!
)

# Use with Unsloth (LoRA + SFT)
lora_sft(
    base_model="meta-llama/Llama-2-7b-hf",
    data_path="./data.jsonl",
    backend="unsloth",
    callbacks=[audit, cost],  # Still work!
)
```

### Example 2: Custom Unified Callback

```python
from training_hub.callbacks import TrainingHubCallback, TrainingHubContext

class CustomMetricCallback(TrainingHubCallback):
    """Track custom metric across all backends."""

    def __init__(self):
        self.metric_history = []

    def on_evaluate(self, ctx: TrainingHubContext, eval_metrics: dict):
        # Works regardless of backend!
        custom_metric = eval_metrics.get('custom_score', 0.0)
        self.metric_history.append(custom_metric)

        print(f"[{ctx.backend_name}] Custom metric: {custom_metric:.4f}")

# Use with any backend
from training_hub import sft

sft(
    base_model="meta-llama/Llama-2-7b-hf",
    data_path="./data.jsonl",
    backend="instructlab-training",  # or "mini-trainer" or "unsloth"
    callbacks=[CustomMetricCallback()],
)
```

---

## END OF RFE
