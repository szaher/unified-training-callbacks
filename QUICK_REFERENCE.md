# TRL Callback System - Quick Reference

## 🎯 Key Findings (TL;DR)

1. **NO custom callback events** - Uses only standard Transformers hooks
2. **NO state/control extensions** - Uses standard TrainerState/TrainerControl
3. **100% compatibility** - All standard callbacks work with TRL trainers
4. **7 custom callbacks** - RLHF-focused implementations
5. **Conditional addition** - Callbacks added based on config

## 📋 TRL Callbacks Cheat Sheet

### SyncRefModelCallback
```python
# Auto-added when sync_ref_model=True
config = DPOConfig(sync_ref_model=True, ref_model_sync_steps=100)
trainer = DPOTrainer(model=model, ref_model=ref_model, args=config)

# Or manual:
trainer.add_callback(SyncRefModelCallback(ref_model=ref_model, accelerator=accelerator))
```
**Purpose**: Syncs reference model with training model
**Hooks**: `on_step_end`
**Used by**: DPOTrainer, KTOTrainer, RLOOTrainer

### WinRateCallback
```python
callback = WinRateCallback(
    judge=judge_model,
    trainer=trainer,
    num_prompts=64,
    shuffle_order=True,
    use_soft_judge=False,
)
trainer.add_callback(callback)
```
**Purpose**: LLM-as-judge evaluation, tracks win rates
**Hooks**: `on_train_begin`, `on_evaluate`
**Logs to**: trainer, wandb, comet

### LogCompletionsCallback
```python
callback = LogCompletionsCallback(
    trainer=trainer,
    num_prompts=8,
    freq=100,
)
trainer.add_callback(callback)
```
**Purpose**: Logs model generations
**Hooks**: `on_step_end`, `on_evaluate`
**Logs to**: wandb, comet

### WeaveCallback
```python
callback = WeaveCallback(
    trainer=trainer,
    project_name="my-project",
    scorers={"coherence": coherence_fn, "relevance": relevance_fn},
    num_prompts=50,
)
trainer.add_callback(callback)
```
**Purpose**: W&B Weave integration
**Hooks**: `on_train_begin`, `on_evaluate`
**Logs to**: W&B Weave

### BEMACallback
```python
callback = BEMACallback(
    bema_start_step=1000,
    bema_alpha=0.9,
    bema_warmup_steps=100,
)
trainer.add_callback(callback)
```
**Purpose**: Bias-corrected exponential moving average
**Hooks**: `on_train_begin`, `on_step_end`, `on_train_end`
**Saves**: Averaged model to `output_dir/bema_model/`

### MergeModelCallback
```python
callback = MergeModelCallback(
    merge_config=merge_config,
    push_to_hub=True,
)
trainer.add_callback(callback)
```
**Purpose**: Merges model checkpoints
**Hooks**: `on_save`, `on_train_end`
**Requires**: mergekit

### RichProgressCallback
```python
callback = RichProgressCallback()
trainer.add_callback(callback)
```
**Purpose**: Enhanced progress visualization
**Hooks**: `on_train_begin`, `on_step_end`, `on_log`, `on_evaluate`, `on_train_end`
**Requires**: rich library

## 🔌 Standard Callback Hooks

All TRL callbacks use **only** these standard hooks:

| Hook | When Called | Used By |
|------|-------------|---------|
| `on_train_begin` | Start of training | WinRateCallback, BEMACallback, WeaveCallback |
| `on_step_end` | After each training step | SyncRefModelCallback, BEMACallback, LogCompletionsCallback, RichProgressCallback |
| `on_evaluate` | During evaluation | WinRateCallback, LogCompletionsCallback, WeaveCallback |
| `on_log` | When metrics are logged | RichProgressCallback |
| `on_save` | When checkpoint is saved | MergeModelCallback |
| `on_train_end` | End of training | BEMACallback, MergeModelCallback |
| `on_prediction_step` | During prediction | RichProgressCallback |

**No custom events!** ✓

## 🏗️ Trainer Inheritance

```
Trainer (Transformers)
  └── BaseTrainer (TRL)
      ├── DPOTrainer
      ├── SFTTrainer
      ├── KTOTrainer
      ├── RLOOTrainer
      ├── ORPOTrainer
      └── RewardTrainer
```

**Exception**: PPOTrainer (doesn't inherit from Trainer)

## ✅ Compatibility

### Standard → TRL
```python
from transformers import EarlyStoppingCallback, TensorBoardCallback
from trl import DPOTrainer

# All standard callbacks work!
trainer = DPOTrainer(
    callbacks=[
        EarlyStoppingCallback(patience=3),
        TensorBoardCallback(),
    ]
)
```

### TRL → Standard
```python
from trl.trainer.callbacks import BEMACallback, RichProgressCallback
from transformers import Trainer

# Some TRL callbacks work with standard Trainer
trainer = Trainer(
    callbacks=[
        BEMACallback(),  # ✓ Works
        RichProgressCallback(),  # ✓ Works
        # SyncRefModelCallback(),  # ✗ Needs ref_model
    ]
)
```

## 💡 Usage Patterns

### Pattern 1: Auto-Addition
```python
config = DPOConfig(sync_ref_model=True)
trainer = DPOTrainer(args=config, ...)
# SyncRefModelCallback auto-added!
```

### Pattern 2: Manual Addition
```python
trainer = DPOTrainer(...)
trainer.add_callback(WinRateCallback(judge=judge, trainer=trainer))
```

### Pattern 3: Init with Callbacks
```python
callbacks = [EarlyStoppingCallback(), WinRateCallback(...)]
trainer = DPOTrainer(callbacks=callbacks, ...)
```

### Pattern 4: Multiple Callbacks
```python
trainer = DPOTrainer(...)
trainer.add_callback(WinRateCallback(judge=judge, trainer=trainer))
trainer.add_callback(LogCompletionsCallback(trainer=trainer, freq=100))
trainer.add_callback(BEMACallback(bema_alpha=0.9))
trainer.add_callback(EarlyStoppingCallback(patience=3))
```

## 🎨 Common Combinations

### RLHF Training with Full Observability
```python
from trl import DPOTrainer, DPOConfig
from trl.trainer.callbacks import WinRateCallback, LogCompletionsCallback
from transformers import EarlyStoppingCallback

config = DPOConfig(
    sync_ref_model=True,
    ref_model_sync_steps=100,
    evaluation_strategy="steps",
    eval_steps=200,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Add evaluation callbacks
trainer.add_callback(WinRateCallback(
    judge=judge_model,
    trainer=trainer,
    num_prompts=64,
))

trainer.add_callback(LogCompletionsCallback(
    trainer=trainer,
    num_prompts=16,
    freq=50,
))

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

trainer.train()
```

### Training with BEMA
```python
from trl import SFTTrainer
from trl.trainer.callbacks import BEMACallback, RichProgressCallback

trainer = SFTTrainer(...)

trainer.add_callback(BEMACallback(
    bema_start_step=500,
    bema_alpha=0.9,
    bema_warmup_steps=100,
))

trainer.add_callback(RichProgressCallback())

trainer.train()
# BEMA model saved to output_dir/bema_model/
```

## 🔍 Debugging

### Check Active Callbacks
```python
# List all callbacks
for callback in trainer.callback_handler.callbacks:
    print(type(callback).__name__)
```

### Remove Callback
```python
trainer.remove_callback(WinRateCallback)
```

### Pop Callback
```python
callback = trainer.pop_callback(WinRateCallback)
```

## 📊 Callback State

### Callbacks Maintain Internal State
```python
class WinRateCallback(TrainerCallback):
    def __init__(self, ...):
        self.reference_completions = None  # Internal state
        self.prompts = [...]  # Internal state

    def on_train_begin(self, ...):
        # Initialize state
        self.reference_completions = self._generate(...)
```

### No TrainerState Extensions
```python
# ✗ TRL does NOT do this
state.ref_model_sync_count  # Doesn't exist
state.current_win_rate  # Doesn't exist

# ✓ Instead callbacks use:
state.global_step  # Standard attribute
state.epoch  # Standard attribute
```

## 🚀 Quick Start

### Minimal Example
```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    output_dir="./output",
    sync_ref_model=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_dataset,
)

trainer.train()
```

### With Custom Callbacks
```python
from trl import DPOTrainer
from trl.trainer.callbacks import WinRateCallback, LogCompletionsCallback

trainer = DPOTrainer(...)

trainer.add_callback(WinRateCallback(judge=judge, trainer=trainer))
trainer.add_callback(LogCompletionsCallback(trainer=trainer))

trainer.train()
```

## 📚 More Information

- **Full Analysis**: See `trl_callback_analysis.md`
- **Code Examples**: See `trl_callback_code_examples.md`
- **Comparison**: See `trl_vs_transformers_callbacks.md`
- **Visual Guide**: See `trl_callback_visual_summary.md`

## ⚡ Key Takeaways

1. **Standard events are enough** - No need for custom callback events
2. **Callbacks as composition** - Add features via callbacks, not event system changes
3. **Maintain compatibility** - Full Transformers ecosystem compatibility
4. **Internal state is fine** - Callbacks can maintain their own state
5. **Config-driven** - Add callbacks conditionally based on config
6. **Trainer-aware** - Callbacks can reference trainer for access to models/datasets

## 🎓 Design Lessons

For framework designers:

1. **Don't extend event systems unnecessarily** - Standard events are powerful
2. **Composition > Extension** - Add features via callback composition
3. **Compatibility matters** - Maintain standard interfaces for ecosystem benefits
4. **Domain-specific callbacks > domain-specific events** - Specialize implementations, not interfaces
5. **Keep core simple** - Add complexity in callbacks, not core trainer
