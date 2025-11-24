# TRL Callback System - Visual Summary

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   HuggingFace Transformers                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  transformers.Trainer                 │  │
│  │                                                       │  │
│  │  • Standard training loop                            │  │
│  │  • Callback system (on_*, callback_handler)          │  │
│  │  • TrainerState / TrainerControl                     │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
└───────────────────────┼─────────────────────────────────────┘
                        │ inherits from
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                         TRL Library                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              trl.trainer.BaseTrainer                  │  │
│  │                                                       │  │
│  │  • Inherits Trainer callback system (unchanged)      │  │
│  │  • Adds create_model_card() customization           │  │
│  │  • No new callback events                           │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│           ┌───────────┼───────────┐                        │
│           │           │           │                         │
│           ▼           ▼           ▼                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │   DPO    │ │   SFT    │ │   KTO    │  ... (6 trainers) │
│  │ Trainer  │ │ Trainer  │ │ Trainer  │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
│                                                             │
│  All trainers:                                             │
│  • Inherit full callback system                           │
│  • Add domain callbacks conditionally                     │
│  • Keep 100% Transformers compatibility                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Callback Event Flow

```
Training Lifecycle                TRL Callbacks Using Event
════════════════════              ═════════════════════════

on_init_end                       (not used by TRL callbacks)
    │
    ▼
on_train_begin ─────────────────► WeaveCallback.on_train_begin()
    │                             BEMACallback.on_train_begin()
    │                             WinRateCallback.on_train_begin()
    │
    ▼
╔═══════════════════════════╗
║      Training Loop        ║
║   ┌───────────────────┐   ║
║   │  For each epoch   │   ║
║   │  ┌─────────────┐  │   ║
║   │  │ For each    │  │   ║
║   │  │ step        │  │   ║
║   │  │             │  │   ║
║   │  │ on_step_    │  │   ║
║   │  │ begin       │  │   ║
║   │  │             │  │   ║
║   │  │ on_substep_ │  │   ║
║   │  │ end         │  │   ║
║   │  │             │  │   ║
║   │  │ on_step_    │────► SyncRefModelCallback.on_step_end()
║   │  │ end         │  │   RichProgressCallback.on_step_end()
║   │  │             │  │   LogCompletionsCallback.on_step_end()
║   │  │             │  │   BEMACallback.on_step_end()
║   │  └─────────────┘  │
║   │                   │
║   │  on_epoch_end     │
║   │                   │
║   │  on_evaluate ─────┼──► WinRateCallback.on_evaluate()
║   │                   │    LogCompletionsCallback.on_evaluate()
║   │                   │    WeaveCallback.on_evaluate()
║   │                   │
║   │  on_log ──────────┼──► RichProgressCallback.on_log()
║   │                   │
║   │  on_save ─────────┼──► MergeModelCallback.on_save()
║   │                   │
║   └───────────────────┘
║
╚═══════════════════════════╝
    │
    ▼
on_train_end ────────────────────► MergeModelCallback.on_train_end()
                                   BEMACallback.on_train_end()
```

## TRL Callback Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                    TRL Custom Callbacks                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────┐
│  RLHF-Specific (4)      │
├─────────────────────────┴──────────────────────────────────┐
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ SyncRefModelCallback                              │    │
│  │ ─────────────────────────────────────────────     │    │
│  │ • Syncs reference model with training model       │    │
│  │ • Uses EMA (exponential moving average)           │    │
│  │ • Essential for DPO, KTO, RLOO                    │    │
│  │ • Handles DeepSpeed models                        │    │
│  │                                                    │    │
│  │ Hooks: on_step_end                                │    │
│  │ Config: ref_model_sync_steps, ref_model_mixup_alpha │  │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ WinRateCallback                                   │    │
│  │ ─────────────────────────────────────────────     │    │
│  │ • LLM-as-judge evaluation                         │    │
│  │ • Compares model vs reference completions         │    │
│  │ • Tracks win rate over training                   │    │
│  │ • Supports soft (probability) judging             │    │
│  │                                                    │    │
│  │ Hooks: on_train_begin, on_evaluate                │    │
│  │ Config: judge, num_prompts, shuffle_order         │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ LogCompletionsCallback                            │    │
│  │ ─────────────────────────────────────────────     │    │
│  │ • Logs model generations                          │    │
│  │ • Integrates with wandb, comet                    │    │
│  │ • Periodic logging during training                │    │
│  │ • Evaluation-time logging                         │    │
│  │                                                    │    │
│  │ Hooks: on_step_end, on_evaluate                   │    │
│  │ Config: num_prompts, freq                         │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ WeaveCallback                                     │    │
│  │ ─────────────────────────────────────────────     │    │
│  │ • W&B Weave integration                           │    │
│  │ • Logs predictions with optional scores           │    │
│  │ • Custom scorer functions support                 │    │
│  │ • Evaluation tracking                             │    │
│  │                                                    │    │
│  │ Hooks: on_train_begin, on_evaluate                │    │
│  │ Config: project_name, scorers, num_prompts        │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────┐
│  Training Enhancement (3)│
├─────────────────────────┴──────────────────────────────────┐
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ BEMACallback                                      │    │
│  │ ─────────────────────────────────────────────     │    │
│  │ • Bias-Corrected Exponential Moving Average       │    │
│  │ • Improves model generalization                   │    │
│  │ • Adaptive weight averaging                       │    │
│  │ • Saves averaged model separately                 │    │
│  │                                                    │    │
│  │ Hooks: on_train_begin, on_step_end, on_train_end  │    │
│  │ Config: bema_alpha, bema_start_step, bema_warmup  │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ MergeModelCallback                                │    │
│  │ ─────────────────────────────────────────────     │    │
│  │ • Merges model checkpoints using mergekit         │    │
│  │ • Automatic merging at save points                │    │
│  │ • Optional hub upload                             │    │
│  │ • Configurable merge strategies                   │    │
│  │                                                    │    │
│  │ Hooks: on_save, on_train_end                      │    │
│  │ Config: merge_config, push_to_hub                 │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ RichProgressCallback                              │    │
│  │ ─────────────────────────────────────────────     │    │
│  │ • Enhanced progress visualization                 │    │
│  │ • Rich console formatting                         │    │
│  │ • Training & evaluation progress bars             │    │
│  │ • Formatted metric tables                         │    │
│  │                                                    │    │
│  │ Hooks: on_train_begin, on_step_end, on_log,       │    │
│  │        on_prediction_step, on_evaluate, on_train_end│  │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Callback Hook Usage Matrix

```
┌──────────────────────┬─────────────────────────────────────────┐
│                      │           Callback Hooks                │
│   TRL Callbacks      ├──────┬──────┬──────┬──────┬──────┬─────┤
│                      │train │step  │eval  │log   │save  │train│
│                      │begin │end   │      │      │      │end  │
├──────────────────────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ SyncRefModel         │      │  ✓   │      │      │      │     │
├──────────────────────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ RichProgress         │  ✓   │  ✓   │  ✓   │  ✓   │      │  ✓  │
├──────────────────────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ WinRate              │  ✓   │      │  ✓   │      │      │     │
├──────────────────────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ LogCompletions       │      │  ✓   │  ✓   │      │      │     │
├──────────────────────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ Weave                │  ✓   │      │  ✓   │      │      │     │
├──────────────────────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ MergeModel           │      │      │      │      │  ✓   │  ✓  │
├──────────────────────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ BEMA                 │  ✓   │  ✓   │      │      │      │  ✓  │
└──────────────────────┴──────┴──────┴──────┴──────┴──────┴─────┘

Legend:
  ✓ = Callback implements this hook
```

## Trainer Hierarchy

```
transformers.Trainer (Standard HF)
        │
        │ No modifications to callback system
        │
        ▼
trl.trainer.BaseTrainer
        │
        │ • Inherits callback system as-is
        │ • No new callback events
        │ • Only adds create_model_card()
        │
        ├─────────────────┬─────────────┬─────────────┐
        │                 │             │             │
        ▼                 ▼             ▼             ▼
    DPOTrainer      SFTTrainer    KTOTrainer    RLOOTrainer
        │
        │ • Accepts callbacks parameter
        │ • Conditionally adds SyncRefModelCallback
        │ • Uses standard callback lifecycle
        │
        └─► Can use ANY Transformers callback ✓
            Can use ANY TRL callback ✓
            100% compatible ✓
```

## Callback Addition Patterns

```
┌─────────────────────────────────────────────────────────────┐
│  Pattern 1: Automatic Callback (Conditional)                │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  config = DPOConfig(                                        │
│      sync_ref_model=True,  ◄── Config flag                 │
│      ref_model_sync_steps=100,                              │
│  )                                                          │
│                                                              │
│  trainer = DPOTrainer(                                      │
│      model=model,                                           │
│      ref_model=ref_model,                                   │
│      args=config,                                           │
│  )                                                          │
│                                                              │
│  # SyncRefModelCallback automatically added! ✓              │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Pattern 2: Manual Callback Addition                        │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  trainer = DPOTrainer(...)                                  │
│                                                              │
│  trainer.add_callback(WinRateCallback(                      │
│      judge=judge_model,                                     │
│      trainer=trainer,                                       │
│      num_prompts=64,                                        │
│  ))                                                         │
│                                                              │
│  trainer.add_callback(LogCompletionsCallback(               │
│      trainer=trainer,                                       │
│      freq=100,                                              │
│  ))                                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Pattern 3: Initialization with Callbacks                   │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  from transformers import EarlyStoppingCallback             │
│                                                              │
│  callbacks = [                                              │
│      EarlyStoppingCallback(patience=3),                     │
│      TensorBoardCallback(),                                 │
│  ]                                                          │
│                                                              │
│  trainer = DPOTrainer(                                      │
│      model=model,                                           │
│      callbacks=callbacks,  ◄── Pass at init                │
│      ...                                                    │
│  )                                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow in Key Callbacks

### SyncRefModelCallback
```
Training Step
     │
     ▼
on_step_end() fired
     │
     ▼
Check: step % sync_steps == 0?
     │
     ├─► No: return
     │
     └─► Yes:
          │
          ├─► Unwrap model (if using accelerator)
          │
          ├─► For each parameter:
          │       ref_param = ref_param * (1-α) + model_param * α
          │
          └─► Reference model updated ✓
```

### WinRateCallback
```
on_train_begin()
     │
     ├─► Sample prompts from eval_dataset
     │
     └─► Generate reference completions
         (cache for later comparison)

...training happens...

on_evaluate()
     │
     ├─► Generate current model completions
     │
     ├─► For each prompt:
     │       │
     │       ├─► Shuffle order (reduce bias)
     │       │
     │       ├─► Judge: Which is better?
     │       │       • Hard judge: A or B
     │       │       • Soft judge: probability
     │       │
     │       └─► Count wins
     │
     ├─► Compute win_rate = wins / total
     │
     └─► Log to trainer, wandb, comet
```

### BEMACallback
```
on_train_begin()
     │
     └─► Create running_model = copy.deepcopy(model)

on_step_end() (for each step)
     │
     ├─► step < start_step? ──Yes──► return
     │
     ├─► step >= end_step? ──Yes──► return
     │
     └─► No (in BEMA range):
          │
          ├─► Compute alpha:
          │       • If warmup: α = 1 / step
          │       • Else: α = bema_alpha
          │
          └─► Update running average:
                  running_param = running_param * (1-α) + model_param * α

on_train_end()
     │
     └─► Save running_model to output_dir/bema_model/
```

## Compatibility Summary

```
┌─────────────────────────────────────────────────────────────┐
│                 Compatibility Matrix                         │
└─────────────────────────────────────────────────────────────┘

Standard Transformers Callbacks → TRL Trainers
───────────────────────────────────────────────
✓ EarlyStoppingCallback      ✓ Works perfectly
✓ TensorBoardCallback         ✓ Works perfectly
✓ WandbCallback               ✓ Works perfectly
✓ MLflowCallback              ✓ Works perfectly
✓ CometCallback               ✓ Works perfectly
✓ PrinterCallback             ✓ Works perfectly
✓ ProgressCallback            ✓ Works perfectly
✓ Custom TrainerCallback      ✓ Works perfectly

Result: 100% compatibility ✓


TRL Callbacks → Standard Transformers Trainer
───────────────────────────────────────────────
⚠️  SyncRefModelCallback      Needs ref_model
⚠️  WinRateCallback           Needs ref_model, generation_config
⚠️  LogCompletionsCallback    Needs generation_config
⚠️  WeaveCallback             Needs generation_config
✓ BEMACallback               ✓ Works (pure EMA)
✓ RichProgressCallback       ✓ Works (pure visualization)
✓ MergeModelCallback         ✓ Works (checkpoint merging)

Result: Partial compatibility (some need TRL features)


TRL Trainers → Standard Transformers Ecosystem
───────────────────────────────────────────────
✓ Can replace Trainer          ✓ Drop-in replacement
✓ Same callback interface      ✓ No changes needed
✓ Same state/control objects   ✓ No extensions
✓ Same event lifecycle         ✓ No custom events

Result: Full LSP compliance ✓
```

## Key Design Decisions

```
┌─────────────────────────────────────────────────────────────┐
│  Decision 1: NO Custom Callback Events                      │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  ❌ Could have done:                                        │
│     • on_generation_step()                                  │
│     • on_reward_computation()                               │
│     • on_preference_update()                                │
│                                                              │
│  ✓ Instead did:                                             │
│     • Used standard on_evaluate() for generation            │
│     • Used standard on_step_end() for model sync            │
│     • Used standard on_log() for metrics                    │
│                                                              │
│  Result: 100% Transformers compatibility ✓                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Decision 2: NO State/Control Extensions                    │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  ❌ Could have done:                                        │
│     • TrainerState.ref_model_sync_count                     │
│     • TrainerState.current_win_rate                         │
│     • TrainerControl.should_sync_ref_model                  │
│                                                              │
│  ✓ Instead did:                                             │
│     • Callbacks maintain internal state                     │
│     • Use existing state.global_step for timing             │
│     • Use standard control flags                            │
│                                                              │
│  Result: No breaking changes to core objects ✓              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Decision 3: Conditional Callback Addition                  │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  ✓ Pattern:                                                 │
│     if args.sync_ref_model:                                 │
│         self.add_callback(SyncRefModelCallback(...))        │
│                                                              │
│  Benefits:                                                  │
│     • Only add callbacks when needed                        │
│     • Config-driven behavior                                │
│     • No overhead when features disabled                    │
│     • Clean separation of concerns                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Decision 4: Trainer-Aware Callbacks                        │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  ✓ Pattern:                                                 │
│     callback = WinRateCallback(                             │
│         judge=judge,                                        │
│         trainer=trainer,  # Pass trainer reference          │
│         ...                                                 │
│     )                                                       │
│                                                              │
│  Enables:                                                   │
│     • Access to trainer.eval_dataset                        │
│     • Access to trainer.ref_model                           │
│     • Access to trainer.generation_config                   │
│     • Ability to call trainer.log()                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Summary: What Makes TRL Callbacks Special

```
NOT Special:
━━━━━━━━━━━━
✗ Custom callback events
✗ Extended state objects
✗ Extended control objects
✗ Modified callback lifecycle
✗ New callback interface

IS Special:
━━━━━━━━━━━━
✓ RLHF-focused implementations
✓ Reference model management
✓ Generation quality evaluation
✓ LLM-as-judge integration
✓ Advanced weight averaging
✓ Model merging automation
✓ Rich experiment tracking

The Magic:
━━━━━━━━━━━━
TRL achieves rich RLHF functionality using ONLY
standard Transformers callback hooks through:
  • Clever callback implementations
  • Internal state management
  • Trainer-aware design
  • Conditional addition patterns

Result: Full compatibility + Rich features ✓
```
