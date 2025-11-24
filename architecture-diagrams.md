# Unified Callback System: Architecture Diagrams & Visualizations

**Version:** 1.0
**Date:** 2025-01-24

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Component Layer Diagram](#2-component-layer-diagram)
3. [Callback Execution Flow](#3-callback-execution-flow)
4. [State Translation Pipeline](#4-state-translation-pipeline)
5. [Event Lifecycle Diagram](#5-event-lifecycle-diagram)
6. [Backend Injection Patterns](#6-backend-injection-patterns)
7. [Adapter Pattern Deep Dive](#7-adapter-pattern-deep-dive)
8. [Sequence Diagrams](#8-sequence-diagrams)
9. [Data Flow Diagrams](#9-data-flow-diagrams)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                           USER APPLICATION LAYER                            │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  class EarlyStoppingCallback(UnifiedCallback):                     │    │
│  │      def __init__(self, patience=3):                               │    │
│  │          self.patience = patience                                  │    │
│  │          self.best_loss = float('inf')                             │    │
│  │          self.wait = 0                                             │    │
│  │                                                                     │    │
│  │      def on_evaluate(self, ctx: UnifiedContext, metrics: dict):   │    │
│  │          loss = metrics.get('eval_loss', float('inf'))            │    │
│  │          if loss < self.best_loss:                                 │    │
│  │              self.best_loss = loss                                 │    │
│  │              self.wait = 0                                         │    │
│  │          else:                                                      │    │
│  │              self.wait += 1                                        │    │
│  │              if self.wait >= self.patience:                        │    │
│  │                  ctx.abort_training()  ◄────── Simple API          │    │
│  │                                                                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Unified Interface
                                      │ (UnifiedCallback ABC)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         ABSTRACTION LAYER (Your Library)                    │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │  UnifiedCallback     │    │  UnifiedContext      │                      │
│  │  (Abstract Base)     │    │  (Data Class)        │                      │
│  ├──────────────────────┤    ├──────────────────────┤                      │
│  │ + on_init_end()      │    │ + global_step        │                      │
│  │ + on_train_begin()   │    │ + epoch              │                      │
│  │ + on_train_end()     │    │ + current_metrics    │                      │
│  │ + on_epoch_begin()   │    │ + best_metric        │                      │
│  │ + on_epoch_end()     │    │ + model              │                      │
│  │ + on_step_begin()    │    │ + tokenizer          │                      │
│  │ + on_step_end()      │    │ + optimizer          │                      │
│  │ + on_substep_end()   │    │ ─────────────────────│                      │
│  │ + on_log()           │    │ + abort_training()   │                      │
│  │ + on_save()          │    │ + trigger_checkpoint()│                     │
│  │ + on_evaluate()      │    │ + trigger_evaluation()│                     │
│  │ + ... (15 methods)   │    │ + get_metric(key)    │                      │
│  └──────────────────────┘    └──────────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Translation Layer
                                      │ (Adapter Pattern)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                           ADAPTER LAYER (Translation)                       │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │        HuggingFaceAdapterCallback(TrainerCallback)                 │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │  - unified_callback: UnifiedCallback                               │    │
│  │                                                                     │    │
│  │  [State Translation]                                               │    │
│  │  def _state_to_context(args, state, control, **kwargs):           │    │
│  │      return UnifiedContext(                                        │    │
│  │          global_step = state.global_step,                          │    │
│  │          epoch = state.epoch,                                      │    │
│  │          current_metrics = state.log_history[-1],                  │    │
│  │          model = kwargs.get('model'),                              │    │
│  │          ...                                                        │    │
│  │      )                                                              │    │
│  │                                                                     │    │
│  │  [Control Translation]                                             │    │
│  │  def _context_to_control(ctx, control):                           │    │
│  │      if ctx._should_stop_training:                                 │    │
│  │          control.should_training_stop = True                       │    │
│  │      if ctx._should_save_now:                                      │    │
│  │          control.should_save = True                                │    │
│  │      return control                                                │    │
│  │                                                                     │    │
│  │  [Event Forwarding - Example]                                      │    │
│  │  def on_evaluate(self, args, state, control, metrics, **kwargs):  │    │
│  │      ctx = self._state_to_context(args, state, control, **kwargs) │    │
│  │      self.unified_callback.on_evaluate(ctx, metrics)  ◄─ Invoke   │    │
│  │      return self._context_to_control(ctx, control)                │    │
│  │                                                                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │        UnifiedCallbackPlugin(BasePlugin) [Axolotl Only]            │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │  def add_callbacks_pre_trainer(cfg, model):                        │    │
│  │      return [HuggingFaceAdapterCallback(unified_callback)]         │    │
│  │                                                                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Backend-Specific API
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│                      │  │                      │  │                      │
│   TRANSFORMERS       │  │        TRL           │  │      AXOLOTL         │
│                      │  │                      │  │                      │
│  Trainer(            │  │  SFTTrainer(         │  │  TrainerBuilder      │
│    callbacks=[       │  │    callbacks=[       │  │    .get_callbacks()  │
│      adapter         │  │      adapter         │  │      → [adapter]     │
│    ]                 │  │    ]                 │  │                      │
│  )                   │  │  )                   │  │  PluginManager       │
│                      │  │                      │  │    .register(plugin) │
│                      │  │                      │  │                      │
│  ┌────────────────┐ │  │  ┌────────────────┐ │  │  ┌────────────────┐  │
│  │ TrainerCallback│ │  │  │ TrainerCallback│ │  │  │ TrainerCallback│  │
│  │ Interface      │ │  │  │ Interface      │ │  │  │ Interface      │  │
│  │                │ │  │  │ (inherited)    │ │  │  │ (inherited)    │  │
│  │ • on_step_end  │ │  │  │ • on_step_end  │ │  │  │ • on_step_end  │  │
│  │ • on_evaluate  │ │  │  │ • on_evaluate  │ │  │  │ • on_evaluate  │  │
│  │ • ... (15)     │ │  │  │ • ... (15)     │ │  │  │ • ... (15)     │  │
│  └────────────────┘ │  │  └────────────────┘ │  │  └────────────────┘  │
│                      │  │                      │  │                      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
            │                         │                         │
            │                         │                         │
            └─────────────────────────┴─────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │                          │
                        │    UNSLOTH (Wrapper)     │
                        │                          │
                        │  FastLanguageModel +     │
                        │  SFTTrainer (from TRL)   │
                        │                          │
                        │  100% HF Compatible      │
                        │  No special handling     │
                        │                          │
                        └──────────────────────────┘
```

### 1.2 Key Design Principles

```
┌────────────────────────────────────────────────────────────────┐
│  1. SINGLE RESPONSIBILITY                                      │
│     ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│     │  User Layer  │   │Adapter Layer │   │Backend Layer │   │
│     │              │   │              │   │              │   │
│     │  Business    │   │ Translation  │   │  Training    │   │
│     │  Logic       │   │ Logic        │   │  Execution   │   │
│     └──────────────┘   └──────────────┘   └──────────────┘   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  2. ADAPTER PATTERN (Gang of Four)                             │
│                                                                 │
│     ┌──────────────────────┐                                   │
│     │  UnifiedCallback     │  Target Interface (What user      │
│     │  (Target)            │  expects)                         │
│     └──────────────────────┘                                   │
│              △                                                  │
│              │ implements                                       │
│              │                                                  │
│     ┌──────────────────────┐                                   │
│     │  UserCallback        │  Client (User's code)             │
│     └──────────────────────┘                                   │
│                                                                 │
│     ┌──────────────────────────────────────────┐               │
│     │  HuggingFaceAdapterCallback              │  Adapter      │
│     │  (Adapter)                               │  (Translates) │
│     │  - wraps: UnifiedCallback                │               │
│     │  - implements: TrainerCallback           │               │
│     └──────────────────────────────────────────┘               │
│              │                                                  │
│              │ uses                                             │
│              ▼                                                  │
│     ┌──────────────────────┐                                   │
│     │  TrainerCallback     │  Adaptee (Backend's interface)    │
│     │  (Adaptee)           │                                   │
│     └──────────────────────┘                                   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  3. FACADE PATTERN (Simplified Interface)                      │
│                                                                 │
│  Complex Backend APIs          Simple Unified API              │
│  ┌──────────────────┐          ┌──────────────────┐           │
│  │ TrainerState     │          │ UnifiedContext   │           │
│  │ • 20 attributes  │          │ • 15 core fields │           │
│  │ • log_history[]  │   ───>   │ • current_metrics│           │
│  │ • stateful_cbs{} │          │ • abort_training()│          │
│  │ • trial_params{} │          │ • get_metric()   │           │
│  └──────────────────┘          └──────────────────┘           │
│                                                                 │
│  ┌──────────────────┐          ┌──────────────────┐           │
│  │ TrainerControl   │          │ UnifiedContext   │           │
│  │ • 5 bool flags   │   ───>   │ • abort_training()│          │
│  │ • auto-reset     │          │ • trigger_save() │           │
│  │ • threading      │          │ (no flag mgmt)   │           │
│  └──────────────────┘          └──────────────────┘           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  4. DEPENDENCY INVERSION                                        │
│                                                                 │
│  High-level (User Code)                                         │
│          │                                                      │
│          │ depends on                                           │
│          ▼                                                      │
│  ┌──────────────────┐                                          │
│  │ UnifiedCallback  │  ◄─── Abstraction (Interface)            │
│  └──────────────────┘                                          │
│          △                                                      │
│          │ depends on                                           │
│          │                                                      │
│  Low-level (Adapters)                                           │
│                                                                 │
│  NOT: User Code ──> HuggingFace API                            │
│  BUT: User Code ──> Abstraction <── Adapter ──> HuggingFace    │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Layer Diagram

### 2.1 Detailed Component Breakdown

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: USER APPLICATION                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────┐  ┌───────────────────────┐                 │
│  │ EarlyStoppingCallback │  │ CustomLoggerCallback  │  ... User       │
│  │ (UnifiedCallback)     │  │ (UnifiedCallback)     │      Callbacks  │
│  └───────────────────────┘  └───────────────────────┘                 │
│             │                           │                               │
│             └────────────┬──────────────┘                               │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           │ implements UnifiedCallback
                           │
┌──────────────────────────┼──────────────────────────────────────────────┐
│ LAYER 2: ABSTRACTION     │                                              │
├──────────────────────────┼──────────────────────────────────────────────┤
│                          ▼                                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ UnifiedCallback (ABC)                                          │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │ Event Hooks (15 methods):                                      │    │
│  │  • on_init_end(ctx: UnifiedContext)                           │    │
│  │  • on_train_begin(ctx: UnifiedContext)                        │    │
│  │  • on_train_end(ctx: UnifiedContext)                          │    │
│  │  • on_epoch_begin(ctx: UnifiedContext)                        │    │
│  │  • on_epoch_end(ctx: UnifiedContext)                          │    │
│  │  • on_step_begin(ctx: UnifiedContext)                         │    │
│  │  • on_step_end(ctx: UnifiedContext)                           │    │
│  │  • on_substep_end(ctx: UnifiedContext)                        │    │
│  │  • on_pre_optimizer_step(ctx: UnifiedContext)                 │    │
│  │  • on_optimizer_step(ctx: UnifiedContext)                     │    │
│  │  • on_log(ctx: UnifiedContext, metrics: dict)                 │    │
│  │  • on_save(ctx: UnifiedContext, metrics: dict)                │    │
│  │  • on_evaluate(ctx: UnifiedContext, metrics: dict)            │    │
│  │  • on_predict(ctx: UnifiedContext, metrics: dict)             │    │
│  │  • on_prediction_step(ctx: UnifiedContext, inputs: dict)      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ UnifiedContext (dataclass)                                     │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │ State Fields:                                                  │    │
│  │  • global_step: int                                           │    │
│  │  • epoch: float                                               │    │
│  │  • max_steps: int                                             │    │
│  │  • current_metrics: Dict[str, float]                          │    │
│  │  • metric_history: List[Dict[str, float]]                     │    │
│  │  • best_metric: Optional[float]                               │    │
│  │  • best_checkpoint_path: Optional[str]                        │    │
│  │  • tokens_seen: int                                           │    │
│  │  • is_main_process: bool                                      │    │
│  │  • model, tokenizer, optimizer: Optional[Any]                 │    │
│  │                                                                │    │
│  │ Control Methods:                                               │    │
│  │  • abort_training()                                           │    │
│  │  • stop_current_epoch()                                       │    │
│  │  • trigger_checkpoint()                                       │    │
│  │  • trigger_evaluation()                                       │    │
│  │  • trigger_logging()                                          │    │
│  │  • get_metric(key: str) → Any                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ uses
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: ADAPTER (Translation)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ HuggingFaceAdapterCallback                                     │    │
│  │ extends: TrainerCallback (HF)                                  │    │
│  │ wraps: UnifiedCallback                                         │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │                                                                │    │
│  │ Private Methods:                                               │    │
│  │                                                                │    │
│  │  _state_to_context(args, state, control, **kwargs)            │    │
│  │  ┌──────────────────────────────────────────────────┐         │    │
│  │  │ INPUT:                                            │         │    │
│  │  │  • args: TrainingArguments                        │         │    │
│  │  │  • state: TrainerState (20 fields)               │         │    │
│  │  │  • control: TrainerControl (5 flags)             │         │    │
│  │  │  • kwargs: {model, tokenizer, optimizer, ...}    │         │    │
│  │  │                                                   │         │    │
│  │  │ OUTPUT:                                           │         │    │
│  │  │  • UnifiedContext (15 fields + methods)          │         │    │
│  │  │                                                   │         │    │
│  │  │ TRANSLATION LOGIC:                                │         │    │
│  │  │  • Extract state.global_step → ctx.global_step   │         │    │
│  │  │  • Extract state.epoch → ctx.epoch               │         │    │
│  │  │  • Parse state.log_history[-1] → current_metrics │         │    │
│  │  │  • Map kwargs['model'] → ctx.model               │         │    │
│  │  │  • Set ctx.backend_name = "transformers"         │         │    │
│  │  └──────────────────────────────────────────────────┘         │    │
│  │                                                                │    │
│  │  _context_to_control(ctx, control)                            │    │
│  │  ┌──────────────────────────────────────────────────┐         │    │
│  │  │ INPUT:                                            │         │    │
│  │  │  • ctx: UnifiedContext (with user modifications) │         │    │
│  │  │  • control: TrainerControl (to modify)           │         │    │
│  │  │                                                   │         │    │
│  │  │ OUTPUT:                                           │         │    │
│  │  │  • Modified TrainerControl                       │         │    │
│  │  │                                                   │         │    │
│  │  │ TRANSLATION LOGIC:                                │         │    │
│  │  │  • ctx._should_stop_training → control.should_   │         │    │
│  │  │                                 training_stop     │         │    │
│  │  │  • ctx._should_save_now → control.should_save    │         │    │
│  │  │  • ctx._should_evaluate_now → control.should_    │         │    │
│  │  │                                evaluate           │         │    │
│  │  └──────────────────────────────────────────────────┘         │    │
│  │                                                                │    │
│  │ Public Methods (15 event forwarders):                         │    │
│  │                                                                │    │
│  │  on_step_end(args, state, control, **kwargs):                │    │
│  │      ctx = self._state_to_context(args, state, control, **kw)│    │
│  │      self.unified_callback.on_step_end(ctx)                  │    │
│  │      return self._context_to_control(ctx, control)           │    │
│  │                                                                │    │
│  │  [... 14 more event methods following same pattern ...]       │    │
│  │                                                                │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ UnifiedCallbackPlugin (Axolotl-specific)                       │    │
│  │ extends: BasePlugin (Axolotl)                                  │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │                                                                │    │
│  │  add_callbacks_pre_trainer(cfg, model):                       │    │
│  │      return [HuggingFaceAdapterCallback(self.unified_cb)]     │    │
│  │                                                                │    │
│  │  add_callbacks_post_trainer(cfg, trainer):                    │    │
│  │      return []  # or trainer-dependent callbacks              │    │
│  │                                                                │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ CallbackInjector (Utility)                                     │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │                                                                │    │
│  │  inject(trainer_or_config, callbacks, backend):               │    │
│  │      if backend in (TRANSFORMERS, TRL, UNSLOTH):              │    │
│  │          adapters = [HFAdapter(cb) for cb in callbacks]       │    │
│  │          trainer_kwargs['callbacks'].extend(adapters)         │    │
│  │      elif backend == AXOLOTL:                                 │    │
│  │          return UnifiedCallbackPlugin(callbacks)              │    │
│  │                                                                │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ implements backend interface
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: BACKEND FRAMEWORKS                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────┐   │
│  │ TrainerCallback    │  │ BasePlugin         │  │ SFTTrainer     │   │
│  │ (Transformers)     │  │ (Axolotl)          │  │ (TRL)          │   │
│  ├────────────────────┤  ├────────────────────┤  ├────────────────┤   │
│  │ • 15 event hooks   │  │ • lifecycle hooks  │  │ extends Trainer│   │
│  │ • TrainerState     │  │ • plugin registry  │  │ • inherits all │   │
│  │ • TrainerControl   │  │                    │  │   callbacks    │   │
│  └────────────────────┘  └────────────────────┘  └────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Callback Execution Flow

### 3.1 Single Event Execution Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ EVENT: on_step_end                                                       │
└──────────────────────────────────────────────────────────────────────────┘

USER CALLBACK                ADAPTER                    BACKEND (Trainer)
     │                          │                              │
     │                          │                              │
     │                          │    ┌─────────────────────────┤
     │                          │    │ 1. Optimizer step done  │
     │                          │    │    global_step += 1     │
     │                          │    └────────────────────────>│
     │                          │                              │
     │                          │    ┌─────────────────────────┤
     │                          │    │ 2. Invoke callbacks     │
     │                          │<───┤    callback_handler.    │
     │                          │    │    on_step_end(         │
     │                          │    │      args,              │
     │                          │    │      state,             │
     │                          │    │      control,           │
     │                          │    │      **kwargs           │
     │                          │    │    )                    │
     │                          │    └─────────────────────────┘
     │                          │                              │
     │      ┌───────────────────┤                              │
     │      │ 3. Translate      │                              │
     │      │    state_to_ctx   │                              │
     │      │                   │                              │
     │      │  ctx = UnifiedContext(                           │
     │      │    global_step = state.global_step,              │
     │      │    epoch = state.epoch,                          │
     │      │    current_metrics = state.log_history[-1],      │
     │      │    model = kwargs['model'],                      │
     │      │    ...                                            │
     │      │  )                                                │
     │      └──────────────────>│                              │
     │                          │                              │
     │      ┌───────────────────┤                              │
     │      │ 4. Invoke user    │                              │
     │<─────┤    callback       │                              │
     │      │    unified_       │                              │
     │      │    callback.      │                              │
     │      │    on_step_end(   │                              │
     │      │      ctx          │                              │
     │      │    )              │                              │
     │      └───────────────────┘                              │
     │                                                          │
     │ 5. User logic executes:                                 │
     │    • Check ctx.global_step                              │
     │    • Read ctx.current_metrics                           │
     │    • Decide to stop: ctx.abort_training()               │
     │    • Set ctx._should_stop_training = True               │
     │                                                          │
     │──────────────────────────>                              │
     │      (returns)            │                              │
     │                          │                              │
     │                          │                              │
     │      ┌───────────────────┤                              │
     │      │ 6. Translate      │                              │
     │      │    ctx_to_control │                              │
     │      │                   │                              │
     │      │  if ctx._should_stop_training:                   │
     │      │    control.should_training_stop = True           │
     │      │  if ctx._should_save_now:                        │
     │      │    control.should_save = True                    │
     │      │  return control                                  │
     │      └──────────────────>│                              │
     │                          │                              │
     │                          │     ┌────────────────────────┤
     │                          │─────> 7. Return control      │
     │                          │     │    (modified)          │
     │                          │     └───────────────────────>│
     │                          │                              │
     │                          │     ┌────────────────────────┤
     │                          │     │ 8. Check control flags │
     │                          │     │                        │
     │                          │     │ if control.should_     │
     │                          │     │    training_stop:      │
     │                          │     │      break training    │
     │                          │     │                        │
     │                          │     │ if control.should_save:│
     │                          │     │      save_checkpoint() │
     │                          │     │      control.should_   │
     │                          │     │      save = False      │
     │                          │     │                        │
     │                          │     └───────────────────────>│
     │                          │                              │
```

### 3.2 Multiple Callbacks Execution (Sequential)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Callback Chaining: Multiple callbacks registered                    │
│ callbacks = [Adapter1(EarlyStopping), Adapter2(CustomLogger)]       │
└──────────────────────────────────────────────────────────────────────┘

TRAINER
   │
   │ on_step_end(args, state, control, **kwargs)
   │
   ├──> CallbackHandler.call_event("on_step_end", ...)
   │
   │    ┌─────────────────────────────────────────────────────┐
   │    │ FOR each callback in self.callbacks:                │
   │    │     control = callback.on_step_end(args, state,     │
   │    │                                     control, **kw)   │
   │    └─────────────────────────────────────────────────────┘
   │
   │
   ├──> 1. Adapter1.on_step_end(args, state, control_v0, **kw)
   │       │
   │       ├─> ctx = state_to_context(state, control_v0)
   │       ├─> EarlyStopping.on_step_end(ctx)
   │       │   └─> ctx._should_stop_training = True
   │       │
   │       └─> control_v1 = context_to_control(ctx, control_v0)
   │           └─> control_v1.should_training_stop = True
   │
   │       return control_v1
   │
   │
   ├──> 2. Adapter2.on_step_end(args, state, control_v1, **kw)
   │       │                                   ^
   │       │                                   │
   │       │                          (sees previous modification!)
   │       │
   │       ├─> ctx = state_to_context(state, control_v1)
   │       ├─> CustomLogger.on_step_end(ctx)
   │       │   └─> print(f"Step {ctx.global_step}")
   │       │   └─> (doesn't modify control)
   │       │
   │       └─> control_v2 = context_to_control(ctx, control_v1)
   │           └─> control_v2.should_training_stop = True (unchanged)
   │
   │       return control_v2
   │
   │
   ├──> 3. Final control returned: control_v2
   │
   └──> Check control_v2.should_training_stop == True
        └──> Break training loop
```

**Key Insight:** Callbacks execute **sequentially**, not in parallel. Later callbacks see the control modifications of earlier callbacks. This is why callback order matters.

---

## 4. State Translation Pipeline

### 4.1 State to Context Translation

```
┌──────────────────────────────────────────────────────────────────────────┐
│ INPUT: Backend State Objects                                            │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ TrainingArguments           │
├─────────────────────────────┤   ┌─────────────────────────────┐
│ • learning_rate: float      │   │ TrainerState                │
│ • num_train_epochs: int     │   ├─────────────────────────────┤
│ • per_device_batch_size: int│   │ • epoch: float              │
│ • gradient_accum_steps: int │   │ • global_step: int          │
│ • logging_steps: int        │   │ • max_steps: int            │
│ • eval_steps: int           │   │ • logging_steps: int        │
│ • save_steps: int           │   │ • eval_steps: int           │
│ • output_dir: str           │   │ • save_steps: int           │
│ • ... (100+ fields)         │   │ • log_history: List[Dict]   │
└─────────────────────────────┘   │ • best_metric: float        │
                                   │ • best_model_checkpoint: str│
┌─────────────────────────────┐   │ • num_input_tokens_seen: int│
│ TrainerControl              │   │ • total_flos: float         │
├─────────────────────────────┤   │ • is_world_process_zero: bool│
│ • should_training_stop: bool│   │ • is_local_process_zero: bool│
│ • should_epoch_stop: bool   │   │ • stateful_callbacks: Dict  │
│ • should_save: bool         │   │ • trial_name: str           │
│ • should_evaluate: bool     │   │ • trial_params: Dict        │
│ • should_log: bool          │   │ • ... (20 fields total)     │
└─────────────────────────────┘   └─────────────────────────────┘

┌─────────────────────────────┐
│ **kwargs                    │
├─────────────────────────────┤
│ • model: PreTrainedModel    │
│ • tokenizer: Tokenizer      │
│ • optimizer: Optimizer      │
│ • lr_scheduler: Scheduler   │
│ • train_dataloader: DataLoader│
│ • eval_dataloader: DataLoader│
└─────────────────────────────┘

                 ▼ ▼ ▼ ▼ ▼

┌──────────────────────────────────────────────────────────────────────────┐
│ TRANSLATION FUNCTION: _state_to_context()                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  def _state_to_context(args, state, control, **kwargs):                 │
│      return UnifiedContext(                                              │
│          # Core progress (direct mapping)                                │
│          global_step = state.global_step,                                │
│          epoch = state.epoch or 0.0,                                     │
│          max_steps = state.max_steps,                                    │
│                                                                          │
│          # Metrics (extract from log_history)                            │
│          current_metrics = (                                             │
│              state.log_history[-1]                                       │
│              if state.log_history                                        │
│              else {}                                                     │
│          ),                                                              │
│          metric_history = state.log_history.copy(),                      │
│          best_metric = state.best_metric,                                │
│          best_checkpoint_path = state.best_model_checkpoint,             │
│                                                                          │
│          # Resources (direct mapping)                                    │
│          tokens_seen = state.num_input_tokens_seen,                      │
│          total_flops = state.total_flos,                                 │
│                                                                          │
│          # Distributed (direct mapping)                                  │
│          is_main_process = state.is_world_process_zero,                  │
│          is_local_main_process = state.is_local_process_zero,            │
│                                                                          │
│          # References (extract from kwargs)                              │
│          model = kwargs.get("model"),                                    │
│          tokenizer = kwargs.get("tokenizer"),                            │
│          optimizer = kwargs.get("optimizer"),                            │
│                                                                          │
│          # Backend info                                                  │
│          backend_name = "transformers",                                  │
│          backend_args = args,                                            │
│                                                                          │
│          # Control flags (initialized to False)                          │
│          _should_stop_training = False,                                  │
│          _should_stop_epoch = False,                                     │
│          _should_save_now = False,                                       │
│          _should_evaluate_now = False,                                   │
│          _should_log_now = False,                                        │
│      )                                                                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

                 ▼ ▼ ▼ ▼ ▼

┌──────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Unified Context                                                  │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ UnifiedContext              │
├─────────────────────────────┤
│ State (read-only):          │
│ • global_step: 1000         │
│ • epoch: 2.5                │
│ • max_steps: 10000          │
│ • current_metrics: {        │
│     "loss": 0.342,          │
│     "learning_rate": 2e-5   │
│   }                         │
│ • metric_history: [...]     │
│ • best_metric: 0.287        │
│ • tokens_seen: 5000000      │
│ • is_main_process: True     │
│                             │
│ References:                 │
│ • model: LlamaForCausalLM   │
│ • tokenizer: LlamaTokenizer │
│ • optimizer: AdamW          │
│                             │
│ Control (user-modifiable):  │
│ • abort_training()          │
│ • trigger_checkpoint()      │
│ • get_metric(key)           │
└─────────────────────────────┘
```

### 4.2 Context to Control Translation

```
┌──────────────────────────────────────────────────────────────────────────┐
│ INPUT: Modified Unified Context (after user callback)                   │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ UnifiedContext              │
├─────────────────────────────┤
│ (User called methods)       │
│                             │
│ ctx.abort_training()        │
│   └─> _should_stop_training │
│       = True                │
│                             │
│ ctx.trigger_checkpoint()    │
│   └─> _should_save_now      │
│       = True                │
│                             │
│ Internal flags:             │
│ • _should_stop_training: ✓  │
│ • _should_stop_epoch: ✗     │
│ • _should_save_now: ✓       │
│ • _should_evaluate_now: ✗   │
│ • _should_log_now: ✗        │
└─────────────────────────────┘

                 ▼ ▼ ▼

┌──────────────────────────────────────────────────────────────────────────┐
│ TRANSLATION FUNCTION: _context_to_control()                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  def _context_to_control(ctx, control):                                  │
│      # Map boolean flags to control flags                                │
│      if ctx._should_stop_training:                                       │
│          control.should_training_stop = True                             │
│                                                                          │
│      if ctx._should_stop_epoch:                                          │
│          control.should_epoch_stop = True                                │
│                                                                          │
│      if ctx._should_save_now:                                            │
│          control.should_save = True                                      │
│                                                                          │
│      if ctx._should_evaluate_now:                                        │
│          control.should_evaluate = True                                  │
│                                                                          │
│      if ctx._should_log_now:                                             │
│          control.should_log = True                                       │
│                                                                          │
│      return control  # Modified in-place                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

                 ▼ ▼ ▼

┌──────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Modified TrainerControl                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ TrainerControl              │
├─────────────────────────────┤
│ BEFORE:                     │
│ • should_training_stop: ✗   │
│ • should_epoch_stop: ✗      │
│ • should_save: ✗            │
│ • should_evaluate: ✗        │
│ • should_log: ✗             │
│                             │
│ AFTER:                      │
│ • should_training_stop: ✓   │ ◄─── Mapped from ctx
│ • should_epoch_stop: ✗      │
│ • should_save: ✓            │ ◄─── Mapped from ctx
│ • should_evaluate: ✗        │
│ • should_log: ✗             │
└─────────────────────────────┘
```

### 4.3 Field Mapping Table

```
┌────────────────────────────────────────────────────────────────────────┐
│ State Translation: TrainerState → UnifiedContext                      │
├─────────────────────────────┬──────────────────────────────────────────┤
│ TrainerState Field          │ UnifiedContext Field                     │
├─────────────────────────────┼──────────────────────────────────────────┤
│ global_step                 │ global_step (direct)                     │
│ epoch                       │ epoch (direct, default 0.0)              │
│ max_steps                   │ max_steps (direct)                       │
│ log_history[-1]             │ current_metrics (extract last entry)     │
│ log_history                 │ metric_history (copy list)               │
│ best_metric                 │ best_metric (direct)                     │
│ best_model_checkpoint       │ best_checkpoint_path (direct)            │
│ num_input_tokens_seen       │ tokens_seen (direct)                     │
│ total_flos                  │ total_flops (direct)                     │
│ is_world_process_zero       │ is_main_process (direct)                 │
│ is_local_process_zero       │ is_local_main_process (direct)           │
├─────────────────────────────┼──────────────────────────────────────────┤
│ **kwargs['model']           │ model (extract from kwargs)              │
│ **kwargs['tokenizer']       │ tokenizer (extract from kwargs)          │
│ **kwargs['optimizer']       │ optimizer (extract from kwargs)          │
├─────────────────────────────┼──────────────────────────────────────────┤
│ (backend detection)         │ backend_name = "transformers"            │
│ args (TrainingArguments)    │ backend_args (store reference)           │
└─────────────────────────────┴──────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Control Translation: UnifiedContext → TrainerControl                   │
├─────────────────────────────┬──────────────────────────────────────────┤
│ UnifiedContext Flag         │ TrainerControl Flag                      │
├─────────────────────────────┼──────────────────────────────────────────┤
│ _should_stop_training       │ should_training_stop                     │
│ _should_stop_epoch          │ should_epoch_stop                        │
│ _should_save_now            │ should_save                              │
│ _should_evaluate_now        │ should_evaluate                          │
│ _should_log_now             │ should_log                               │
└─────────────────────────────┴──────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ User Methods → Internal Flags                                          │
├─────────────────────────────┬──────────────────────────────────────────┤
│ User Method Call            │ Internal Flag Set                        │
├─────────────────────────────┼──────────────────────────────────────────┤
│ ctx.abort_training()        │ _should_stop_training = True             │
│ ctx.stop_current_epoch()    │ _should_stop_epoch = True                │
│ ctx.trigger_checkpoint()    │ _should_save_now = True                  │
│ ctx.trigger_evaluation()    │ _should_evaluate_now = True              │
│ ctx.trigger_logging()       │ _should_log_now = True                   │
└─────────────────────────────┴──────────────────────────────────────────┘
```

---

## 5. Event Lifecycle Diagram

### 5.1 Complete Training Lifecycle with Events

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING LIFECYCLE                                │
└──────────────────────────────────────────────────────────────────────────┘

Trainer.__init__(model, args, callbacks=[adapter])
    │
    ├─> Initialize optimizer, scheduler, dataloaders
    │
    ├─> CallbackHandler.__init__(callbacks)
    │
    └─> ◆ EVENT: on_init_end
        ├─> kwargs: {model, tokenizer, train_dataloader,
        │            eval_dataloader, optimizer, lr_scheduler}
        └─> Use case: Setup logging, initialize custom state

────────────────────────────────────────────────────────────────────────────

Trainer.train()
    │
    └─> ◆ EVENT: on_train_begin
        ├─> state.max_steps initialized
        ├─> state.num_train_epochs initialized
        └─> Use case: Start timers, log training config

    ┌────────────────────────────────────────────────────────────────────┐
    │ EPOCH LOOP: for epoch in range(num_epochs)                        │
    │                                                                    │
    │   ◆ EVENT: on_epoch_begin                                         │
    │   ├─> state.epoch = current epoch (float)                         │
    │   └─> Use case: Reset epoch-level metrics                         │
    │                                                                    │
    │   ┌────────────────────────────────────────────────────────────┐  │
    │   │ STEP LOOP: for step, batch in enumerate(dataloader)       │  │
    │   │                                                            │  │
    │   │   ┌────────────────────────────────────────────────────┐  │  │
    │   │   │ GRADIENT ACCUMULATION LOOP                        │  │  │
    │   │   │ for micro_batch in gradient_accumulation_batches: │  │  │
    │   │   │                                                    │  │  │
    │   │   │   Forward pass                                     │  │  │
    │   │   │   Backward pass                                    │  │  │
    │   │   │                                                    │  │  │
    │   │   │   ◆ EVENT: on_substep_end                         │  │  │
    │   │   │   ├─> Fires ONCE per micro-batch                  │  │  │
    │   │   │   ├─> state.global_step NOT incremented yet       │  │  │
    │   │   │   └─> Use case: Log per-batch metrics             │  │  │
    │   │   │                                                    │  │  │
    │   │   └────────────────────────────────────────────────────┘  │  │
    │   │                                                            │  │
    │   │   ◆ EVENT: on_step_begin                                   │  │
    │   │   ├─> After gradient accumulation completes                │  │
    │   │   ├─> Before optimizer step                                │  │
    │   │   └─> Use case: Modify gradients                           │  │
    │   │                                                            │  │
    │   │   Gradient clipping (if enabled)                           │  │
    │   │                                                            │  │
    │   │   ◆ EVENT: on_pre_optimizer_step (NEW in v4.52+)          │  │
    │   │   ├─> After gradient clipping                              │  │
    │   │   ├─> Before optimizer.step()                              │  │
    │   │   └─> Use case: Inspect clipped gradients                  │  │
    │   │                                                            │  │
    │   │   optimizer.step()                                         │  │
    │   │                                                            │  │
    │   │   ◆ EVENT: on_optimizer_step                               │  │
    │   │   ├─> After optimizer.step()                               │  │
    │   │   ├─> Before optimizer.zero_grad()                         │  │
    │   │   ├─> Gradients still available                            │  │
    │   │   └─> Use case: Custom optimizer logic                     │  │
    │   │                                                            │  │
    │   │   optimizer.zero_grad()                                    │  │
    │   │   state.global_step += 1                                   │  │
    │   │                                                            │  │
    │   │   ◆ EVENT: on_step_end                                     │  │
    │   │   ├─> state.global_step NOW incremented                    │  │
    │   │   ├─> Most common event for custom logic                   │  │
    │   │   └─> Use case: Checkpointing, early stopping              │  │
    │   │                                                            │  │
    │   │   ─────────────────────────────────────────────────────    │  │
    │   │                                                            │  │
    │   │   if should_log(step):                                     │  │
    │   │       metrics = compute_metrics()                          │  │
    │   │       state.log_history.append(metrics)                    │  │
    │   │                                                            │  │
    │   │       ◆ EVENT: on_log                                      │  │
    │   │       ├─> logs: Dict[str, float] (current metrics)        │  │
    │   │       └─> Use case: Custom logging to W&B, MLflow         │  │
    │   │                                                            │  │
    │   │       control.should_log = False  # Auto-reset            │  │
    │   │                                                            │  │
    │   │   ─────────────────────────────────────────────────────    │  │
    │   │                                                            │  │
    │   │   if should_evaluate(step):                                │  │
    │   │       ┌────────────────────────────────────────────────┐  │  │
    │   │       │ EVALUATION LOOP                                │  │  │
    │   │       │ for eval_batch in eval_dataloader:            │  │  │
    │   │       │                                                │  │  │
    │   │       │   ◆ EVENT: on_prediction_step                 │  │  │
    │   │       │   ├─> kwargs: {inputs} (batch dict)           │  │  │
    │   │       │   └─> Use case: Monitor predictions           │  │  │
    │   │       │                                                │  │  │
    │   │       │   model(eval_batch)                           │  │  │
    │   │       │                                                │  │  │
    │   │       └────────────────────────────────────────────────┘  │  │
    │   │                                                            │  │
    │   │       eval_metrics = compute_eval_metrics()                │  │
    │   │                                                            │  │
    │   │       ◆ EVENT: on_evaluate                                 │  │
    │   │       ├─> metrics: Dict[str, float] (eval results)        │  │
    │   │       ├─> state.best_metric updated (if improved)         │  │
    │   │       └─> Use case: Early stopping, model selection       │  │
    │   │                                                            │  │
    │   │       control.should_evaluate = False  # Auto-reset       │  │
    │   │                                                            │  │
    │   │   ─────────────────────────────────────────────────────    │  │
    │   │                                                            │  │
    │   │   if should_save(step):                                    │  │
    │   │       ◆ EVENT: on_save                                     │  │
    │   │       ├─> kwargs: {metrics} (current metrics)             │  │
    │   │       └─> Use case: Custom checkpoint logic               │  │
    │   │                                                            │  │
    │   │       save_checkpoint()                                    │  │
    │   │       control.should_save = False  # Auto-reset           │  │
    │   │                                                            │  │
    │   │   ─────────────────────────────────────────────────────    │  │
    │   │                                                            │  │
    │   │   if control.should_training_stop:                         │  │
    │   │       break  # Exit training                               │  │
    │   │                                                            │  │
    │   │   if control.should_epoch_stop:                            │  │
    │   │       break  # Exit epoch loop                             │  │
    │   │                                                            │  │
    │   └────────────────────────────────────────────────────────────┘  │
    │                                                                    │
    │   ◆ EVENT: on_epoch_end                                           │
    │   ├─> state.epoch incremented                                     │
    │   └─> Use case: Log epoch summary                                 │
    │                                                                    │
    │   control.should_epoch_stop = False  # Reset for next epoch       │
    │                                                                    │
    │   if control.should_training_stop:                                │
    │       break  # Exit epoch loop                                    │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

    ◆ EVENT: on_train_end
    ├─> state contains final metrics
    └─> Use case: Cleanup, final logging

────────────────────────────────────────────────────────────────────────────

Trainer.predict(test_dataset)  # Optional
    │
    │   ┌────────────────────────────────────────────────────────┐
    │   │ PREDICTION LOOP                                        │
    │   │ for batch in test_dataloader:                          │
    │   │                                                        │
    │   │   ◆ EVENT: on_prediction_step                         │
    │   │   ├─> kwargs: {inputs}                                │
    │   │   └─> Use case: Monitor predictions                   │
    │   │                                                        │
    │   │   predictions = model(batch)                          │
    │   │                                                        │
    │   └────────────────────────────────────────────────────────┘
    │
    └─> ◆ EVENT: on_predict
        ├─> metrics: Dict[str, float]
        └─> Use case: Log prediction results
```

### 5.2 Event Frequency Analysis

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Event Firing Frequency (per training run)                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Event                    │ Frequency                                   │
│  ────────────────────────────────────────────────────────────────────   │
│  on_init_end              │ 1 time (at start)                           │
│  on_train_begin           │ 1 time (at start)                           │
│  on_train_end             │ 1 time (at end)                             │
│  ────────────────────────────────────────────────────────────────────   │
│  on_epoch_begin           │ num_epochs times                            │
│  on_epoch_end             │ num_epochs times                            │
│  ────────────────────────────────────────────────────────────────────   │
│  on_step_begin            │ total_steps times                           │
│  on_step_end              │ total_steps times                           │
│  on_substep_end           │ total_steps × gradient_accum_steps          │
│  on_pre_optimizer_step    │ total_steps times (v4.52+)                  │
│  on_optimizer_step        │ total_steps times                           │
│  ────────────────────────────────────────────────────────────────────   │
│  on_log                   │ total_steps / logging_steps                 │
│  on_save                  │ total_steps / save_steps                    │
│  on_evaluate              │ total_steps / eval_steps                    │
│  ────────────────────────────────────────────────────────────────────   │
│  on_prediction_step       │ num_eval_batches (during eval/predict)      │
│  on_predict               │ 1 time (if Trainer.predict() called)        │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                          │
│  Example: 10,000 steps, gradient_accum=4, logging=100, eval=500:       │
│  • on_step_end: 10,000 times                                            │
│  • on_substep_end: 40,000 times (4× more!)                              │
│  • on_log: 100 times                                                    │
│  • on_evaluate: 20 times                                                │
│                                                                          │
│  ⚠️  Performance Impact: on_substep_end can fire very frequently!       │
│      Keep callback logic lightweight.                                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Backend Injection Patterns

### 6.1 Standard Injection (Transformers, TRL, Unsloth)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ PATTERN: Constructor Parameter Injection                                │
└──────────────────────────────────────────────────────────────────────────┘

USER CODE
    │
    │ 1. Create user callback
    │
    ├─> my_callback = EarlyStoppingCallback(patience=3)
    │
    │ 2. Wrap in adapter
    │
    ├─> adapter = HuggingFaceAdapterCallback(my_callback)
    │
    │ 3. Pass to trainer constructor
    │
    ├─> trainer = Trainer(
    │       model=model,
    │       args=training_args,
    │       train_dataset=dataset,
    │       callbacks=[adapter],  ◄─── List of callbacks
    │   )
    │
    ▼

TRAINER (HuggingFace)
    │
    │ Trainer.__init__(self, ..., callbacks=None, ...)
    │     │
    │     ├─> self.callback_handler = CallbackHandler(
    │     │       callbacks,
    │     │       self.model,
    │     │       self.tokenizer,
    │     │       self.optimizer,
    │     │       self.lr_scheduler
    │     │   )
    │     │
    │     ├─> CallbackHandler.__init__(self, callbacks, ...)
    │     │       │
    │     │       ├─> self.callbacks = []
    │     │       │
    │     │       ├─> Add default callbacks:
    │     │       │   • DefaultFlowCallback (REQUIRED)
    │     │       │   • ProgressCallback
    │     │       │   • PrinterCallback
    │     │       │
    │     │       └─> for cb in callbacks:
    │     │               self.callbacks.append(cb)
    │     │
    │     └─> self.callback_handler.callbacks = [
    │             DefaultFlowCallback(),
    │             ProgressCallback(),
    │             PrinterCallback(),
    │             HuggingFaceAdapterCallback(my_callback),  ◄─── Your callback
    │         ]
    │
    ▼

TRAINING LOOP
    │
    ├─> callback_handler.on_step_end(args, state, control)
    │       │
    │       └─> for callback in self.callbacks:
    │               control = callback.on_step_end(args, state, control)
    │                             │
    │                             └─> HuggingFaceAdapterCallback.on_step_end()
    │                                     │
    │                                     └─> my_callback.on_step_end(ctx)
    │
    ▼

┌──────────────────────────────────────────────────────────────────────────┐
│ Alternative: Add callback after construction                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  trainer = Trainer(model, args, train_dataset)                           │
│  trainer.add_callback(adapter)  ◄─── Can add more callbacks later       │
│                                                                          │
│  # Also can remove:                                                      │
│  trainer.remove_callback(SomeCallback)                                   │
│  trainer.pop_callback(SomeCallback)  # Remove and return                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Axolotl Two-Phase Injection

```
┌──────────────────────────────────────────────────────────────────────────┐
│ PATTERN: Plugin-Based Two-Phase Injection                               │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Pre-Trainer Callbacks (before Trainer.__init__)           │
└─────────────────────────────────────────────────────────────────────┘

USER CODE
    │
    │ 1. Create plugin
    │
    ├─> my_callback = EarlyStoppingCallback(patience=3)
    ├─> plugin = UnifiedCallbackPlugin([my_callback])
    │
    │ 2. Register plugin
    │
    ├─> from axolotl.integrations.base import PluginManager
    ├─> PluginManager.get_instance().register_plugin(plugin)
    │
    │ 3. Load config and build trainer
    │
    ├─> from axolotl.cli import load_cfg
    ├─> from axolotl.core.builders import causal
    │
    ├─> cfg = load_cfg("axolotl_config.yaml")
    │
    ├─> trainer_builder = causal.HFCausalTrainerBuilder(cfg, ...)
    │
    ▼

TRAINER BUILDER (Axolotl)
    │
    │ HFCausalTrainerBuilder.build()
    │     │
    │     ├─> 1. Build model
    │     │
    │     ├─> 2. Build datasets
    │     │
    │     ├─> 3. Get pre-trainer callbacks
    │     │
    │     ├─> callbacks = self.get_callbacks()
    │     │       │
    │     │       ├─> callbacks = []
    │     │       │
    │     │       ├─> # Plugin callbacks (PRE-TRAINER)
    │     │       ├─> plugin_manager = PluginManager.get_instance()
    │     │       ├─> callbacks.extend(
    │     │       │       plugin_manager.add_callbacks_pre_trainer(
    │     │       │           cfg=self.cfg,
    │     │       │           model=self.model
    │     │       │       )
    │     │       │   )
    │     │       │       │
    │     │       │       └─> UnifiedCallbackPlugin.add_callbacks_pre_trainer()
    │     │       │               │
    │     │       │               └─> return [
    │     │       │                       HuggingFaceAdapterCallback(my_callback)
    │     │       │                   ]
    │     │       │
    │     │       ├─> # Config-driven callbacks
    │     │       ├─> if cfg.gc_steps:
    │     │       │       callbacks.append(GCCallback(cfg.gc_steps))
    │     │       │
    │     │       ├─> if cfg.use_wandb:
    │     │       │       callbacks.append(SaveAxolotlConfigtoWandBCallback(...))
    │     │       │
    │     │       └─> return callbacks
    │     │
    │     ├─> 4. Create trainer with pre-trainer callbacks
    │     │
    │     ├─> trainer = Trainer(
    │     │       model=self.model,
    │     │       args=training_args,
    │     │       train_dataset=self.train_dataset,
    │     │       callbacks=callbacks,  ◄─── PRE-TRAINER CALLBACKS
    │     │   )
    │     │
    │     ▼

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Post-Trainer Callbacks (after Trainer.__init__)           │
└─────────────────────────────────────────────────────────────────────┘

TRAINER BUILDER (continued)
    │
    │     ├─> 5. Apply post-create hook
    │     │
    │     ├─> trainer = self.hook_post_create_trainer(trainer)
    │     │
    │     ├─> 6. Get post-trainer callbacks
    │     │
    │     ├─> post_callbacks = self.get_post_trainer_create_callbacks(trainer)
    │     │       │
    │     │       ├─> callbacks = []
    │     │       │
    │     │       ├─> # Callbacks requiring trainer access
    │     │       ├─> if cfg.do_bench_eval:
    │     │       │       callbacks.append(
    │     │       │           bench_eval_callback_factory(trainer, tokenizer)
    │     │       │       )
    │     │       │
    │     │       ├─> if cfg.lisa_n_layers:
    │     │       │       callbacks.append(
    │     │       │           lisa_callback_factory(trainer)
    │     │       │       )
    │     │       │
    │     │       ├─> # Plugin callbacks (POST-TRAINER)
    │     │       ├─> callbacks.extend(
    │     │       │       plugin_manager.add_callbacks_post_trainer(
    │     │       │           cfg=self.cfg,
    │     │       │           trainer=trainer
    │     │       │       )
    │     │       │   )
    │     │       │       │
    │     │       │       └─> UnifiedCallbackPlugin.add_callbacks_post_trainer()
    │     │       │               │
    │     │       │               └─> return []  # or trainer-dependent callbacks
    │     │       │
    │     │       └─> return callbacks
    │     │
    │     ├─> 7. Add post-trainer callbacks
    │     │
    │     └─> for callback in post_callbacks:
    │             trainer.add_callback(callback)  ◄─── POST-TRAINER CALLBACKS
    │
    ├─> return trainer
    │
    ▼

FINAL TRAINER STATE
    │
    ├─> trainer.callback_handler.callbacks = [
    │       DefaultFlowCallback(),
    │       ProgressCallback(),
    │       PrinterCallback(),
    │       HuggingFaceAdapterCallback(my_callback),  ◄─── From pre-trainer
    │       GCCallback(),                              ◄─── From config
    │       BenchEvalCallback(),                       ◄─── From post-trainer
    │       LISACallback(),                            ◄─── From post-trainer
    │   ]
    │
    ▼

┌──────────────────────────────────────────────────────────────────────────┐
│ Why Two Phases?                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ PRE-TRAINER:                                                             │
│  • Callbacks that need model but NOT trainer                             │
│  • Standard callbacks (logging, checkpointing, etc.)                     │
│  • Passed to Trainer(..., callbacks=[...]) constructor                   │
│                                                                          │
│ POST-TRAINER:                                                            │
│  • Callbacks that need trainer reference                                 │
│  • Example: LISACallback needs trainer.model.layers                      │
│  • Example: LogPredictionCallback needs trainer.generate()               │
│  • Added via trainer.add_callback() after construction                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Comparison Table

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Injection Pattern Comparison                                              │
├──────────────────┬─────────────────────────────────────────────────────────┤
│ Backend          │ Injection Mechanism                                     │
├──────────────────┼─────────────────────────────────────────────────────────┤
│ Transformers     │ ✓ Trainer(callbacks=[...])                              │
│                  │ ✓ trainer.add_callback(cb)                              │
│                  │ • Simple, direct                                        │
│                  │ • Well-documented                                       │
├──────────────────┼─────────────────────────────────────────────────────────┤
│ TRL              │ ✓ SFTTrainer(callbacks=[...])                           │
│                  │ ✓ trainer.add_callback(cb)                              │
│                  │ • Identical to Transformers (inheritance)               │
│                  │ • No special handling                                   │
├──────────────────┼─────────────────────────────────────────────────────────┤
│ Unsloth          │ ✓ SFTTrainer(callbacks=[...])                           │
│                  │ ✓ trainer.add_callback(cb)                              │
│                  │ • 100% HF-compatible                                    │
│                  │ • Thin wrapper, no modifications                        │
├──────────────────┼─────────────────────────────────────────────────────────┤
│ Axolotl          │ ◆ Config-driven (YAML)                                  │
│                  │   use_wandb: true                                       │
│                  │   gc_steps: 100                                         │
│                  │                                                         │
│                  │ ◆ Plugin system (Programmatic)                          │
│                  │   plugin.add_callbacks_pre_trainer(cfg, model)          │
│                  │   plugin.add_callbacks_post_trainer(cfg, trainer)       │
│                  │                                                         │
│                  │ ◆ TrainerBuilder override (Advanced)                    │
│                  │   class CustomBuilder(HFCausalTrainerBuilder):          │
│                  │       def get_callbacks(self): ...                      │
│                  │                                                         │
│                  │ • Two-phase injection (pre/post trainer creation)       │
│                  │ • More complex but more flexible                        │
│                  │ • Plugin system is recommended approach                 │
├──────────────────┴─────────────────────────────────────────────────────────┤
│                                                                            │
│ UNIFIED LIBRARY ABSTRACTION:                                              │
│                                                                            │
│  CallbackInjector.inject(trainer_or_config, callbacks, backend)           │
│      ├─> Transformers/TRL/Unsloth: Add to callbacks list                  │
│      └─> Axolotl: Return plugin instance to register                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Adapter Pattern Deep Dive

### 7.1 Class Relationships

```
┌────────────────────────────────────────────────────────────────────────┐
│ Target Interface (What user programs against)                         │
└────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │         «interface»                                 │
    │         UnifiedCallback                             │
    ├─────────────────────────────────────────────────────┤
    │ + on_init_end(ctx: UnifiedContext)                  │
    │ + on_train_begin(ctx: UnifiedContext)               │
    │ + on_train_end(ctx: UnifiedContext)                 │
    │ + on_epoch_begin(ctx: UnifiedContext)               │
    │ + on_epoch_end(ctx: UnifiedContext)                 │
    │ + on_step_begin(ctx: UnifiedContext)                │
    │ + on_step_end(ctx: UnifiedContext)                  │
    │ + on_substep_end(ctx: UnifiedContext)               │
    │ + on_pre_optimizer_step(ctx: UnifiedContext)        │
    │ + on_optimizer_step(ctx: UnifiedContext)            │
    │ + on_log(ctx: UnifiedContext, metrics: Dict)        │
    │ + on_save(ctx: UnifiedContext, metrics: Dict)       │
    │ + on_evaluate(ctx: UnifiedContext, metrics: Dict)   │
    │ + on_predict(ctx: UnifiedContext, metrics: Dict)    │
    │ + on_prediction_step(ctx: UnifiedContext, inputs)   │
    └─────────────────────────────────────────────────────┘
                        △
                        │ implements
                        │
    ┌─────────────────────────────────────────────────────┐
    │         EarlyStoppingCallback                       │
    ├─────────────────────────────────────────────────────┤
    │ - patience: int                                     │
    │ - best_loss: float                                  │
    │ - wait: int                                         │
    ├─────────────────────────────────────────────────────┤
    │ + on_evaluate(ctx, metrics)                         │
    │     if metrics['eval_loss'] >= self.best_loss:      │
    │         self.wait += 1                              │
    │         if self.wait >= self.patience:              │
    │             ctx.abort_training()                    │
    └─────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Adaptee Interface (What backend provides)                             │
└────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │         «interface»                                             │
    │         TrainerCallback                                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ + on_init_end(args, state, control, **kwargs)                   │
    │ + on_train_begin(args, state, control, **kwargs)                │
    │ + on_train_end(args, state, control, **kwargs)                  │
    │ + on_epoch_begin(args, state, control, **kwargs)                │
    │ + on_epoch_end(args, state, control, **kwargs)                  │
    │ + on_step_begin(args, state, control, **kwargs)                 │
    │ + on_step_end(args, state, control, **kwargs)                   │
    │ + on_substep_end(args, state, control, **kwargs)                │
    │ + on_pre_optimizer_step(args, state, control, **kwargs)         │
    │ + on_optimizer_step(args, state, control, **kwargs)             │
    │ + on_log(args, state, control, logs, **kwargs)                  │
    │ + on_save(args, state, control, **kwargs)                       │
    │ + on_evaluate(args, state, control, metrics, **kwargs)          │
    │ + on_predict(args, state, control, metrics, **kwargs)           │
    │ + on_prediction_step(args, state, control, **kwargs)            │
    └─────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Adapter (Translates between interfaces)                               │
└────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │         HuggingFaceAdapterCallback                               │
    │         extends TrainerCallback                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │ - unified_callback: UnifiedCallback  ◄──────┐                    │
    │                                              │ composition       │
    ├──────────────────────────────────────────────┼──────────────────┤
    │ - _state_to_context(args, state, control, **│kwargs)            │
    │     → UnifiedContext                         │                  │
    │                                              │                  │
    │ - _context_to_control(ctx, control)          │                  │
    │     → TrainerControl                         │                  │
    │                                              │                  │
    ├──────────────────────────────────────────────┼──────────────────┤
    │ + on_step_end(args, state, control, **kwargs)│                  │
    │     ctx = self._state_to_context(...)        │                  │
    │     self.unified_callback.on_step_end(ctx) ──┘                  │
    │     return self._context_to_control(ctx, control)               │
    │                                                                  │
    │ [... 14 more event methods ...]                                 │
    └──────────────────────────────────────────────────────────────────┘
```

### 7.2 Object Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Runtime Object Interactions                                              │
└──────────────────────────────────────────────────────────────────────────┘

    :Trainer                :CallbackHandler          :Adapter           :UserCallback
        │                           │                      │                   │
        │ train()                   │                      │                   │
        ├─────────────────────────> │                      │                   │
        │                           │                      │                   │
        │                           │ on_step_end(         │                   │
        │                           │   args,              │                   │
        │                           │   state,             │                   │
        │                           │   control            │                   │
        │                           │ )                    │                   │
        │                           ├────────────────────> │                   │
        │                           │                      │                   │
        │                           │                      │ _state_to_context(│
        │                           │                      │   args, state,    │
        │                           │                      │   control         │
        │                           │                      │ )                 │
        │                           │                      │                   │
        │                           │                      │ ┌───────────────┐ │
        │                           │                      │ │ Create        │ │
        │                           │                      │ │ UnifiedContext│ │
        │                           │                      │ └───────────────┘ │
        │                           │                      │                   │
        │                           │                      │ :ctx              │
        │                           │                      ├──────────────────>│
        │                           │                      │                   │
        │                           │                      │  on_step_end(ctx) │
        │                           │                      │ ─────────────────>│
        │                           │                      │                   │
        │                           │                      │                   │ ┌──────────────┐
        │                           │                      │                   │ │ User Logic:  │
        │                           │                      │                   │ │ • Read state │
        │                           │                      │                   │ │ • Make       │
        │                           │                      │                   │ │   decisions  │
        │                           │                      │                   │ │ • Modify     │
        │                           │                      │                   │ │   control    │
        │                           │                      │                   │ │   flags      │
        │                           │                      │                   │ └──────────────┘
        │                           │                      │                   │
        │                           │                      │ <─────────────────┤
        │                           │                      │  (return)         │
        │                           │                      │                   │
        │                           │                      │ _context_to_      │
        │                           │                      │ control(ctx,      │
        │                           │                      │ control)          │
        │                           │                      │                   │
        │                           │                      │ ┌───────────────┐ │
        │                           │                      │ │ Apply context │ │
        │                           │                      │ │ flags to      │ │
        │                           │                      │ │ TrainerControl│ │
        │                           │                      │ └───────────────┘ │
        │                           │                      │                   │
        │                           │ <────────────────────┤                   │
        │                           │  return control      │                   │
        │                           │  (modified)          │                   │
        │                           │                      │                   │
        │ <─────────────────────────┤                      │                   │
        │  return control           │                      │                   │
        │                           │                      │                   │
        │ ┌───────────────────────┐ │                      │                   │
        │ │ Check control flags:  │ │                      │                   │
        │ │                       │ │                      │                   │
        │ │ if control.should_    │ │                      │                   │
        │ │    training_stop:     │ │                      │                   │
        │ │     break             │ │                      │                   │
        │ └───────────────────────┘ │                      │                   │
        │                           │                      │                   │
```

---

## 8. Sequence Diagrams

### 8.1 Complete Training Flow with Early Stopping

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Scenario: Training stops early based on eval loss                       │
└──────────────────────────────────────────────────────────────────────────┘

User                Trainer          CallbackHandler    Adapter         EarlyStoppingCB
 │                    │                     │              │                   │
 │ trainer.train()    │                     │              │                   │
 ├───────────────────>│                     │              │                   │
 │                    │                     │              │                   │
 │                    │ on_train_begin()    │              │                   │
 │                    ├────────────────────>│              │                   │
 │                    │                     ├─────────────>│                   │
 │                    │                     │              ├──────────────────>│
 │                    │                     │              │  on_train_begin() │
 │                    │                     │              │ <─────────────────┤
 │                    │                     │ <────────────┤                   │
 │                    │ <───────────────────┤              │                   │
 │                    │                     │              │                   │
 │                    │ ┌─ TRAINING LOOP ──────────────────────────────────┐  │
 │                    │ │                   │              │                │  │
 │                    │ │ Step 1            │              │                │  │
 │                    │ │ optimizer.step()  │              │                │  │
 │                    │ │                   │              │                │  │
 │                    │ │ on_step_end()     │              │                │  │
 │                    │ ├──────────────────>│              │                │  │
 │                    │ │                   ├─────────────>│                │  │
 │                    │ │                   │              ├───────────────>│  │
 │                    │ │                   │              │ on_step_end() │  │
 │                    │ │                   │              │ <──────────────┤  │
 │                    │ │                   │              │ (no action)   │  │
 │                    │ │                   │ <────────────┤              │  │
 │                    │ │ <─────────────────┤              │              │  │
 │                    │ │                   │              │              │  │
 │                    │ │ ... (steps 2-99)  │              │              │  │
 │                    │ │                   │              │              │  │
 │                    │ │ Step 100          │              │              │  │
 │                    │ │ ┌─ EVALUATION ────────────────────────────────┐│  │
 │                    │ │ │                 │              │            ││  │
 │                    │ │ │ evaluate()      │              │            ││  │
 │                    │ │ │ metrics = {     │              │            ││  │
 │                    │ │ │   "eval_loss": 0.45           │            ││  │
 │                    │ │ │ }               │              │            ││  │
 │                    │ │ │                 │              │            ││  │
 │                    │ │ │ on_evaluate()   │              │            ││  │
 │                    │ │ ├────────────────>│              │            ││  │
 │                    │ │ │                 ├─────────────>│            ││  │
 │                    │ │ │                 │              ├───────────>││  │
 │                    │ │ │                 │              │on_evaluate││  │
 │                    │ │ │                 │              │(ctx, {    ││  │
 │                    │ │ │                 │              │  "eval_   ││  │
 │                    │ │ │                 │              │   loss":  ││  │
 │                    │ │ │                 │              │   0.45    ││  │
 │                    │ │ │                 │              │})         ││  │
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ ┌──────────────┐
 │                    │ │ │                 │              │ │best_loss=inf │
 │                    │ │ │                 │              │ │0.45 < inf    │
 │                    │ │ │                 │              │ │best_loss=0.45│
 │                    │ │ │                 │              │ │wait = 0      │
 │                    │ │ │                 │              │ └──────────────┘
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ <──────────││  │
 │                    │ │ │                 │ <────────────┤           ││  │
 │                    │ │ │ <───────────────┤              │           ││  │
 │                    │ │ └──────────────────────────────────────────────┘│  │
 │                    │ │                   │              │              │  │
 │                    │ │ ... (steps 101-199)              │              │  │
 │                    │ │                   │              │              │  │
 │                    │ │ Step 200          │              │              │  │
 │                    │ │ ┌─ EVALUATION ────────────────────────────────┐│  │
 │                    │ │ │ metrics = {     │              │            ││  │
 │                    │ │ │   "eval_loss": 0.48  (WORSE!) │            ││  │
 │                    │ │ │ }               │              │            ││  │
 │                    │ │ │                 │              │            ││  │
 │                    │ │ │ on_evaluate()   │              │            ││  │
 │                    │ │ ├────────────────>│              │            ││  │
 │                    │ │ │                 ├─────────────>│            ││  │
 │                    │ │ │                 │              ├───────────>││  │
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ ┌──────────────┐
 │                    │ │ │                 │              │ │0.48 >= 0.45  │
 │                    │ │ │                 │              │ │wait += 1     │
 │                    │ │ │                 │              │ │wait = 1      │
 │                    │ │ │                 │              │ │1 < 3         │
 │                    │ │ │                 │              │ │(continue)    │
 │                    │ │ │                 │              │ └──────────────┘
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ <──────────││  │
 │                    │ │ │                 │ <────────────┤           ││  │
 │                    │ │ │ <───────────────┤              │           ││  │
 │                    │ │ └──────────────────────────────────────────────┘│  │
 │                    │ │                   │              │              │  │
 │                    │ │ ... (steps 201-299)              │              │  │
 │                    │ │                   │              │              │  │
 │                    │ │ Step 300          │              │              │  │
 │                    │ │ ┌─ EVALUATION ────────────────────────────────┐│  │
 │                    │ │ │ metrics = {     │              │            ││  │
 │                    │ │ │   "eval_loss": 0.50  (WORSE AGAIN!)        ││  │
 │                    │ │ │ }               │              │            ││  │
 │                    │ │ │                 │              │            ││  │
 │                    │ │ │ on_evaluate()   │              │            ││  │
 │                    │ │ ├────────────────>│              │            ││  │
 │                    │ │ │                 ├─────────────>│            ││  │
 │                    │ │ │                 │              ├───────────>││  │
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ ┌──────────────┐
 │                    │ │ │                 │              │ │0.50 >= 0.45  │
 │                    │ │ │                 │              │ │wait += 1     │
 │                    │ │ │                 │              │ │wait = 2      │
 │                    │ │ │                 │              │ │2 < 3         │
 │                    │ │ │                 │              │ │(continue)    │
 │                    │ │ │                 │              │ └──────────────┘
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ <──────────││  │
 │                    │ │ │                 │ <────────────┤           ││  │
 │                    │ │ │ <───────────────┤              │           ││  │
 │                    │ │ └──────────────────────────────────────────────┘│  │
 │                    │ │                   │              │              │  │
 │                    │ │ ... (steps 301-399)              │              │  │
 │                    │ │                   │              │              │  │
 │                    │ │ Step 400          │              │              │  │
 │                    │ │ ┌─ EVALUATION ────────────────────────────────┐│  │
 │                    │ │ │ metrics = {     │              │            ││  │
 │                    │ │ │   "eval_loss": 0.52  (WORSE THIRD TIME!)   ││  │
 │                    │ │ │ }               │              │            ││  │
 │                    │ │ │                 │              │            ││  │
 │                    │ │ │ on_evaluate()   │              │            ││  │
 │                    │ │ ├────────────────>│              │            ││  │
 │                    │ │ │                 ├─────────────>│            ││  │
 │                    │ │ │                 │              ├───────────>││  │
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ ┌──────────────┐
 │                    │ │ │                 │              │ │0.52 >= 0.45  │
 │                    │ │ │                 │              │ │wait += 1     │
 │                    │ │ │                 │              │ │wait = 3      │
 │                    │ │ │                 │              │ │3 >= 3 ✓      │
 │                    │ │ │                 │              │ │              │
 │                    │ │ │                 │              │ │ctx.abort_    │
 │                    │ │ │                 │              │ │  training()  │
 │                    │ │ │                 │              │ │              │
 │                    │ │ │                 │              │ │ctx._should_  │
 │                    │ │ │                 │              │ │  stop_       │
 │                    │ │ │                 │              │ │  training    │
 │                    │ │ │                 │              │ │  = True      │
 │                    │ │ │                 │              │ └──────────────┘
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │              │ <──────────││  │
 │                    │ │ │                 │              │           ││  │
 │                    │ │ │                 │ control.should_training_ ││  │
 │                    │ │ │                 │        stop = True       ││  │
 │                    │ │ │                 │ <────────────┤           ││  │
 │                    │ │ │ <───────────────┤              │           ││  │
 │                    │ │ │ control.should_ │              │           ││  │
 │                    │ │ │   training_stop │              │           ││  │
 │                    │ │ │   = True        │              │           ││  │
 │                    │ │ └──────────────────────────────────────────────┘│  │
 │                    │ │                   │              │              │  │
 │                    │ │ if control.should_training_stop: │              │  │
 │                    │ │     break  ◄─────────────────────────────────────  │
 │                    │ └───────────────────────────────────────────────────┘  │
 │                    │                     │              │                   │
 │                    │ on_train_end()      │              │                   │
 │                    ├────────────────────>│              │                   │
 │                    │                     ├─────────────>│                   │
 │                    │                     │              ├──────────────────>│
 │                    │                     │              │  on_train_end()   │
 │                    │                     │              │ <─────────────────┤
 │                    │                     │ <────────────┤                   │
 │                    │ <───────────────────┤              │                   │
 │                    │                     │              │                   │
 │ <──────────────────┤                     │              │                   │
 │ (training stopped  │                     │              │                   │
 │  at step 400)      │                     │              │                   │
 │                    │                     │              │                   │
```

---

## 9. Data Flow Diagrams

### 9.1 State Data Flow (Read Path)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA FLOW: Backend State → User Callback (Read Path)                    │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│ TRAINER             │
│ Internal State      │
├─────────────────────┤
│ • optimizer         │───┐
│ • lr_scheduler      │   │
│ • model             │   │
│ • tokenizer         │   │
│ • dataloaders       │   │
└─────────────────────┘   │
                          │
                          ├─> **kwargs
                          │
┌─────────────────────┐   │   ┌─────────────────────┐
│ TrainingArguments   │───┼──>│ args                │
├─────────────────────┤   │   └─────────────────────┘
│ • learning_rate     │   │
│ • batch_size        │   │
│ • num_epochs        │   │
│ • logging_steps     │   │
│ • eval_steps        │   │
│ • save_steps        │   │
│ • output_dir        │   │
│ • ... (100+ fields) │   │
└─────────────────────┘   │
                          │
┌─────────────────────┐   │   ┌─────────────────────┐
│ TrainerState        │───┼──>│ state               │
├─────────────────────┤   │   └─────────────────────┘
│ • global_step: 1000 │   │
│ • epoch: 2.5        │   │
│ • max_steps: 10000  │   │
│ • log_history: [    │   │
│     {               │   │
│       "loss": 0.34, │   │
│       "lr": 2e-5    │   │
│     },              │   │
│     ...             │   │
│   ]                 │   │
│ • best_metric: 0.28 │   │
│ • best_checkpoint:  │   │
│   "ckpt-800"        │   │
│ • tokens_seen:      │   │
│   5000000           │   │
│ • is_world_process_ │   │
│   zero: True        │   │
└─────────────────────┘   │
                          │
┌─────────────────────┐   │
│ TrainerControl      │   │   (not used in read path)
├─────────────────────┤   │
│ • should_training_  │   │
│   stop: False       │   │
│ • should_save: False│   │
│ • ...               │   │
└─────────────────────┘   │
                          │
          ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
     ┌────────────────────────────┐
     │ ADAPTER                    │
     │ _state_to_context()        │
     └────────────────────────────┘
          │
          │ Extract & Transform
          │
          ├─> global_step = state.global_step
          ├─> epoch = state.epoch
          ├─> current_metrics = state.log_history[-1]
          ├─> metric_history = state.log_history.copy()
          ├─> best_metric = state.best_metric
          ├─> best_checkpoint_path = state.best_model_checkpoint
          ├─> tokens_seen = state.num_input_tokens_seen
          ├─> is_main_process = state.is_world_process_zero
          ├─> model = kwargs['model']
          ├─> tokenizer = kwargs['tokenizer']
          └─> backend_args = args

          ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼

┌─────────────────────────────────────────┐
│ UnifiedContext                          │
├─────────────────────────────────────────┤
│ State (Read-Only for User):             │
│ • global_step: 1000                     │
│ • epoch: 2.5                            │
│ • max_steps: 10000                      │
│ • current_metrics: {                    │
│     "loss": 0.34,                       │
│     "learning_rate": 2e-5               │
│   }                                     │
│ • metric_history: [                     │
│     {"loss": 0.50, "lr": 2e-5},        │
│     {"loss": 0.42, "lr": 1.9e-5},      │
│     {"loss": 0.34, "lr": 2e-5},        │
│   ]                                     │
│ • best_metric: 0.28                     │
│ • best_checkpoint_path: "ckpt-800"      │
│ • tokens_seen: 5000000                  │
│ • is_main_process: True                 │
│                                         │
│ References:                             │
│ • model: LlamaForCausalLM(...)          │
│ • tokenizer: LlamaTokenizer(...)        │
│ • backend_args: TrainingArguments(...)  │
│                                         │
│ Control Methods:                        │
│ • abort_training()                      │
│ • trigger_checkpoint()                  │
│ • get_metric(key)                       │
└─────────────────────────────────────────┘
          │
          │ Passed to user callback
          ▼

┌─────────────────────────────────────────┐
│ USER CALLBACK                           │
│ on_step_end(ctx: UnifiedContext)        │
├─────────────────────────────────────────┤
│                                         │
│ # Read state                            │
│ step = ctx.global_step                  │
│ loss = ctx.get_metric("loss")           │
│ best = ctx.best_metric                  │
│                                         │
│ # Make decisions                        │
│ if step > 1000:                         │
│     ctx.abort_training()                │
│                                         │
│ if step % 100 == 0:                     │
│     ctx.trigger_checkpoint()            │
│                                         │
└─────────────────────────────────────────┘
```

### 9.2 Control Data Flow (Write Path)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA FLOW: User Callback → Backend Control (Write Path)                 │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ USER CALLBACK                           │
│ on_evaluate(ctx, metrics)               │
├─────────────────────────────────────────┤
│                                         │
│ loss = metrics.get("eval_loss")         │
│ if loss < 0.1:                          │
│     ctx.abort_training()  ◄─────────────┼─── User Decision
│     ctx.trigger_checkpoint()            │
│                                         │
└─────────────────────────────────────────┘
          │
          │ Returns modified ctx
          ▼

┌─────────────────────────────────────────┐
│ UnifiedContext (Modified)               │
├─────────────────────────────────────────┤
│ Internal Flags (Private):               │
│ • _should_stop_training: True  ◄────────┼─── Set by abort_training()
│ • _should_stop_epoch: False             │
│ • _should_save_now: True  ◄─────────────┼─── Set by trigger_checkpoint()
│ • _should_evaluate_now: False           │
│ • _should_log_now: False                │
└─────────────────────────────────────────┘
          │
          │ Passed to adapter
          ▼

     ┌────────────────────────────┐
     │ ADAPTER                    │
     │ _context_to_control()      │
     └────────────────────────────┘
          │
          │ Map flags
          │
          ├─> if ctx._should_stop_training:
          │       control.should_training_stop = True
          │
          ├─> if ctx._should_stop_epoch:
          │       control.should_epoch_stop = True
          │
          ├─> if ctx._should_save_now:
          │       control.should_save = True  ◄────┐
          │                                        │
          ├─> if ctx._should_evaluate_now:         │
          │       control.should_evaluate = True   │
          │                                        │
          └─> if ctx._should_log_now:              │
                  control.should_log = True        │

          ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼                         │
                                                   │
┌─────────────────────┐                            │
│ TrainerControl      │◄───────────────────────────┘
│ (Modified)          │
├─────────────────────┤
│ BEFORE:             │         AFTER:
│ • should_training_  │         • should_training_
│   stop: False       │  ────>    stop: True  ◄─── Mapped
│ • should_epoch_stop:│         • should_epoch_stop:
│   False             │           False
│ • should_save:      │         • should_save:
│   False             │  ────>    True  ◄─── Mapped
│ • should_evaluate:  │         • should_evaluate:
│   False             │           False
│ • should_log:       │         • should_log:
│   False             │           False
└─────────────────────┘
          │
          │ Returned to trainer
          ▼

┌─────────────────────────────────────────┐
│ TRAINER                                 │
│ Training Loop                           │
├─────────────────────────────────────────┤
│                                         │
│ control = callback_handler.on_evaluate( │
│     args, state, control, metrics       │
│ )                                       │
│                                         │
│ # Check control flags                   │
│ if control.should_training_stop:        │◄─── Persistent flag
│     break  # Stop training              │     triggers training stop
│                                         │
│ if control.should_save:                 │◄─── Transient flag
│     self.save_checkpoint()              │     triggers checkpoint
│     control.should_save = False         │     then auto-resets
│                                         │
│ if control.should_evaluate:             │
│     self.evaluate()                     │
│     control.should_evaluate = False     │
│                                         │
└─────────────────────────────────────────┘
          │
          │ Actions executed:
          ▼

┌─────────────────────────────────────────┐
│ EFFECTS                                 │
├─────────────────────────────────────────┤
│ ✓ Checkpoint saved to disk              │
│ ✓ Training loop breaks                  │
│ ✓ on_train_end() callbacks fire         │
│ ✓ Trainer.train() returns               │
└─────────────────────────────────────────┘
```

---

## Summary

This document provides comprehensive visual representations of the Unified Callback System architecture:

1. **System Architecture**: 4-layer design (User → Abstraction → Adapter → Backend)
2. **Component Layers**: Detailed class responsibilities and data flow
3. **Callback Execution**: Sequential event handling with control threading
4. **State Translation**: Bidirectional mapping between backend and unified interfaces
5. **Event Lifecycle**: Complete training flow with 15 event hooks
6. **Backend Injection**: Standard vs. two-phase patterns across frameworks
7. **Adapter Pattern**: Gang of Four design pattern implementation
8. **Sequence Diagrams**: Real-world early stopping scenario
9. **Data Flow**: Read path (state) and write path (control) separation

The architecture ensures:
- **Single responsibility**: Each layer has one job
- **Abstraction**: Users never see backend complexity
- **Compatibility**: Works across 4 different frameworks
- **Extensibility**: Easy to add new backends or callbacks
- **Performance**: Minimal overhead (< 1% training time)
