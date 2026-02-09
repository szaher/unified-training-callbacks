# RFE: Callback Mechanism Implementation for InstructLab Training and Mini-Trainer

## JIRA RFE Template

---

## Summary

Add callback hook mechanisms to InstructLab Training and Mini-Trainer libraries to enable extensible training event monitoring, logging, and control. This RFE focuses on implementing the foundational callback support in these two training backends and integrating them with Training-Hub, creating the infrastructure needed for the unified callback system (covered in a separate RFE).

---

## Component

**Product**: Red Hat OpenShift AI (RHOAI)

**Components**:
- InstructLab Training Library (https://github.com/instructlab/training)
- Mini-Trainer Library (https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer)
- Training-Hub Library (https://github.com/Red-Hat-AI-Innovation-Team/training_hub)

**Delivery**: Feature enhancement to training libraries and Training-Hub backend integration

**Version Targets**:
- InstructLab Training >= 0.4.0
- Mini-Trainer >= 0.2.0
- Training-Hub >= 1.0.0

---

## Description

### Problem Statement

InstructLab Training and Mini-Trainer currently lack formal callback mechanisms, making it impossible for users to:

**Current Limitations:**

| Backend | Current State | Pain Points |
|---------|--------------|-------------|
| **InstructLab Training** | No formal callbacks, hardcoded events in training loop | Users cannot inject custom monitoring, early stopping, or compliance logging without modifying source code |
| **Mini-Trainer** | No formal callbacks, hardcoded logging/checkpointing | Teams must fork the repository to add custom instrumentation |

**Impact on Users:**

1. **No Extensibility**: Cannot add custom logging, monitoring, or compliance tracking
   - Enterprise users need audit trails but can't inject logging without code modification
   - Research teams want custom metrics but must modify library source
   - FinOps teams need cost tracking but have no hook points

2. **Code Modification Required**: Users must fork or modify library code
   - Forks diverge from upstream, missing bug fixes and features
   - Code modifications break with library updates
   - No standard extension mechanism

3. **Training-Hub Integration Gap**: Training-Hub cannot provide consistent callback support
   - Unsloth backend supports callbacks (via HuggingFace), but InstructLab and Mini-Trainer don't
   - Users cannot pass callbacks through Training-Hub API
   - Backend-specific instrumentation required

4. **Enterprise Adoption Blocker**: Fortune 500 companies require callback systems
   - Compliance: SOC2, ISO27001, HIPAA require audit trails
   - Cost Control: FinOps teams need GPU cost tracking
   - Security: Security teams need training event hooks for SIEM integration
   - No callbacks = limited enterprise adoption

**Customer Impact:**

- **Financial Services**: Cannot add compliance logging without forking
- **Healthcare/Life Sciences**: Cannot implement HIPAA validation hooks
- **Government/Defense**: Cannot add security event logging
- **Research Teams**: Cannot track custom metrics without code modification

### Why Callback Mechanisms Are Essential

Callback mechanisms are **architectural necessities** for production ML libraries. See `RFE_Training_Hub_Unified_Callbacks.md` for detailed justification, but key points:

1. **Separation of Concerns**: Training libraries should focus on training; everything else (monitoring, logging, compliance) should be externalized through callbacks

2. **Extensibility Without Code Modification**: Production ML systems require features the library author never anticipated (audit logging, cost tracking, bias detection)

3. **Industry Standard**: All major ML frameworks have callback systems (HuggingFace Transformers, PyTorch Lightning, Keras, TensorFlow)

4. **Enterprise Requirement**: Fortune 500 companies will not adopt training libraries without callbacks

**Without callbacks, users resort to anti-patterns:**
- Forking the repository (100+ hours/year merge conflicts)
- Wrapper scripts (can't access internal state)
- Monkey patching (breaks with updates)
- Copy-paste programming (technical debt)

### Proposed Solution

Implement callback hook mechanisms in InstructLab Training and Mini-Trainer training loops, and integrate them with Training-Hub backends.

**Approach:**

1. **Code Enhancement Strategy**: Add callback hooks to training loops
   - InstructLab Training: Enhance `main_ds.py` with callback hooks
   - Mini-Trainer: Enhance `train.py` with callback hooks
   - Minimal, localized changes compatible with upstream updates

2. **Standard Lifecycle Events**: Support 15 standard callback hooks
   - `on_init_end`, `on_train_begin`, `on_epoch_begin`, `on_step_begin`
   - `on_substep_end`, `on_pre_optimizer_step`, `on_optimizer_step`
   - `on_log`, `on_save`, `on_evaluate`, `on_step_end`
   - `on_epoch_end`, `on_train_end`, `on_predict`, `on_prediction_step`

3. **Training-Hub Integration**: Add callback injection in Training-Hub backends
   - InstructLab backend: Inject CallbackHandler into training loop
   - Mini-Trainer backend: Inject CallbackHandler into training loop
   - Callbacks passed through `sft()`, `osft()`, `lora_sft()` APIs

4. **Minimal Interface**: Simple callback handler and event dispatcher
   - `CallbackHandler` class for event dispatching
   - Callback exception handling (callbacks don't crash training)
   - Distributed training support (main process detection)

### Technical Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   TRAINING-HUB PUBLIC API                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  from training_hub import sft, osft                      │  │
│  │                                                          │  │
│  │  sft(model, data, callbacks=[...],                      │  │
│  │      backend='instructlab-training')                     │  │
│  │  osft(model, data, callbacks=[...])  # mini-trainer     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ inject callbacks
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│            TRAINING-HUB BACKEND CALLBACK INJECTION              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  InstructLabBackend.execute_training(callbacks=[...])    │  │
│  │  MiniTrainerBackend.execute_training(callbacks=[...])    │  │
│  │                                                          │  │
│  │  - Creates CallbackHandler                              │  │
│  │  - Passes handler to training loop                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ invoke training with callbacks
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│         TRAINER LIBRARIES (ENHANCED WITH CALLBACKS)             │
│  ┌────────────────┐              ┌─────────────────────────┐   │
│  │  InstructLab   │              │    Mini-Trainer         │   │
│  │   Training     │              │     (OSFT)              │   │
│  │  - Enhanced    │              │   - Enhanced            │   │
│  │    main_ds.py  │              │     train.py            │   │
│  │                │              │                         │   │
│  │  Hooks Added:  │              │   Hooks Added:          │   │
│  │  - train begin │              │   - train begin         │   │
│  │  - step end    │              │   - step end            │   │
│  │  - on_log      │              │   - on_log              │   │
│  │  - on_save     │              │   - on_save             │   │
│  │  - on_evaluate │              │   - on_evaluate         │   │
│  │  - train end   │              │   - train end           │   │
│  └────────────────┘              └─────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Core Components

**1. CallbackHandler (Event Dispatcher)**

```python
# File: training_hub/callbacks/handler.py

class CallbackHandler:
    """Manages callback execution for training events."""

    def __init__(self, callbacks: List):
        self.callbacks = callbacks

    def fire_event(self, event_name: str, *args, **kwargs):
        """Fire an event to all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, event_name, None)
            if method and callable(method):
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    # Log error but don't stop training
                    print(f"Callback {callback.__class__.__name__}.{event_name} failed: {e}")
```

**2. InstructLab Training Enhancement**

```python
# File: instructlab/training/main_ds.py (ENHANCED)

def train_with_callbacks(model, train_loader, callback_handler=None, **kwargs):
    """Enhanced InstructLab training loop with callback support."""

    # Existing InstructLab initialization...
    accelerator = AcceleratorWrapper(...)
    optimizer = torch.optim.AdamW(...)

    # Callback: on_init_end
    if callback_handler:
        callback_handler.fire_event('on_init_end', model=model, optimizer=optimizer)

    # Callback: on_train_begin
    model.train()
    if callback_handler:
        callback_handler.fire_event('on_train_begin', model=model)

    for epoch in range(num_epochs):
        # Callback: on_epoch_begin
        if callback_handler:
            callback_handler.fire_event('on_epoch_begin', epoch=epoch)

        for batch in train_loader:
            global_step += 1

            # Callback: on_step_begin
            if callback_handler:
                callback_handler.fire_event('on_step_begin', step=global_step)

            # Existing InstructLab batch processing
            loss = batch_loss_manager.process_batch(batch)

            # Callback: on_step_end
            if callback_handler:
                callback_handler.fire_event('on_step_end', step=global_step, loss=loss.item())

            # Logging (with callback)
            if step % logging_steps == 0:
                metrics = {"loss": loss.item(), "lr": lr, "step": step}
                if callback_handler:
                    callback_handler.fire_event('on_log', metrics=metrics)

            # Checkpointing (with callback)
            if should_save_checkpoint:
                save_checkpoint(...)
                if callback_handler:
                    callback_handler.fire_event('on_save', step=global_step)

            # Evaluation (with callback)
            if should_evaluate:
                eval_metrics = compute_validation_loss(...)
                if callback_handler:
                    callback_handler.fire_event('on_evaluate', metrics=eval_metrics)

        # Callback: on_epoch_end
        if callback_handler:
            callback_handler.fire_event('on_epoch_end', epoch=epoch)

    # Callback: on_train_end
    if callback_handler:
        callback_handler.fire_event('on_train_end')
```

**3. Mini-Trainer Enhancement**

```python
# File: mini_trainer/train.py (ENHANCED)

def train_with_callbacks(model, data, callback_handler=None, **kwargs):
    """Enhanced mini_trainer loop with callback support."""

    # Similar structure to InstructLab integration
    # Add callback hooks at key points:
    # - on_init_end
    # - on_train_begin
    # - on_step_begin
    # - on_step_end
    # - on_log (hook into AsyncStructuredLogger)
    # - on_save (hook into Checkpointer)
    # - on_evaluate (hook into compute_validation_loss)
    # - on_train_end
```

**4. Training-Hub Backend Integration**

```python
# File: training_hub/backends/instructlab.py

from training_hub.callbacks.handler import CallbackHandler

class InstructLabBackend(Backend):
    def execute_training(self, ..., callbacks=None, **kwargs):
        # Initialize callback handler
        handler = CallbackHandler(callbacks or [])

        # Call modified InstructLab training function with callback support
        train_with_callbacks(
            model=model,
            train_loader=train_loader,
            callback_handler=handler,
            **kwargs
        )

# File: training_hub/backends/mini_trainer.py

class MiniTrainerBackend(Backend):
    def execute_training(self, ..., callbacks=None, **kwargs):
        # Similar to InstructLab approach
        handler = CallbackHandler(callbacks or [])

        # Call enhanced mini_trainer training function
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

**For InstructLab Training Users:**
- **Extensibility**: Add custom monitoring, early stopping, compliance logging without forking
- **Enterprise Ready**: Meet SOC2, HIPAA, ISO27001 requirements via callback-based audit logging
- **Research Velocity**: Experiment with custom metrics, dynamic hyperparameters

**For Mini-Trainer Users:**
- **Continual Learning Instrumentation**: Track custom metrics for OSFT workflows
- **Async Logger Integration**: Callbacks work alongside existing AsyncStructuredLogger
- **Cost Tracking**: Monitor GPU costs for continual learning experiments

**For Training-Hub:**
- **Consistent API**: Callbacks can be passed to both backends through Training-Hub
- **Foundation for Unification**: Enables future unified callback system (separate RFE)
- **Backend Parity**: InstructLab and Mini-Trainer reach feature parity with Unsloth (which has callbacks via HuggingFace)

**For Enterprise Organizations:**
- **Compliance**: Unified audit trails across InstructLab and Mini-Trainer training runs
- **Cost Control**: Consistent GPU cost tracking regardless of backend
- **Risk Reduction**: Standardized security event logging, model governance

### Competitive Positioning

- **InstructLab Training**: Currently behind HuggingFace, TRL, Axolotl in extensibility → Callbacks make it enterprise-ready
- **Mini-Trainer**: Unique continual learning focus → Callbacks enable custom metric tracking for research
- **Training-Hub**: First to provide callback support across diverse backends → Competitive differentiation

---

## Acceptance Criteria

### Functional Requirements

**FR1: InstructLab Training Callback Hooks**
- [ ] Enhanced `main_ds.py` with 15 callback hooks
- [ ] CallbackHandler integrated into training loop
- [ ] Distributed training support (main process detection)
- [ ] Backward compatible (training works without callbacks)

**FR2: Mini-Trainer Callback Hooks**
- [ ] Enhanced `train.py` with 15 callback hooks
- [ ] CallbackHandler integrated into training loop
- [ ] AsyncStructuredLogger compatibility maintained
- [ ] Backward compatible (training works without callbacks)

**FR3: Training-Hub Backend Integration**
- [ ] InstructLabBackend accepts `callbacks` parameter
- [ ] MiniTrainerBackend accepts `callbacks` parameter
- [ ] `sft()` function accepts `callbacks` parameter (InstructLab)
- [ ] `osft()` function accepts `callbacks` parameter (Mini-Trainer)

**FR4: Documentation**
- [ ] InstructLab Training callback API documentation
- [ ] Mini-Trainer callback API documentation
- [ ] Training-Hub integration examples
- [ ] Migration guide for existing users

### Non-Functional Requirements

**NFR1: Performance**
- [ ] Callback overhead < 2% of total training time
- [ ] Memory overhead < 10MB per callback instance
- [ ] No impact on distributed training scalability (tested up to 64 GPUs with InstructLab)

**NFR2: Compatibility**
- [ ] Works with InstructLab Training >= 0.3.0
- [ ] Works with Mini-Trainer >= 0.1.0
- [ ] Does not break existing Training-Hub API (backward compatible)
- [ ] Python 3.10+ support

**NFR3: Reliability**
- [ ] 90%+ unit test coverage for callback hooks
- [ ] Integration tests for InstructLab and Mini-Trainer
- [ ] Distributed training tests (multi-GPU, multi-node)
- [ ] Callback exception handling (callbacks don't crash training)

**NFR4: Maintainability**
- [ ] Code changes are minimal and localized
- [ ] Compatible with upstream updates
- [ ] Clear separation between callback system and training logic
- [ ] No breaking changes to existing trainer APIs

### Success Metrics

**Adoption Metrics:**
- 30%+ of InstructLab users adopt callbacks within 6 months
- 30%+ of Mini-Trainer users adopt callbacks within 6 months
- Featured in InstructLab and Mini-Trainer documentation

**Technical Metrics:**
- Zero reported bugs causing training failures
- < 2% performance overhead
- 100% backward compatibility

**Business Metrics:**
- Enables unified callback RFE (separate initiative)
- Unblocks 3+ enterprise customer use cases
- Community feedback positive (GitHub issues, discussions)

---

## Dependencies

### Technical Dependencies

**Required:**
- Python >= 3.10
- InstructLab Training >= 0.3.0
- Mini-Trainer >= 0.1.0
- Training-Hub (this repository)
- torch >= 2.0

**Development:**
- pytest >= 7.0
- pytest-cov >= 4.0
- black, ruff (code formatting)

### Product Dependencies

**Upstream Coordination:**
- InstructLab Training maintainers: Approve callback hook additions to `main_ds.py`
- Mini-Trainer maintainers: Approve callback hook additions to `train.py`
- Red Hat AI Innovation Team: Coordinate changes across repositories

### Blocking Dependencies

**None** - This RFE can be implemented independently. The unified callback system (separate RFE) will depend on this RFE's completion.

---

## Implementation Plan

### Phase 1: InstructLab Training Enhancement (Weeks 1-3)

**Deliverables:**
- Enhanced `main_ds.py` with callback hooks
- CallbackHandler implementation
- InstructLab backend integration in Training-Hub
- Unit and integration tests

**Critical Files:**
```
instructlab/training/
└── main_ds.py               # ENHANCED with callback hooks

training_hub/
├── callbacks/
│   ├── __init__.py
│   └── handler.py           # CallbackHandler
├── backends/
│   └── instructlab.py       # Callback injection
└── tests/
    └── test_instructlab_callbacks.py
```

**Milestones:**
- [ ] Week 1: CallbackHandler implementation and unit tests
- [ ] Week 2: InstructLab main_ds.py enhancements
- [ ] Week 3: Training-Hub backend integration and testing
- [ ] Pull request to InstructLab repository

### Phase 2: Mini-Trainer Enhancement (Weeks 4-6)

**Deliverables:**
- Enhanced `train.py` with callback hooks
- Mini-Trainer backend integration in Training-Hub
- OSFT workflow testing
- Integration tests

**Critical Files:**
```
mini_trainer/
└── train.py                 # ENHANCED with callback hooks

training_hub/
├── backends/
│   └── mini_trainer.py      # Callback injection
└── tests/
    └── test_mini_trainer_callbacks.py
```

**Milestones:**
- [ ] Week 4: Mini-Trainer train.py enhancements
- [ ] Week 5: Training-Hub backend integration
- [ ] Week 6: OSFT workflow testing and performance benchmarks
- [ ] Pull request to Mini-Trainer repository

### Phase 3: Documentation & Testing (Week 7)

**Deliverables:**
- Comprehensive documentation (API ref, user guide, examples)
- Example notebooks
- Performance benchmarks
- Migration guide

**Critical Files:**
```
instructlab/training/docs/
└── callbacks_guide.md

mini_trainer/docs/
└── callbacks_guide.md

training_hub/
├── examples/
│   ├── instructlab_callbacks_example.ipynb
│   └── mini_trainer_callbacks_example.ipynb
└── docs/
    └── backend_callbacks_guide.md
```

**Milestones:**
- [ ] Documentation complete and reviewed
- [ ] Example notebooks runnable on RHOAI
- [ ] Performance benchmarks published
- [ ] Migration guide for existing users

---

## Testing Requirements

### Unit Tests

**Coverage Target: 90%+**

**CallbackHandler Tests:**
- Event dispatching (all 15 events)
- Exception handling (callbacks don't crash training)
- Multiple callback coordination
- Empty callback list handling

### Integration Tests

**InstructLab Training Integration:**
- Train with InstructLab backend + callbacks
- Distributed training test (2 GPUs minimum)
- Verify callbacks only execute on main process
- Test callback exception handling
- Verify all 15 events fire in correct order

**Mini-Trainer Integration:**
- Train with Mini-Trainer backend + callbacks
- OSFT workflow with custom callbacks
- Verify AsyncStructuredLogger compatibility
- Test callback overhead (< 2%)
- Verify all 15 events fire in correct order

### Distributed Training Tests

**Multi-GPU Validation:**
- Verify main process detection works correctly
- Ensure callbacks don't duplicate logs across processes
- Test distributed checkpoint saving with callbacks
- Validate metric aggregation across ranks

### Performance Tests

**Benchmarking:**
- Measure callback overhead on InstructLab (4x A100 GPUs)
- Measure callback overhead on Mini-Trainer (single GPU)
- Target: < 2% overhead with typical callbacks

### Regression Tests

**Backward Compatibility:**
- Existing InstructLab code works without `callbacks` parameter
- Existing Mini-Trainer code works without `callbacks` parameter
- No performance regression when callbacks not used
- Backend selection logic unchanged

---

## Documentation Requirements

### User-Facing Documentation

**1. InstructLab Training Callback Guide** (`instructlab/training/docs/callbacks_guide.md`)
- What are callbacks and why use them?
- Writing custom callbacks for InstructLab
- Available callback hooks
- Distributed training considerations
- Best practices

**2. Mini-Trainer Callback Guide** (`mini_trainer/docs/callbacks_guide.md`)
- What are callbacks and why use them?
- Writing custom callbacks for Mini-Trainer
- Available callback hooks
- OSFT workflow integration
- Best practices

**3. Training-Hub Integration Guide** (`training_hub/docs/backend_callbacks_guide.md`)
- Using callbacks with InstructLab backend
- Using callbacks with Mini-Trainer backend
- Passing callbacks through Training-Hub API
- Example notebooks

**4. Example Notebooks**
- `examples/instructlab_callbacks_example.ipynb`: Distributed SFT with custom logging
- `examples/mini_trainer_callbacks_example.ipynb`: OSFT with custom metrics

### Developer Documentation

**1. Architecture Document** (`docs/callback_implementation_architecture.md`)
- Why callbacks for InstructLab and Mini-Trainer?
- Code enhancement strategy
- Lifecycle event mapping
- Design decisions and trade-offs

**2. Contributing Guide** (`docs/contributing_callbacks.md`)
- How to add new callback hooks
- Testing requirements
- Code review process
- Pull request guidelines

---

## Risks and Mitigation

### Technical Risks

**Risk 1: Upstream Rejection**
- **Description**: InstructLab or Mini-Trainer maintainers reject callback hook PRs
- **Likelihood**: Low (same Red Hat organization)
- **Impact**: High
- **Mitigation**:
  - Engage Red Hat AI Innovation Team early
  - Demonstrate value with working prototype
  - Keep changes minimal and localized
  - Offer to maintain callback code

**Risk 2: Performance Degradation**
- **Description**: Callback overhead > 2% in InstructLab or Mini-Trainer
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**:
  - Profile callback dispatch thoroughly
  - Make callback execution conditional (can be disabled)
  - Optimize hot paths in CallbackHandler
  - Benchmark against baseline

**Risk 3: Distributed Training Edge Cases**
- **Description**: Callbacks behave incorrectly in multi-node InstructLab training
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**:
  - Extensive multi-GPU/multi-node testing
  - Clear documentation of main-process-only behavior
  - Distributed training examples
  - Callback author guidelines

**Risk 4: Breaking Changes**
- **Description**: Callback enhancements break existing trainer functionality
- **Likelihood**: Low
- **Impact**: High
- **Mitigation**:
  - Callbacks are optional (default None)
  - Extensive regression testing
  - Backward compatibility tests
  - Careful code review

### Product Risks

**Risk 5: Low Adoption**
- **Description**: Users continue modifying source code instead of using callbacks
- **Likelihood**: Low
- **Impact**: Medium
- **Mitigation**:
  - Include in official documentation
  - Provide compelling examples
  - Demonstrate value in Training-Hub examples
  - Outreach to top users

**Risk 6: Maintenance Burden**
- **Description**: Callback system increases maintenance burden
- **Likelihood**: Medium
- **Impact**: Low
- **Mitigation**:
  - Keep callback system design simple
  - Automated testing reduces manual validation
  - Clear ownership and documentation
  - Minimize changes to core training logic

---

## Future Enhancements (Post-MVP)

### Follow-On Work

**1. Unified Callback System** (Separate RFE)
- Build on this RFE to create unified callback abstraction
- TrainingHubCallback base class
- TrainingHubContext for normalized state
- Enterprise callback library

**2. Additional Callback Hooks**
- More granular hooks based on user feedback
- Custom events for InstructLab-specific features
- OSFT-specific events for Mini-Trainer

**3. Callback Debugging Tools**
- Callback execution tracing
- Performance profiling per callback
- Visual callback flow diagrams

---

## References

### Design Documents

- **Unified Callback RFE**: `./RFE_Training_Hub_Unified_Callbacks.md` (separate, follow-on RFE)
- **Architecture Specification**: `./architecture.md`
- **TRL Callback Analysis**: `./trl_callback_analysis.md`
- **HuggingFace Callback Analysis**: `./huggingface_transformers_callback_analysis.md`

### Repositories

- **InstructLab Training**: https://github.com/instructlab/training
- **Mini-Trainer**: https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer
- **Training-Hub**: https://github.com/Red-Hat-AI-Innovation-Team/training_hub

### Upstream Projects

- **HuggingFace Transformers**: https://github.com/huggingface/transformers
- **TRL**: https://github.com/huggingface/trl

---

## Contact / Stakeholders

**Product Owner**: [TBD - RHOAI Product Manager]
**Engineering Lead**: [TBD - Red Hat AI Innovation Team Lead]
**Technical Lead**: Sameh Zaher (szaher@redhat.com)
**Documentation Lead**: [TBD - RHOAI Technical Writer]
**QE Lead**: [TBD - RHOAI QE Manager]

**Reviewers:**
- Red Hat AI Innovation Team (Training-Hub maintainers)
- InstructLab Training maintainers
- Mini-Trainer maintainers

**Critical Stakeholders:**
- **InstructLab Team**: Approve callback hooks in training loop
- **Mini-Trainer Team**: Approve callback hooks in training loop

---

## Appendix A: Standard Callback Events

All 15 standard callback events to be supported:

1. **on_init_end**: Called after backend initialization, before training
2. **on_train_begin**: Called at the start of training
3. **on_epoch_begin**: Called at the beginning of an epoch
4. **on_step_begin**: Called at the beginning of a training step
5. **on_substep_end**: Called after each gradient accumulation substep
6. **on_pre_optimizer_step**: Called before optimizer.step() (after gradient clipping)
7. **on_optimizer_step**: Called after optimizer.step()
8. **on_log**: Called when metrics are logged
9. **on_save**: Called when a checkpoint is saved
10. **on_evaluate**: Called after evaluation
11. **on_step_end**: Called at the end of a training step
12. **on_epoch_end**: Called at the end of an epoch
13. **on_train_end**: Called at the end of training
14. **on_predict**: Called before prediction/inference
15. **on_prediction_step**: Called during each prediction step

---

## Appendix B: Example Usage

### Example 1: Simple Logging Callback (InstructLab)

```python
class SimpleLoggingCallback:
    """Simple callback that logs training events."""

    def on_train_begin(self, **kwargs):
        print("Training started!")

    def on_step_end(self, step, loss, **kwargs):
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss:.4f}")

    def on_train_end(self, **kwargs):
        print("Training completed!")

# Use with InstructLab via Training-Hub
from training_hub import sft

sft(
    base_model="meta-llama/Llama-2-7b-hf",
    data_path="./data.jsonl",
    backend="instructlab-training",
    callbacks=[SimpleLoggingCallback()],
)
```

### Example 2: Cost Tracking Callback (Mini-Trainer)

```python
import time

class CostTrackingCallback:
    """Track GPU costs during training."""

    def __init__(self, gpu_cost_per_hour=2.50):
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.start_time = None

    def on_train_begin(self, **kwargs):
        self.start_time = time.time()
        print(f"Starting cost tracking at ${self.gpu_cost_per_hour}/hour")

    def on_train_end(self, **kwargs):
        duration_hours = (time.time() - self.start_time) / 3600
        total_cost = duration_hours * self.gpu_cost_per_hour
        print(f"Training Cost: ${total_cost:.2f} ({duration_hours:.2f} hours)")

# Use with Mini-Trainer via Training-Hub
from training_hub import osft

osft(
    base_model="meta-llama/Llama-2-7b-hf",
    data_path="./data.jsonl",
    backend="mini-trainer",
    callbacks=[CostTrackingCallback(gpu_cost_per_hour=2.50)],
)
```

---

## END OF RFE
