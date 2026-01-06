# RFE: Ray Train Support for Training-Hub Distributed LLM Training

## JIRA RFE Template

---

## Summary

Add Ray Train support to the Training-Hub library via a unified `RayTrainReportCallback`, enabling data scientists to seamlessly scale LLM fine-tuning workloads from single-node to multi-node distributed clusters on Ray. This RFE proposes a **callback-only approach** that integrates existing Training-Hub backends (InstructLab, Mini-Trainer, Unsloth) with Ray Train's orchestration capabilities while maintaining clear separation of concerns.

---

## Component

**Product**: Red Hat OpenShift AI (RHOAI)
**Component**: Training-Hub Library
**Repository**: https://github.com/Red-Hat-AI-Innovation-Team/training_hub
**Documentation**: https://deepwiki.com/Red-Hat-AI-Innovation-Team/training_hub
**Delivery**: Feature enhancement to existing training_hub Python library
**Version Target**: training_hub 1.1.0 (after unified callbacks in 1.0.0)
**Dependency**: Requires RFE "Unified Callback System" (training_hub 1.0.0)

---

## Description

### Problem Statement

Training-Hub provides a unified API for multiple LLM training backends (InstructLab Training, Mini-Trainer, Unsloth, VERL), but currently lacks native support for distributed training on Ray clusters. This creates significant challenges for enterprise AI/ML teams scaling to production:

**Current State of Distributed Training:**

| Backend | Distributed Support | Mechanism | Ray Cluster Support | Limitations |
|---------|-------------------|-----------|-------------------|-------------|
| **InstructLab Training** | ✅ Native | DeepSpeed/FSDP | ❌ No | Requires manual cluster setup, no fault tolerance |
| **Mini-Trainer** | ⚠️ Limited | PyTorch DDP | ❌ No | Single-node only, no fault tolerance |
| **Unsloth** | ✅ Via Transformers | PyTorch DDP/FSDP | ❌ No | Limited to HuggingFace Accelerate patterns |
| **VERL** | ✅ Native | Ray (internal) | ⚠️ Hardcoded | Cannot use with other backends, vendor lock-in |
| **Ray Train** | ✅ Universal | PyTorch DDP/FSDP/DeepSpeed | ✅ Yes | **Not integrated with Training-Hub** |

**Pain Points:**

1. **No Universal Distributed Training**: Training-Hub users cannot easily scale to multi-node Ray clusters
   - InstructLab Training requires manual DeepSpeed setup across nodes
   - Mini-Trainer limited to single-node training
   - Unsloth relies on HuggingFace Accelerate patterns only
   - No unified way to leverage Ray's fault tolerance and resource management

2. **Limited Fault Tolerance**: Production training runs lack automatic recovery
   - Multi-hour/multi-day LLM training runs fail on single node failures
   - No automatic checkpoint resumption across cluster failures
   - Teams manually implement retry logic and monitoring

3. **Resource Management Gaps**: Cannot leverage Ray's dynamic resource allocation
   - No automatic GPU/CPU allocation across heterogeneous clusters
   - Cannot mix different GPU types (A100, H100, L40S) efficiently
   - No elastic scaling during training

4. **Vendor Lock-In Risk**: VERL has Ray support, but it's backend-specific
   - Users cannot use Ray with InstructLab, Mini-Trainer, or Unsloth
   - Switching to VERL requires rewriting training code
   - No portability across deployment environments

5. **Cloud Deployment Friction**: Difficult to deploy Training-Hub on cloud Ray clusters
   - AWS Anyscale, Azure ML Ray clusters, GCP Vertex AI Ray require custom integration
   - No standard pattern for Training-Hub + Ray deployment
   - Teams build custom wrappers for each cloud provider

**Customer Impact:**

- **Fortune 500 Financial Services**: Need fault-tolerant distributed training for compliance models (72-hour training runs cannot fail at hour 68)
- **Healthcare/Life Sciences**: Require scalable multi-node training for biomedical LLMs (100B+ parameter models)
- **Research Institutions**: Want elastic cluster usage for cost optimization (scale up/down based on cluster availability)
- **Cloud-First Enterprises**: Need seamless deployment on AWS Anyscale, Azure ML Ray, GCP Vertex AI Ray clusters

### Why Ray Train is Essential for Training-Hub

Ray Train is the industry-standard framework for distributed ML training orchestration:

#### 1. **Universal Framework Support**

Ray Train supports ALL major ML frameworks through a unified API:
- PyTorch (via TorchTrainer)
- TensorFlow (via TensorflowTrainer)
- HuggingFace Transformers (native integration)
- HuggingFace Accelerate (native integration)
- PyTorch Lightning (via LightningTrainer)
- DeepSpeed (via TorchTrainer + DeepSpeed config)

**For Training-Hub**: This means ALL backends (InstructLab, Mini-Trainer, Unsloth) can benefit from Ray Train.

#### 2. **Fault Tolerance & Automatic Recovery**

Ray Train provides production-grade fault tolerance:
- **Automatic node failure detection**: Detects and handles worker failures mid-training
- **Checkpoint-based resumption**: Automatically resumes from last checkpoint after failure
- **Health monitoring**: Continuous health checks on distributed workers
- **Gang scheduling**: Ensures all workers start together or none start at all

**Impact**: A 72-hour LLM training run doesn't fail at hour 68 due to single node failure.

#### 3. **Resource Management & Elastic Scaling**

Ray's cluster management enables:
- **Dynamic resource allocation**: Request GPUs/CPUs as needed
- **Heterogeneous cluster support**: Mix A100, H100, L40S GPUs in same cluster
- **Elastic scaling**: Add/remove workers during training (experimental)
- **Resource scheduling**: Queue training jobs when resources unavailable

**Impact**: Maximize GPU utilization and minimize costs.

#### 4. **Cloud Provider Integration**

Ray Train works out-of-box on:
- **AWS**: Anyscale hosted platform, Amazon SageMaker with Ray
- **Azure**: Azure ML with Ray integration
- **GCP**: Google Cloud Vertex AI with Ray
- **On-Prem**: RayCluster on Kubernetes (OpenShift)

**For RHOAI**: Seamless deployment on OpenShift AI + Ray Operator.

#### 5. **Distributed Data Loading**

Ray Train integrates with Ray Data for distributed data preprocessing:
- **Parallel data loading**: Each worker loads different data shards
- **Streaming datasets**: Handle datasets larger than cluster memory
- **Preprocessing pipelines**: Distributed tokenization and augmentation
- **Cache management**: Automatic caching for repeated access

**Impact**: Eliminate data loading bottlenecks in distributed training.

#### 6. **Observability & Monitoring**

Built-in integrations with:
- **Ray Dashboard**: Real-time training metrics across cluster
- **TensorBoard**: Automatic TensorBoard integration
- **Weights & Biases**: W&B experiment tracking
- **MLflow**: Experiment tracking and model registry

**For Training-Hub**: Users get enterprise-grade observability for free.

---

### Proposed Solution: Callback-Only Integration

Implement `RayTrainReportCallback` that integrates Training-Hub backends with Ray Train orchestration.

#### **Why Callback-Only (Not a Dedicated RayBackend)?**

After thorough analysis, a callback-only approach is the optimal solution because:

**1. Ray Train is Orchestration, Not a Training Framework**

Ray Train is fundamentally different from Training-Hub backends:
- **Ray Train** = Distributed orchestration layer (HOW to distribute workers)
- **Training-Hub backends** = Training frameworks (WHICH framework/algorithm to use)
- **These are orthogonal concerns**, not alternatives

Analogies:
- You don't pick "DeepSpeed OR InstructLab" → You pick "InstructLab WITH DeepSpeed"
- You don't pick "DDP OR Unsloth" → You pick "Unsloth WITH DDP"
- You don't pick "Ray OR Unsloth" → You pick "Unsloth ON Ray"

Creating a `RayBackend` conflates orchestration with training frameworks, violating separation of concerns.

**2. Users Need Full Control Over Ray Train Configuration**

Teams deploying on Ray clusters need fine-grained control over:
- Scaling configuration (workers, resources per worker)
- Fault tolerance settings (checkpoint frequency, failure handling)
- Integration with Ray Tune (hyperparameter optimization)
- Custom Ray cluster configuration
- Resource scheduling and placement groups

Abstracting this behind `backend_config` either:
- **Limits functionality** (if we only expose common options)
- **Becomes a pass-through** (if we expose everything), adding no value

**3. Ray Train's API is Already User-Friendly**

Ray Train's `TorchTrainer` API is already simple and well-documented. Adding another abstraction layer increases complexity without corresponding benefit.

**4. Simpler Architecture = Less Maintenance**

A dedicated `RayBackend` would require:
- Dynamic generation of `train_loop_per_worker` functions
- Configuration translation between Training-Hub and Ray Train
- Ray Train lifecycle management
- Testing across all backend combinations
- Ongoing maintenance as Ray Train evolves

Callback-only approach requires:
- Single callback class (`RayTrainReportCallback`)
- Integration via standard callback mechanism
- Documentation and examples

**5. Philosophical Clarity**

Training-Hub abstracts **training algorithms and frameworks**, not **distributed orchestration strategies**. Ray Train is more analogous to DeepSpeed/FSDP/DDP (distributed strategies) than to Unsloth/InstructLab (training frameworks).

---

### How It Works

Users wrap their Training-Hub training call in a `train_loop_per_worker` function and add `RayTrainReportCallback`:

```python
from training_hub import sft
from training_hub.callbacks import RayTrainReportCallback
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

# Define training function to run on each Ray worker
def train_on_ray_worker(config):
    # Use ANY Training-Hub backend inside Ray Train
    sft(
        base_model=config["base_model"],
        data_path=config["data_path"],
        backend="unsloth",  # or "instructlab-training" or "mini-trainer"
        callbacks=[
            RayTrainReportCallback(),  # Reports checkpoints/metrics to Ray Train
        ],
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
    )

# Initialize Ray (connect to existing cluster)
ray.init(address="auto")

# Configure Ray Train
trainer = TorchTrainer(
    train_loop_per_worker=train_on_ray_worker,
    train_loop_config={
        "base_model": "meta-llama/Llama-2-7b-hf",
        "data_path": "./data.jsonl",
        "learning_rate": 2e-5,
        "num_epochs": 3,
    },
    scaling_config=ScalingConfig(
        num_workers=8,  # 8 GPUs across cluster
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )
)

# Launch distributed training on Ray cluster
result = trainer.fit()
print(f"Training completed! Checkpoint: {result.checkpoint}")
```

**What Happens:**
1. `TorchTrainer` launches 8 workers across Ray cluster
2. Each worker executes `train_on_ray_worker(config)`
3. Each worker runs Training-Hub `sft()` with the specified backend
4. `RayTrainReportCallback` reports checkpoints/metrics to Ray Train via `ray.train.report()`
5. Ray Train handles fault tolerance, resource management, checkpoint aggregation
6. Training completes and returns unified results

---

### Technical Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   TRAINING-HUB PUBLIC API                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  from training_hub import sft, osft, lora_sft           │  │
│  │                                                          │  │
│  │  def train_fn(config):                                  │  │
│  │      sft(                                               │  │
│  │          backend="unsloth",                             │  │
│  │          callbacks=[RayTrainReportCallback()],          │  │
│  │          **config                                       │  │
│  │      )                                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ user wraps in train_loop_per_worker
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│                   RAY TRAIN FRAMEWORK                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TorchTrainer(                                           │  │
│  │      train_loop_per_worker=train_fn,                    │  │
│  │      scaling_config=ScalingConfig(num_workers=8, ...)   │  │
│  │  )                                                       │  │
│  │                                                          │  │
│  │  - Launches workers across Ray cluster                  │  │
│  │  - Fault tolerance & automatic recovery                 │  │
│  │  - Resource management (GPU/CPU allocation)             │  │
│  │  - Checkpoint aggregation & resumption                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ runs on each worker
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│            TRAINING-HUB + RAYTRAIN CALLBACK                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RayTrainReportCallback (Unified Callback)               │  │
│  │  - Reports checkpoints via ray.train.report()           │  │
│  │  - Reports metrics to Ray Train                         │  │
│  │  - Handles distributed synchronization                  │  │
│  │  - Works with ALL backends                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ integrates with
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│              TRAINING-HUB BACKEND IMPLEMENTATIONS               │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  InstructLab   │  │  Mini-Trainer  │  │    Unsloth      │  │
│  │   Training     │  │   (OSFT)       │  │  (LoRA + SFT)   │  │
│  │  - Runs on     │  │  - Runs on     │  │  - Runs on      │  │
│  │    each Ray    │  │    each Ray    │  │    each Ray     │  │
│  │    worker      │  │    worker      │  │    worker       │  │
│  └────────────────┘  └────────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

### Core Component: RayTrainReportCallback

```python
# File: training_hub/callbacks/ray_train.py

from training_hub.callbacks.base import TrainingHubCallback
from training_hub.callbacks.context import TrainingHubContext
from typing import Dict, Any
import ray.train

class RayTrainReportCallback(TrainingHubCallback):
    """
    Callback that reports checkpoints and metrics to Ray Train.

    Works with ALL Training-Hub backends when running inside Ray Train's
    train_loop_per_worker function. Provides the essential integration
    between Training-Hub and Ray Train for fault-tolerant distributed training.

    Key Features:
    - Reports checkpoints to Ray Train for fault tolerance
    - Reports training metrics for monitoring
    - Handles distributed coordination (only main process reports)
    - Configurable reporting frequency

    Usage:
        def train_fn(config):
            from training_hub import sft
            from training_hub.callbacks import RayTrainReportCallback

            sft(
                backend="unsloth",
                callbacks=[RayTrainReportCallback()],
                **config
            )

        from ray.train.torch import TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_fn,
            scaling_config=ScalingConfig(num_workers=8, use_gpu=True)
        )
        trainer.fit()
    """

    def __init__(
        self,
        checkpoint_frequency: int = 1,  # Report checkpoint every N saves
        metrics_frequency: int = 1,     # Report metrics every N log events
    ):
        """
        Initialize RayTrainReportCallback.

        Args:
            checkpoint_frequency: Report checkpoint every N backend checkpoint saves.
                Default: 1 (report every checkpoint)
            metrics_frequency: Report metrics every N backend log events.
                Default: 1 (report every log event)
        """
        self.checkpoint_frequency = checkpoint_frequency
        self.metrics_frequency = metrics_frequency
        self._checkpoint_count = 0
        self._metrics_count = 0

    def on_save(self, ctx: TrainingHubContext, checkpoint_metrics: Dict[str, Any]):
        """
        Called when backend saves a checkpoint.
        Report checkpoint to Ray Train for fault tolerance.

        Ray Train will:
        - Store checkpoint in distributed storage
        - Enable automatic resumption on failure
        - Track best checkpoint across all workers
        """
        # Only report from main process in distributed training
        if not ctx.is_main_process:
            return

        # Check if we should report based on frequency
        self._checkpoint_count += 1
        if self._checkpoint_count % self.checkpoint_frequency != 0:
            return

        # Create Ray Train checkpoint from backend checkpoint path
        from ray.train import Checkpoint

        checkpoint_path = ctx.best_checkpoint_path or checkpoint_metrics.get("checkpoint_path")
        if checkpoint_path:
            checkpoint = Checkpoint.from_directory(checkpoint_path)
            ray.train.report(
                metrics=checkpoint_metrics,
                checkpoint=checkpoint
            )

    def on_log(self, ctx: TrainingHubContext, metrics: Dict[str, float]):
        """
        Called when backend logs metrics.
        Report metrics to Ray Train for monitoring.

        Metrics will be visible in:
        - Ray Dashboard
        - TensorBoard (if configured)
        - W&B/MLflow (if configured)
        """
        # Only report from main process
        if not ctx.is_main_process:
            return

        # Check if we should report based on frequency
        self._metrics_count += 1
        if self._metrics_count % self.metrics_frequency != 0:
            return

        # Add training progress info
        metrics_with_progress = {
            **metrics,
            "global_step": ctx.global_step,
            "epoch": ctx.epoch,
        }

        # Report to Ray Train
        ray.train.report(metrics=metrics_with_progress)

    def on_evaluate(self, ctx: TrainingHubContext, eval_metrics: Dict[str, float]):
        """
        Called after evaluation.
        Report evaluation metrics to Ray Train.

        Evaluation metrics are always reported regardless of metrics_frequency
        since they occur less frequently and are important for tracking.
        """
        if not ctx.is_main_process:
            return

        # Always report evaluation metrics
        eval_metrics_with_progress = {
            **eval_metrics,
            "global_step": ctx.global_step,
            "epoch": ctx.epoch,
        }

        ray.train.report(metrics=eval_metrics_with_progress)
```

---

## Business Justification

### Value Proposition

**For Data Scientists:**
- **Seamless Ray integration**: Add one callback to run on Ray clusters
- **Backend flexibility**: Use ANY Training-Hub backend (InstructLab, Mini-Trainer, Unsloth) on Ray
- **Fault tolerance**: Multi-day training runs recover automatically from node failures
- **Resource flexibility**: Run on any Ray cluster (AWS, Azure, GCP, on-prem OpenShift)
- **Full control**: Complete access to Ray Train's configuration options

**For ML Platform Teams:**
- **Unified deployment**: One codebase for local dev and production Ray clusters
- **Resource management**: Ray handles GPU allocation and scheduling
- **Observability**: Built-in Ray Dashboard, TensorBoard, W&B integration
- **Multi-cloud**: Same code on AWS Anyscale, Azure ML Ray, GCP Vertex AI Ray
- **Minimal maintenance**: Simple callback implementation, no complex backend abstraction

**For Enterprise Organizations:**
- **Production-grade**: Fault-tolerant distributed training for mission-critical models
- **Compliance**: Checkpoint every N steps for audit trails and recovery
- **Scalability**: Train 100B+ parameter models across multi-node clusters
- **Cloud-native**: First-class support for cloud Ray deployments (OpenShift AI + Ray Operator)
- **Clear architecture**: Separation between orchestration (Ray) and training (Training-Hub)

### Competitive Differentiation

**vs. Standalone Ray Train:**
- Ray Train requires custom code for each training framework
- Training-Hub + RayTrainReportCallback works with all backends via unified callback
- Users get multi-backend abstraction (InstructLab, Mini-Trainer, Unsloth) on Ray

**vs. Other Training Frameworks:**
- AWS SageMaker Training: Vendor lock-in, limited to AWS
- Azure ML Training: Vendor lock-in, limited to Azure
- Training-Hub + Ray: Multi-cloud, open-source, portable

**Red Hat OpenShift AI Advantage:**
- First AI platform with unified multi-backend Ray integration via callbacks
- Seamless deployment on OpenShift + Ray Operator
- Enterprise support for both Training-Hub and Ray

---

## Acceptance Criteria

### Functional Requirements

**FR1: RayTrainReportCallback Implementation**
- [ ] `RayTrainReportCallback` class with `on_save`, `on_log`, `on_evaluate` methods
- [ ] Reports checkpoints to Ray Train via `ray.train.report()` + `Checkpoint.from_directory()`
- [ ] Reports training metrics to Ray Train for monitoring
- [ ] Reports evaluation metrics to Ray Train
- [ ] Respects `is_main_process` flag (only main process reports)
- [ ] Configurable checkpoint and metrics reporting frequency
- [ ] Works with all Training-Hub backends (InstructLab, Mini-Trainer, Unsloth)

**FR2: Documentation & Examples**
- [ ] User guide for Ray Train integration with Training-Hub
- [ ] Example notebook: Unsloth on Ray (2-4 GPUs)
- [ ] Example notebook: InstructLab on Ray (8+ GPUs)
- [ ] Example notebook: Mini-Trainer on Ray (4 GPUs)
- [ ] Deployment guide for OpenShift AI + Ray Operator
- [ ] Troubleshooting guide for common Ray Train issues
- [ ] Performance benchmarks (Ray overhead vs. native backend)

**FR3: Testing Coverage**
- [ ] Unit tests for RayTrainReportCallback
- [ ] Integration tests: Unsloth + Ray (2 GPUs)
- [ ] Integration tests: InstructLab + Ray (4+ GPUs)
- [ ] Integration tests: Mini-Trainer + Ray (2 GPUs)
- [ ] Fault tolerance test (simulated node failure)
- [ ] Checkpoint resumption test
- [ ] Multi-node test (8+ GPUs across 2+ nodes)

### Non-Functional Requirements

**NFR1: Compatibility**
- [ ] Python 3.10+ support
- [ ] Ray >= 2.10.0 (latest stable as of 2026-01)
- [ ] Compatible with Ray Train TorchTrainer API
- [ ] Works with all existing Training-Hub backends
- [ ] No breaking changes to Training-Hub core API

**NFR2: Performance**
- [ ] Callback overhead < 2% of total training time
- [ ] No regression in non-Ray training performance
- [ ] Efficient checkpoint serialization to Ray storage
- [ ] Minimal memory overhead

**NFR3: Reliability**
- [ ] 90%+ unit test coverage for callback
- [ ] Integration tests pass on Ray clusters
- [ ] Fault tolerance tests demonstrate recovery
- [ ] No race conditions in distributed reporting

---

## Dependencies

### Technical Dependencies

**Required:**
- Python >= 3.10
- Training-Hub >= 1.0.0 (with Unified Callback System)
- Ray >= 2.10.0
- Ray Train (included in Ray >= 2.10.0)
- PyTorch >= 2.0

**Backend-Specific:**
- InstructLab Training >= 0.3.0 (for InstructLab backend on Ray)
- Mini-Trainer >= 0.1.0 (for Mini-Trainer backend on Ray)
- Unsloth >= 2024.11 (for Unsloth backend on Ray)

**Optional:**
- Ray Dashboard (for monitoring)
- Ray Operator (for Kubernetes/OpenShift deployment)

### Product Dependencies

**Required:**
- RFE "Unified Callback System" (training_hub 1.0.0) MUST be implemented first
  - `RayTrainReportCallback` depends on `TrainingHubCallback` base class
  - `TrainingHubContext` provides normalized training state

**Optional:**
- RHOAI Ray Operator integration (for OpenShift deployment)
- RHOAI Model Registry (for checkpoint management)

---

## Implementation Plan

### Phase 1: RayTrainReportCallback Implementation (Weeks 1-2)

**Deliverables:**
- [ ] `RayTrainReportCallback` class implementation
- [ ] Unit tests for callback (checkpoint reporting, metrics reporting, main process filtering)
- [ ] Basic integration test with Unsloth on 2-GPU Ray cluster
- [ ] Code review and feedback incorporation

**Validation:**
- Callback reports checkpoints correctly to Ray Train
- Callback reports metrics correctly to Ray Train
- Only main process reports (distributed training)
- Unit tests achieve 90%+ coverage

---

### Phase 2: Multi-Backend Integration Testing (Weeks 3-4)

**Deliverables:**
- [ ] Integration test: Unsloth + Ray (2 GPUs)
- [ ] Integration test: InstructLab + Ray (8 GPUs)
- [ ] Integration test: Mini-Trainer + Ray (4 GPUs)
- [ ] Fault tolerance test (simulated node failure during training)
- [ ] Checkpoint resumption test

**Validation:**
- All three backends work correctly on Ray clusters
- Checkpoints saved and restored correctly
- Automatic recovery from node failures works
- Metrics visible in Ray Dashboard

---

### Phase 3: Documentation & Examples (Weeks 5-6)

**Deliverables:**
- [ ] User guide: "Using Training-Hub with Ray Train"
- [ ] Example notebook: Unsloth on Ray (complete walkthrough)
- [ ] Example notebook: InstructLab on Ray (multi-node setup)
- [ ] Example notebook: Mini-Trainer on Ray (OSFT workflow)
- [ ] Deployment guide: OpenShift AI + Ray Operator
- [ ] Troubleshooting guide
- [ ] Performance benchmarks document

**Validation:**
- Documentation is clear and complete
- Examples run successfully on Ray clusters
- Users can follow guides to deploy on OpenShift AI
- Performance benchmarks show < 2% overhead

---

### Phase 4: Production Hardening (Weeks 7-8)

**Deliverables:**
- [ ] Multi-node testing (8+ GPUs across 2+ nodes)
- [ ] Performance optimization (if overhead > 2%)
- [ ] Error handling improvements
- [ ] Edge case testing (checkpoint failures, network issues, etc.)
- [ ] Final documentation review and polish

**Validation:**
- Multi-node training works reliably
- Performance overhead < 2%
- Graceful error handling
- Production-ready quality

---

## Testing Requirements

### Unit Tests

**RayTrainReportCallback Tests:**
- [ ] Test checkpoint reporting with valid checkpoint path
- [ ] Test checkpoint reporting with missing checkpoint path
- [ ] Test metrics reporting with various metric types
- [ ] Test main process filtering (only rank 0 reports)
- [ ] Test checkpoint frequency configuration
- [ ] Test metrics frequency configuration
- [ ] Test evaluation metrics reporting

**Coverage Target:** 90%+

---

### Integration Tests

**Single-Node Ray Tests (2-4 GPUs):**
- [ ] Unsloth backend + RayTrainReportCallback
- [ ] InstructLab backend + RayTrainReportCallback
- [ ] Mini-Trainer backend + RayTrainReportCallback
- [ ] Checkpoint saving and loading
- [ ] Metrics collection and reporting

**Multi-Node Ray Tests (8+ GPUs across 2+ nodes):**
- [ ] InstructLab distributed training (8 GPUs, 2 nodes)
- [ ] Checkpoint synchronization across nodes
- [ ] Metrics aggregation across workers
- [ ] Fault tolerance (simulated node failure)
- [ ] Checkpoint resumption after failure

---

### Performance Tests

**Benchmarking:**
- [ ] Measure callback overhead: Ray vs. native backend
  - Target: < 2% overhead
  - Test on Unsloth (LoRA), InstructLab (SFT), Mini-Trainer (OSFT)
- [ ] Measure checkpoint serialization performance
  - Target: < 30 seconds for 7B model checkpoint
- [ ] Measure multi-node communication overhead
  - Target: < 5% overhead vs. single-node

---

## Risks and Mitigation

### Technical Risks

**Risk 1: Ray Train API Changes**
- **Likelihood**: Low (Ray Train has stable API)
- **Impact**: Medium (would require callback updates)
- **Mitigation**: Pin Ray version, monitor Ray releases, maintain compatibility layer

**Risk 2: Backend-Specific Issues**
- **Likelihood**: Medium (backends may have Ray-specific quirks)
- **Impact**: Medium (may limit Ray support for some backends)
- **Mitigation**: Extensive testing, document known limitations, work with backend maintainers

**Risk 3: Performance Overhead**
- **Likelihood**: Low (callback is lightweight)
- **Impact**: High (users won't adopt if slow)
- **Mitigation**: Benchmark thoroughly, optimize checkpoint serialization, configurable reporting frequency

**Risk 4: Checkpoint Compatibility**
- **Likelihood**: Medium (different backends save checkpoints differently)
- **Impact**: Medium (may require backend-specific handling)
- **Mitigation**: Test thoroughly, add backend-specific checkpoint handling if needed

---

### Product Risks

**Risk 5: User Adoption**
- **Likelihood**: Medium (requires learning Ray Train concepts)
- **Impact**: Medium (investment not utilized)
- **Mitigation**: Excellent documentation, clear examples, RHOAI integration guide, workshops/webinars

**Risk 6: Complexity for Simple Use Cases**
- **Likelihood**: Medium (users may find TorchTrainer setup complex)
- **Impact**: Low (users can choose to use Ray or not)
- **Mitigation**: Provide simple examples, templates for common scenarios, consider future helper utilities

---

## Future Enhancements (Post-MVP)

### Phase 2 Features (6-12 months)

1. **Ray Data Integration**
   - Distributed data loading and preprocessing
   - Integration with Ray Datasets for efficient data pipelines

2. **Ray Tune Integration**
   - Hyperparameter tuning on Ray clusters
   - Example notebooks combining Training-Hub + Ray Tune

3. **Helper Utilities**
   - `training_hub.ray_utils.create_trainer()` - simplified TorchTrainer creation
   - `training_hub.ray_utils.launch_on_ray()` - one-liner for common scenarios
   - Template configurations for common Ray cluster setups

4. **Advanced Fault Tolerance**
   - Fine-grained checkpoint strategies
   - Partial checkpoint resumption
   - Configurable failure handling policies

5. **Multi-Cloud Deployment Automation**
   - Automated deployment scripts for AWS Anyscale, Azure ML Ray, GCP Vertex AI Ray
   - CI/CD integration examples
   - Infrastructure-as-code templates

---

## References

### Design Documents
- **Unified Callback RFE**: `./RFE_Training_Hub_Unified_Callbacks.md`
- **Architecture Specification**: `./architecture.md`
- **TRL Callback Analysis**: `./trl_callback_analysis.md`

### Training-Hub
- **Repository**: https://github.com/Red-Hat-AI-Innovation-Team/training_hub
- **Documentation**: https://deepwiki.com/Red-Hat-AI-Innovation-Team/training_hub

### Ray Train
- **Ray Train Documentation**: https://docs.ray.io/en/latest/train/train.html
- **Ray Train HuggingFace Integration**: https://docs.ray.io/en/latest/train/getting-started-transformers.html
- **Ray Train PyTorch Guide**: https://docs.ray.io/en/latest/train/getting-started-pytorch.html
- **TorchTrainer API**: https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
- **Ray Checkpoints**: https://docs.ray.io/en/latest/train/user-guides/checkpoints.html
- **Ray Metrics Reporting**: https://docs.ray.io/en/latest/train/user-guides/monitoring-logging.html

### External Resources
- [Announcing Ray 2.4.0: Infrastructure for LLM training](https://www.anyscale.com/blog/announcing-ray-2-4-0-infrastructure-for-llm-training-tuning-inference-and)
- [Ray for Fault-Tolerant Distributed LLM Fine-Tuning](https://latitude-blog.ghost.io/blog/ray-for-fault-tolerant-distributed-llm-fine-tuning/)
- [Fine-tune an LLM with Ray Train and DeepSpeed](https://docs.ray.io/en/latest/train/examples/pytorch/deepspeed_finetune/README.html)

---

## Contact / Stakeholders

**Product Owner**: [TBD - RHOAI Product Manager]
**Engineering Lead**: [TBD - Red Hat AI Innovation Team Lead]
**Technical Lead**: Sameh Zaher (szaher@redhat.com)
**Documentation Lead**: [TBD - RHOAI Technical Writer]
**QE Lead**: [TBD - RHOAI QE Manager]

**Reviewers:**
- Red Hat AI Innovation Team (Training-Hub maintainers)
- Ray Train community (for best practices)
- RHOAI AI/ML Solutions Architects
- OpenShift AI product team

---

## Appendix A: Design Decision Rationale

### Why Callback-Only Instead of Dedicated Backend?

This decision was made after careful analysis of Ray Train's role in the ML training ecosystem.

#### Key Insights

**1. Orthogonal Concerns**

Ray Train and Training-Hub backends address different problems:

| Concern | What It Addresses | Examples |
|---------|------------------|----------|
| **Orchestration** (Ray Train) | HOW to distribute training across workers | Resource allocation, fault tolerance, checkpoint aggregation |
| **Training Framework** (Training-Hub backends) | WHICH framework/algorithm to use | InstructLab (DeepSpeed), Unsloth (LoRA), Mini-Trainer (OSFT) |

These are independent choices:
- Choose orchestration: Ray Train, Kubernetes Jobs, SLURM, standalone
- Choose framework: InstructLab, Mini-Trainer, Unsloth
- They combine: "InstructLab ON Ray", "Unsloth ON Ray", etc.

Creating a `RayBackend` treats Ray Train as an alternative to Unsloth/InstructLab, which is conceptually incorrect.

**2. User Control Requirements**

Ray Train deployment requires configuration that varies by use case:

```python
# Different teams need different Ray Train configurations

# Team A: Cost optimization with spot instances
TorchTrainer(
    ...,
    scaling_config=ScalingConfig(
        num_workers=16,
        use_gpu=True,
        resources_per_worker={"GPU": 1},
        placement_strategy="SPREAD",  # Distribute across nodes
    ),
    run_config=RunConfig(
        failure_config=FailureConfig(max_failures=3),  # Tolerate spot interruptions
        checkpoint_config=CheckpointConfig(num_to_keep=5),
    )
)

# Team B: Maximum performance, no fault tolerance
TorchTrainer(
    ...,
    scaling_config=ScalingConfig(
        num_workers=64,
        use_gpu=True,
        resources_per_worker={"GPU": 8},  # Multi-GPU nodes
        placement_strategy="PACK",  # Minimize communication
    ),
    run_config=RunConfig(
        failure_config=FailureConfig(max_failures=0),  # Fail fast
    )
)

# Team C: Hyperparameter tuning with Ray Tune
TorchTrainer(
    ...,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    run_config=RunConfig(
        callbacks=[TuneReportCallback()],  # Ray Tune integration
    )
)
```

Exposing all these options through `backend_config` provides no value over using TorchTrainer directly.

**3. Simplicity Analysis**

| Approach | Lines of Code | Complexity | Maintenance Burden |
|----------|--------------|------------|-------------------|
| **Callback-only** | ~100 (callback class) | Low (single integration point) | Low (stable callback interface) |
| **Backend + Callback** | ~400 (backend + callback) | High (two integration points) | High (backend evolution + callback) |

Callback-only is 75% less code with significantly lower complexity.

**4. Ray Train's API is Already Simple**

Ray Train's `TorchTrainer` API is already user-friendly:

```python
# This is already simple - no abstraction needed
trainer = TorchTrainer(
    train_loop_per_worker=my_training_function,
    train_loop_config={...},
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True)
)
trainer.fit()
```

Adding a Training-Hub backend layer doesn't make this simpler.

**5. Precedent from Other Frameworks**

How do other multi-backend training frameworks handle Ray Train?

- **HuggingFace Transformers**: Callback-based integration (`RayTrainReportCallback`)
- **PyTorch Lightning**: Callback-based integration (`RayPlugin`)
- **None use a "Ray backend"** - all recognize Ray Train as orthogonal to framework choice

---

## Appendix B: Complete Usage Examples

### Example 1: Unsloth on Ray (2-4 GPUs)

```python
from training_hub import sft
from training_hub.callbacks import RayTrainReportCallback
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_unsloth_on_ray(config):
    """Training function executed on each Ray worker."""
    sft(
        base_model="unsloth/llama-2-7b-bnb-4bit",
        data_path="./data.jsonl",
        backend="unsloth",
        callbacks=[
            RayTrainReportCallback(
                checkpoint_frequency=100,  # Report every 100 checkpoints
                metrics_frequency=10,      # Report every 10 log events
            )
        ],
        max_seq_length=2048,
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        output_dir="/tmp/output",  # Local output (Ray aggregates checkpoints)
    )

# Initialize Ray (connect to existing cluster)
ray.init(address="auto")  # or ray.init() for local cluster

# Configure Ray Train
trainer = TorchTrainer(
    train_loop_per_worker=train_unsloth_on_ray,
    train_loop_config={
        "learning_rate": 2e-4,
        "num_epochs": 3,
    },
    scaling_config=ScalingConfig(
        num_workers=4,  # 4 GPUs
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )
)

# Launch training
result = trainer.fit()
print(f"Training completed! Best checkpoint: {result.checkpoint}")
print(f"Final metrics: {result.metrics}")
```

---

### Example 2: InstructLab on Ray (8+ GPUs, Multi-Node)

```python
from training_hub import sft
from training_hub.callbacks import RayTrainReportCallback, AuditLoggingCallback
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

def train_instructlab_on_ray(config):
    """Training function for InstructLab backend on Ray."""
    sft(
        base_model="ibm-granite/granite-7b-base",
        data_path=config["data_path"],
        backend="instructlab-training",
        callbacks=[
            RayTrainReportCallback(),
            AuditLoggingCallback(),  # Enterprise compliance logging
        ],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        output_dir="/tmp/output",
    )

# Initialize Ray (connect to multi-node cluster)
ray.init(address="auto")

# Configure Ray Train for multi-node training
trainer = TorchTrainer(
    train_loop_per_worker=train_instructlab_on_ray,
    train_loop_config={
        "data_path": "s3://my-bucket/training-data.jsonl",
        "num_epochs": 3,
        "learning_rate": 2e-5,
    },
    scaling_config=ScalingConfig(
        num_workers=16,  # 16 GPUs across multiple nodes
        use_gpu=True,
        resources_per_worker={"GPU": 1},
        placement_strategy="SPREAD",  # Distribute across nodes
    ),
    run_config=RunConfig(
        name="instructlab-granite-7b",
        storage_path="s3://my-bucket/ray-results",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,  # Keep 3 best checkpoints
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    )
)

# Launch training
result = trainer.fit()
print(f"Training completed!")
print(f"Best checkpoint: {result.checkpoint}")
print(f"Best eval_loss: {result.metrics['eval_loss']}")
```

---

### Example 3: Mini-Trainer OSFT on Ray

```python
from training_hub import osft  # Online SFT
from training_hub.callbacks import RayTrainReportCallback, CostTrackingCallback
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_minitrain_osft_on_ray(config):
    """Training function for Mini-Trainer OSFT on Ray."""
    osft(
        base_model="meta-llama/Llama-2-7b-hf",
        data_path=config["data_path"],
        backend="mini-trainer",
        callbacks=[
            RayTrainReportCallback(),
            CostTrackingCallback(gpu_cost_per_hour=2.50),  # Track costs
        ],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        output_dir="/tmp/output",
    )

# Initialize Ray
ray.init(address="auto")

# Configure Ray Train
trainer = TorchTrainer(
    train_loop_per_worker=train_minitrain_osft_on_ray,
    train_loop_config={
        "data_path": "./streaming_data.jsonl",
        "num_epochs": 5,
        "learning_rate": 1e-5,
    },
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )
)

# Launch training
result = trainer.fit()
```

---

### Example 4: Fault Tolerance Demo

```python
from training_hub import sft
from training_hub.callbacks import RayTrainReportCallback
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, FailureConfig

def train_with_fault_tolerance(config):
    """Training with automatic fault recovery."""
    sft(
        base_model="meta-llama/Llama-2-7b-hf",
        data_path="./data.jsonl",
        backend="unsloth",
        callbacks=[
            RayTrainReportCallback(checkpoint_frequency=1),  # Checkpoint frequently
        ],
        save_steps=100,  # Save every 100 steps
        **config
    )

ray.init(address="auto")

trainer = TorchTrainer(
    train_loop_per_worker=train_with_fault_tolerance,
    train_loop_config={"num_epochs": 10, "learning_rate": 2e-5},
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    ),
    run_config=RunConfig(
        failure_config=FailureConfig(
            max_failures=3,  # Retry up to 3 times on failure
        ),
        checkpoint_config=CheckpointConfig(
            num_to_keep=5,  # Keep 5 checkpoints for recovery
        )
    )
)

# If a node fails during training, Ray will:
# 1. Detect the failure
# 2. Load the last checkpoint
# 3. Resume training from that point
# 4. All workers synchronize automatically
result = trainer.fit()
```

---

## Appendix C: FAQ

**Q: Do I need to change my existing Training-Hub code to use Ray?**

A: No. Existing code works unchanged. Ray integration is opt-in. To use Ray:
1. Wrap your Training-Hub code in a `train_loop_per_worker` function
2. Add `RayTrainReportCallback` to callbacks list
3. Create a `TorchTrainer` and call `trainer.fit()`

**Q: Does Ray Train work with all Training-Hub backends?**

A: Yes. InstructLab, Mini-Trainer, and Unsloth all work on Ray clusters. The callback-based integration is backend-agnostic.

**Q: What's the performance overhead of using Ray Train?**

A: Typically < 2% for checkpoint/metrics reporting. Ray's distributed orchestration overhead is minimal. We'll provide detailed benchmarks in documentation.

**Q: Can I use Ray Train for local single-GPU training?**

A: Yes, but it's unnecessary. Use native backends for single-GPU training. Use Ray for multi-GPU/multi-node scenarios where you need fault tolerance and distributed orchestration.

**Q: Does this work on OpenShift AI?**

A: Yes. Deploy Ray Operator on OpenShift AI, create a RayCluster, then use Training-Hub + RayTrainReportCallback as described. We'll provide a complete deployment guide.

**Q: What happens if a node fails during training?**

A: Ray Train automatically:
1. Detects the node failure
2. Loads the last reported checkpoint
3. Resumes training from that checkpoint
4. Synchronizes all workers

You'll see a brief pause (checkpoint loading), then training continues.

**Q: How do I monitor training on Ray clusters?**

A: Ray Train integrates with:
- Ray Dashboard (built-in, shows real-time metrics)
- TensorBoard (configure via `run_config`)
- Weights & Biases (add W&B callback to Training-Hub)
- MLflow (add MLflow callback to Training-Hub)

**Q: Can I use this with Ray Tune for hyperparameter tuning?**

A: Yes. Add `TuneReportCallback` to your TorchTrainer's `run_config.callbacks`. Example will be provided in Phase 2 enhancements.

**Q: Why not create a `backend="ray"` option?**

A: After analysis, we determined that Ray Train is an orchestration layer (HOW to distribute), not a training framework (WHICH framework to use). Creating a Ray backend would conflate these orthogonal concerns. See Appendix A for detailed rationale.

**Q: Is there a simpler way than writing train_loop_per_worker?**

A: Currently, wrapping your training code is required by Ray Train's architecture. In Phase 2, we may add helper utilities like `training_hub.ray_utils.launch_on_ray()` for common scenarios, but the callback approach will remain the core integration.

---

## END OF RFE
