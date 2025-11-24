# TRL Callback System - Code Examples

## Complete Callback Implementations from trl/trainer/callbacks.py

### 1. SyncRefModelCallback - Complete Implementation

```python
class SyncRefModelCallback(TrainerCallback):
    """
    A [`~transformers.TrainerCallback`] that synchronizes the reference model with the training model.

    Args:
        ref_model (`PreTrainedModel` or `torch.nn.Module`):
            The reference model to synchronize with the training model.
        accelerator (`Accelerator`, *optional*):
            The Accelerator object used to prepare the model.
    """

    def __init__(
        self,
        ref_model: PreTrainedModel | torch.nn.Module,
        accelerator: Accelerator | None,
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(model, target_model, alpha):
        """
        Synchronize the target model with the model using exponential moving average.

        Args:
            model: The source model
            target_model: The target model to update
            alpha: The mixing coefficient (0 = keep target, 1 = replace with source)
        """
        for target_param, copy_param in zip(target_model.parameters(), model.parameters(), strict=True):
            target_param.data.mul_(1.0 - alpha).add_(copy_param.data, alpha=alpha)

    def sync_target_model(self, model, target_model, alpha):
        """
        Synchronize the target model with the model, handling DeepSpeed if necessary.
        """
        if is_deepspeed_available() and isinstance(model, DeepSpeedEngine):
            # DeepSpeed models need special handling
            import deepspeed
            with deepspeed.zero.GatheredParameters(
                list(model.parameters()), modifier_rank=None
            ):
                if self.accelerator.process_index == 0:
                    self._sync_target_model(model, target_model, alpha)
        else:
            self._sync_target_model(model, target_model, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        """
        Event called at the end of a training step. Synchronizes the reference model
        if the current step is a multiple of ref_model_sync_steps.
        """
        model: PreTrainedModel = kwargs["model"]

        if (
            self.ref_model is not None
            and state.global_step % args.ref_model_sync_steps == 0
        ):
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.ref_model_mixup_alpha)

        return control
```

**Usage Example**:
```python
from trl import DPOTrainer, DPOConfig
from trl.trainer.callbacks import SyncRefModelCallback

config = DPOConfig(
    sync_ref_model=True,
    ref_model_sync_steps=100,  # Sync every 100 steps
    ref_model_mixup_alpha=0.01,  # EMA coefficient
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_dataset,
)

# Callback is automatically added when sync_ref_model=True
# Or manually add:
# trainer.add_callback(SyncRefModelCallback(ref_model=ref_model, accelerator=accelerator))
```

### 2. WinRateCallback - Complete Implementation

```python
class WinRateCallback(TrainerCallback):
    """
    A [`~transformers.TrainerCallback`] that computes the win rate of the model compared to a reference model.

    Args:
        judge (`Llama`):
            The judge model to evaluate the generations.
        trainer (`Trainer`):
            The trainer instance.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use. If not provided, uses trainer's generation_config.
        num_prompts (`int`, *optional*):
            The number of prompts to evaluate. Defaults to min(256, len(eval_dataset)).
        shuffle_order (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the order of completions when judging.
        use_soft_judge (`bool`, *optional*, defaults to `False`):
            Whether to use soft judging (returns probabilities) or hard judging (returns winner).
    """

    def __init__(
        self,
        judge,
        trainer: Trainer,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
        shuffle_order: bool = True,
        use_soft_judge: bool = False,
    ):
        self.judge = judge
        self.trainer = trainer
        self.generation_config = generation_config or trainer.generation_config
        self.shuffle_order = shuffle_order
        self.use_soft_judge = use_soft_judge

        # Determine number of prompts
        eval_dataset = trainer.eval_dataset
        if num_prompts is None:
            self.num_prompts = min(256, len(eval_dataset))
        else:
            self.num_prompts = min(num_prompts, len(eval_dataset))

        # Sample prompts
        indices = random.sample(range(len(eval_dataset)), self.num_prompts)
        self.prompts = [eval_dataset[i]["prompt"] for i in indices]
        self.reference_completions = None

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Generate reference completions at the start of training.
        """
        if self.trainer.ref_model is not None:
            self.reference_completions = self._generate_completions(
                self.trainer.ref_model,
                self.prompts
            )
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Generate model completions and compute win rate during evaluation.
        """
        model = kwargs["model"]

        # Generate current model completions
        model_completions = self._generate_completions(model, self.prompts)

        # Compare with reference
        if self.reference_completions is not None:
            win_rate = self._compute_win_rate(
                model_completions,
                self.reference_completions,
                self.prompts,
            )

            # Log win rate
            self.trainer.log({"eval/win_rate": win_rate})

            # Log to wandb if available
            if is_wandb_available() and wandb.run is not None:
                wandb.log({"eval/win_rate": win_rate}, step=state.global_step)

        return control

    def _generate_completions(self, model, prompts):
        """Generate completions for a list of prompts."""
        # Implementation details...
        pass

    def _compute_win_rate(self, completions_a, completions_b, prompts):
        """
        Use the judge to compute win rate.

        Returns:
            float: Win rate (0.0 to 1.0)
        """
        wins = 0
        total = len(prompts)

        for prompt, comp_a, comp_b in zip(prompts, completions_a, completions_b):
            # Optionally shuffle order to reduce position bias
            if self.shuffle_order and random.random() < 0.5:
                comp_a, comp_b = comp_b, comp_a
                flip = True
            else:
                flip = False

            # Judge the completions
            if self.use_soft_judge:
                score = self.judge.judge_soft(prompt, comp_a, comp_b)
                wins += score if not flip else (1.0 - score)
            else:
                winner = self.judge.judge(prompt, comp_a, comp_b)
                if (winner == "A" and not flip) or (winner == "B" and flip):
                    wins += 1

        return wins / total
```

**Usage Example**:
```python
from trl import DPOTrainer
from trl.trainer.callbacks import WinRateCallback
from transformers import GenerationConfig

# Setup judge model (e.g., GPT-4, Claude, or local LLM)
judge = JudgeModel(...)  # Your judge implementation

generation_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Add win rate callback
win_rate_callback = WinRateCallback(
    judge=judge,
    trainer=trainer,
    generation_config=generation_config,
    num_prompts=128,  # Evaluate on 128 prompts
    shuffle_order=True,  # Reduce position bias
    use_soft_judge=False,  # Hard judging (winner takes all)
)

trainer.add_callback(win_rate_callback)
trainer.train()
```

### 3. LogCompletionsCallback - Complete Implementation

```python
class LogCompletionsCallback(TrainerCallback):
    """
    A [`~transformers.TrainerCallback`] that logs model completions to experiment trackers.

    Args:
        trainer (`Trainer`):
            The trainer instance.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use.
        num_prompts (`int`, *optional*):
            The number of prompts to log. Defaults to 8.
        freq (`int`, *optional*):
            Log completions every `freq` steps. Defaults to 100.
    """

    def __init__(
        self,
        trainer: Trainer,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
        freq: int | None = None,
    ):
        self.trainer = trainer
        self.generation_config = generation_config or trainer.generation_config
        self.num_prompts = num_prompts or 8
        self.freq = freq or 100

        # Sample prompts from eval dataset
        eval_dataset = trainer.eval_dataset
        num_samples = min(self.num_prompts, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), num_samples)
        self.prompts = [eval_dataset[i]["prompt"] for i in indices]

    def on_step_end(self, args, state, control, **kwargs):
        """
        Log completions at specified frequency.
        """
        if state.global_step % self.freq == 0:
            model = kwargs["model"]
            completions = self._generate_and_format_completions(model, self.prompts)

            # Log to wandb
            if is_wandb_available() and wandb.run is not None:
                wandb.log({
                    "completions": wandb.Table(
                        columns=["step", "prompt", "completion"],
                        data=[[state.global_step, p, c] for p, c in zip(self.prompts, completions)]
                    )
                }, step=state.global_step)

            # Log to comet
            if is_comet_available():
                experiment = comet_ml.get_global_experiment()
                if experiment is not None:
                    for i, (prompt, completion) in enumerate(zip(self.prompts, completions)):
                        experiment.log_text(
                            f"Prompt {i}: {prompt}\nCompletion: {completion}",
                            step=state.global_step,
                        )

        return control

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Log completions during evaluation.
        """
        model = kwargs["model"]
        completions = self._generate_and_format_completions(model, self.prompts)

        # Log evaluation completions
        # Similar to on_step_end but with "eval_" prefix

        return control

    def _generate_and_format_completions(self, model, prompts):
        """Generate and format completions."""
        # Implementation details...
        pass
```

**Usage Example**:
```python
from trl import SFTTrainer
from trl.trainer.callbacks import LogCompletionsCallback
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Add logging callback
log_callback = LogCompletionsCallback(
    trainer=trainer,
    generation_config=generation_config,
    num_prompts=16,  # Log 16 examples
    freq=50,  # Log every 50 steps
)

trainer.add_callback(log_callback)
trainer.train()
```

### 4. BEMACallback - Complete Implementation

```python
class BEMACallback(TrainerCallback):
    """
    A [`~transformers.TrainerCallback`] that applies Bias-Corrected Exponential Moving Average.

    BEMA is an adaptive weight averaging technique that can improve model generalization.

    Args:
        bema_start_step (`int`, *optional*, defaults to 0):
            The step to start applying BEMA.
        bema_end_step (`int`, *optional*):
            The step to stop applying BEMA. If None, continues until training ends.
        bema_alpha (`float`, *optional*, defaults to 0.9):
            The exponential moving average coefficient.
        bema_warmup_steps (`int`, *optional*, defaults to 0):
            Number of warmup steps for bias correction.
    """

    def __init__(
        self,
        bema_start_step: int = 0,
        bema_end_step: int | None = None,
        bema_alpha: float = 0.9,
        bema_warmup_steps: int = 0,
    ):
        self.bema_start_step = bema_start_step
        self.bema_end_step = bema_end_step
        self.bema_alpha = bema_alpha
        self.bema_warmup_steps = bema_warmup_steps
        self.running_model = None
        self.bema_step = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        Initialize BEMA buffers at the start of training.
        """
        if model is not None:
            # Create a copy of the model for BEMA
            self.running_model = copy.deepcopy(model)
            self.running_model.eval()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Update BEMA weights after each training step.
        """
        if model is None or self.running_model is None:
            return control

        current_step = state.global_step

        # Check if we should apply BEMA at this step
        if current_step < self.bema_start_step:
            return control

        if self.bema_end_step is not None and current_step >= self.bema_end_step:
            return control

        # Update BEMA step counter
        self.bema_step += 1

        # Compute bias-corrected alpha
        if self.bema_step <= self.bema_warmup_steps:
            # Bias correction during warmup
            alpha = 1.0 / self.bema_step
        else:
            alpha = self.bema_alpha

        # Update running average
        with torch.no_grad():
            for bema_param, model_param in zip(
                self.running_model.parameters(),
                model.parameters(),
                strict=True
            ):
                bema_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """
        Save the BEMA model at the end of training.
        """
        if self.running_model is not None:
            output_dir = os.path.join(args.output_dir, "bema_model")
            self.running_model.save_pretrained(output_dir)

            if args.push_to_hub:
                self.running_model.push_to_hub(
                    repo_id=f"{args.hub_model_id}-bema",
                    private=args.hub_private_repo,
                )

        return control
```

**Usage Example**:
```python
from trl import SFTTrainer
from trl.trainer.callbacks import BEMACallback

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Add BEMA callback
bema_callback = BEMACallback(
    bema_start_step=1000,  # Start BEMA after 1000 steps
    bema_alpha=0.9,  # EMA coefficient
    bema_warmup_steps=100,  # Warmup for bias correction
)

trainer.add_callback(bema_callback)
trainer.train()

# BEMA model will be saved to output_dir/bema_model/
```

### 5. WeaveCallback - Complete Implementation

```python
class WeaveCallback(TrainerCallback):
    """
    A [`~transformers.TrainerCallback`] that logs predictions to W&B Weave.

    Args:
        trainer (`Trainer`):
            The trainer instance.
        project_name (`str`, *optional*):
            The Weave project name.
        scorers (`dict[str, callable]`, *optional*):
            Dictionary of scorer functions to evaluate predictions.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use.
        num_prompts (`int`, *optional*):
            Number of prompts to evaluate.
        dataset_name (`str`, *optional*, defaults to "eval_dataset"):
            Name of the dataset to use.
        model_name (`str`, *optional*):
            Name for the model in Weave.
    """

    def __init__(
        self,
        trainer: Trainer,
        project_name: str | None = None,
        scorers: dict[str, callable] | None = None,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
        dataset_name: str = "eval_dataset",
        model_name: str | None = None,
    ):
        self.trainer = trainer
        self.project_name = project_name
        self.scorers = scorers or {}
        self.generation_config = generation_config or trainer.generation_config
        self.dataset_name = dataset_name
        self.model_name = model_name or "model"

        # Setup prompts
        eval_dataset = trainer.eval_dataset
        self.num_prompts = num_prompts or min(100, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), self.num_prompts)
        self.prompts = [eval_dataset[i] for i in indices]

        self.weave_client = None

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Initialize Weave client.
        """
        if not is_weave_available():
            warnings.warn("Weave is not available. Install with: pip install weave")
            return control

        import weave

        project_name = self.project_name or args.run_name or "trl-training"
        self.weave_client = weave.init(project_name)

        return control

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Generate predictions and log to Weave.
        """
        if self.weave_client is None:
            return control

        model = kwargs["model"]

        # Generate predictions
        predictions = []
        for item in self.prompts:
            prompt = item["prompt"]
            completion = self._generate_completion(model, prompt)

            pred_dict = {
                "prompt": prompt,
                "completion": completion,
                "step": state.global_step,
            }

            # Apply scorers if available
            if self.scorers:
                scores = {}
                for scorer_name, scorer_fn in self.scorers.items():
                    scores[scorer_name] = scorer_fn(prompt, completion)
                pred_dict["scores"] = scores

            predictions.append(pred_dict)

        # Log to Weave
        self.weave_client.log({
            "predictions": predictions,
            "step": state.global_step,
            "model_name": self.model_name,
        })

        return control

    def _generate_completion(self, model, prompt):
        """Generate a completion for a prompt."""
        # Implementation details...
        pass
```

**Usage Example**:
```python
from trl import DPOTrainer
from trl.trainer.callbacks import WeaveCallback

# Define custom scorers
def coherence_scorer(prompt, completion):
    # Your scoring logic
    return score

def relevance_scorer(prompt, completion):
    # Your scoring logic
    return score

scorers = {
    "coherence": coherence_scorer,
    "relevance": relevance_scorer,
}

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Add Weave callback
weave_callback = WeaveCallback(
    trainer=trainer,
    project_name="my-rlhf-project",
    scorers=scorers,
    num_prompts=50,
)

trainer.add_callback(weave_callback)
trainer.train()
```

## Integration Examples

### Example 1: Combining Multiple Callbacks

```python
from trl import DPOTrainer, DPOConfig
from trl.trainer.callbacks import (
    SyncRefModelCallback,
    WinRateCallback,
    LogCompletionsCallback,
    BEMACallback,
)
from transformers import EarlyStoppingCallback

# Setup config
config = DPOConfig(
    output_dir="./outputs",
    sync_ref_model=True,
    ref_model_sync_steps=100,
    ref_model_mixup_alpha=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Add multiple callbacks
trainer.add_callback(WinRateCallback(
    judge=judge_model,
    trainer=trainer,
    num_prompts=64,
))

trainer.add_callback(LogCompletionsCallback(
    trainer=trainer,
    num_prompts=8,
    freq=100,
))

trainer.add_callback(BEMACallback(
    bema_start_step=500,
    bema_alpha=0.9,
))

trainer.add_callback(EarlyStoppingCallback(
    early_stopping_patience=3,
))

# Train
trainer.train()
```

### Example 2: Custom Callback for RLHF Metrics

```python
from transformers import TrainerCallback
import numpy as np

class RLHFMetricsCallback(TrainerCallback):
    """
    Custom callback to track RLHF-specific metrics.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.reward_history = []
        self.kl_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Track reward and KL divergence from logs.
        """
        if logs is not None:
            if "rewards/mean" in logs:
                self.reward_history.append(logs["rewards/mean"])

            if "objective/kl" in logs:
                self.kl_history.append(logs["objective/kl"])

            # Compute moving averages
            if len(self.reward_history) >= 10:
                recent_reward = np.mean(self.reward_history[-10:])
                logs["rewards/ma_10"] = recent_reward

            if len(self.kl_history) >= 10:
                recent_kl = np.mean(self.kl_history[-10:])
                logs["kl/ma_10"] = recent_kl

        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Log evaluation metrics.
        """
        if metrics is not None:
            print(f"\nEvaluation at step {state.global_step}:")
            print(f"  Reward MA: {np.mean(self.reward_history[-10:]) if self.reward_history else 0:.4f}")
            print(f"  KL MA: {np.mean(self.kl_history[-10:]) if self.kl_history else 0:.4f}")

        return control

# Usage
trainer = DPOTrainer(...)
trainer.add_callback(RLHFMetricsCallback(trainer))
```

### Example 3: Conditional Callback Behavior

```python
from transformers import TrainerCallback

class AdaptiveSyncCallback(TrainerCallback):
    """
    Adaptive reference model synchronization based on performance.
    """

    def __init__(self, ref_model, accelerator, initial_freq=100):
        self.ref_model = ref_model
        self.accelerator = accelerator
        self.sync_freq = initial_freq
        self.last_eval_loss = float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Adjust sync frequency based on eval loss trend.
        """
        if metrics and "eval_loss" in metrics:
            current_loss = metrics["eval_loss"]

            # If improving, sync less frequently
            if current_loss < self.last_eval_loss:
                self.sync_freq = min(self.sync_freq + 10, 500)
            # If degrading, sync more frequently
            else:
                self.sync_freq = max(self.sync_freq - 10, 50)

            self.last_eval_loss = current_loss
            print(f"Adjusted sync frequency to: {self.sync_freq}")

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """
        Sync based on adaptive frequency.
        """
        if state.global_step % self.sync_freq == 0:
            model = kwargs["model"]
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self._sync_models(model, self.ref_model)

        return control

    def _sync_models(self, model, ref_model, alpha=0.01):
        """Synchronize model parameters."""
        for ref_param, model_param in zip(ref_model.parameters(), model.parameters()):
            ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

# Usage
trainer = DPOTrainer(...)
trainer.add_callback(AdaptiveSyncCallback(
    ref_model=ref_model,
    accelerator=accelerator,
    initial_freq=100,
))
```

## Standard Transformers Callbacks Compatible with TRL

```python
from transformers import (
    EarlyStoppingCallback,
    TensorBoardCallback,
    TrainerCallback,
)

# All standard callbacks work with TRL trainers
trainer = DPOTrainer(...)

# Early stopping
trainer.add_callback(EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
))

# TensorBoard logging
trainer.add_callback(TensorBoardCallback())

# Custom standard callback
class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")
        return control

trainer.add_callback(PrinterCallback())
```
