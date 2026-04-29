"""
InsightSerenity AI Engine — Learning Rate Schedulers
=====================================================
Learning rate schedulers control how the learning rate changes during
training. The right schedule is often as important as the choice of optimizer
— a good schedule can mean the difference between successful training and
divergence or underfitting.

Schedulers implemented:

1. LinearWarmupCosineDecay
   The standard LLM training schedule:
   - Phase 1 (warmup):  lr linearly increases from 0 to peak_lr
                         over `warmup_steps` steps
   - Phase 2 (cosine):  lr follows a half-cosine wave from peak_lr
                         down to min_lr over the remaining steps

   Used by: GPT-2, GPT-3, LLaMA, Mistral, Chinchilla

2. LinearWarmupLinearDecay
   Linear warmup followed by linear decay to zero.
   Simpler than cosine but less smooth.

3. CosineAnnealingWithRestarts (SGDR)
   Cosine decay with periodic restarts. Helps escape local minima and
   enables "snapshot ensemble" — save model at each restart trough.

4. ReduceLROnPlateau
   Reduces LR when a monitored metric stops improving.
   Heuristic-driven — good when you don't know the total steps ahead.

5. ConstantWithWarmup
   Constant learning rate after linear warmup. Minimal schedule.
   Useful for short fine-tuning runs.
"""

import math
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class LinearWarmupCosineDecay(LambdaLR):
    """
    Linear warmup followed by cosine decay to min_lr.

    This is the canonical schedule for LLM pretraining. The warmup phase
    prevents early training instability by gradually increasing the step
    size before the model has learned useful gradients.

    The cosine shape of the decay:
        lr(t) = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
    where progress = (t - warmup_steps) / max(1, total_steps - warmup_steps).

    Args:
        optimizer:     Optimizer whose lr will be scheduled.
        warmup_steps:  Number of linear warmup steps.
        total_steps:   Total training steps (warmup + cosine decay).
        min_lr:        Minimum (final) learning rate. Default: 10% of peak_lr.
        last_step:     Starting step for resuming. Default -1 (start fresh).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_step: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            # Phase 1: linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Phase 2: cosine decay
            decay_steps = max(1, total_steps - warmup_steps)
            progress    = float(current_step - warmup_steps) / float(decay_steps)
            progress    = min(progress, 1.0)

            # Cosine wave from 1.0 → min_lr_ratio
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_step)


class LinearWarmupLinearDecay(LambdaLR):
    """
    Linear warmup followed by linear decay to zero.

    Simpler than cosine decay, useful for short fine-tuning runs where
    the smooth cosine tail doesn't matter much.

    Args:
        optimizer:     Optimizer to schedule.
        warmup_steps:  Linear warmup steps.
        total_steps:   Total training steps.
        last_step:     Starting step for resuming.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_step: int = -1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            remaining = total_steps - warmup_steps
            completed = current_step - warmup_steps
            return max(0.0, 1.0 - float(completed) / float(max(1, remaining)))

        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_step)


class ConstantWithWarmup(LambdaLR):
    """
    Linear warmup followed by constant learning rate.

    The minimal schedule for fine-tuning. The constant phase means no
    learning rate decay — useful when you only want a short burst of
    fine-tuning without altering the base rate too aggressively.

    Args:
        optimizer:    Optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        last_step:    Starting step for resuming.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_step: int = -1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_step)


class CosineAnnealingWithRestarts(LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR — Loshchilov & Hutter, 2016).

    The learning rate follows a cosine wave that decays to min_lr, then
    restarts from max_lr. Each cycle can be longer than the previous one
    (controlled by T_mult) to give the model time to settle.

    lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t_cur / T_cur))

    After each cycle, the cycle length is multiplied by T_mult.

    The model at each trough (lr = min_lr) is a good snapshot to save —
    these snapshots can be ensembled for free extra performance.

    Args:
        optimizer: Optimizer to schedule.
        T_0:       Initial cycle length (steps). E.g., 1000.
        T_mult:    Cycle length multiplier after each restart. Default 2.
        min_lr:    Minimum learning rate (lr at the end of each cycle).
        last_step: Last step for resuming.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 2,
        min_lr: float = 0.0,
        last_step: int = -1,
    ) -> None:
        self.T_0      = T_0
        self.T_mult   = T_mult
        self.min_lr   = min_lr
        self.T_cur    = 0
        self.T_i      = T_0   # Current cycle length
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self) -> List[float]:
        """Compute current learning rate for each parameter group."""
        if self.T_cur == self.T_i:
            # Restart: reset cycle counter and extend cycle length
            self.T_cur = 0
            self.T_i  *= self.T_mult

        progress = self.T_cur / self.T_i
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.T_cur += 1

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine
            for base_lr in self.base_lrs
        ]


class ReduceLROnPlateau:
    """
    Thin wrapper around torch's ReduceLROnPlateau with sensible defaults.

    Reduces the learning rate when a monitored metric (e.g. validation loss)
    stops improving for `patience` consecutive evaluations.

    Unlike the other schedulers here, this one requires calling .step(metric)
    with the current metric value rather than .step() with no arguments.

    Args:
        optimizer: Optimizer to schedule.
        mode:      "min" (reduce when metric stops decreasing, e.g. loss)
                   "max" (reduce when metric stops increasing, e.g. accuracy).
        factor:    Factor by which lr is reduced. Default 0.5 (halve it).
        patience:  Number of evaluations with no improvement before reducing.
        min_lr:    Lower bound on learning rate.
        verbose:   Print a message when lr is reduced.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        verbose: bool = True,
    ) -> None:
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=verbose,
        )

    def step(self, metric: float) -> None:
        """
        Call after each evaluation with the current metric value.

        Args:
            metric: The value to monitor (e.g. validation loss).
        """
        self._scheduler.step(metric)

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self._scheduler.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs,
) -> LRScheduler:
    """
    Retrieve a learning rate scheduler by name.

    Args:
        name:      Scheduler name (case-insensitive).
                   Supported: "cosine_warmup", "linear_warmup", "constant_warmup",
                   "cosine_restarts".
        optimizer: The optimizer to schedule.
        **kwargs:  Scheduler-specific arguments.

    Returns:
        An instantiated LRScheduler.

    Raises:
        ValueError: If the name is not recognised.
    """
    name_lower = name.lower()
    registry = {
        "cosine_warmup":   LinearWarmupCosineDecay,
        "linear_warmup":   LinearWarmupLinearDecay,
        "constant_warmup": ConstantWithWarmup,
        "cosine_restarts": CosineAnnealingWithRestarts,
    }
    cls = registry.get(name_lower)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Supported: {sorted(registry.keys())}"
        )
    return cls(optimizer=optimizer, **kwargs)
