"""
InsightSerenity AI Engine — Base Trainer
=========================================
Abstract base class for all trainers. Defines the contract between
the training loop and the rest of the system: models, datasets,
optimizers, schedulers, callbacks, and checkpointing.

Design principles:
    - Separation of concerns: the trainer orchestrates; models compute;
      callbacks observe; checkpointing persists.
    - Every training state variable lives in TrainState — a single
      dataclass that can be saved and restored atomically.
    - Callbacks are the extension point: all logging, early stopping,
      LR scheduling observations, and progress reporting go through them.
    - No global state — everything is an explicit argument or attribute.

Training loop (one epoch):
    for batch in train_loader:
        on_before_step(batch)
        loss = train_step(batch)
        on_after_step(loss)
    on_epoch_end(metrics)
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.training.callbacks.base_callback import BaseCallback
    from src.training.checkpointing.checkpoint_manager import CheckpointManager

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    """
    All hyperparameters and infrastructure settings for a training run.
    Passed to the trainer at construction time — nothing is hardcoded inside
    the training loop.
    """
    # ── Compute ───────────────────────────────────────────────────────────────
    max_epochs:            int   = 3
    max_steps:             Optional[int] = None   # Overrides max_epochs if set
    device:                str   = "auto"         # "auto" | "cuda" | "mps" | "cpu"

    # ── Gradient ──────────────────────────────────────────────────────────────
    gradient_clip_norm:    float = 1.0      # Max gradient L2 norm (0 = disabled)
    gradient_accumulation: int   = 1        # Accumulate gradients over N steps

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp:               bool  = True     # Automatic mixed precision (AMP)
    amp_dtype:             str   = "bfloat16"  # "float16" or "bfloat16"

    # ── Validation ────────────────────────────────────────────────────────────
    eval_every_n_steps:    int   = 500      # Validate every N gradient steps
    eval_max_batches:      Optional[int] = None   # Cap evaluation batches

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps:     int   = 10       # Log metrics every N steps

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_every_n_steps:    int   = 1000     # Save checkpoint every N steps
    output_dir:            str   = "storage/checkpoints"
    run_name:              str   = "run"    # Subdirectory for this run's files


# ─────────────────────────────────────────────────────────────────────────────
# Training state — everything that must be checkpointed for resumability
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainState:
    """
    Mutable training state. This is the single source of truth for where
    training is at any moment. Serialised to disk on every checkpoint.
    """
    global_step:  int   = 0      # Total gradient steps taken so far
    epoch:        int   = 0      # Current epoch index (0-based)
    best_metric:  float = float("inf")  # Best validation metric (lower = better)
    best_step:    int   = 0      # Step at which best_metric was achieved
    train_loss:   float = 0.0    # Most recent training loss
    val_loss:     float = 0.0    # Most recent validation loss
    elapsed_secs: float = 0.0    # Total wall-clock training time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_step":  self.global_step,
            "epoch":        self.epoch,
            "best_metric":  self.best_metric,
            "best_step":    self.best_step,
            "train_loss":   self.train_loss,
            "val_loss":     self.val_loss,
            "elapsed_secs": self.elapsed_secs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
# Base Trainer
# ─────────────────────────────────────────────────────────────────────────────

class BaseTrainer(ABC):
    """
    Abstract base trainer. Provides the complete training loop skeleton.

    Subclasses must implement:
        train_step(batch) → Dict[str, float]
        val_step(batch)   → Dict[str, float]

    Everything else — gradient clipping, AMP, accumulation, callback
    dispatch, checkpointing, device management — is handled here.

    Args:
        model:          The nn.Module to train.
        optimizer:      Optimizer wrapping model.parameters().
        train_loader:   DataLoader for training data.
        config:         TrainerConfig with all hyperparameters.
        scheduler:      Optional LR scheduler (called after each step).
        val_loader:     Optional DataLoader for validation.
        callbacks:      List of BaseCallback instances.
        checkpoint_mgr: Optional CheckpointManager for save/resume.
    """

    def __init__(
        self,
        model:          nn.Module,
        optimizer:      Optimizer,
        train_loader:   DataLoader,
        config:         TrainerConfig,
        scheduler:      Optional[LRScheduler] = None,
        val_loader:     Optional[DataLoader]  = None,
        callbacks:      Optional[List["BaseCallback"]] = None,
        checkpoint_mgr: Optional["CheckpointManager"] = None,
    ) -> None:
        self.model          = model
        self.optimizer      = optimizer
        self.train_loader   = train_loader
        self.config         = config
        self.scheduler      = scheduler
        self.val_loader     = val_loader
        self.callbacks      = callbacks or []
        self.checkpoint_mgr = checkpoint_mgr
        self.state          = TrainState()

        # Resolve device
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        logger.info("Trainer initialised", device=str(self.device))

        # AMP scaler (only valid for float16; bfloat16 doesn't need it)
        if config.use_amp and config.amp_dtype == "float16":
            self.scaler: Optional[torch.cuda.amp.GradScaler] = \
                torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self._amp_dtype = (
            torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        )

        # Gradient norm EMA for explosion detection (updated every optimizer step)
        self._grad_norm_ema: Optional[float] = None
        self._nan_steps_skipped: int = 0

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single forward + backward step on one batch.

        Must NOT call optimizer.step() or optimizer.zero_grad() —
        the base class handles gradient accumulation, clipping, and the
        actual optimizer step.

        Args:
            batch: Dict of tensors from the DataLoader (already on device).

        Returns:
            Dict of scalar metrics, e.g. {"loss": 2.34, "accuracy": 0.45}.
            Must include a "loss" key — that value will be backward()'d.
        """
        ...

    @abstractmethod
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run one validation batch without gradient computation.

        Args:
            batch: Dict of tensors from the validation DataLoader.

        Returns:
            Dict of scalar metrics including "loss".
        """
        ...

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None) -> TrainState:
        """
        Execute the complete training run.

        Args:
            resume_from: Optional path to a checkpoint directory to resume from.

        Returns:
            Final TrainState after training completes.
        """
        if resume_from and self.checkpoint_mgr:
            self._resume(resume_from)

        self._call_callbacks("on_train_begin", self.state)
        start_time = time.perf_counter()

        try:
            for epoch in range(self.state.epoch, self.config.max_epochs):
                self.state.epoch = epoch
                self._call_callbacks("on_epoch_begin", self.state)

                epoch_metrics = self._run_epoch(epoch)

                self.state.val_loss = epoch_metrics.get("val_loss", self.state.val_loss)
                self.state.elapsed_secs += time.perf_counter() - start_time
                self._call_callbacks("on_epoch_end", self.state, epoch_metrics)

                # Check if max_steps reached
                if (
                    self.config.max_steps is not None
                    and self.state.global_step >= self.config.max_steps
                ):
                    logger.info("max_steps reached, stopping", step=self.state.global_step)
                    break

                # Check early stopping signal from callbacks
                if self._should_stop():
                    logger.info("Early stopping triggered", epoch=epoch)
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")

        self._call_callbacks("on_train_end", self.state)
        return self.state

    def validate(self) -> Dict[str, float]:
        """
        Run a full validation pass and return aggregated metrics.
        Can be called independently (e.g. for evaluation after training).
        """
        if not self.val_loader:
            return {}

        self.model.eval()
        all_metrics: Dict[str, List[float]] = {}
        n_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if (
                    self.config.eval_max_batches is not None
                    and i >= self.config.eval_max_batches
                ):
                    break

                batch = self._to_device(batch)
                metrics = self.val_step(batch)

                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)
                n_batches += 1

        # Average all metrics over batches
        avg = {k: sum(vs) / max(len(vs), 1) for k, vs in all_metrics.items()}

        logger.info("Validation complete", batches=n_batches, **{f"val_{k}": round(v, 4) for k, v in avg.items()})
        self.model.train()
        return {f"val_{k}": v for k, v in avg.items()}

    # ── Training epoch ─────────────────────────────────────────────────────────

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one full epoch of training."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_loss   = 0.0
        n_steps_done = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            # ── Forward + backward (AMP-aware) ─────────────────────────────
            if self.config.use_amp:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._amp_dtype,
                ):
                    metrics = self.train_step(batch)
            else:
                metrics = self.train_step(batch)

            loss = metrics["loss"]

            # NaN/Inf guard: the subclass detected a bad loss before calling
            # backward — zero grads and skip this step entirely.
            if math.isnan(loss) or math.isinf(loss):
                self._nan_steps_skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    "Skipping step — NaN/Inf loss",
                    batch_idx=batch_idx,
                    total_skipped=self._nan_steps_skipped,
                )
                if self._nan_steps_skipped > 10:
                    raise RuntimeError(
                        "Training aborted: more than 10 consecutive NaN/Inf loss steps. "
                        "Check your learning rate, data quality, and AMP settings."
                    )
                continue

            self._nan_steps_skipped = 0   # Reset streak counter on good step
            epoch_loss += loss

            # The backward pass is handled inside train_step (already called).
            # Base class only handles gradient accumulation bookkeeping.

            # ── Gradient step (every N accumulation steps) ─────────────────
            is_accumulation_step = (batch_idx + 1) % self.config.gradient_accumulation == 0

            if is_accumulation_step:
                grad_norm = self._optimizer_step()
                self.state.global_step += 1
                self.state.train_loss   = loss
                n_steps_done += 1

                # Callbacks: after each optimizer step
                self._call_callbacks("on_step_end", self.state, metrics)

                # Periodic logging — includes grad norm for anomaly monitoring
                if self.state.global_step % self.config.log_every_n_steps == 0:
                    lr = self._get_current_lr()
                    logger.info(
                        "Step",
                        step=self.state.global_step,
                        epoch=epoch,
                        loss=round(loss, 4),
                        lr=round(lr, 8),
                        grad_norm=round(grad_norm, 3) if grad_norm else None,
                        grad_norm_ema=round(self._grad_norm_ema, 3) if self._grad_norm_ema else None,
                    )

                # Periodic validation
                if (
                    self.val_loader
                    and self.state.global_step % self.config.eval_every_n_steps == 0
                ):
                    val_metrics = self.validate()
                    self.state.val_loss = val_metrics.get("val_loss", self.state.val_loss)
                    self._update_best(val_metrics)
                    self._call_callbacks("on_evaluate", self.state, val_metrics)

                # Periodic checkpointing
                if (
                    self.checkpoint_mgr
                    and self.state.global_step % self.config.save_every_n_steps == 0
                ):
                    self.checkpoint_mgr.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        state=self.state,
                    )

                # Check max_steps
                if (
                    self.config.max_steps is not None
                    and self.state.global_step >= self.config.max_steps
                ):
                    break

        epoch_avg_loss = epoch_loss / max(n_steps_done, 1)
        epoch_metrics  = {"train_loss": epoch_avg_loss}

        if self.val_loader:
            val_metrics = self.validate()
            epoch_metrics.update(val_metrics)
            self.state.val_loss = val_metrics.get("val_loss", self.state.val_loss)
            self._update_best(val_metrics)

        return epoch_metrics

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _optimizer_step(self) -> float:
        """
        Clip gradients, detect explosions, step optimizer and scheduler.

        Returns the pre-clip gradient L2 norm so callers can log it.
        Skips the optimizer step (but zeros grads) if an explosion is detected.
        """
        # Unscale before measuring norm (required for AMP)
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Compute raw gradient norm before clipping
        raw_norm: float = 0.0
        if self.config.gradient_clip_norm > 0:
            raw_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            ).item()
        else:
            # Compute norm without clipping for monitoring only
            total_sq = sum(
                p.grad.detach().norm().item() ** 2
                for p in self.model.parameters()
                if p.grad is not None
            )
            raw_norm = math.sqrt(total_sq)

        # Update EMA of gradient norm (α = 0.01 for slow-moving baseline)
        if self._grad_norm_ema is None:
            self._grad_norm_ema = raw_norm
        else:
            self._grad_norm_ema = 0.99 * self._grad_norm_ema + 0.01 * raw_norm

        # Explosion detection: norm > 10× EMA baseline triggers a skip
        if self._grad_norm_ema > 0 and raw_norm > 10.0 * self._grad_norm_ema:
            logger.warning(
                "Gradient explosion detected — skipping optimizer step",
                raw_norm=round(raw_norm, 3),
                ema_norm=round(self._grad_norm_ema, 3),
                step=self.state.global_step,
            )
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.update()   # Keep scaler state consistent
            return raw_norm

        # Normal step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        return raw_norm

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move all tensors in a batch dict to the training device."""
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _get_current_lr(self) -> float:
        """Return the current learning rate from the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    def _update_best(self, val_metrics: Dict[str, float]) -> None:
        """Update best metric and step if current validation is better."""
        val_loss = val_metrics.get("val_loss", float("inf"))
        if val_loss < self.state.best_metric:
            self.state.best_metric = val_loss
            self.state.best_step   = self.state.global_step

    def _should_stop(self) -> bool:
        """Ask callbacks if early stopping has been triggered."""
        return any(getattr(cb, "_stop_training", False) for cb in self.callbacks)

    def _call_callbacks(self, event: str, *args, **kwargs) -> None:
        """Dispatch a lifecycle event to all registered callbacks."""
        for cb in self.callbacks:
            if hasattr(cb, event):
                getattr(cb, event)(*args, **kwargs)

    def _resume(self, checkpoint_path: str) -> None:
        """Restore model, optimizer, scheduler, and TrainState from a checkpoint."""
        if self.checkpoint_mgr:
            restored = self.checkpoint_mgr.load(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            if restored:
                self.state = restored
                logger.info(
                    "Resumed training",
                    step=self.state.global_step,
                    epoch=self.state.epoch,
                )

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve "auto" to the best available hardware."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)
