"""
InsightSerenity AI Engine — Checkpoint Manager
===============================================
Manages saving and loading of training checkpoints. A checkpoint is a
complete snapshot of training state that allows resuming from an exact
mid-run position:
    - Model weights (state_dict)
    - Optimizer state (momentum, variance accumulators for Adam)
    - LR scheduler state
    - TrainState (step, epoch, best metric, elapsed time)
    - Training config

Checkpoint file structure:
    {output_dir}/{run_name}/
        step_001000/
            model.pt              ← model state_dict
            optimizer.pt          ← optimizer state_dict
            scheduler.pt          ← scheduler state_dict (if any)
            train_state.json      ← TrainState as JSON
            config.json           ← TrainerConfig as JSON
        step_002000/
            ...
        best/
            → symlink or copy of the best checkpoint directory
        latest/
            → symlink or copy of the most recent checkpoint

Pruning: by default only the last N checkpoints are kept to prevent
disk exhaustion. The "best" checkpoint is always preserved regardless.
"""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.training.trainer.base_trainer import TrainState


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Saves, loads, and prunes training checkpoints.

    Args:
        output_dir:          Root directory for all checkpoints.
        run_name:            Name of this training run (subdirectory).
        keep_last_n:         Retain only the N most recent step checkpoints.
                             The best checkpoint is always retained. Default 5.
        save_best:           If True, copy the best-metric checkpoint to a
                             "best" subdirectory for easy loading. Default True.
        best_metric:         Name of the metric to use for "best" tracking.
                             Default "val_loss" (lower is better).
        best_metric_mode:    "min" (loss) or "max" (accuracy). Default "min".
    """

    def __init__(
        self,
        output_dir:       str,
        run_name:         str   = "run",
        keep_last_n:      int   = 5,
        save_best:        bool  = True,
        best_metric:      str   = "val_loss",
        best_metric_mode: str   = "min",
    ) -> None:
        self._root           = Path(output_dir) / run_name
        self._root.mkdir(parents=True, exist_ok=True)

        self._keep_last_n        = keep_last_n
        self._save_best          = save_best
        self._best_metric        = best_metric
        self._best_metric_mode   = best_metric_mode
        self._best_metric_val    = float("inf") if best_metric_mode == "min" else float("-inf")

        # Ordered list of step checkpoint directories (oldest first)
        self._step_checkpoints: List[Path] = []

        logger.info("CheckpointManager ready", output_dir=str(self._root))

    # ── Saving ─────────────────────────────────────────────────────────────────

    def save(
        self,
        model:     nn.Module,
        optimizer: Optimizer,
        state:     TrainState,
        scheduler: Optional[LRScheduler] = None,
        metrics:   Optional[Dict[str, float]] = None,
        config:    Optional[Any] = None,
    ) -> Path:
        """
        Save a complete training checkpoint.

        Args:
            model:     The model being trained.
            optimizer: The optimizer.
            state:     Current TrainState (step, epoch, best metric, etc.)
            scheduler: Optional LR scheduler.
            metrics:   Optional dict of current validation metrics.
            config:    Optional TrainerConfig (serialised as JSON).

        Returns:
            Path to the saved checkpoint directory.
        """
        step_dir = self._root / f"step_{state.global_step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # ── Model weights ──────────────────────────────────────────────────
        # Unwrap DDP if wrapped
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), step_dir / "model.pt")

        # ── Optimizer state ────────────────────────────────────────────────
        torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")

        # ── Scheduler state ────────────────────────────────────────────────
        if scheduler is not None:
            torch.save(scheduler.state_dict(), step_dir / "scheduler.pt")

        # ── Train state (includes SHA-256 checksum of model.pt) ───────────
        state_dict = state.to_dict()
        if metrics:
            state_dict["last_metrics"] = metrics
        state_dict["model_checksum_sha256"] = _sha256(step_dir / "model.pt")
        with open(step_dir / "train_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        # ── Config ────────────────────────────────────────────────────────
        if config is not None:
            config_dict = (
                vars(config) if hasattr(config, "__dict__") else config
            )
            with open(step_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

        logger.info(
            "Checkpoint saved",
            step=state.global_step,
            path=str(step_dir),
        )

        # Track and prune
        self._step_checkpoints.append(step_dir)
        self._prune_old_checkpoints()

        # Update "best" checkpoint
        if self._save_best and metrics:
            current_metric = metrics.get(self._best_metric)
            if current_metric is not None:
                improved = (
                    current_metric < self._best_metric_val
                    if self._best_metric_mode == "min"
                    else current_metric > self._best_metric_val
                )
                if improved:
                    self._best_metric_val = current_metric
                    self._update_best_symlink(step_dir)

        # Update "latest" symlink/copy
        self._update_latest_symlink(step_dir)

        return step_dir

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(
        self,
        checkpoint_path: str,
        model:     nn.Module,
        optimizer: Optional[Optimizer]      = None,
        scheduler: Optional[LRScheduler]   = None,
        strict:    bool                     = True,
    ) -> Optional[TrainState]:
        """
        Restore a checkpoint into the given model (and optionally optimizer/scheduler).

        Args:
            checkpoint_path: Path to the checkpoint directory
                             (e.g. "storage/checkpoints/run1/step_00001000").
            model:     The model to load weights into.
            optimizer: If provided, also restore optimizer state.
            scheduler: If provided, also restore scheduler state.
            strict:    Pass to model.load_state_dict — False allows partial loads.

        Returns:
            Restored TrainState, or None if train_state.json is missing.
        """
        ckpt_dir = Path(checkpoint_path)

        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Load model weights
        model_path = ckpt_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pt missing from checkpoint: {ckpt_dir}")

        # ── Integrity check: verify SHA-256 before loading into memory ─────
        state_path_for_check = ckpt_dir / "train_state.json"
        if state_path_for_check.exists():
            with open(state_path_for_check) as f:
                state_meta = json.load(f)
            expected_checksum = state_meta.get("model_checksum_sha256")
            if expected_checksum:
                actual_checksum = _sha256(model_path)
                if actual_checksum != expected_checksum:
                    raise ValueError(
                        f"Checkpoint integrity FAILED for {ckpt_dir}.\n"
                        f"  Expected SHA-256: {expected_checksum}\n"
                        f"  Actual   SHA-256: {actual_checksum}\n"
                        f"The model.pt file may be corrupted or tampered with."
                    )
                logger.info("Checkpoint integrity verified", checksum=expected_checksum[:16])

        # Handle DDP-wrapped models
        target_model = model.module if hasattr(model, "module") else model
        state_dict   = torch.load(model_path, map_location="cpu", weights_only=True)
        target_model.load_state_dict(state_dict, strict=strict)
        logger.info("Model weights loaded", path=str(model_path))

        # Load optimizer state
        if optimizer is not None:
            opt_path = ckpt_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                logger.info("Optimizer state loaded")

        # Load scheduler state
        if scheduler is not None:
            sch_path = ckpt_dir / "scheduler.pt"
            if sch_path.exists():
                scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
                logger.info("Scheduler state loaded")

        # Load train state
        state_path = ckpt_dir / "train_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state_dict_json = json.load(f)
            train_state = TrainState.from_dict(state_dict_json)
            logger.info(
                "TrainState restored",
                step=train_state.global_step,
                epoch=train_state.epoch,
            )
            return train_state

        return None

    def load_best(
        self,
        model:     nn.Module,
        optimizer: Optional[Optimizer]    = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> Optional[TrainState]:
        """Load the best checkpoint (by validation metric)."""
        best_dir = self._root / "best"
        if not best_dir.exists():
            logger.warning("No 'best' checkpoint found")
            return None
        return self.load(str(best_dir), model, optimizer, scheduler)

    def load_latest(
        self,
        model:     nn.Module,
        optimizer: Optional[Optimizer]    = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> Optional[TrainState]:
        """Load the most recent checkpoint."""
        latest_dir = self._root / "latest"
        if not latest_dir.exists():
            logger.warning("No 'latest' checkpoint found")
            return None
        return self.load(str(latest_dir), model, optimizer, scheduler)

    def list_checkpoints(self) -> List[Path]:
        """Return all checkpoint directories sorted by step number."""
        return sorted(
            [d for d in self._root.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _prune_old_checkpoints(self) -> None:
        """
        Delete old step checkpoints beyond the keep_last_n limit.
        Always preserves the "best" and "latest" symlinks/copies.
        """
        if len(self._step_checkpoints) <= self._keep_last_n:
            return

        to_delete = self._step_checkpoints[: -self._keep_last_n]
        for ckpt_dir in to_delete:
            # Never delete the "best" checkpoint
            best_path = self._root / "best"
            if best_path.exists() and ckpt_dir.resolve() == best_path.resolve():
                continue
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
                logger.debug("Old checkpoint pruned", path=str(ckpt_dir))

        self._step_checkpoints = self._step_checkpoints[-self._keep_last_n:]

    def _update_best_symlink(self, step_dir: Path) -> None:
        """Update the 'best' pointer to `step_dir`."""
        best_path = self._root / "best"
        if best_path.exists():
            shutil.rmtree(best_path)
        # Copy instead of symlink for portability (symlinks can break on Windows)
        shutil.copytree(step_dir, best_path)
        logger.info("Best checkpoint updated", path=str(step_dir))

    def _update_latest_symlink(self, step_dir: Path) -> None:
        """Update the 'latest' pointer to `step_dir`."""
        latest_path = self._root / "latest"
        if latest_path.exists():
            shutil.rmtree(latest_path)
        shutil.copytree(step_dir, latest_path)
