"""
InsightSerenity AI Engine — Callback Base Class
================================================
Callbacks are the observation and intervention layer of the training loop.
They receive notifications at every lifecycle event (step begin/end, epoch
begin/end, evaluate, train begin/end) and can:
    - Log metrics to disk, console, or experiment trackers
    - Save checkpoints
    - Modify training state (e.g. early stopping)
    - Visualise training progress

The trainer calls each registered callback at each lifecycle event.
The callback signature uses keyword-only arguments so new events can be
added without breaking existing callbacks.

Lifecycle events (in order):
    on_train_begin(state)
    on_epoch_begin(state)
        on_step_end(state, metrics)        ← after each optimizer step
        on_evaluate(state, val_metrics)    ← after each validation run
    on_epoch_end(state, epoch_metrics)
    on_train_end(state)
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.trainer.base_trainer import TrainState


class BaseCallback:
    """
    Abstract base class for all training callbacks.

    Default implementation of every lifecycle method is a no-op.
    Subclasses override only the events they care about.

    The attribute `_stop_training` can be set to True by a subclass
    (e.g. EarlyStoppingCallback) to signal the trainer to stop after
    the current epoch completes.
    """

    def __init__(self) -> None:
        # Signal to the trainer that it should stop training
        self._stop_training: bool = False

    def on_train_begin(self, state: "TrainState") -> None:
        """Called once before the first training epoch."""

    def on_train_end(self, state: "TrainState") -> None:
        """Called once after training completes (or is interrupted)."""

    def on_epoch_begin(self, state: "TrainState") -> None:
        """Called at the start of each epoch before iterating the DataLoader."""

    def on_epoch_end(
        self, state: "TrainState", epoch_metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch with aggregated epoch metrics."""

    def on_step_end(
        self, state: "TrainState", step_metrics: Dict[str, float]
    ) -> None:
        """Called after every optimizer step with per-step metrics."""

    def on_evaluate(
        self, state: "TrainState", val_metrics: Dict[str, float]
    ) -> None:
        """Called after every validation run with validation metrics."""


class CallbackList:
    """
    Container that dispatches lifecycle events to a list of callbacks.

    Using a CallbackList instead of iterating callbacks in the trainer
    keeps the trainer code clean and makes it easy to add/remove callbacks.

    Args:
        callbacks: List of BaseCallback instances.
    """

    def __init__(self, callbacks: Optional[list] = None) -> None:
        self.callbacks = callbacks or []

    def add(self, callback: BaseCallback) -> None:
        """Append a callback to the list."""
        self.callbacks.append(callback)

    def __call__(self, event: str, *args, **kwargs) -> None:
        """Dispatch `event` to all callbacks."""
        for cb in self.callbacks:
            if hasattr(cb, event):
                getattr(cb, event)(*args, **kwargs)

    def should_stop(self) -> bool:
        """Return True if any callback has set _stop_training = True."""
        return any(getattr(cb, "_stop_training", False) for cb in self.callbacks)
