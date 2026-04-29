"""
InsightSerenity AI Engine — Early Stopping Callback
====================================================
Monitors a validation metric and signals the trainer to stop when the
metric has not improved for `patience` consecutive evaluations.

This prevents overfitting by stopping before the model memorises the
training data beyond the point where it generalises.

The `min_delta` parameter prevents stopping on trivially small improvements.
E.g. if val_loss goes from 2.500 → 2.499, that's not meaningful — set
min_delta=0.001 to require at least 0.001 improvement to count.
"""

from typing import Optional

from src.training.callbacks.base_callback import BaseCallback
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when the monitored metric stops improving.

    Args:
        monitor:   Metric name to watch (key in val_metrics dict).
                   Default "val_loss".
        patience:  Evaluations with no improvement before stopping. Default 5.
        min_delta: Minimum change to count as an improvement. Default 1e-4.
        mode:      "min" — lower is better (loss).
                   "max" — higher is better (accuracy, F1).
        verbose:   Log a message when patience counter increments.
    """

    def __init__(
        self,
        monitor:   str   = "val_loss",
        patience:  int   = 5,
        min_delta: float = 1e-4,
        mode:      str   = "min",
        verbose:   bool  = True,
    ) -> None:
        super().__init__()

        self._monitor   = monitor
        self._patience  = patience
        self._min_delta = min_delta
        self._verbose   = verbose

        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self._mode = mode

        # Best value seen so far (inf for "min", -inf for "max")
        self._best_val: float = float("inf") if mode == "min" else float("-inf")
        self._wait:     int   = 0   # Evaluations since last improvement

    def on_evaluate(self, state, val_metrics: dict) -> None:
        """
        Check if the monitored metric has improved.
        If not improved for `patience` evaluations, set _stop_training = True.
        """
        current = val_metrics.get(self._monitor)

        if current is None:
            logger.warning(
                "EarlyStopping: monitored metric not found in val_metrics",
                monitor=self._monitor,
                available=list(val_metrics.keys()),
            )
            return

        # Check improvement
        improved = (
            current < self._best_val - self._min_delta
            if self._mode == "min"
            else current > self._best_val + self._min_delta
        )

        if improved:
            self._best_val = current
            self._wait     = 0
            if self._verbose:
                logger.info(
                    "EarlyStopping: improvement",
                    monitor=self._monitor,
                    value=round(current, 6),
                    step=state.global_step,
                )
        else:
            self._wait += 1
            if self._verbose:
                logger.info(
                    "EarlyStopping: no improvement",
                    monitor=self._monitor,
                    value=round(current, 6),
                    patience_used=f"{self._wait}/{self._patience}",
                )

            if self._wait >= self._patience:
                logger.warning(
                    "EarlyStopping: stopping training",
                    monitor=self._monitor,
                    best=round(self._best_val, 6),
                    patience=self._patience,
                )
                self._stop_training = True
