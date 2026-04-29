"""
InsightSerenity AI Engine — Progress Callback
=============================================
Displays a live progress bar during training using tqdm.
Shows: current step, epoch, loss, perplexity, learning rate, and ETA.

The progress bar is optional — if tqdm is not installed (e.g. in
headless server environments), this callback degrades gracefully to
simple log-line output every N steps.
"""

import sys
from typing import Dict, Optional

from src.training.callbacks.base_callback import BaseCallback
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


class ProgressCallback(BaseCallback):
    """
    Training progress bar with live metric updates.

    Args:
        total_steps: Total number of optimizer steps (used to set tqdm total).
                     If None, shows step count without an ETA.
        log_every:   Fallback: log every N steps if tqdm is unavailable.
    """

    def __init__(
        self,
        total_steps: Optional[int] = None,
        log_every:   int           = 50,
    ) -> None:
        super().__init__()
        self._total_steps = total_steps
        self._log_every   = log_every
        self._pbar        = None

    def on_train_begin(self, state) -> None:
        if _TQDM_AVAILABLE:
            self._pbar = tqdm(
                total=self._total_steps,
                initial=state.global_step,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
                file=sys.stdout,
                leave=True,
            )
        else:
            logger.info("Progress: training started", total_steps=self._total_steps)

    def on_step_end(self, state, step_metrics: Dict[str, float]) -> None:
        postfix = {
            "loss": f"{step_metrics.get('loss', 0):.4f}",
            "ppl":  f"{step_metrics.get('perplexity', 0):.2f}",
            "epoch": state.epoch,
        }

        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(postfix)
        elif state.global_step % self._log_every == 0:
            logger.info("Progress", step=state.global_step, **postfix)

    def on_evaluate(self, state, val_metrics: Dict[str, float]) -> None:
        val_loss = val_metrics.get("val_loss", 0)
        val_ppl  = val_metrics.get("val_perplexity", 0)

        if self._pbar is not None:
            self._pbar.set_postfix({
                "val_loss": f"{val_loss:.4f}",
                "val_ppl":  f"{val_ppl:.2f}",
            })
        else:
            logger.info("Evaluation", val_loss=round(val_loss, 4))

    def on_train_end(self, state) -> None:
        if self._pbar is not None:
            self._pbar.close()
        logger.info(
            "Training complete",
            total_steps=state.global_step,
            best_val_loss=round(state.best_metric, 4),
        )
