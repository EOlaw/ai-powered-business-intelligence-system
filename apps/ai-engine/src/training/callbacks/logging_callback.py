"""
InsightSerenity AI Engine — Logging Callback
=============================================
Writes training metrics to a JSONL log file and optionally to MLflow.

Every step and epoch event is persisted immediately — if training crashes,
the log up to that point is recoverable. The JSONL format means every line
is a self-contained JSON record that can be streamed, grepped, or loaded
into pandas without loading the entire file.

Each log record has this shape:
    {
        "event":       "step" | "epoch" | "evaluate",
        "step":        int,
        "epoch":       int,
        "timestamp":   "ISO-8601",
        "loss":        float,
        "perplexity":  float,
        "lr":          float,
        ...other metrics...
    }
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.training.callbacks.base_callback import BaseCallback
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingCallback(BaseCallback):
    """
    Logs all training metrics to a JSONL file and structured logger.

    Args:
        log_path:     Path to the output JSONL log file.
        log_mlflow:   If True, also log metrics to MLflow (must be installed).
        mlflow_run_id: Resume an existing MLflow run rather than starting new.
    """

    def __init__(
        self,
        log_path: str,
        log_mlflow: bool = False,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._log_mlflow   = log_mlflow
        self._mlflow_run   = None

        if log_mlflow:
            try:
                import mlflow
                if mlflow_run_id:
                    self._mlflow_run = mlflow.start_run(run_id=mlflow_run_id)
                else:
                    self._mlflow_run = mlflow.start_run()
                logger.info("MLflow run started", run_id=self._mlflow_run.info.run_id)
            except ImportError:
                logger.warning("MLflow not installed — skipping MLflow logging")
                self._log_mlflow = False

    def on_train_begin(self, state) -> None:
        self._write({
            "event": "train_begin",
            "step":  state.global_step,
            "epoch": state.epoch,
        })

    def on_step_end(self, state, step_metrics: Dict[str, float]) -> None:
        record = {
            "event": "step",
            "step":  state.global_step,
            "epoch": state.epoch,
            **step_metrics,
        }
        self._write(record)

        if self._log_mlflow and self._mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics(step_metrics, step=state.global_step)
            except Exception:
                pass

    def on_evaluate(self, state, val_metrics: Dict[str, float]) -> None:
        record = {
            "event": "evaluate",
            "step":  state.global_step,
            "epoch": state.epoch,
            **val_metrics,
        }
        self._write(record)

        if self._log_mlflow and self._mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics(val_metrics, step=state.global_step)
            except Exception:
                pass

    def on_epoch_end(self, state, epoch_metrics: Dict[str, float]) -> None:
        record = {
            "event": "epoch",
            "step":  state.global_step,
            "epoch": state.epoch,
            **epoch_metrics,
        }
        self._write(record)

    def on_train_end(self, state) -> None:
        self._write({
            "event":       "train_end",
            "step":        state.global_step,
            "epoch":       state.epoch,
            "best_metric": state.best_metric,
            "best_step":   state.best_step,
            "elapsed_secs": round(state.elapsed_secs, 2),
        })

        if self._log_mlflow and self._mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

    def _write(self, record: Dict[str, Any]) -> None:
        """Append one JSON record to the log file."""
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
