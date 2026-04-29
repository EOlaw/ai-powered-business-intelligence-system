"""
InsightSerenity AI Engine — Training Package
=============================================
Single import surface for the training infrastructure.

Usage:
    from src.training import LanguageModelTrainer, LMTrainerConfig
    from src.training import CheckpointManager
    from src.training import Evaluator, EvalResult
    from src.training import LoggingCallback, EarlyStoppingCallback
    from src.training import count_parameters, get_memory_stats
"""

from src.training.trainer.base_trainer import (
    BaseTrainer, TrainerConfig, TrainState,
)
from src.training.trainer.language_model_trainer import (
    LanguageModelTrainer, LMTrainerConfig,
)
from src.training.callbacks.base_callback import BaseCallback, CallbackList
from src.training.callbacks.logging_callback import LoggingCallback
from src.training.callbacks.early_stopping_callback import EarlyStoppingCallback
from src.training.callbacks.progress_callback import ProgressCallback
from src.training.checkpointing.checkpoint_manager import CheckpointManager
from src.training.evaluation.metrics import (
    perplexity, accuracy, top_k_accuracy,
    bleu_score, rouge_l, rouge_l_corpus, f1_score,
)
from src.training.evaluation.evaluator import Evaluator, EvalResult
from src.training.distributed.ddp_wrapper import (
    setup_distributed, cleanup_distributed, wrap_model_ddp,
    is_main_process, get_rank, get_world_size, barrier, all_reduce_mean,
)
from src.training.profiling.profiler import (
    count_parameters, parameter_summary, format_param_count,
    estimate_flops_transformer, get_memory_stats, reset_peak_memory_stats,
    measure_throughput, estimate_training_time,
)

__all__ = [
    # Trainer
    "BaseTrainer", "TrainerConfig", "TrainState",
    "LanguageModelTrainer", "LMTrainerConfig",
    # Callbacks
    "BaseCallback", "CallbackList",
    "LoggingCallback", "EarlyStoppingCallback", "ProgressCallback",
    # Checkpointing
    "CheckpointManager",
    # Evaluation
    "perplexity", "accuracy", "top_k_accuracy",
    "bleu_score", "rouge_l", "rouge_l_corpus", "f1_score",
    "Evaluator", "EvalResult",
    # Distributed
    "setup_distributed", "cleanup_distributed", "wrap_model_ddp",
    "is_main_process", "get_rank", "get_world_size", "barrier", "all_reduce_mean",
    # Profiling
    "count_parameters", "parameter_summary", "format_param_count",
    "estimate_flops_transformer", "get_memory_stats", "reset_peak_memory_stats",
    "measure_throughput", "estimate_training_time",
]
