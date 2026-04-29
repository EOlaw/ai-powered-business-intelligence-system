"""
InsightSerenity AI Engine — Language Model Trainer
====================================================
Concrete trainer for autoregressive (GPT-style) and masked (BERT-style)
language model training. Inherits from BaseTrainer and implements the
two required abstract methods: train_step and val_step.

The trainer handles:
    - Cross-entropy loss over vocabulary logits (CLM)
    - Perplexity calculation from the loss
    - Gradient backward call (inside train_step, before the base class clips)
    - AMP-safe loss computation

The loss computation for CLM:
    - Input:  token IDs [BOS, w1, w2, ..., wN]
    - Target: token IDs [w1, w2, ..., wN, EOS]  (shifted left by 1)
    - Label -100 at padding positions → CrossEntropyLoss ignores them
    - Loss = mean(-log P(w_{t+1} | w_1..w_t)) over all non-padding positions

Usage:
    from src.training.trainer.language_model_trainer import LMTrainer, LMTrainerConfig

    trainer = LMTrainer(
        model=gpt_model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=LMTrainerConfig(max_epochs=3, use_amp=True),
        scheduler=scheduler,
        val_loader=val_loader,
    )
    state = trainer.train()
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.training.trainer.base_trainer import BaseTrainer, TrainerConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LMTrainerConfig(TrainerConfig):
    """
    Extended configuration for language model training.
    Inherits all fields from TrainerConfig.
    """
    # Vocabulary size — used to validate logit shape during debugging
    vocab_size:      Optional[int] = None
    # Token ID to ignore in the loss (padding / masked positions)
    ignore_index:    int           = -100
    # Label smoothing: prevents overconfidence (0 = disabled)
    label_smoothing: float         = 0.0


class LanguageModelTrainer(BaseTrainer):
    """
    Trainer for autoregressive (causal) language models.

    The training objective is next-token prediction:
        L = -1/N * sum_{t=1}^{N} log P(x_{t+1} | x_1, ..., x_t)

    The loss is computed with label smoothing and the ignore_index mask
    so padding tokens do not contribute to the gradient.

    Args:
        model:       GPTDecoder or any model whose forward() returns a dict
                     with a "logits" key of shape (B, T, vocab_size).
        optimizer:   Optimizer instance.
        train_loader: DataLoader yielding dicts with "input_ids" and "labels".
        config:      LMTrainerConfig.
        scheduler:   Optional LR scheduler.
        val_loader:  Optional validation DataLoader.
        callbacks:   List of callbacks.
        checkpoint_mgr: Optional CheckpointManager.
    """

    def __init__(
        self,
        model:          nn.Module,
        optimizer:      Optimizer,
        train_loader:   DataLoader,
        config:         LMTrainerConfig,
        scheduler:      Optional[LRScheduler] = None,
        val_loader:     Optional[DataLoader]  = None,
        callbacks:      Optional[list]        = None,
        checkpoint_mgr: Optional[object]      = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            scheduler=scheduler,
            val_loader=val_loader,
            callbacks=callbacks,
            checkpoint_mgr=checkpoint_mgr,
        )
        self.lm_config = config
        self.loss_fn   = nn.CrossEntropyLoss(
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing,
        )

    # ── Abstract implementations ───────────────────────────────────────────────

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        One forward + backward pass on a training batch.

        Expected batch keys:
            input_ids:      (B, T) — token IDs
            labels:         (B, T) — target token IDs (-100 at pad positions)
            attention_mask: (B, T) — optional padding mask

        Returns:
            {"loss": float, "perplexity": float, "tokens_per_sec": float}
        """
        self.model.train()

        input_ids       = batch["input_ids"]
        labels          = batch["labels"]
        attention_mask  = batch.get("attention_mask")

        # Forward pass
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = output["logits"]   # (B, T, V)

        # Compute cross-entropy loss
        B, T, V = logits.shape
        loss = self.loss_fn(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
        )

        # NaN/Inf guard: detect before backward to prevent corrupting model weights.
        # Return a sentinel value; base_trainer._run_epoch sees it and skips the step.
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "NaN/Inf loss detected — skipping backward",
                step=getattr(self, "state", None) and getattr(self.state, "global_step", "?"),
            )
            return {"loss": float("nan"), "perplexity": float("nan")}

        # Backward: scale by gradient accumulation steps so effective loss
        # is the mean over accumulated steps (not the sum)
        scaled_loss = loss / self.config.gradient_accumulation
        scaled_loss.backward()

        perplexity = math.exp(min(loss.item(), 20.0))

        return {
            "loss":       loss.item(),
            "perplexity": perplexity,
        }

    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        One forward pass on a validation batch (no gradients).

        Args:
            batch: Same format as train_step batches.

        Returns:
            {"loss": float, "perplexity": float}
        """
        input_ids      = batch["input_ids"]
        labels         = batch["labels"]
        attention_mask = batch.get("attention_mask")

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = output["logits"]
        B, T, V = logits.shape

        loss = self.loss_fn(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
        )

        perplexity = math.exp(min(loss.item(), 20.0))

        return {
            "loss":       loss.item(),
            "perplexity": perplexity,
        }

    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """
        Compute perplexity over an entire dataset.

        Perplexity = exp(mean cross-entropy loss).
        Lower perplexity = better language model.

        Args:
            dataloader: DataLoader to evaluate on.

        Returns:
            Perplexity as a float.
        """
        self.model.eval()
        total_loss  = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch)

                input_ids      = batch["input_ids"]
                labels         = batch["labels"]
                attention_mask = batch.get("attention_mask")

                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output["logits"]
                B, T, V = logits.shape

                # Sum loss (not mean) so we can average over tokens correctly
                loss_per_token = nn.CrossEntropyLoss(
                    ignore_index=self.lm_config.ignore_index,
                    reduction="sum",
                )(logits.reshape(B * T, V), labels.reshape(B * T))

                # Count non-ignored tokens
                non_pad = (labels != self.lm_config.ignore_index).sum().item()

                total_loss   += loss_per_token.item()
                total_tokens += non_pad

        if total_tokens == 0:
            return float("inf")

        mean_loss  = total_loss / total_tokens
        perplexity = math.exp(min(mean_loss, 20.0))

        logger.info("Perplexity computed", perplexity=round(perplexity, 4), tokens=total_tokens)
        self.model.train()
        return perplexity
