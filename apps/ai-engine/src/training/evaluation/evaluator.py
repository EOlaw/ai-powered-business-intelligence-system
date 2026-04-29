"""
InsightSerenity AI Engine — Evaluator
======================================
Orchestrates a complete evaluation pass over a dataset and computes
all configured metrics in a single run.

The Evaluator is used:
    1. During training (called by the trainer after each N steps)
    2. After training (standalone evaluation of a checkpoint)
    3. For benchmarking (compare multiple checkpoints)

It decouples metric computation from the trainer, so you can evaluate
any model on any dataset without setting up a full training run.

Usage:
    from src.training.evaluation.evaluator import Evaluator

    evaluator = Evaluator(model, tokenizer, device="cuda")
    results   = evaluator.evaluate(val_loader)
    print(results)
    # {
    #     "loss":       2.345,
    #     "perplexity": 10.43,
    #     "accuracy":   0.451,
    #     "bleu":       0.123,
    # }
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor

from src.training.evaluation.metrics import (
    perplexity, accuracy, top_k_accuracy,
    bleu_score, rouge_l_corpus,
)
from src.utils.logger import get_logger, LogTimer

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """
    Container for all metrics from one evaluation run.

    Attributes:
        loss:        Average cross-entropy loss over the eval dataset.
        perplexity:  exp(loss) — language model quality.
        accuracy:    Token prediction accuracy (for CLM).
        top5_acc:    Top-5 token prediction accuracy.
        bleu:        BLEU score (only if generated text is available).
        rouge_l:     ROUGE-L F1 (only if generated text is available).
        num_examples: Total examples evaluated.
        num_batches:  Total batches evaluated.
        extra:       Any additional task-specific metrics.
    """
    loss:         float = 0.0
    perplexity:   float = 0.0
    accuracy:     float = 0.0
    top5_acc:     float = 0.0
    bleu:         Optional[float] = None
    rouge_l:      Optional[float] = None
    num_examples: int   = 0
    num_batches:  int   = 0
    extra:        Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to a flat dict of metric name → value."""
        result = {
            "loss":       self.loss,
            "perplexity": self.perplexity,
            "accuracy":   self.accuracy,
            "top5_acc":   self.top5_acc,
        }
        if self.bleu is not None:
            result["bleu"] = self.bleu
        if self.rouge_l is not None:
            result["rouge_l"] = self.rouge_l
        result.update(self.extra)
        return result

    def __str__(self) -> str:
        parts = [
            f"loss={self.loss:.4f}",
            f"ppl={self.perplexity:.2f}",
            f"acc={self.accuracy:.4f}",
        ]
        if self.bleu is not None:
            parts.append(f"bleu={self.bleu:.4f}")
        return " | ".join(parts)


class Evaluator:
    """
    Runs evaluation on a language model over an entire dataset.

    Args:
        model:         The trained model (must return dict with "logits" key).
        device:        Compute device. Default "cpu".
        ignore_index:  Label index to ignore in loss/accuracy. Default -100.
        max_batches:   Cap the evaluation at N batches (for quick checks).
    """

    def __init__(
        self,
        model:        nn.Module,
        device:       str = "cpu",
        ignore_index: int = -100,
        max_batches:  Optional[int] = None,
    ) -> None:
        self.model        = model
        self.device       = torch.device(device)
        self.ignore_index = ignore_index
        self.max_batches  = max_batches

        self.model.to(self.device)

    def evaluate(self, dataloader: DataLoader) -> EvalResult:
        """
        Run a full evaluation pass over `dataloader`.

        For each batch:
            1. Forward pass (no gradients)
            2. Compute loss, accuracy
            3. Accumulate metrics

        Returns:
            EvalResult with averaged metrics.
        """
        self.model.eval()

        total_loss      = 0.0
        total_correct   = 0
        total_top5      = 0
        total_valid_tok = 0
        n_batches       = 0
        n_examples      = 0

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction="sum",   # Sum for accurate per-token averaging
        )

        with LogTimer(logger, "Evaluation"):
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if self.max_batches is not None and batch_idx >= self.max_batches:
                        break

                    # Move to device
                    batch = {
                        k: v.to(self.device) if isinstance(v, Tensor) else v
                        for k, v in batch.items()
                    }

                    input_ids      = batch["input_ids"]
                    labels         = batch["labels"]
                    attention_mask = batch.get("attention_mask")

                    # Forward pass
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = output["logits"]   # (B, T, V)
                    B, T, V = logits.shape

                    flat_logits = logits.reshape(B * T, V)
                    flat_labels = labels.reshape(B * T)

                    # Loss (summed, not mean, for correct aggregation)
                    total_loss += loss_fn(flat_logits, flat_labels).item()

                    # Valid token count
                    valid_mask   = flat_labels != self.ignore_index
                    n_valid      = valid_mask.sum().item()
                    total_valid_tok += n_valid

                    # Accuracy
                    preds   = flat_logits.argmax(dim=-1)
                    correct = (preds[valid_mask] == flat_labels[valid_mask]).sum().item()
                    total_correct += correct

                    # Top-5 accuracy
                    if V >= 5:
                        top5_preds = flat_logits[valid_mask].topk(5, dim=-1).indices
                        targets    = flat_labels[valid_mask].unsqueeze(-1).expand_as(top5_preds)
                        top5_correct = top5_preds.eq(targets).any(dim=-1).sum().item()
                        total_top5 += top5_correct

                    n_batches  += 1
                    n_examples += B

        # Aggregate
        if total_valid_tok == 0:
            return EvalResult()

        mean_loss   = total_loss / total_valid_tok
        ppl         = perplexity(mean_loss)
        acc         = total_correct / total_valid_tok
        top5        = total_top5 / total_valid_tok if total_valid_tok > 0 else 0.0

        result = EvalResult(
            loss=round(mean_loss, 6),
            perplexity=round(ppl, 4),
            accuracy=round(acc, 6),
            top5_acc=round(top5, 6),
            num_examples=n_examples,
            num_batches=n_batches,
        )

        logger.info(
            "Evaluation complete",
            batches=n_batches,
            examples=n_examples,
            loss=result.loss,
            perplexity=result.perplexity,
            accuracy=result.accuracy,
        )
        self.model.train()
        return result

    def evaluate_generation(
        self,
        prompts:    List[str],
        references: List[str],
        tokenizer:  Any,
        generator:  Any,
    ) -> Dict[str, float]:
        """
        Evaluate text generation quality using BLEU and ROUGE-L.

        Args:
            prompts:    Input prompts to generate from.
            references: Expected output texts.
            tokenizer:  Tokenizer with encode/decode methods.
            generator:  TextGenerator (from src.llm.inference.generator).

        Returns:
            Dict with "bleu" and "rouge_l" scores.
        """
        predictions: List[str] = []

        for prompt in prompts:
            generated = generator.generate(prompt, max_new_tokens=128)
            predictions.append(generated)

        bleu   = bleu_score(predictions, references)
        rougeL = rouge_l_corpus(predictions, references)

        logger.info(
            "Generation evaluation complete",
            bleu=round(bleu, 4),
            rouge_l=round(rougeL, 4),
        )
        return {"bleu": bleu, "rouge_l": rougeL}
