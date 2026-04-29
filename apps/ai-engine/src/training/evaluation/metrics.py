"""
InsightSerenity AI Engine — Evaluation Metrics
===============================================
Pure functions for computing evaluation metrics on model predictions.
All functions are stateless — they take prediction tensors or lists
and return scalar floats.

Metrics implemented:

1. perplexity(loss)
   The standard language model metric. Lower = better.
   Perplexity = exp(mean cross-entropy loss)
   A perplexity of K means the model is as uncertain as if it had to
   choose uniformly among K options for each token.

2. accuracy(logits, labels)
   Fraction of tokens predicted correctly. Ignores -100 positions.

3. top_k_accuracy(logits, labels, k)
   Fraction of tokens where the true label is in the top-k predictions.

4. bleu_score(predictions, references)
   Bilingual Evaluation Understudy — measures n-gram overlap.
   Used for translation and summarisation quality.
   Returns BLEU-4 (geometric mean of 1,2,3,4-gram precisions).

5. rouge_l(prediction, reference)
   Recall-Oriented Understudy for Gisting Evaluation — Longest Common
   Subsequence based. Better for summarisation than BLEU.

6. f1_score(predictions, labels, num_classes)
   Macro-averaged F1 for classification tasks.
"""

import math
import re
from collections import Counter
from typing import List, Optional, Tuple

import torch
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Language model metrics
# ─────────────────────────────────────────────────────────────────────────────

def perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss). Clamped to avoid numerical overflow.

    Args:
        loss: Mean cross-entropy loss per token (scalar float).

    Returns:
        Perplexity as a float. Lower is better.
    """
    return math.exp(min(loss, 100.0))


def accuracy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy, ignoring padding positions.

    Args:
        logits:       (B, T, V) or (N, V) — raw model output.
        labels:       (B, T) or (N,) — integer class indices.
        ignore_index: Positions with this label are excluded. Default -100.

    Returns:
        Accuracy in [0, 1].
    """
    if logits.dim() == 3:
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        labels = labels.reshape(B * T)

    predictions = logits.argmax(dim=-1)
    valid_mask  = labels != ignore_index

    if valid_mask.sum() == 0:
        return 0.0

    correct = (predictions[valid_mask] == labels[valid_mask]).float().sum()
    total   = valid_mask.float().sum()
    return (correct / total).item()


def top_k_accuracy(
    logits: Tensor,
    labels: Tensor,
    k: int = 5,
    ignore_index: int = -100,
) -> float:
    """
    Compute top-K accuracy: fraction of examples where the true label
    is among the K highest-probability predictions.

    Args:
        logits:       (N, C) or (B, T, C) raw scores.
        labels:       (N,) or (B, T) integer indices.
        k:            Number of top predictions to consider.
        ignore_index: Positions to exclude.

    Returns:
        Top-K accuracy in [0, 1].
    """
    if logits.dim() == 3:
        B, T, C = logits.shape
        logits = logits.reshape(B * T, C)
        labels = labels.reshape(B * T)

    valid_mask = labels != ignore_index
    if valid_mask.sum() == 0:
        return 0.0

    top_k_preds = logits[valid_mask].topk(k, dim=-1).indices
    targets     = labels[valid_mask].unsqueeze(-1).expand_as(top_k_preds)
    correct     = top_k_preds.eq(targets).any(dim=-1).float().sum()
    total       = valid_mask.float().sum()
    return (correct / total).item()


# ─────────────────────────────────────────────────────────────────────────────
# Text generation quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_text(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenisation for metric computation."""
    return re.findall(r"\w+", text.lower())


def _count_ngrams(tokens: List[str], n: int) -> Counter:
    """Count all n-grams in a token list."""
    return Counter(
        tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)
    )


def bleu_score(
    predictions: List[str],
    references:  List[str],
    max_n:       int  = 4,
    smooth:      bool = True,
) -> float:
    """
    Compute corpus-level BLEU score.

    BLEU measures how much of the model's predicted text (n-grams) appears
    in the reference text. BLEU-4 is the standard for machine translation.

    Args:
        predictions: List of predicted text strings (one per example).
        references:  List of reference (ground truth) text strings.
        max_n:       Maximum n-gram order to consider. Default 4 (BLEU-4).
        smooth:      Apply NIST smoothing to avoid 0 for unseen n-grams.

    Returns:
        BLEU score in [0, 1]. Higher is better.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    # Aggregate counts across the corpus
    clipped_counts = [0] * max_n
    total_counts   = [0] * max_n
    pred_len_total = 0
    ref_len_total  = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize_text(pred)
        ref_tokens  = _tokenize_text(ref)

        pred_len_total += len(pred_tokens)
        ref_len_total  += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = _count_ngrams(pred_tokens, n)
            ref_ngrams  = _count_ngrams(ref_tokens,  n)

            # Clipped precision: count only ngrams that appear in reference
            for ngram, count in pred_ngrams.items():
                clipped_counts[n-1] += min(count, ref_ngrams.get(ngram, 0))
            total_counts[n-1] += max(len(pred_tokens) - n + 1, 0)

    # Compute n-gram precisions
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            p = clipped_counts[n] / total_counts[n]
            if smooth and p == 0:
                p = 1.0 / (total_counts[n] + 1)   # Laplace smoothing
            precisions.append(p)

    if all(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of n-gram precisions
    log_avg = sum(
        math.log(p) for p in precisions if p > 0
    ) / max_n

    # Brevity penalty: penalise predictions shorter than the reference
    bp = (
        1.0 if pred_len_total >= ref_len_total
        else math.exp(1 - ref_len_total / pred_len_total)
    )

    return bp * math.exp(log_avg)


def rouge_l(prediction: str, reference: str) -> Tuple[float, float, float]:
    """
    Compute ROUGE-L: Longest Common Subsequence based F1 score.

    ROUGE-L measures how much of the reference appears (in order) in the
    prediction. Unlike BLEU, it does not require exact n-gram matches.

    Args:
        prediction: Generated text.
        reference:  Ground truth text.

    Returns:
        Tuple (precision, recall, f1) — all in [0, 1].
        Use f1 (third element) for the standard ROUGE-L score.
    """
    pred_tokens = _tokenize_text(prediction)
    ref_tokens  = _tokenize_text(reference)

    if not pred_tokens or not ref_tokens:
        return (0.0, 0.0, 0.0)

    # Dynamic programming LCS
    m, n = len(pred_tokens), len(ref_tokens)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len   = dp[m][n]
    precision = lcs_len / max(m, 1)
    recall    = lcs_len / max(n, 1)
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return (precision, recall, f1)


def rouge_l_corpus(
    predictions: List[str],
    references:  List[str],
) -> float:
    """Average ROUGE-L F1 across a list of prediction/reference pairs."""
    if not predictions:
        return 0.0
    scores = [rouge_l(p, r)[2] for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def f1_score(
    predictions: List[int],
    labels:      List[int],
    num_classes: int,
    average:     str = "macro",
) -> float:
    """
    Compute F1 score for multi-class classification.

    Args:
        predictions: Predicted class indices.
        labels:      True class indices.
        num_classes: Total number of classes.
        average:     "macro" (unweighted mean) or "micro" (global).

    Returns:
        F1 score in [0, 1].
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    if average == "micro":
        tp = sum(p == l for p, l in zip(predictions, labels))
        fp = len(predictions) - tp
        fn = fp
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Macro: compute per-class F1, then average
    class_f1s = []
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(predictions, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(predictions, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(predictions, labels) if p != c and l == c)
        if tp + fp == 0 or tp + fn == 0:
            class_f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_f1s.append(f1)

    return sum(class_f1s) / max(len(class_f1s), 1)
