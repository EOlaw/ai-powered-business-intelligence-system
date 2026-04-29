"""
InsightSerenity AI Engine — Loss Functions
==========================================
Loss functions measure how far the model's predictions are from the ground
truth. The gradient of the loss with respect to model parameters is what
drives learning during backpropagation.

Losses implemented:

1. CrossEntropyLoss
   Standard classification loss: -sum(y * log(ŷ)).
   With label smoothing: prevents overconfidence by softening the target
   distribution from one-hot to (1-ε, ε/(K-1), ...).
   Language model training uses this with ignore_index=-100 to skip padding.

2. MSELoss (Mean Squared Error)
   Regression loss: mean((y - ŷ)²). Sensitive to outliers.

3. HuberLoss (Smooth L1)
   Combines MSE for small errors (quadratic) and MAE for large errors
   (linear), making it robust to outliers. Used in DQN reward learning.

4. KLDivergenceLoss
   KL(P || Q) = sum(P * log(P/Q)). Measures how much distribution P
   differs from Q. Used in VAE training and knowledge distillation.

5. FocalLoss (Lin et al., 2017)
   FL = -(1 - p_t)^γ * log(p_t).
   Down-weights easy examples (high p_t) and focuses on hard ones.
   Solves the class imbalance problem better than re-weighting.

6. ContrastiveLoss
   Pulls similar pairs together and pushes dissimilar pairs apart in
   embedding space. Foundation for SimCLR, CLIP, and sentence embeddings.

7. LabelSmoothingCrossEntropy
   Explicit label smoothing implementation with temperature scaling.
   More flexible than the built-in label_smoothing parameter.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for multi-class classification.

    This wraps nn.CrossEntropyLoss but exposes all options clearly and
    adds an explicit description of what each parameter does.

    For language model training: pass ignore_index=-100 to skip the loss
    on padding positions (the datasets set labels[pad_pos] = -100).

    Args:
        weight:        Optional (C,) tensor of per-class weights.
                       Useful for imbalanced datasets.
        ignore_index:  Class index to exclude from loss computation.
                       Default -100 (padding convention).
        label_smoothing: Smooth the target distribution. 0.0 = one-hot,
                         0.1 = common for transformer training.
        reduction:     "mean", "sum", or "none".
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  (B, C) or (B, T, C) — raw unnormalised scores.
            targets: (B,)  or (B, T)     — integer class indices.

        Returns:
            Scalar loss (if reduction="mean" or "sum") or (B,) tensor.
        """
        # For sequence tasks: flatten (B, T, C) → (B*T, C) and (B, T) → (B*T,)
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits  = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

        if (
            self.loss_fn.reduction == "mean"
            and self.loss_fn.ignore_index is not None
            and torch.all(targets == self.loss_fn.ignore_index)
        ):
            return logits.sum() * 0.0

        return self.loss_fn(logits, targets)


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for regression.

    MSE(y, ŷ) = mean((y - ŷ)²)

    Args:
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            predictions: Any shape float tensor.
            targets:     Same shape as predictions.

        Returns:
            Scalar MSE loss.
        """
        return self.loss_fn(predictions, targets)


class HuberLoss(nn.Module):
    """
    Huber loss (Smooth L1) — robust regression loss.

    For |error| < delta: loss = 0.5 * error²     (quadratic, like MSE)
    For |error| >= delta: loss = delta * (|error| - delta/2)  (linear, like MAE)

    More robust to outliers than MSE while still having zero gradient
    at the minimum. Standard loss for DQN reinforcement learning.

    Args:
        delta:     Threshold between quadratic and linear regimes. Default 1.0.
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta, reduction=reduction)
        self.delta   = delta

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            predictions: Float tensor of any shape.
            targets:     Same shape as predictions.

        Returns:
            Huber loss scalar.
        """
        return self.loss_fn(predictions, targets)

    def extra_repr(self) -> str:
        return f"delta={self.delta}"


class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence loss: KL(P || Q) = sum(P * log(P / Q)).

    Measures how much distribution P differs from Q. Backpropagation
    flows through Q (the model's predicted distribution) — P is treated
    as a fixed target.

    Used in:
        - VAE training: KL between posterior q(z|x) and prior p(z)
        - Knowledge distillation: match student distribution to teacher
        - Soft-label training: match to temperature-scaled teacher logits

    Args:
        reduction: "batchmean" (correct KL), "mean", "sum", or "none".
                   Use "batchmean" for correct mathematical KL divergence.
        log_target: If True, the target P is already in log-space.
    """

    def __init__(
        self,
        reduction: str = "batchmean",
        log_target: bool = False,
    ) -> None:
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction=reduction, log_target=log_target)

    def forward(self, log_probs: Tensor, target_probs: Tensor) -> Tensor:
        """
        Args:
            log_probs:    Log-probabilities from model (output of log_softmax).
            target_probs: Target probability distribution (not log-space by default).

        Returns:
            KL divergence scalar.
        """
        return self.loss_fn(log_probs, target_probs)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017 — RetinaNet).

    FL(p_t) = -(1 - p_t)^γ * log(p_t)

    The modulating factor (1 - p_t)^γ:
        - Reduces the loss contribution from easy, well-classified examples
        - Increases the relative loss for misclassified examples
        - γ = 0 → standard CrossEntropy; γ = 2 is the typical default

    Solves the class imbalance problem by dynamically scaling down the
    loss for easy examples instead of re-weighting classes statically.

    Args:
        gamma:        Focusing parameter. Higher = more focus on hard examples.
                      Typical range: 0.5–5. Default 2.
        alpha:        Optional (C,) tensor of class weights. If a scalar,
                      applied to the positive class (binary focal loss).
        ignore_index: Class index to exclude.
        reduction:    "mean", "sum", or "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  (B, C) or (B, T, C) — raw scores.
            targets: (B,) or (B, T) — integer class indices.

        Returns:
            Focal loss scalar.
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits  = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

        # Standard cross-entropy (per sample, not reduced)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Probability of the true class
        probs    = F.softmax(logits, dim=-1)
        # Gather the probability of the correct class for each sample
        valid    = targets != self.ignore_index
        p_t      = torch.zeros_like(ce_loss)
        p_t[valid] = probs[valid].gather(1, targets[valid].unsqueeze(1)).squeeze(1)

        # Modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss   = focal_weight * ce_loss

        # Zero out ignored positions
        focal_loss = focal_loss * valid.float()

        if self.reduction == "mean":
            return focal_loss.sum() / valid.float().sum().clamp(min=1.0)
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction={self.reduction}"


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for embedding learning.

    L = (1 - y) * 0.5 * D² + y * 0.5 * max(0, margin - D)²

    where D = ||e_a - e_b||₂ (Euclidean distance between two embeddings)
    and y = 0 if the pair is similar, 1 if dissimilar.

    For similar pairs (y=0): minimise D (pull together).
    For dissimilar pairs (y=1): maximise D up to `margin` (push apart).

    Args:
        margin:    Minimum distance enforced for dissimilar pairs. Default 1.0.
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin    = margin
        self.reduction = reduction

    def forward(
        self,
        embedding_a: Tensor,
        embedding_b: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Args:
            embedding_a: (B, D) — first embedding in each pair.
            embedding_b: (B, D) — second embedding in each pair.
            labels:      (B,) float — 0 for similar, 1 for dissimilar.

        Returns:
            Contrastive loss scalar.
        """
        distances = F.pairwise_distance(embedding_a, embedding_b, p=2)

        # Similar pair loss: minimise distance
        sim_loss  = (1.0 - labels) * 0.5 * distances.pow(2)

        # Dissimilar pair loss: push apart up to margin
        dissim_loss = labels * 0.5 * F.relu(self.margin - distances).pow(2)

        loss = sim_loss + dissim_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"margin={self.margin}"


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with explicit label smoothing and optional temperature scaling.

    Target distribution: (1 - ε) * one_hot + ε / K
    where ε = smoothing and K = number of classes.

    Temperature scaling: divide logits by T before softmax.
    T > 1 produces softer distributions (more uncertainty).
    T < 1 produces sharper distributions (more confident).
    Used in knowledge distillation.

    Args:
        smoothing:    Label smoothing factor ε in [0, 1). Default 0.1.
        temperature:  Temperature T for logit scaling. Default 1.0.
        ignore_index: Class to ignore. Default -100.
        reduction:    "mean" or "sum".
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        temperature: float = 1.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smoothing    = smoothing
        self.temperature  = temperature
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  (B, C) or (B, T, C).
            targets: (B,) or (B, T) integer class indices.

        Returns:
            Label-smoothed cross-entropy scalar.
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits  = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

        C = logits.size(-1)

        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Compute log-probabilities
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        # Build smooth target distribution: (B, C) float tensor
        # Start with uniform distribution of ε/K
        smooth_targets = torch.full_like(log_probs, self.smoothing / C)

        # Add (1-ε) to the one-hot positions (valid targets only)
        valid_mask = targets != self.ignore_index
        smooth_targets[valid_mask] = smooth_targets[valid_mask].scatter(
            1,
            targets[valid_mask].unsqueeze(1),
            1.0 - self.smoothing + self.smoothing / C,
        )

        # KL divergence: -sum(P * log(Q))
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Zero out ignored positions
        loss = loss * valid_mask.float()

        if self.reduction == "mean":
            return loss.sum() / valid_mask.float().sum().clamp(min=1.0)
        return loss.sum()

    def extra_repr(self) -> str:
        return (
            f"smoothing={self.smoothing}, "
            f"temperature={self.temperature}"
        )
