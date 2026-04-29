"""
InsightSerenity AI Engine — Normalization Layers
=================================================
Normalisation is essential for training stability in deep networks. Without
it, activations grow or shrink exponentially with depth, making gradients
either vanish or explode.

Layers implemented:

1. LayerNorm (Ba et al., 2016)
   Normalises over the FEATURE dimension of each example independently.
   Shape: (B, T, D) → normalise over D.
   The default choice for transformers. Every example is normalised
   independently — no dependency on batch size.

2. RMSNorm (Zhang & Sennrich, 2019)
   Root Mean Square normalisation — a simplified LayerNorm that only
   rescales without centring (no mean subtraction).
   RMSNorm(x) = x / RMS(x) * g    where  RMS(x) = sqrt(mean(x^2))
   Used in: LLaMA-1/2/3, Mistral, T5, Gemma, Falcon.
   Faster than LayerNorm (no mean computation) and equally effective.

3. BatchNorm1d / BatchNorm2d
   Normalises over the BATCH dimension for each feature channel.
   Shape (1D): (B, C, L) → normalise over B for each (C, l).
   Requires batch size > 1. Better for CNNs than transformers.

4. GroupNorm (Wu & He, 2018)
   Divides channels into groups, normalises within each group.
   Works well with small batches (single images). Used in vision models.

Pre-norm vs Post-norm:
   Original Transformer: Post-norm (attention → residual → LayerNorm)
   Modern practice:      Pre-norm  (LayerNorm → attention → residual)
   Pre-norm is more stable and generally easier to train deep models with.
"""

import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    """
    Layer Normalisation (Ba et al., 2016).

    Normalises the input to zero mean and unit variance over the last
    `normalised_shape` dimensions, then applies a learned scale (gamma)
    and optional shift (beta).

    This is equivalent to nn.LayerNorm but with explicit documentation
    of what is actually being computed.

    Formula:
        y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    Args:
        normalised_shape: The shape of the dimensions to normalise.
                          For transformers: typically (d_model,) — normalise
                          the last dimension (the feature dimension).
        eps:              Small constant for numerical stability. Default 1e-5.
        elementwise_affine: If True, learn per-feature gamma and beta.
    """

    def __init__(
        self,
        normalised_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.normalised_shape = (normalised_shape,) if isinstance(normalised_shape, int) \
            else tuple(normalised_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            # gamma (scale): initialised to 1 — no rescaling at the start
            self.weight = nn.Parameter(torch.ones(self.normalised_shape))
            # beta (shift): initialised to 0 — no shift at the start
            self.bias   = nn.Parameter(torch.zeros(self.normalised_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of any shape ending in normalised_shape.
               Typical: (B, T, d_model)

        Returns:
            Normalised tensor of the same shape.
        """
        return nn.functional.layer_norm(
            x,
            self.normalised_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"normalised_shape={self.normalised_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalisation (Zhang & Sennrich, 2019).

    Simpler and slightly faster than LayerNorm. Removes the mean-centring step
    and only applies RMS-based rescaling. Works as well as LayerNorm in practice
    for transformer architectures.

    Formula:
        RMS(x) = sqrt( mean(x_i^2) + eps )
        RMSNorm(x) = x / RMS(x) * gamma

    Used by: LLaMA-1/2/3, Mistral, Gemma, Falcon, T5 variants.

    Args:
        d_model: Feature dimension (last dimension of input).
        eps:     Numerical stability constant. Default 1e-6.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps     = eps
        # Learned per-feature scale
        self.weight  = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model) or (*, d_model).

        Returns:
            RMS-normalised tensor of the same shape.
        """
        # Compute RMS over the last dimension
        # keepdim=True to allow broadcasting over all positions
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # Normalise and scale; cast back to original dtype
        return (x.float() / rms).to(x.dtype) * self.weight

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


class BatchNorm1d(nn.Module):
    """
    Batch Normalisation for 1D sequential data (B, C, L).

    Normalises over the batch dimension B for each (channel, position) pair.
    Tracks running mean and variance for use during inference (eval mode)
    without needing to see the batch.

    Warning: requires batch size > 1 during training. Not suitable for
    transformer architectures — use LayerNorm or RMSNorm instead.

    Args:
        num_features: Number of channels C.
        eps:          Numerical stability constant. Default 1e-5.
        momentum:     Momentum for running statistics update. Default 0.1.
        affine:       If True, learn per-channel gamma and beta.
        track_running_stats: If True, maintain running mean/var for inference.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C) or (B, C, L).

        Returns:
            Batch-normalised tensor of the same shape.
        """
        return self.norm(x)


class BatchNorm2d(nn.Module):
    """
    Batch Normalisation for 2D spatial data (B, C, H, W).

    Args:
        num_features: Number of channels C.
        eps:          Numerical stability constant.
        momentum:     Running statistics momentum.
        affine:       Learn per-channel gamma and beta.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W).

        Returns:
            Batch-normalised tensor.
        """
        return self.norm(x)


class GroupNorm(nn.Module):
    """
    Group Normalisation (Wu & He, 2018).

    Divides channels into `num_groups` groups and normalises within each group.
    Unlike BatchNorm, GroupNorm does not depend on batch size — it can work
    with batch_size=1. This makes it the preferred choice when batch sizes
    are small (e.g. object detection, high-resolution images).

    When num_groups=1: equivalent to LayerNorm over channel dimension.
    When num_groups=C: equivalent to InstanceNorm.

    Args:
        num_groups:   Number of groups to divide channels into.
        num_channels: Total number of channels (must be divisible by num_groups).
        eps:          Numerical stability constant.
        affine:       Learn per-channel gamma and beta.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, *) — any shape with batch and channel dimensions.

        Returns:
            Group-normalised tensor.
        """
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_norm(name: str, d_model: int, **kwargs) -> nn.Module:
    """
    Retrieve a normalisation layer by name.

    Args:
        name:    "layer", "rms", "batch1d", "batch2d", "group".
        d_model: The primary dimension (feature size or channel count).
        **kwargs: Additional arguments passed to the norm constructor.

    Returns:
        An instantiated normalisation nn.Module.

    Raises:
        ValueError: If the name is not recognised.
    """
    name_lower = name.lower()

    if name_lower == "layer":
        return LayerNorm(d_model, **kwargs)
    elif name_lower == "rms":
        return RMSNorm(d_model, **kwargs)
    elif name_lower == "batch1d":
        return BatchNorm1d(d_model, **kwargs)
    elif name_lower == "batch2d":
        return BatchNorm2d(d_model, **kwargs)
    elif name_lower == "group":
        return GroupNorm(num_groups=kwargs.pop("num_groups", 8), num_channels=d_model, **kwargs)
    else:
        raise ValueError(
            f"Unknown norm '{name}'. "
            f"Supported: layer, rms, batch1d, batch2d, group."
        )
