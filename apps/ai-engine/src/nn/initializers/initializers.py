"""
InsightSerenity AI Engine — Weight Initializers
================================================
Weight initialisation sets the starting point for all model parameters.
Poor initialisation causes training to diverge or converge very slowly.
Good initialisation preserves signal magnitude through the forward pass
and gradient magnitude through the backward pass.

Key insight (Glorot & Bengio, 2010):
    If weights are too large → activations and gradients explode
    If weights are too small → activations and gradients vanish
    Good init keeps the variance of activations roughly constant with depth.

Initializers implemented:

1. xavier_uniform_ / xavier_normal_   (Glorot & Bengio, 2010)
   For activations without curvature (tanh, sigmoid, linear).
   Var(W) = 2 / (fan_in + fan_out)

2. kaiming_uniform_ / kaiming_normal_ (He et al., 2015)
   For ReLU and its variants (accounts for the ~half-dead neurons).
   Var(W) = 2 / fan_in   [mode="fan_in"]
   The correct default for any ReLU-like activation.

3. orthogonal_    (Saxe et al., 2013)
   Initialises weight matrices as random orthogonal matrices.
   Good for RNNs — preserves gradient magnitude over many time steps.

4. truncated_normal_
   Normal distribution with values beyond ±2σ resampled.
   Used by BERT, T5, and many vision transformers.

5. initialize_weights()
   Applies sensible default init to an entire model based on layer type.
   Call this once after model construction.

6. scale_residual_weights()
   GPT-2 trick: scale down weights in residual paths by 1/sqrt(num_layers)
   to counteract the accumulation of residual updates across depth.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Individual initializer functions
# ─────────────────────────────────────────────────────────────────────────────

def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Xavier (Glorot) uniform initialisation.

    Fills tensor with values drawn from U(-a, a) where:
        a = gain * sqrt(6 / (fan_in + fan_out))

    Best for: Linear layers with tanh/sigmoid activations.

    Args:
        tensor: Parameter tensor to initialise in-place.
        gain:   Scaling factor (e.g. nn.init.calculate_gain('tanh') ≈ 5/3).

    Returns:
        The initialised tensor.
    """
    return nn.init.xavier_uniform_(tensor, gain=gain)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Xavier (Glorot) normal initialisation.

    Fills tensor with values drawn from N(0, std²) where:
        std = gain * sqrt(2 / (fan_in + fan_out))

    Args:
        tensor: Parameter tensor.
        gain:   Scaling factor.

    Returns:
        The initialised tensor.
    """
    return nn.init.xavier_normal_(tensor, gain=gain)


def kaiming_uniform_(
    tensor: Tensor,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
) -> Tensor:
    """
    Kaiming (He) uniform initialisation.

    Fills tensor with values from U(-bound, bound) where:
        bound = sqrt(3) * gain / sqrt(fan_mode)

    Best for: Linear and conv layers with ReLU / LeakyReLU activations.

    Args:
        tensor:        Parameter tensor.
        mode:          "fan_in" (preserves variance of forward pass) or
                       "fan_out" (preserves variance of backward pass).
        nonlinearity:  Activation name for gain computation: "relu", "leaky_relu".

    Returns:
        The initialised tensor.
    """
    return nn.init.kaiming_uniform_(tensor, mode=mode, nonlinearity=nonlinearity)


def kaiming_normal_(
    tensor: Tensor,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
) -> Tensor:
    """
    Kaiming (He) normal initialisation.

    Fills tensor with values from N(0, std²) where:
        std = gain / sqrt(fan_mode)

    Args:
        tensor:        Parameter tensor.
        mode:          "fan_in" or "fan_out".
        nonlinearity:  Activation name.

    Returns:
        The initialised tensor.
    """
    return nn.init.kaiming_normal_(tensor, mode=mode, nonlinearity=nonlinearity)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Orthogonal initialisation.

    Fills tensor with a (semi-)orthogonal matrix via SVD.
    For a matrix W of shape (m, n):
        - If m >= n: columns are orthonormal (W.T @ W = I_n)
        - If m < n:  rows are orthonormal    (W @ W.T = I_m)

    Best for: RNN weight matrices — preserves gradient magnitude over time.

    Args:
        tensor: 2D (or higher) parameter tensor.
        gain:   Multiplicative scaling factor.

    Returns:
        The initialised tensor.
    """
    return nn.init.orthogonal_(tensor, gain=gain)


def truncated_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    a: float = -2.0,
    b: float = 2.0,
) -> Tensor:
    """
    Truncated normal initialisation.

    Draws from N(mean, std²) but resamples values outside [mean+a*std, mean+b*std].
    This prevents extreme initial values that could cause instability.

    Used by: BERT, T5, ViT, most modern vision and language transformers.

    Args:
        tensor: Parameter tensor to initialise in-place.
        mean:   Distribution mean. Default 0.
        std:    Standard deviation. Default 0.02 (BERT/GPT-2 default).
        a:      Lower cut-off in units of std. Default -2 → mean - 2*std.
        b:      Upper cut-off in units of std. Default  2 → mean + 2*std.

    Returns:
        The initialised tensor.
    """
    with torch.no_grad():
        # Use PyTorch's trunc_normal_ if available (1.8+), else manual
        try:
            nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a * std + mean, b=b * std + mean)
        except AttributeError:
            # Fallback for older PyTorch
            size = tensor.numel()
            tmp  = tensor.new_empty(size + int(size * 0.1)).normal_(mean=mean, std=std)
            valid = tmp[(tmp >= mean + a * std) & (tmp <= mean + b * std)]
            if valid.numel() < size:
                # Very rare: regenerate
                extra = tensor.new_empty(size).normal_(mean=mean, std=std).clamp_(
                    mean + a * std, mean + b * std
                )
                valid = torch.cat([valid, extra])
            tensor.copy_(valid[:size].reshape(tensor.shape))
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Model-level init
# ─────────────────────────────────────────────────────────────────────────────

def initialize_weights(
    model: nn.Module,
    init_std: float = 0.02,
    norm_eps: float = 1e-5,
) -> None:
    """
    Apply sensible default initialisation to every layer in a model.

    Strategy (GPT-2 / LLaMA convention):
        Linear layers:    Truncated normal N(0, 0.02²), bias → 0
        Embedding layers: Truncated normal N(0, 0.02²), padding → 0
        LayerNorm / RMSNorm: weight → 1, bias → 0
        BatchNorm:        weight → 1, bias → 0

    This is safe to call on any model at construction time. Individual
    layers can override their own initialisation after this call.

    Args:
        model:    The nn.Module to initialise.
        init_std: Standard deviation for weight normal distributions.
        norm_eps: Not used here; exposed for documentation completeness.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, std=init_std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

        elif isinstance(module, (nn.LayerNorm,)):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if module.affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def scale_residual_weights(model: nn.Module, num_layers: int) -> None:
    """
    GPT-2 residual path scaling: multiply weights in residual output
    projections by 1 / sqrt(2 * num_layers).

    When a transformer has N layers and each layer adds to the residual
    stream, the variance of the accumulated residual grows as O(N).
    Pre-scaling output projections by 1/sqrt(2N) counteracts this and
    stabilises training of very deep models.

    Convention: output projection layers are named "out_proj" or "c_proj".
    Call this AFTER initialize_weights().

    Args:
        model:      The transformer model.
        num_layers: Number of transformer layers (used to compute the scale).
    """
    scale = 1.0 / math.sqrt(2.0 * num_layers)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            s in name for s in ["out_proj", "c_proj", "w_down"]
        ):
            with torch.no_grad():
                module.weight.mul_(scale)
