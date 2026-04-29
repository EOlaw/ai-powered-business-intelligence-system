"""
InsightSerenity AI Engine — Activation Functions
=================================================
Activation functions introduce the non-linearity that makes neural networks
capable of approximating arbitrary functions. Without them, a stack of linear
layers collapses to a single linear transformation regardless of depth.

Each activation is provided in two forms:
    1. A standalone function operating on tensors
    2. An nn.Module wrapping the function for use in nn.Sequential

Activations implemented:
    ReLU       — Rectified Linear Unit. Simple, fast, no vanishing gradient
                 above 0. Has "dying ReLU" problem when inputs < 0.
    GELU       — Gaussian Error Linear Unit. Smooth approximation of ReLU.
                 Used by BERT, GPT-2/3/4, and most modern transformers.
    SiLU/Swish — Self-gated: x * sigmoid(x). Smooth, non-monotonic.
                 Used by EfficientNet, LLaMA FFN gates.
    Mish       — x * tanh(softplus(x)). Smooth, self-regularising.
                 Common in vision models.
    ELU        — Exponential Linear Unit. Negative values saturate to -alpha
                 rather than dying to 0.
    Softmax    — Converts a vector to a probability distribution.
                 Used in attention and output layers.
    LogSoftmax — log(Softmax(x)). More numerically stable than log(Softmax).
                 Used with NLLLoss.
    Sigmoid    — Maps to (0, 1). Used for binary classification output.
    Tanh       — Maps to (-1, 1). Zero-centred; used in RNN gates.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Activation functions (stateless, operate on tensors)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x: Tensor) -> Tensor:
    """max(0, x) — the workhorse activation for deep nets."""
    return F.relu(x)


def gelu(x: Tensor, approximate: str = "tanh") -> Tensor:
    """
    Gaussian Error Linear Unit: x * Φ(x) where Φ is the standard normal CDF.

    The "tanh" approximation is:
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    This is the version used by GPT-2, BERT, and most modern transformers.
    The "none" option uses the exact erf-based computation.

    Args:
        x:           Input tensor.
        approximate: "tanh" (default, faster) or "none" (exact).
    """
    return F.gelu(x, approximate=approximate)


def silu(x: Tensor) -> Tensor:
    """
    Sigmoid Linear Unit (also called Swish): x * sigmoid(x).
    Self-gated, smooth, non-monotonic. Used in LLaMA SwiGLU gates.
    """
    return F.silu(x)


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    """
    Generalised Swish: x * sigmoid(beta * x).
    beta=1 reduces to SiLU. beta can be learned in some architectures.
    """
    return x * torch.sigmoid(beta * x)


def mish(x: Tensor) -> Tensor:
    """
    Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).
    Smooth, self-regularising, no upper bound. Used in Mish-activated nets.
    """
    return x * torch.tanh(F.softplus(x))


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Exponential Linear Unit:
        ELU(x) = x       if x >= 0
               = α(e^x - 1) if x < 0
    Negative part saturates to -alpha, avoiding dead neurons.
    """
    return F.elu(x, alpha=alpha)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """
    Leaky ReLU: allows small gradient when x < 0, fixing the dying ReLU problem.
        LeakyReLU(x) = x           if x >= 0
                      = negative_slope * x  if x < 0
    """
    return F.leaky_relu(x, negative_slope=negative_slope)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Softmax over dimension `dim`.
    Converts a vector of raw scores to a probability distribution.
    Numerically stable (PyTorch subtracts max before exp).
    """
    return F.softmax(x, dim=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    log(Softmax(x)). More numerically stable than computing log after softmax.
    Pair with nn.NLLLoss for language model training.
    """
    return F.log_softmax(x, dim=dim)


def sigmoid(x: Tensor) -> Tensor:
    """σ(x) = 1 / (1 + e^{-x}). Maps ℝ → (0, 1). Binary output layer."""
    return torch.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    """tanh(x). Maps ℝ → (-1, 1). Zero-centred sigmoid variant. RNN gates."""
    return torch.tanh(x)


# ─────────────────────────────────────────────────────────────────────────────
# nn.Module wrappers (for use in nn.Sequential and model configs)
# ─────────────────────────────────────────────────────────────────────────────

class ReLU(nn.Module):
    """ReLU activation as nn.Module."""
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class GELU(nn.Module):
    """
    GELU activation as nn.Module.

    Args:
        approximate: "tanh" (faster) or "none" (exact). Default "tanh".
    """
    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x, self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate}"


class SiLU(nn.Module):
    """SiLU (Swish) activation as nn.Module."""
    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


class Mish(nn.Module):
    """Mish activation as nn.Module."""
    def forward(self, x: Tensor) -> Tensor:
        return mish(x)


class ELU(nn.Module):
    """ELU activation as nn.Module."""
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return elu(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class LeakyReLU(nn.Module):
    """Leaky ReLU activation as nn.Module."""
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class Softmax(nn.Module):
    """Softmax over a configurable dimension."""
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Sigmoid(nn.Module):
    """Sigmoid activation as nn.Module."""
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(nn.Module):
    """Tanh activation as nn.Module."""
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory: get activation by name (used in config-driven model construction)
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVATION_REGISTRY = {
    "relu":        ReLU,
    "gelu":        GELU,
    "silu":        SiLU,
    "swish":       SiLU,        # Alias
    "mish":        Mish,
    "elu":         ELU,
    "leaky_relu":  LeakyReLU,
    "softmax":     Softmax,
    "sigmoid":     Sigmoid,
    "tanh":        Tanh,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Retrieve an activation module by name.

    Args:
        name:    Activation name (case-insensitive).
                 Supported: relu, gelu, silu, swish, mish, elu, leaky_relu,
                 softmax, sigmoid, tanh.
        **kwargs: Forwarded to the activation constructor.

    Returns:
        An instantiated nn.Module activation.

    Raises:
        ValueError: If the name is not recognised.
    """
    name_lower = name.lower()
    cls = _ACTIVATION_REGISTRY.get(name_lower)
    if cls is None:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Supported: {sorted(_ACTIVATION_REGISTRY.keys())}"
        )
    return cls(**kwargs)
