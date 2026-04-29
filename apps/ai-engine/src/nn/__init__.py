"""
InsightSerenity AI Engine — Neural Network Primitives Package
=============================================================
Single import surface for all nn building blocks.

Usage:
    from src.nn import Linear, MultiHeadAttention, RMSNorm, GELU
    from src.nn import AdamW, LinearWarmupCosineDecay
    from src.nn import CrossEntropyLoss, FocalLoss
    from src.nn import initialize_weights, get_activation
"""

# ── Layers ─────────────────────────────────────────────────────────────────────
from src.nn.layers.linear import Linear
from src.nn.layers.embedding import (
    TokenEmbedding,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
)
from src.nn.layers.conv import CausalConv1d, Conv1dBlock, Conv2dBlock
from src.nn.layers.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    MultiHeadCausalAttention,
    build_causal_mask,
    build_padding_mask,
)
from src.nn.layers.feedforward import FeedForward, GatedFeedForward

# ── Activations ────────────────────────────────────────────────────────────────
from src.nn.activations.activations import (
    ReLU, GELU, SiLU, Mish, ELU, LeakyReLU, Softmax, Sigmoid, Tanh,
    relu, gelu, silu, mish, elu, leaky_relu, softmax, sigmoid, tanh,
    get_activation,
)

# ── Normalization ──────────────────────────────────────────────────────────────
from src.nn.normalization.normalization import (
    LayerNorm, RMSNorm, BatchNorm1d, BatchNorm2d, GroupNorm,
    get_norm,
)

# ── Losses ────────────────────────────────────────────────────────────────────
from src.nn.losses.losses import (
    CrossEntropyLoss,
    MSELoss,
    HuberLoss,
    KLDivergenceLoss,
    FocalLoss,
    ContrastiveLoss,
    LabelSmoothingCrossEntropy,
)

# ── Optimizers ────────────────────────────────────────────────────────────────
from src.nn.optimizers.optimizers import (
    SGD, Adam, AdamW, Lion,
    get_optimizer,
)

# ── Schedulers ────────────────────────────────────────────────────────────────
from src.nn.schedulers.schedulers import (
    LinearWarmupCosineDecay,
    LinearWarmupLinearDecay,
    ConstantWithWarmup,
    CosineAnnealingWithRestarts,
    ReduceLROnPlateau,
    get_scheduler,
)

# ── Initializers ──────────────────────────────────────────────────────────────
from src.nn.initializers.initializers import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
    orthogonal_,
    truncated_normal_,
    initialize_weights,
    scale_residual_weights,
)

__all__ = [
    # Layers
    "Linear",
    "TokenEmbedding", "SinusoidalPositionalEmbedding",
    "LearnedPositionalEmbedding", "RotaryPositionalEmbedding",
    "CausalConv1d", "Conv1dBlock", "Conv2dBlock",
    "scaled_dot_product_attention", "MultiHeadAttention",
    "MultiHeadCausalAttention", "build_causal_mask", "build_padding_mask",
    "FeedForward", "GatedFeedForward",
    # Activations
    "ReLU", "GELU", "SiLU", "Mish", "ELU", "LeakyReLU",
    "Softmax", "Sigmoid", "Tanh",
    "relu", "gelu", "silu", "mish", "elu", "leaky_relu",
    "softmax", "sigmoid", "tanh", "get_activation",
    # Normalization
    "LayerNorm", "RMSNorm", "BatchNorm1d", "BatchNorm2d", "GroupNorm",
    "get_norm",
    # Losses
    "CrossEntropyLoss", "MSELoss", "HuberLoss", "KLDivergenceLoss",
    "FocalLoss", "ContrastiveLoss", "LabelSmoothingCrossEntropy",
    # Optimizers
    "SGD", "Adam", "AdamW", "Lion", "get_optimizer",
    # Schedulers
    "LinearWarmupCosineDecay", "LinearWarmupLinearDecay",
    "ConstantWithWarmup", "CosineAnnealingWithRestarts",
    "ReduceLROnPlateau", "get_scheduler",
    # Initializers
    "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_",
    "orthogonal_", "truncated_normal_",
    "initialize_weights", "scale_residual_weights",
]
