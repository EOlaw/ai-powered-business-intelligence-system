"""
InsightSerenity AI Engine — Feed-Forward Sublayers
====================================================
The position-wise feed-forward network (FFN) is the second sublayer in every
transformer block. After self-attention allows tokens to communicate with each
other, the FFN processes each token's representation independently through a
small MLP.

Implemented variants:

1. FeedForward (standard)
   FFN(x) = dropout(activation(Linear_1(x))) @ W_2 + b_2
   Typically: d_model → 4 × d_model → d_model
   Used in: original Transformer, BERT, GPT-2

2. GatedFeedForward (SwiGLU variant)
   FFN_SwiGLU(x) = (SiLU(W_1 x) ⊗ W_3 x) @ W_2
   Two independent linear projections, one gated by SiLU, element-wise
   multiplied before the down-projection.
   Used in: LLaMA-1/2/3, Mistral, PaLM, Gemma
   Advantage: better performance with fewer parameters than plain FFN.

Why is the FFN important?
   Attention is a routing mechanism — it decides which tokens to mix.
   The FFN is where most of the actual computation happens. Approximately
   2/3 of transformer parameters are in the FFN layers. The FFN acts as
   a key-value memory: rows of W_2 store "facts" and W_1 rows are queries.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    """
    Standard position-wise feed-forward network used in transformer blocks.

    Architecture:
        x → Linear(d_model, d_ff) → Activation → Dropout → Linear(d_ff, d_model)

    Args:
        d_model:    Input and output dimension.
        d_ff:       Hidden (expansion) dimension. Typically 4 × d_model.
        activation: Activation module. Default nn.GELU() (standard for transformers).
        dropout:    Dropout applied between the two linear layers.
        bias:       Add bias terms. Default True.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff    = d_ff

        self.linear_1  = nn.Linear(d_model, d_ff, bias=bias)
        self.linear_2  = nn.Linear(d_ff, d_model, bias=bias)
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout   = nn.Dropout(p=dropout)

        # Standard init for transformer FFN
        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.normal_(self.linear_2.weight, std=0.02)
        if bias:
            nn.init.zeros_(self.linear_1.bias)
            nn.init.zeros_(self.linear_2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model) — input from attention sublayer.

        Returns:
            (B, T, d_model) — transformed output.
        """
        x = self.linear_1(x)       # (B, T, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)       # (B, T, d_model)
        return x

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class GatedFeedForward(nn.Module):
    """
    SwiGLU-style gated feed-forward network.

    Architecture (Noam Shazeer, 2020):
        gate  = SiLU(W_gate(x))          # (B, T, d_ff)
        up    = W_up(x)                   # (B, T, d_ff)
        hidden = gate ⊗ up               # element-wise product (gating)
        output = W_down(hidden)           # (B, T, d_model)

    The gate controls how much information flows through the up-projection,
    acting as a learned filter. This variant typically achieves better
    perplexity than standard FFN at the same parameter count.

    Note: to match parameter count of FeedForward(d_model, 4*d_model), set
    d_ff = int(8/3 * d_model) because we use two up-projections.
    LLaMA rounds this to the nearest multiple of 256.

    Args:
        d_model:  Input and output dimension.
        d_ff:     Hidden dimension for gate and up projections.
                  Default: int(8/3 * d_model) rounded up to multiple of 256.
        dropout:  Dropout after down projection.
        bias:     Add bias terms (usually False for SwiGLU).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if d_ff is None:
            # LLaMA-style: 8/3 * d_model, rounded to nearest 256
            raw = int(8 / 3 * d_model)
            d_ff = ((raw + 255) // 256) * 256

        self.d_model   = d_model
        self.d_ff      = d_ff

        self.w_gate    = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up      = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down    = nn.Linear(d_ff, d_model, bias=bias)
        self.activation = nn.SiLU()
        self.dropout   = nn.Dropout(p=dropout)

        for layer in [self.w_gate, self.w_up, self.w_down]:
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model).

        Returns:
            (B, T, d_model).
        """
        gate   = self.activation(self.w_gate(x))   # (B, T, d_ff) — gating signal
        up     = self.w_up(x)                       # (B, T, d_ff) — content signal
        hidden = gate * up                          # element-wise: (B, T, d_ff)
        hidden = self.dropout(hidden)
        return self.w_down(hidden)                  # (B, T, d_model)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff} [SwiGLU]"
