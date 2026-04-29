"""
InsightSerenity AI Engine — Linear Layer
=========================================
The fundamental building block of every neural network: a fully-connected
(dense) linear transformation  y = x @ W.T + b.

Why implement our own instead of using nn.Linear directly?
    - We attach custom weight initialisation strategies at construction time
    - We expose explicit shape documentation on every forward call
    - We make the bias optional and controllable in a single place
    - It gives us a consistent interface to build every higher-level layer on

This is NOT a re-implementation of autograd — we use PyTorch's Parameter and
autograd machinery. What we own is the *interface*, *initialisation*, and
the ability to extend without touching framework internals.

Shape conventions used throughout the nn package:
    B  = batch size
    T  = sequence length (tokens)
    D  = model / embedding dimension
    H  = number of attention heads
    C  = channels (conv)
    *  = any number of leading dimensions
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    """
    Fully-connected linear transformation: y = x @ W.T + b

    Equivalent to nn.Linear but with explicit initialisation control and
    inline shape documentation. Use this as the default dense layer across
    the entire codebase so initialisation strategy is centralised.

    Args:
        in_features:  Size of each input sample (D_in).
        out_features: Size of each output sample (D_out).
        bias:         If True, add a learnable bias term. Default True.
        init:         Weight initialisation strategy.
                        "kaiming_uniform" — default PyTorch init (good for ReLU)
                        "kaiming_normal"  — Kaiming with normal distribution
                        "xavier_uniform"  — good for tanh / sigmoid
                        "xavier_normal"   — Xavier with normal distribution
                        "zeros"           — initialise to zero (output projections)
                        "normal"          — small normal N(0, 0.02), GPT-style
        device:       Target device for parameter creation.
        dtype:        Parameter dtype (default float32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "kaiming_uniform",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # Weight matrix: shape (out_features, in_features)
        # Stored transposed so the forward pass is x @ W.T (standard convention)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self._init_weights(init)

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_weights(self, strategy: str) -> None:
        """Apply the chosen weight initialisation to self.weight."""
        with torch.no_grad():
            if strategy == "kaiming_uniform":
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in = self.in_features
                    bound  = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(self.bias, -bound, bound)

            elif strategy == "kaiming_normal":
                nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

            elif strategy == "xavier_uniform":
                nn.init.xavier_uniform_(self.weight)

            elif strategy == "xavier_normal":
                nn.init.xavier_normal_(self.weight)

            elif strategy == "zeros":
                nn.init.zeros_(self.weight)

            elif strategy == "normal":
                # GPT-2 style: small normal with std=0.02
                nn.init.normal_(self.weight, mean=0.0, std=0.02)
                # Scale down output projections by 1/sqrt(num_layers)
                # (caller's responsibility to set this via apply_output_scaling)

            elif strategy == "identity":
                # Useful for residual connection bypass layers
                nn.init.eye_(self.weight) if self.weight.shape[0] == self.weight.shape[1] \
                    else nn.init.normal_(self.weight, std=0.02)

            else:
                raise ValueError(
                    f"Unknown init strategy '{strategy}'. "
                    f"Expected: kaiming_uniform, kaiming_normal, xavier_uniform, "
                    f"xavier_normal, zeros, normal, identity."
                )

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply linear transformation to input.

        Args:
            x: Input tensor of shape (*, in_features) where * is any
               number of leading dimensions (batch, sequence, etc.)

        Returns:
            Output tensor of shape (*, out_features).
        """
        return nn.functional.linear(x, self.weight, self.bias)

    # ── Extra ──────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
