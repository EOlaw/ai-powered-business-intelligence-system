"""
InsightSerenity AI Engine — Multi-Layer Perceptron
====================================================
The simplest complete neural network: a stack of fully-connected (dense)
layers with non-linear activations between them. Despite its simplicity,
the MLP is the foundation of every more complex architecture.

Universal Approximation Theorem: a sufficiently wide MLP with a non-linear
activation can approximate any continuous function on a compact domain.
This is WHY neural networks can learn.

Architecture:
    Input → [Linear → Activation → Dropout]ₙ → Linear → Output

Options:
    - Residual connections (ResNet-style) to allow very deep MLPs
    - Batch/Layer normalisation
    - Configurable activation per layer or uniform
    - Configurable hidden dimensions (constant width or funnel/pyramid)

Usage:
    from src.architectures.feedforward.mlp import MLP

    # Classification head
    mlp = MLP(
        in_dim=512, hidden_dims=[256, 128], out_dim=10,
        activation="gelu", dropout=0.1
    )
    logits = mlp(features)   # (B, 10)

    # Feature extractor (no final linear)
    encoder = MLP(in_dim=784, hidden_dims=[512, 256], out_dim=256)
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.nn.activations.activations import get_activation
from src.nn.normalization.normalization import LayerNorm
from src.nn.initializers.initializers import initialize_weights


class MLPBlock(nn.Module):
    """
    A single MLP block: Linear → [Norm] → Activation → Dropout.

    Used as the reusable building unit inside MLP. Separated here so it
    can be imported independently when you need just one block.

    Args:
        in_dim:     Input feature dimension.
        out_dim:    Output feature dimension.
        activation: Activation name (e.g. "gelu", "relu") or nn.Module.
        dropout:    Dropout probability. Default 0.0.
        norm:       If True, apply LayerNorm after the linear projection
                    (pre-norm convention: norm before activation).
        bias:       Add bias to linear layer. Default True.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, nn.Module] = "gelu",
        dropout: float = 0.0,
        norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm   = LayerNorm(out_dim) if norm else None
        self.act    = get_activation(activation) if isinstance(activation, str) \
                      else activation
        self.drop   = nn.Dropout(p=dropout) if dropout > 0.0 else None

        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (*, in_dim)

        Returns:
            (*, out_dim)
        """
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable depth, width, and options.

    The standard building block for:
        - Classification heads (e.g. on top of a transformer encoder)
        - Value networks in RL (critic)
        - Regression networks
        - Feature transformation pipelines

    Args:
        in_dim:        Input feature dimension.
        hidden_dims:   List of hidden layer widths. E.g. [256, 128].
                       Empty list → single linear layer (logistic regression).
        out_dim:       Output dimension.
        activation:    Activation name or module. Default "gelu".
        dropout:       Dropout between all hidden layers.
        norm:          Apply LayerNorm in each hidden block.
        residual:      Add skip connections when in_dim == out_dim of a block.
                       Enables very deep MLPs without vanishing gradients.
        output_bias:   Add bias to the final output layer. Default True.
        output_activation: If not None, apply this activation to the output.
                       E.g. nn.Sigmoid() for binary outputs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: Union[str, nn.Module] = "gelu",
        dropout: float = 0.0,
        norm: bool = False,
        residual: bool = False,
        output_bias: bool = True,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.residual    = residual

        # Build hidden blocks
        blocks: List[nn.Module] = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            blocks.append(
                MLPBlock(
                    in_dim=prev_dim,
                    out_dim=h_dim,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                )
            )
            prev_dim = h_dim

        self.hidden_blocks = nn.ModuleList(blocks)

        # Final linear projection
        self.output_layer = nn.Linear(prev_dim, out_dim, bias=output_bias)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        if output_bias:
            nn.init.zeros_(self.output_layer.bias)

        self.output_activation = output_activation

        # Store dimensions for residual connection checks
        self._block_dims = list(zip(
            [in_dim] + hidden_dims,
            hidden_dims,
        ))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (*, in_dim) — input features.

        Returns:
            (*, out_dim) — output predictions or features.
        """
        for i, block in enumerate(self.hidden_blocks):
            if self.residual and self._block_dims[i][0] == self._block_dims[i][1]:
                x = x + block(x)   # Residual skip when dimensions match
            else:
                x = block(x)

        x = self.output_layer(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

    def extra_repr(self) -> str:
        hidden = [d[1] for d in self._block_dims]
        return (
            f"in_dim={self.in_dim}, "
            f"hidden={hidden}, "
            f"out_dim={self.out_dim}"
        )
