"""
InsightSerenity AI Engine — Convolutional Layers
=================================================
1D and 2D convolutional building blocks with explicit shape documentation.

Convolutions scan a learnable filter across the input, detecting local
patterns regardless of where they appear. In NLP, 1D convolutions over
token sequences detect n-gram features. In vision and structured data,
2D convolutions detect spatial features.

Shapes:
    Conv1D input:  (B, C_in, L)    — batch, channels, sequence length
    Conv1D output: (B, C_out, L')  — L' depends on kernel, stride, padding
    Conv2D input:  (B, C_in, H, W)
    Conv2D output: (B, C_out, H', W')

Both layers support:
    - Configurable kernel size, stride, padding, dilation, groups
    - Optional batch normalisation applied before the activation
    - Optional activation function (passed as a module)
    - Causal padding for Conv1D (no future leakage in sequence models)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: each output position at time t sees only inputs
    at times ≤ t. This is the correct convolution for autoregressive models.

    Implemented by left-padding the input by (kernel_size - 1) × dilation
    so the output has the same length as the input.

    Args:
        in_channels:  Number of input channels (C_in).
        out_channels: Number of output channels (C_out).
        kernel_size:  Width of the convolution window.
        dilation:     Dilation factor for dilated (atrous) convolutions.
        bias:         If True, add a learnable bias. Default True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Total left-padding to maintain causal property
        self._pad_len = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,         # We handle padding manually
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C_in, L) — input tensor.

        Returns:
            (B, C_out, L) — same length as input (causal).
        """
        # Left-pad only: ensures no future information leaks
        x = nn.functional.pad(x, (self._pad_len, 0))
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    A single 1D convolution block: Conv1D → [Norm] → Activation → [Dropout].

    Used as the reusable unit in ConvNet1d. All hyperparameters are passed
    at construction time; no implicit global state.

    Args:
        in_channels:  Input channels.
        out_channels: Output channels.
        kernel_size:  Convolution window size.
        stride:       Stride. Default 1.
        padding:      Padding. Default "same" (auto-computes same-length output).
        dilation:     Dilation factor. Default 1.
        groups:       Number of groups for grouped/depthwise conv. Default 1.
        bias:         Add bias term. Default True.
        norm:         Normalisation to apply after conv.
                        "batch" — BatchNorm1d
                        "layer" — LayerNorm
                        None    — no normalisation
        activation:   nn.Module to use as activation (e.g. nn.GELU()).
                      If None, no activation is applied.
        dropout:      Dropout rate after activation. Default 0.0.
        causal:       If True, use left-only padding (autoregressive safe).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Optional[str] = "batch",
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        layers: list = []

        if causal:
            # CausalConv1d handles its own padding
            layers.append(
                CausalConv1d(in_channels, out_channels, kernel_size, dilation, bias)
            )
        else:
            layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=bias,
                )
            )
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity="relu")

        if norm == "batch":
            layers.append(nn.BatchNorm1d(out_channels))
        elif norm == "layer":
            # LayerNorm on channel dimension requires a transpose
            # We handle this via a wrapper below
            layers.append(_ChannelLastLayerNorm(out_channels))

        if activation is not None:
            layers.append(activation)

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C_in, L) input tensor.

        Returns:
            (B, C_out, L') output tensor.
        """
        return self.block(x)


class Conv2dBlock(nn.Module):
    """
    A single 2D convolution block: Conv2D → [Norm] → Activation → [Dropout].

    Args:
        in_channels:  Input channels.
        out_channels: Output channels.
        kernel_size:  Convolution kernel size (int or (H, W) tuple).
        stride:       Stride (int or tuple).
        padding:      Padding (int, tuple, or "same").
        norm:         "batch", "instance", or None.
        activation:   Activation module, or None.
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = "same",
        norm: Optional[str] = "batch",
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=(norm is None),
            )
        ]
        nn.init.kaiming_normal_(layers[0].weight, nonlinearity="relu")

        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        if activation is not None:
            layers.append(activation)

        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C_in, H, W) input tensor.

        Returns:
            (B, C_out, H', W') output tensor.
        """
        return self.block(x)


# ── Internal helper ────────────────────────────────────────────────────────────

class _ChannelLastLayerNorm(nn.Module):
    """
    LayerNorm applied over the channel dimension of a (B, C, L) tensor.

    nn.LayerNorm expects the normalised dimension to be last.
    This module transposes (B, C, L) → (B, L, C), applies LayerNorm, then
    transposes back — making LayerNorm compatible with the conv channel layout.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L)
        x = x.transpose(1, 2)   # (B, L, C)
        x = self.norm(x)
        x = x.transpose(1, 2)   # (B, C, L)
        return x
