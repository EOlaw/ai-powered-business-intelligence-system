"""
InsightSerenity AI Engine — Convolutional Network Architectures
================================================================
Builds full CNN models from Conv1dBlock and Conv2dBlock primitives.

CNNs excel at detecting local patterns at multiple scales simultaneously
through a hierarchy of convolutional layers. Each layer has a larger
effective receptive field than the one before it.

Two architecture families:

1. ConvNet1d — for sequential/temporal data
   Input: (B, C_in, L) where L = sequence length
   Use cases: text classification (character-level), time series,
              waveform processing (speech / audio).

2. ConvNet2d — for 2D spatial data
   Input: (B, C_in, H, W) — images or 2D feature maps
   Use cases: image classification, structured data on grids,
              spectrogram analysis.

Both expose a consistent interface:
    - feature_extractor(x)  → feature tensor before the head
    - forward(x)            → logits (if out_classes provided) or features

Design: The channel sizes at each stage follow the standard doubling pattern
(e.g. 32 → 64 → 128 → 256). This halves the spatial dimension while doubling
the feature depth, keeping computation roughly constant per layer.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.nn.layers.conv import Conv1dBlock, Conv2dBlock
from src.nn.activations.activations import get_activation


@dataclass
class ConvNet1dConfig:
    """Configuration for a 1D convolutional network."""
    in_channels:   int          = 1
    channel_sizes: List[int]    = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes:  List[int]    = field(default_factory=lambda: [3, 3, 3])
    strides:       List[int]    = field(default_factory=lambda: [1, 2, 2])
    dropout:       float        = 0.1
    norm:          str          = "batch"
    activation:    str          = "relu"
    causal:        bool         = False
    out_classes:   Optional[int] = None
    pool:          str          = "avg"   # "avg", "max", or "none"


@dataclass
class ConvNet2dConfig:
    """Configuration for a 2D convolutional network."""
    in_channels:   int          = 3
    channel_sizes: List[int]    = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes:  List[int]    = field(default_factory=lambda: [3, 3, 3, 3])
    strides:       List[int]    = field(default_factory=lambda: [1, 2, 2, 2])
    dropout:       float        = 0.1
    norm:          str          = "batch"
    activation:    str          = "relu"
    out_classes:   Optional[int] = None


class ConvNet1d(nn.Module):
    """
    1D Convolutional Network for sequential data.

    Stacks Conv1dBlocks with progressive channel expansion and optional
    strided downsampling. A global pooling operation at the end produces a
    fixed-length feature vector regardless of the input sequence length.

    Args:
        config: ConvNet1dConfig dataclass with all hyperparameters.

    Input:  (B, in_channels, L)
    Output: (B, out_classes)  if out_classes is set
            (B, channel_sizes[-1]) otherwise (feature vector)
    """

    def __init__(self, config: ConvNet1dConfig) -> None:
        super().__init__()
        self.config = config

        # Validate config lists have the same length
        n_layers = len(config.channel_sizes)
        if not (len(config.kernel_sizes) == n_layers == len(config.strides)):
            raise ValueError(
                "channel_sizes, kernel_sizes, and strides must have the same length"
            )

        # Build convolutional blocks
        blocks: List[nn.Module] = []
        in_c = config.in_channels

        for out_c, ks, stride in zip(
            config.channel_sizes, config.kernel_sizes, config.strides
        ):
            blocks.append(
                Conv1dBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=ks,
                    stride=stride,
                    padding="same" if stride == 1 else ks // 2,
                    norm=config.norm,
                    activation=get_activation(config.activation),
                    dropout=config.dropout,
                    causal=config.causal,
                )
            )
            in_c = out_c

        self.blocks = nn.Sequential(*blocks)
        self.feature_dim = config.channel_sizes[-1]

        # Global pooling: reduces (B, C, L') → (B, C)
        if config.pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        elif config.pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        else:
            self.pool = None

        # Classification head
        if config.out_classes is not None:
            self.head = nn.Linear(self.feature_dim, config.out_classes)
        else:
            self.head = None

    def feature_extractor(self, x: Tensor) -> Tensor:
        """
        Extract pooled feature vector from input sequence.

        Args:
            x: (B, in_channels, L)

        Returns:
            (B, feature_dim) — pooled feature vector.
        """
        x = self.blocks(x)          # (B, C_last, L')
        if self.pool is not None:
            x = self.pool(x)        # (B, C_last, 1)
            x = x.squeeze(-1)       # (B, C_last)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, L)

        Returns:
            (B, out_classes) if head exists, else (B, feature_dim).
        """
        features = self.feature_extractor(x)
        if self.head is not None:
            return self.head(features)
        return features


class ConvNet2d(nn.Module):
    """
    2D Convolutional Network for spatial data (images, spectrograms).

    Stacks Conv2dBlocks with progressive channel doubling and strided
    downsampling. A global average pooling at the end followed by a
    fully-connected classification head.

    Args:
        config: ConvNet2dConfig dataclass.

    Input:  (B, in_channels, H, W)
    Output: (B, out_classes) if out_classes is set
            (B, channel_sizes[-1]) otherwise
    """

    def __init__(self, config: ConvNet2dConfig) -> None:
        super().__init__()
        self.config = config

        n_layers = len(config.channel_sizes)
        if not (len(config.kernel_sizes) == n_layers == len(config.strides)):
            raise ValueError(
                "channel_sizes, kernel_sizes, and strides must have the same length"
            )

        blocks: List[nn.Module] = []
        in_c = config.in_channels

        for out_c, ks, stride in zip(
            config.channel_sizes, config.kernel_sizes, config.strides
        ):
            blocks.append(
                Conv2dBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=ks,
                    stride=stride,
                    padding="same" if stride == 1 else ks // 2,
                    norm=config.norm,
                    activation=get_activation(config.activation),
                    dropout=config.dropout,
                )
            )
            in_c = out_c

        self.blocks      = nn.Sequential(*blocks)
        self.feature_dim = config.channel_sizes[-1]

        # Global average pooling: (B, C, H', W') → (B, C)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if config.out_classes is not None:
            self.head: Optional[nn.Linear] = nn.Linear(self.feature_dim, config.out_classes)
        else:
            self.head = None

    def feature_extractor(self, x: Tensor) -> Tensor:
        """
        Extract global feature vector from a spatial input.

        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, feature_dim)
        """
        x = self.blocks(x)               # (B, C_last, H', W')
        x = self.pool(x)                 # (B, C_last, 1, 1)
        x = x.flatten(start_dim=1)       # (B, C_last)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, out_classes) or (B, feature_dim).
        """
        features = self.feature_extractor(x)
        if self.head is not None:
            return self.head(features)
        return features
