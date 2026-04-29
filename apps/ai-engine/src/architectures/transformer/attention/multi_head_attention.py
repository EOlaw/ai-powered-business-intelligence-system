"""
InsightSerenity AI Engine — Transformer Attention Module
=========================================================
Re-exports and augments the attention mechanisms from nn/layers/attention.py
with a complete, standalone transformer-level module that includes all the
variants used across different transformer architectures.

This module exists separately from nn/layers/attention.py because:
    - nn/layers: provides the pure algorithmic primitives
    - architectures/transformer/attention: assembles them into complete
      architectural variants used by specific model families

Variants:
    SelfAttention       — standard self-attention (no masking)
    CausalSelfAttention — autoregressive self-attention (GPT-style)
    CrossAttention      — encoder → decoder attention (T5-style)

All three share the same weight structure; they differ only in:
    - What is passed as query, key, value
    - What mask is applied
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Re-use the battle-tested primitives from the nn layer package
from src.nn.layers.attention import (
    MultiHeadAttention,
    MultiHeadCausalAttention,
    build_causal_mask,
    build_padding_mask,
)
from src.nn.layers.embedding import RotaryPositionalEmbedding
from src.nn.normalization.normalization import RMSNorm


class SelfAttention(nn.Module):
    """
    Bidirectional self-attention (BERT/encoder-style).

    Every position attends to every other position. No causal masking.
    Used in: BERT, RoBERTa, T5 encoder.

    Wraps MultiHeadAttention so the caller only needs to pass one tensor.

    Args:
        d_model:          Model dimension.
        num_heads:        Number of attention heads.
        dropout:          Attention dropout. Default 0.1.
        use_rope:         If True, apply Rotary Position Embeddings.
        max_seq_len:      Maximum sequence length (needed for RoPE buffer).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()

        rope = None
        if use_rope:
            head_dim = d_model // num_heads
            rope = RotaryPositionalEmbedding(head_dim=head_dim, max_seq_len=max_seq_len)

        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope=rope,
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:                (B, T, D) — input sequence.
            key_padding_mask: (B, T) boolean — True for real tokens, False for padding.

        Returns:
            Tuple (output (B, T, D), attention_weights (B, H, T, T)).
        """
        return self.attn(x, x, x, key_padding_mask=key_padding_mask)


class CausalSelfAttention(nn.Module):
    """
    Causal (autoregressive) self-attention (GPT-style).

    Position i can only attend to positions 0..i.
    Used in: GPT-2/3/4, LLaMA, Mistral, decoder blocks in T5.

    Args:
        d_model:     Model dimension.
        num_heads:   Number of attention heads.
        max_seq_len: Maximum supported sequence length.
        dropout:     Attention dropout.
        use_rope:    Apply Rotary Position Embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = True,
    ) -> None:
        super().__init__()

        rope = None
        if use_rope:
            head_dim = d_model // num_heads
            rope = RotaryPositionalEmbedding(head_dim=head_dim, max_seq_len=max_seq_len)

        self.attn = MultiHeadCausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope=rope,
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:                (B, T, D) — input token representations.
            key_padding_mask: (B, T) — padding mask.

        Returns:
            Tuple (output (B, T, D), weights (B, H, T, T)).
        """
        return self.attn(
            query=x,
            key_padding_mask=key_padding_mask,
        )


class CrossAttention(nn.Module):
    """
    Cross-attention: query from decoder, key/value from encoder.

    Used in:
        - T5 / BART encoder-decoder cross-attention
        - Diffusion model conditioning
        - Any architecture where one sequence attends to another

    The decoder queries the encoder context to incorporate source
    information into the target generation.

    Args:
        d_model:    Model dimension.
        num_heads:  Number of attention heads.
        dropout:    Attention dropout.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope=None,   # RoPE is not used in cross-attention
        )

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        context_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query:                (B, T_q, D) — decoder hidden states.
            context:              (B, T_k, D) — encoder hidden states.
            context_padding_mask: (B, T_k) — True for real encoder tokens.

        Returns:
            Tuple (output (B, T_q, D), weights (B, H, T_q, T_k)).
        """
        return self.attn(
            query=query,
            key=context,
            value=context,
            key_padding_mask=context_padding_mask,
        )
