"""
InsightSerenity AI Engine — Attention Mechanisms
=================================================
The attention mechanism is the core innovation that makes transformers work.
It allows every position in a sequence to look at every other position
(subject to masking) and weight them by relevance.

This module implements:

1. scaled_dot_product_attention (function)
   The core mathematical operation: softmax(QK^T / sqrt(d_k)) V
   This is also available as torch.nn.functional.scaled_dot_product_attention
   in PyTorch 2.0+ with Flash Attention support. We implement it explicitly
   for educational clarity and fallback compatibility.

2. MultiHeadAttention
   The full multi-head mechanism from "Attention Is All You Need":
   - Projects Q, K, V into h independent heads of dimension d_head
   - Applies scaled dot-product attention in each head in parallel
   - Concatenates heads and applies output projection
   - Supports: self-attention, cross-attention, causal masking, key/value cache

3. MultiHeadCausalAttention
   Subclass of MultiHeadAttention pre-configured for autoregressive decoding:
   - Always applies a causal (lower-triangular) mask
   - Maintains a KV cache for fast incremental decoding during inference

Shape conventions:
    B   = batch size
    T   = query sequence length
    S   = key/value sequence length (= T for self-attention)
    D   = model dimension (d_model)
    H   = number of attention heads
    Dh  = head dimension = D // H
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Core attention function
# ─────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute scaled dot-product attention.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    The scaling by 1/sqrt(d_k) prevents the dot products from growing
    too large in magnitude, which would push softmax into regions with
    very small gradients (vanishing gradient problem).

    Args:
        q:         Query tensor    (B, H, T, Dh)
        k:         Key tensor      (B, H, S, Dh)
        v:         Value tensor    (B, H, S, Dh)
        mask:      Optional boolean or float mask:
                     bool mask: True positions are KEPT (attended to)
                     float mask: added directly to logits before softmax
                   Shape: broadcastable to (B, H, T, S)
        dropout_p: Attention weight dropout probability.
        training:  Whether the model is in training mode (for dropout).

    Returns:
        Tuple of:
          - output:  (B, H, T, Dh) — weighted sum of values
          - weights: (B, H, T, S)  — attention probabilities (for visualisation)
    """
    d_k = q.size(-1)
    scale = 1.0 / math.sqrt(d_k)

    # Compute raw attention scores: (B, H, T, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply mask before softmax
    if mask is not None:
        if mask.dtype == torch.bool:
            # Boolean mask: mask out False positions with -inf
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            # Float mask: add directly (e.g. ALiBi bias, padding mask as -inf)
            scores = scores + mask

    # Convert to probabilities; -inf → 0 after softmax
    weights = F.softmax(scores, dim=-1)

    # Replace NaN (from all-masked rows where softmax produced 0/0)
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    # Weighted sum of values
    output = torch.matmul(weights, v)   # (B, H, T, Dh)

    return output, weights


# ─────────────────────────────────────────────────────────────────────────────
# Mask builders
# ─────────────────────────────────────────────────────────────────────────────

def build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """
    Build a causal (lower-triangular) boolean mask for autoregressive attention.

    Position i can attend to positions 0..i only (not i+1..T-1).
    mask[i, j] = True  if j <= i (position i CAN attend to position j)
    mask[i, j] = False if j >  i (position i CANNOT attend to position j)

    Args:
        seq_len: Sequence length T.
        device:  Target device.

    Returns:
        BoolTensor of shape (T, T).
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def build_padding_mask(
    attention_mask: Tensor,
    num_heads: int,
) -> Tensor:
    """
    Convert a 2D attention mask (B, T) to a 4D float mask (B, 1, 1, T)
    compatible with the attention score shape (B, H, T, S).

    Positions marked 0 in attention_mask (padding) get -inf so softmax
    zeroes them out. Real positions (1) get 0 (no change to logit).

    Args:
        attention_mask: BoolTensor (B, T) — 1 for real, 0 for padding.
        num_heads:      Number of attention heads (unused, mask broadcasts).

    Returns:
        FloatTensor of shape (B, 1, 1, T) with 0.0 or -inf.
    """
    mask = attention_mask.to(dtype=torch.float)
    mask = (1.0 - mask) * torch.finfo(torch.float32).min
    return mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, T)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al., 2017).

    Projects queries, keys, and values into h parallel attention heads,
    computes scaled dot-product attention in each head, then concatenates
    and projects back to d_model.

    When query=key=value (the same tensor), this is self-attention.
    When query comes from one source and key/value from another, this is
    cross-attention (used in encoder-decoder transformers).

    Args:
        d_model:       Model dimension (must be divisible by num_heads).
        num_heads:     Number of parallel attention heads.
        dropout:       Attention weight dropout. Default 0.1.
        bias:          If True, add bias terms to Q/K/V/output projections.
        rope:          Optional RotaryPositionalEmbedding module. If provided,
                       RoPE is applied to Q and K before computing attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.dropout   = dropout
        self.rope      = rope

        # Q, K, V projections — no separate bias needed if bias=True
        # We use a single combined QKV projection for efficiency
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # GPT-2 style init for Q/K/V
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Scaled-down init for output projection (residual path)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute multi-head attention.

        Args:
            query:            (B, T, D) — queries.
            key:              (B, S, D) — keys.
            value:            (B, S, D) — values.
            mask:             Optional mask of shape (T, S) or (B, 1, T, S).
                              Boolean: True = attend, False = mask out.
                              Float: added to logits before softmax.
            key_padding_mask: Optional (B, S) boolean tensor. True = real token,
                              False = padding. Applied on the key/value dimension.

        Returns:
            Tuple of:
              - output:  (B, T, D) — attended output.
              - weights: (B, H, T, S) — attention weights (for analysis/logging).
        """
        B, T, _ = query.shape
        _, S, _ = key.shape

        # Project to Q, K, V
        q = self.q_proj(query)   # (B, T, D)
        k = self.k_proj(key)     # (B, S, D)
        v = self.v_proj(value)   # (B, S, D)

        # Reshape to (B, H, T, Dh) — split d_model into H heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, Dh)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, Dh)

        # Apply RoPE if configured
        if self.rope is not None:
            q, k = self.rope(q, k)

        # Build combined mask
        combined_mask = self._build_mask(mask, key_padding_mask, B, T, S, query.device)

        # Scaled dot-product attention
        attn_out, weights = scaled_dot_product_attention(
            q, k, v,
            mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )

        # Merge heads: (B, H, T, Dh) → (B, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Output projection
        output = self.out_proj(attn_out)   # (B, T, D)

        return output, weights

    def _build_mask(
        self,
        mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        B: int,
        T: int,
        S: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Combine attention mask and key padding mask into one float mask."""
        combined: Optional[Tensor] = None

        if mask is not None:
            if mask.dtype == torch.bool:
                # Bool mask → float mask: False positions get -inf
                combined = torch.zeros(T, S, device=device, dtype=torch.float)
                combined = combined.masked_fill(~mask, float("-inf"))
            else:
                combined = mask

        if key_padding_mask is not None:
            # (B, S) → (B, 1, 1, S) float mask
            pad_mask = build_padding_mask(key_padding_mask, self.num_heads)
            combined = pad_mask if combined is None else combined + pad_mask

        return combined

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )


class MultiHeadCausalAttention(MultiHeadAttention):
    """
    Multi-head attention with built-in causal (autoregressive) masking.

    Inherits all of MultiHeadAttention but automatically generates and
    applies the causal mask so position i cannot attend to positions > i.

    Used in the decoder of autoregressive language models (GPT-style).

    Args:
        d_model:       Model dimension.
        num_heads:     Number of heads.
        max_seq_len:   Maximum sequence length (used to pre-build the mask).
        dropout:       Attention dropout.
        bias:          Add bias to projections.
        rope:          Optional RoPE module.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        bias: bool = True,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads, dropout=dropout,
                         bias=bias, rope=rope)
        self.max_seq_len = max_seq_len

        # Pre-build the causal mask for the maximum length
        # This avoids re-building it on every forward call
        causal = build_causal_mask(max_seq_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Self-attention with automatic causal masking.

        Args:
            query:            (B, T, D) — input sequence (serves as Q=K=V).
            key, value:       If None, self-attention: key=value=query.
            mask:             Additional mask (added on top of causal mask).
            key_padding_mask: (B, T) padding mask.

        Returns:
            Tuple (output (B, T, D), weights (B, H, T, T)).
        """
        if key is None:
            key = query
        if value is None:
            value = query

        T = query.size(1)

        # Slice the pre-built causal mask to the actual sequence length
        causal = self.causal_mask[:T, :T]   # type: ignore[index]

        # Merge with any additional mask
        if mask is not None:
            causal = causal & mask if mask.dtype == torch.bool else causal

        return super().forward(
            query=query,
            key=key,
            value=value,
            mask=causal,
            key_padding_mask=key_padding_mask,
        )
