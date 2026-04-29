"""
InsightSerenity AI Engine — Embedding Layers
=============================================
Converts discrete token IDs and sequence positions into dense continuous
vectors. These are the first layers in every language model — without them
the model cannot process discrete symbols.

Three embedding types are implemented:

1. TokenEmbedding
   Maps integer token IDs → dense vectors via a learned lookup table.
   Shape: (vocab_size, d_model). This IS a linear layer with a one-hot
   input, but the lookup is O(1) via index operation, not O(vocab_size).

2. SinusoidalPositionalEmbedding
   Fixed (not learned) positional encodings from "Attention Is All You Need".
   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   These are deterministic, require no training, and generalise to unseen
   lengths. Used by the original Transformer, BERT's option, and T5.

3. LearnedPositionalEmbedding
   Learned position embeddings — a simple lookup table indexed by position.
   GPT-2, RoBERTa, and many modern models use this. More expressive than
   sinusoidal but does not extrapolate beyond max_seq_len at training time.

4. RotaryPositionalEmbedding (RoPE)
   Rotates Q and K vectors in attention based on position. Encodes relative
   position information directly into the attention score, not the input.
   Used by LLaMA-1/2/3, Mistral, Falcon, and GPT-NeoX.
   RoPE has better length generalisation than learned absolute positions.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Learned token embedding table: token_id → dense vector.

    The embedding is scaled by sqrt(d_model) following the original
    Transformer paper, which keeps embedding magnitudes comparable to
    the positional encoding magnitudes.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model:    Embedding dimension.
        pad_id:     Token ID to treat as padding (gradients are zeroed for it).
                    Set to None to disable padding masking.
        scale:      If True, multiply embeddings by sqrt(d_model).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: Optional[int] = 0,
        scale: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.scale_val  = math.sqrt(d_model) if scale else 1.0

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        # GPT-2 style init: N(0, 0.02)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if pad_id is not None:
            nn.init.zeros_(self.embedding.weight[pad_id])

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Look up token embeddings and apply scale factor.

        Args:
            token_ids: LongTensor of shape (B, T) — integer token IDs.

        Returns:
            FloatTensor of shape (B, T, d_model) — dense embedding vectors.
        """
        return self.embedding(token_ids) * self.scale_val

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}"


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional encodings (Vaswani et al., 2017).

    The encodings are computed once at construction time and cached as a
    non-trainable buffer. They are added to the token embeddings, not
    concatenated, so the model dimension stays d_model throughout.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        d_model:     Embedding/model dimension. Must be even.
        max_seq_len: Maximum sequence length to pre-compute.
        dropout:     Dropout probability applied to the sum of token +
                     positional embeddings. Default 0.1.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for sinusoidal embeddings, got {d_model}"
            )

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Build the full (max_seq_len, d_model) encoding table once
        pe = self._build_table(max_seq_len, d_model)

        # Register as a buffer: part of model state but not a trainable parameter.
        # shape: (1, max_seq_len, d_model) for broadcasting over batch dimension.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to token embeddings.

        Args:
            x: FloatTensor of shape (B, T, d_model) — token embeddings.

        Returns:
            FloatTensor of shape (B, T, d_model) — embeddings + positions.
        """
        # self.pe shape: (1, max_seq_len, d_model)
        # Slice to the actual sequence length T
        T   = x.size(1)
        out = x + self.pe[:, :T, :]      # type: ignore[index]
        return self.dropout(out)

    @staticmethod
    def _build_table(max_len: int, d_model: int) -> Tensor:
        """Pre-compute the (max_len, d_model) sinusoidal table."""
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        dims      = torch.arange(0, d_model, 2, dtype=torch.float)          # (d_model/2,)

        # Compute dividing terms: 10000^(2i/d_model)
        div_term = torch.exp(dims * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned absolute positional embeddings (GPT-2 style).

    A simple lookup table indexed by position 0..max_seq_len-1.
    Each position gets a unique learned embedding vector added to the
    token embedding at that position.

    Limitation: the model cannot generalise to sequences longer than
    max_seq_len without fine-tuning. Use RoPE for better extrapolation.

    Args:
        d_model:     Embedding dimension.
        max_seq_len: Maximum sequence length supported.
        dropout:     Dropout applied after adding positional embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model    = d_model
        self.max_seq_len = max_seq_len
        self.dropout    = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add learned positional embeddings to token embeddings.

        Args:
            x: FloatTensor of shape (B, T, d_model).

        Returns:
            FloatTensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        pos_emb   = self.embedding(positions)                         # (1, T, d_model)
        return self.dropout(x + pos_emb)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_seq_len={self.max_seq_len}"


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al., 2021.
    Used by LLaMA, Mistral, Falcon, GPT-NeoX.

    Unlike additive position embeddings, RoPE encodes position by rotating
    the query and key vectors in 2D subspaces. This means position is
    encoded in the *relationship* between Q and K, not in the token vector
    itself, giving better generalisation to unseen lengths.

    Key properties:
        - Relative position is encoded: attention(q_i, k_j) depends on (i-j)
        - No position information added to values (V stays unchanged)
        - Allows theoretically infinite sequence extrapolation (in practice,
          requires further tricks like YaRN or ALiBi for very long sequences)

    Args:
        head_dim:    Dimension per attention head (d_model // num_heads).
        max_seq_len: Pre-compute rotation matrices up to this length.
        base:        The base period. Default 10000 (original RoPE).
                     LLaMA-3 and others use larger values (e.g. 500000)
                     for longer context windows.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim   = head_dim
        self.max_seq_len = max_seq_len
        self.base       = base

        # Pre-compute cos and sin tables for positions 0..max_seq_len-1
        cos, sin = self._build_tables(max_seq_len, head_dim, base)
        # shape: (max_seq_len, head_dim) — register as non-trainable buffers
        self.register_buffer("cos_table", cos)
        self.register_buffer("sin_table", sin)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (B, num_heads, T, head_dim).
            k: Key tensor of shape   (B, num_heads, T, head_dim).
            seq_len: Sequence length. If None, inferred from q.size(-2).

        Returns:
            Tuple (q_rotated, k_rotated) with the same shapes as inputs.
        """
        T   = seq_len or q.size(-2)
        cos = self.cos_table[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = self.sin_table[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)

        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _apply_rotation(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Rotate vector x using the rotation matrix defined by cos/sin.

        The rotation operates on pairs of dimensions:
            (x_1, x_2, x_3, x_4, ...) → pairs (x_1, x_2), (x_3, x_4), ...
        Each pair is rotated as a 2D vector:
            rotated = [x_1 * cos - x_2 * sin, x_1 * sin + x_2 * cos]

        This is implemented by splitting x into even/odd halves and
        constructing the rotated x via the negative rotation trick.
        """
        head_dim = x.shape[-1]
        half     = head_dim // 2

        # Split into two halves along the head_dim axis
        x1 = x[..., :half]    # even dimensions
        x2 = x[..., half:]    # odd dimensions

        # Apply rotation: [-x2, x1] produces the orthogonal complement
        x_rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos + x_rotated * sin

    @staticmethod
    def _build_tables(max_len: int, head_dim: int, base: float) -> Tuple[Tensor, Tensor]:
        """Pre-compute cos and sin tables for positions 0..max_len-1."""
        half      = head_dim // 2
        positions = torch.arange(max_len, dtype=torch.float)
        freqs     = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float) / half))

        # Outer product: (max_len, half)
        angles = torch.outer(positions, freqs)

        # Concatenate to full head_dim: (max_len, head_dim)
        angles = torch.cat([angles, angles], dim=-1)

        return angles.cos(), angles.sin()

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base}"
