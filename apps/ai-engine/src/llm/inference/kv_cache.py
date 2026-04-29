"""
InsightSerenity AI Engine — KV Cache
======================================
Key-Value cache for fast autoregressive text generation.

Without KV cache: generating T tokens requires T forward passes, each
processing the entire sequence 1..t. This is O(T²) total compute.

With KV cache: we cache the Key and Value tensors from attention at each
previous position. When generating token t+1, we only need to compute
Q for the new token and look up K,V from the cache. This reduces generation
to O(T) compute — each new token requires only one transformer forward pass
over a single position (plus the O(1) cache lookup).

How it works per layer:
    - First forward pass (prefill): process the full prompt, cache all K,V
    - Subsequent passes (decode): compute K,V only for the new token,
      append to cache, compute attention over all cached K,V

Shape evolution:
    After prefill (seq_len=T):   K shape = (B, H, T, Dh)
    After step 1 of decode:      K shape = (B, H, T+1, Dh)
    After step k of decode:      K shape = (B, H, T+k, Dh)

Memory: KV cache uses B × num_layers × 2 × H × max_len × Dh × 4 bytes.
For GPT-small (12 layers, 12 heads, 64 head_dim, max_len=2048, batch=1):
    ≈ 1 × 12 × 2 × 12 × 2048 × 64 × 4 = ~75 MB

This is why long-context models require significant GPU memory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class LayerKVCache:
    """
    Cached Key and Value tensors for one transformer layer.

    The cache grows incrementally: each decode step appends one new
    (key, value) pair along the sequence dimension.
    """
    keys:   Optional[Tensor] = None   # (B, H, seq_len, Dh)
    values: Optional[Tensor] = None   # (B, H, seq_len, Dh)

    def update(
        self,
        new_key:   Tensor,   # (B, H, 1, Dh) — new token's key
        new_value: Tensor,   # (B, H, 1, Dh) — new token's value
    ) -> Tuple[Tensor, Tensor]:
        """
        Append new key/value to the cache and return the full K, V.

        Args:
            new_key:   Key tensor for the new token(s).
            new_value: Value tensor for the new token(s).

        Returns:
            Tuple (full_keys, full_values) including all cached positions.
        """
        if self.keys is None:
            # First call: initialise the cache
            self.keys   = new_key
            self.values = new_value
        else:
            # Append along the sequence dimension (dim=2)
            self.keys   = torch.cat([self.keys,   new_key],   dim=2)
            self.values = torch.cat([self.values, new_value], dim=2)

        return self.keys, self.values

    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self.keys.size(2) if self.keys is not None else 0

    def clear(self) -> None:
        """Clear the cache (e.g. between unrelated generation requests)."""
        self.keys   = None
        self.values = None


class KVCache:
    """
    Full KV cache for all layers of a transformer model.

    Manages one LayerKVCache per transformer layer. The cache is attached
    to a specific generation request — each new request starts with a fresh
    empty cache.

    Args:
        num_layers: Number of transformer layers (must match the model).
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers   = num_layers
        self._layer_caches: List[LayerKVCache] = [
            LayerKVCache() for _ in range(num_layers)
        ]

    def update(
        self,
        layer_idx: int,
        new_key:   Tensor,
        new_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Update the cache for `layer_idx` and return full K, V.

        Args:
            layer_idx: Which transformer layer this is for.
            new_key:   New key tensor to append.
            new_value: New value tensor to append.

        Returns:
            (full_keys, full_values) for attention computation.
        """
        return self._layer_caches[layer_idx].update(new_key, new_value)

    def get(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return the current cached (keys, values) for a layer without updating."""
        cache = self._layer_caches[layer_idx]
        return cache.keys, cache.values

    def seq_len(self) -> int:
        """Current cached sequence length (same for all layers)."""
        return self._layer_caches[0].seq_len() if self._layer_caches else 0

    def clear(self) -> None:
        """Clear all layer caches."""
        for cache in self._layer_caches:
            cache.clear()

    def is_empty(self) -> bool:
        """True if no tokens have been cached yet."""
        return self._layer_caches[0].keys is None

    def __repr__(self) -> str:
        return (
            f"KVCache(num_layers={self.num_layers}, "
            f"seq_len={self.seq_len()})"
        )
