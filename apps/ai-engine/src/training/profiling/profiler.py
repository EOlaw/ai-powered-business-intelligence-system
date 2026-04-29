"""
InsightSerenity AI Engine — Training Profiler
=============================================
Utilities for measuring model efficiency: parameter count, FLOPs per
forward pass, and memory usage. Essential for:
    - Deciding if a model fits in GPU memory before starting training
    - Comparing architectural choices (more layers vs wider layers)
    - Estimating compute costs (FLOPs → tokens/second → training time)
    - Detecting memory leaks during training

FLOPs estimation uses a simplified model:
    - Linear(in, out): 2 × in × out FLOPs (multiply-add)
    - Attention head: 4 × T² × D FLOPs (approximate)
    - Embedding lookup: 0 FLOPs (table lookup, no multiply)
    This is a lower bound — actual FLOPs depend on the specific operations
    in the forward pass.

Memory units:
    - Parameters stored in float32: 4 bytes per parameter
    - Gradients: same as parameters (4 bytes)
    - Optimizer state (AdamW): 8 bytes per parameter (m and v)
    Total training memory ≈ 16 bytes × num_parameters
"""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    Count the total number of parameters in a model.

    Args:
        model:           The nn.Module to inspect.
        trainable_only:  If True, count only parameters with requires_grad=True.

    Returns:
        Total parameter count as an integer.
    """
    params = (
        (p for p in model.parameters() if p.requires_grad)
        if trainable_only
        else model.parameters()
    )
    return sum(p.numel() for p in params)


def parameter_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Return a comprehensive parameter count breakdown by layer type.

    Returns:
        Dict with keys:
            total:       Total parameter count
            trainable:   Trainable parameter count
            frozen:      Non-trainable parameter count
            by_layer:    Dict of {layer_name: param_count} for named layers
            memory_mb:   Estimated memory usage (float32, params only)
    """
    total     = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    by_layer: Dict[str, int] = {}
    for name, module in model.named_modules():
        if name and list(module.parameters(recurse=False)):
            n = sum(p.numel() for p in module.parameters(recurse=False))
            if n > 0:
                by_layer[name] = n

    # float32 = 4 bytes, convert to MB
    memory_mb = (total * 4) / (1024 ** 2)

    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
        "by_layer":  by_layer,
        "memory_mb": round(memory_mb, 2),
    }


def format_param_count(n: int) -> str:
    """Format a parameter count as a human-readable string (e.g. '125M', '1.3B')."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_flops_transformer(
    d_model:    int,
    num_heads:  int,
    num_layers: int,
    seq_len:    int,
    vocab_size: int,
    ffn_ratio:  float = 4.0,
) -> int:
    """
    Estimate FLOPs for a single forward pass of a transformer decoder.

    Based on the Chinchilla paper's formula (Hoffmann et al., 2022):
        FLOPs per token ≈ 6 × N  (for training: fwd + bwd = 3×fwd)
    where N = number of non-embedding parameters.

    This function computes a more detailed estimate by layer type.

    Args:
        d_model:    Model hidden dimension.
        num_heads:  Number of attention heads.
        num_layers: Number of transformer layers.
        seq_len:    Sequence length (tokens).
        vocab_size: Vocabulary size (for lm_head).
        ffn_ratio:  FFN hidden dim ratio. Default 4.0.

    Returns:
        Estimated FLOPs for one forward pass (not including backward).
    """
    head_dim = d_model // num_heads
    d_ff     = int(d_model * ffn_ratio)

    # Per-layer FLOPs
    # Attention: Q, K, V projections + attention scores + output projection
    attn_qkv_flops  = 3 * seq_len * 2 * d_model * d_model          # Q,K,V projections
    attn_score_flops = seq_len * seq_len * d_model * 2               # QK^T
    attn_val_flops   = seq_len * seq_len * d_model * 2               # AV
    attn_out_flops   = seq_len * 2 * d_model * d_model               # output projection
    attn_total       = attn_qkv_flops + attn_score_flops + attn_val_flops + attn_out_flops

    # FFN: two linear layers (SwiGLU has an extra projection, approximated here)
    ffn_flops = seq_len * 2 * (2 * d_model * d_ff)   # up + down projections

    # Per-layer total
    per_layer = attn_total + ffn_flops

    # All layers
    total_flops = num_layers * per_layer

    # Embedding lookup: 0 FLOPs
    # LM head: vocab projection
    lm_head_flops = seq_len * 2 * d_model * vocab_size

    return total_flops + lm_head_flops


# ─────────────────────────────────────────────────────────────────────────────
# Memory profiling
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryStats:
    """GPU memory statistics at a point in time."""
    allocated_mb:   float = 0.0
    reserved_mb:    float = 0.0
    max_allocated_mb: float = 0.0
    device:         str   = "cpu"

    def __str__(self) -> str:
        return (
            f"GPU Memory [{self.device}]: "
            f"allocated={self.allocated_mb:.1f}MB "
            f"reserved={self.reserved_mb:.1f}MB "
            f"peak={self.max_allocated_mb:.1f}MB"
        )


def get_memory_stats(device: Optional[torch.device] = None) -> MemoryStats:
    """
    Get current GPU memory statistics.

    Args:
        device: GPU device to query. If None, uses the default CUDA device.

    Returns:
        MemoryStats with current allocation figures.
    """
    if not torch.cuda.is_available():
        return MemoryStats()

    device = device or torch.device("cuda")
    MB = 1024 ** 2

    return MemoryStats(
        allocated_mb=   torch.cuda.memory_allocated(device) / MB,
        reserved_mb=    torch.cuda.memory_reserved(device) / MB,
        max_allocated_mb=torch.cuda.max_memory_allocated(device) / MB,
        device=str(device),
    )


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset the peak memory tracking counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


# ─────────────────────────────────────────────────────────────────────────────
# Throughput measurement
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def measure_throughput(
    seq_len:    int,
    batch_size: int,
) -> Generator[Dict[str, float], None, None]:
    """
    Context manager that measures tokens/second throughput.

    Usage:
        with measure_throughput(seq_len=1024, batch_size=8) as result:
            output = model(input_ids)
        print(result["tokens_per_second"])

    Args:
        seq_len:    Sequence length of the batch being processed.
        batch_size: Batch size.

    Yields:
        A mutable dict that will contain "tokens_per_second" and
        "wall_time_ms" after the context exits.
    """
    result: Dict[str, float] = {}

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield result

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    tokens  = seq_len * batch_size

    result["wall_time_ms"]       = round(elapsed * 1000, 2)
    result["tokens_per_second"]  = round(tokens / elapsed, 1)
    result["ms_per_token"]       = round(elapsed * 1000 / max(tokens, 1), 4)


def estimate_training_time(
    tokens_per_second:   float,
    total_training_tokens: int,
    gradient_accumulation: int = 1,
) -> Dict[str, float]:
    """
    Estimate total training time from measured throughput.

    Args:
        tokens_per_second:     Measured training throughput.
        total_training_tokens: Total tokens in the training run.
        gradient_accumulation: Gradient accumulation steps (reduces throughput).

    Returns:
        Dict with "hours", "days", and "tokens_per_second".
    """
    effective_tps = tokens_per_second / gradient_accumulation
    total_secs    = total_training_tokens / max(effective_tps, 1)

    return {
        "total_seconds":        round(total_secs, 0),
        "hours":                round(total_secs / 3600, 2),
        "days":                 round(total_secs / 86400, 3),
        "tokens_per_second":    round(effective_tps, 1),
    }
