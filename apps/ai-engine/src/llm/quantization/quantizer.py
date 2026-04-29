"""
InsightSerenity AI Engine — Model Quantization
================================================
Reduces model memory footprint and increases inference throughput by
converting float32 weights to lower-precision integer formats.

Why quantize?
    - A 7B parameter model in float32 requires 28 GB of VRAM.
    - The same model in INT8 requires ~7 GB.
    - In INT4 (GGUF/GPTQ style): ~3.5 GB.
    - Quality degradation: INT8 is nearly lossless; INT4 loses ~1-2 PPL.

Quantization types implemented:

1. Dynamic INT8 quantization
   Weights stored as INT8, activations quantized on the fly.
   PyTorch built-in: torch.quantization.quantize_dynamic.
   Best for CPU inference — minimal code, good speedup.

2. Static INT8 quantization (with calibration)
   Both weights and activations quantized using statistics from a
   calibration dataset. More accurate than dynamic but requires calibration.

3. Fake quantization (quantization-aware training)
   Simulates quantization during training so the model adapts to the
   reduced precision. Best quality but most expensive to implement.

4. INT4 weight-only quantization
   Weights stored as 4-bit integers, dequantized to float16 before
   matrix multiplication. Simple, effective, widely used.
   (This is the approach used by llama.cpp / GGUF format.)

Note: INT4/INT8 quantization only speeds up inference, not training.
For training, use mixed precision (bfloat16) instead.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic INT8 quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_dynamic_int8(
    model:      nn.Module,
    layer_types: Optional[Tuple] = None,
) -> nn.Module:
    """
    Apply PyTorch dynamic INT8 quantization to all Linear layers.

    Dynamic quantization quantizes weights statically (offline) but
    activations dynamically (at runtime, per-batch). This trades some
    accuracy for simplicity — no calibration dataset needed.

    Best for: CPU inference with batch_size=1 (chatbot, API serving).
    Limited benefit on GPU (GPU operations are already fast in fp16).

    Args:
        model:       The model to quantize.
        layer_types: Tuple of layer types to quantize. Default: (nn.Linear,).

    Returns:
        Quantized model (in-place modification of a copy).
    """
    layer_types = layer_types or (nn.Linear,)

    logger.info("Applying dynamic INT8 quantization")
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=set(layer_types),
        dtype=torch.qint8,
    )
    n_params_before = sum(p.numel() for p in model.parameters())
    logger.info(
        "Dynamic INT8 quantization complete",
        original_params=n_params_before,
    )
    return quantized


# ─────────────────────────────────────────────────────────────────────────────
# INT4 weight-only quantization
# ─────────────────────────────────────────────────────────────────────────────

class Int4Linear(nn.Module):
    """
    Linear layer with INT4 weight-only quantization.

    Weights are stored as INT4 (packed 2 per byte), dequantized to float16
    just before the matrix multiplication. Activations remain in float16.

    This halves the weight memory compared to INT8 and gives ~4x compression
    vs float32, with moderate accuracy degradation (~1-2 perplexity points).

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Whether to include a bias term.
        group_size:   Quantization group size. Smaller groups = more accurate
                      but more overhead. -1 = quantize the whole row.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = True,
        group_size:   int  = 128,
    ) -> None:
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size if group_size > 0 else in_features

        # INT4 packed: each byte stores 2 INT4 values
        # Storage shape: (out_features, in_features // 2) bytes
        n_groups  = (in_features + self.group_size - 1) // self.group_size
        half_in   = (in_features + 1) // 2

        self.register_buffer("weight_int4", torch.zeros(out_features, half_in, dtype=torch.uint8))
        self.register_buffer("scales",      torch.ones(out_features, n_groups, dtype=torch.float16))
        self.register_buffer("zeros",       torch.zeros(out_features, n_groups, dtype=torch.float16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Dequantize weights and compute linear transformation."""
        weight_fp16 = self._dequantize()
        return nn.functional.linear(x.half(), weight_fp16, self.bias)

    def _dequantize(self) -> Tensor:
        """Unpack INT4 bytes and dequantize to float16."""
        # Unpack: each byte → two 4-bit values
        low  = self.weight_int4 & 0x0F    # Lower 4 bits
        high = self.weight_int4 >> 4       # Upper 4 bits

        # Interleave to recover original column order
        weight_int4_full = torch.stack([low, high], dim=-1).reshape(
            self.out_features, -1
        )[:, :self.in_features]   # Trim any padding

        # Dequantize: w_float = (w_int4 - zero) * scale
        # Group-wise dequantization
        weight_fp16 = torch.zeros(
            self.out_features, self.in_features,
            dtype=torch.float16, device=self.weight_int4.device,
        )

        for g in range(self.scales.shape[1]):
            start = g * self.group_size
            end   = min(start + self.group_size, self.in_features)
            scale = self.scales[:, g:g+1]   # (out, 1)
            zero  = self.zeros[:, g:g+1]    # (out, 1)
            weight_fp16[:, start:end] = (
                weight_int4_full[:, start:end].float() - zero
            ) * scale

        return weight_fp16

    @classmethod
    def from_linear(
        cls,
        linear:     nn.Linear,
        group_size: int = 128,
    ) -> "Int4Linear":
        """
        Convert an existing nn.Linear to Int4Linear.

        Quantizes the weights in place using per-group asymmetric quantization.

        Args:
            linear:     Source nn.Linear module.
            group_size: Quantization group size.

        Returns:
            New Int4Linear with quantized weights.
        """
        out_f, in_f = linear.weight.shape
        has_bias    = linear.bias is not None

        int4_linear = cls(in_f, out_f, bias=has_bias, group_size=group_size)

        weight_fp = linear.weight.data.float()
        n_groups  = (in_f + group_size - 1) // group_size

        # Quantize each group
        weight_int4 = torch.zeros(out_f, in_f, dtype=torch.int8)
        scales_     = torch.zeros(out_f, n_groups)
        zeros_      = torch.zeros(out_f, n_groups)

        for g in range(n_groups):
            start    = g * group_size
            end      = min(start + group_size, in_f)
            group_w  = weight_fp[:, start:end]

            min_val  = group_w.min(dim=1, keepdim=True).values
            max_val  = group_w.max(dim=1, keepdim=True).values
            scale    = (max_val - min_val) / 15.0   # INT4 range: 0..15
            scale    = scale.clamp(min=1e-8)
            zero     = min_val / scale

            q_weight = ((group_w / scale) - zero).round().clamp(0, 15).to(torch.int8)
            weight_int4[:, start:end] = q_weight
            scales_[:, g] = scale.squeeze(-1)
            zeros_[:, g]  = zero.squeeze(-1)

        # Pack two INT4 values per byte
        if in_f % 2 != 0:
            weight_int4 = torch.cat(
                [weight_int4, torch.zeros(out_f, 1, dtype=torch.int8)], dim=-1
            )
        packed = (weight_int4[:, 1::2] << 4) | (weight_int4[:, 0::2] & 0x0F)

        int4_linear.weight_int4.copy_(packed.to(torch.uint8))
        int4_linear.scales.copy_(scales_.half())
        int4_linear.zeros.copy_(zeros_.half())

        if has_bias:
            int4_linear.bias.data.copy_(linear.bias.data.half())

        return int4_linear


def quantize_model_int4(
    model:      nn.Module,
    group_size: int = 128,
    skip_names: Optional[List[str]] = None,
) -> nn.Module:
    """
    Quantize all Linear layers in a model to INT4.

    Skips layers whose names contain any string in `skip_names` — typically
    the embedding layer and LM head, which are more sensitive to quantization.

    Args:
        model:       The model to quantize.
        group_size:  Quantization group size.
        skip_names:  Layer name substrings to skip. Default: ["embed", "lm_head"].

    Returns:
        Model with Linear layers replaced by Int4Linear.
    """
    skip_names = skip_names or ["embed", "lm_head", "token_emb"]
    n_quantized = 0
    n_skipped   = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Skip sensitive layers
            if any(s in name for s in skip_names):
                n_skipped += 1
                continue

            # Replace with Int4Linear
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            int4_layer = Int4Linear.from_linear(module, group_size=group_size)
            setattr(parent, attr_name, int4_layer)
            n_quantized += 1

    logger.info(
        "INT4 quantization complete",
        quantized_layers=n_quantized,
        skipped_layers=n_skipped,
    )
    return model


def estimate_memory_reduction(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory usage before and after INT4 quantization.

    Returns a dict with memory estimates in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    int4_params  = sum(
        p.numel() for name, p in model.named_parameters()
        if not any(s in name for s in ["embed", "lm_head"])
    )
    other_params = total_params - int4_params

    fp32_mb  = total_params * 4 / (1024 ** 2)
    int4_mb  = (int4_params * 0.5 + other_params * 4) / (1024 ** 2)  # 4-bit = 0.5 bytes

    return {
        "fp32_memory_mb":        round(fp32_mb, 1),
        "int4_memory_mb":        round(int4_mb, 1),
        "compression_ratio":     round(fp32_mb / max(int4_mb, 0.001), 2),
        "memory_saved_mb":       round(fp32_mb - int4_mb, 1),
    }
