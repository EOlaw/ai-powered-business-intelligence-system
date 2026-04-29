"""
InsightSerenity AI Engine — GPT-Style Transformer Decoder
==========================================================
The autoregressive decoder is the backbone of every generation-focused
language model: GPT-2, GPT-3/4, LLaMA, Mistral, Falcon, and our own LLM.

Architecture of one decoder layer (Pre-LN):
    x → RMSNorm → CausalSelfAttention → + → RMSNorm → FFN → +

"Decoder-only" means there is no cross-attention to an encoder.
The model predicts the next token given all previous tokens.

Full GPT decoder:
    Token IDs
    → TokenEmbedding + [PositionalEmbedding]
    → Stack of N GPTDecoderLayers (each with causal self-attention)
    → Final RMSNorm
    → LM head: Linear(d_model, vocab_size)
    → logits for next-token prediction

Training objective: cross-entropy loss on shifted targets (teacher forcing).
    Loss = -sum(log P(x_{t+1} | x_1, ..., x_t))

This is the architecture we will pretrain on our corpus in Phase 6.

Key design decisions:
    - Pre-norm (RMSNorm before each sublayer): more stable training
    - RoPE positional embeddings: better length generalisation
    - SwiGLU FFN: better performance at same parameter count
    - Weight tying: embedding and LM head share weights (reduces params)
    - Residual scaling: output projections scaled by 1/sqrt(2N)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from src.nn.layers.feedforward import FeedForward, GatedFeedForward
from src.nn.normalization.normalization import RMSNorm, LayerNorm
from src.nn.layers.embedding import TokenEmbedding, RotaryPositionalEmbedding
from src.architectures.transformer.attention.multi_head_attention import CausalSelfAttention
from src.nn.initializers.initializers import initialize_weights, scale_residual_weights
from src.tokenizer.special_tokens import SpecialTokens as ST


@dataclass
class GPTConfig:
    """Configuration for a GPT-style decoder-only language model."""
    vocab_size:       int   = 50_257   # GPT-2 default; set to our BPE vocab size
    d_model:          int   = 768      # "small" GPT-2 hidden size
    num_heads:        int   = 12
    num_layers:       int   = 12
    d_ff:             Optional[int] = None   # Defaults to 4×d_model for FFN or 8/3×d_model for SwiGLU
    max_seq_len:      int   = 1024
    dropout:          float = 0.1
    attention_dropout: float = 0.0
    ffn_type:         str   = "swiglu"   # "standard" or "swiglu"
    norm_type:        str   = "rms"      # "rms" or "layer"
    use_rope:         bool  = True
    tie_weights:      bool  = True       # Share embedding ↔ LM head weights
    pad_id:           int   = ST.PAD_ID


# ── Pre-defined model sizes ─────────────────────────────────────────────────

GPT_SMALL = GPTConfig(d_model=768,  num_heads=12, num_layers=12, max_seq_len=1024)
GPT_MEDIUM = GPTConfig(d_model=1024, num_heads=16, num_layers=24, max_seq_len=1024)
GPT_LARGE  = GPTConfig(d_model=1280, num_heads=20, num_layers=36, max_seq_len=1024)


class GPTDecoderLayer(nn.Module):
    """
    One Pre-LN GPT decoder layer.

    Architecture:
        x → Norm → CausalSelfAttention → + → Norm → FFN → +

    Args:
        d_model:           Model dimension.
        num_heads:         Attention heads.
        d_ff:              FFN hidden dim (None = auto).
        max_seq_len:       Maximum sequence length for causal mask.
        dropout:           Residual dropout.
        attention_dropout: Attention weight dropout.
        ffn_type:          "standard" (GELU FFN) or "swiglu" (SwiGLU FFN).
        norm_type:         "rms" or "layer".
        use_rope:          Apply RoPE to Q and K.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        ffn_type: str = "swiglu",
        norm_type: str = "rms",
        use_rope: bool = True,
    ) -> None:
        super().__init__()

        NormClass = RMSNorm if norm_type == "rms" else LayerNorm

        self.norm_1 = NormClass(d_model)
        self.norm_2 = NormClass(d_model)

        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=attention_dropout,
            use_rope=use_rope,
        )

        if ffn_type == "swiglu":
            self.ffn: nn.Module = GatedFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        else:
            self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.drop = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:                (B, T, D) — input token representations.
            key_padding_mask: (B, T) — True for real tokens.

        Returns:
            Tuple (output (B, T, D), attention_weights (B, H, T, T)).
        """
        # Pre-norm attention sublayer
        residual = x
        x, weights = self.attn(self.norm_1(x), key_padding_mask=key_padding_mask)
        x = self.drop(x)
        x = residual + x

        # Pre-norm FFN sublayer
        residual = x
        x = self.ffn(self.norm_2(x))
        x = self.drop(x)
        x = residual + x

        return x, weights


class GPTDecoder(nn.Module):
    """
    Full GPT-style autoregressive language model.

    This is the primary model architecture for our LLM. After Phase 6
    (pretraining on our crawled corpus), this model will be the backbone
    for all generation tasks.

    The model can be used for:
        - Next-token prediction (causal LM pretraining)
        - Text completion (inference: greedy/sampling)
        - Fine-tuning (SFT, RLHF)

    Args:
        config: GPTConfig dataclass.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # ── Token embeddings ──────────────────────────────────────────────
        # Scale=False because we use pre-norm, not post-norm
        self.token_emb = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            pad_id=config.pad_id,
            scale=False,
        )

        # If NOT using RoPE, use learned positional embeddings
        if not config.use_rope:
            from src.nn.layers.embedding import LearnedPositionalEmbedding
            self.pos_emb: Optional[nn.Module] = LearnedPositionalEmbedding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            self.pos_emb = None   # RoPE is handled inside each attention layer

        self.emb_dropout = nn.Dropout(p=config.dropout)

        # ── Transformer decoder layers ────────────────────────────────────
        self.layers = nn.ModuleList([
            GPTDecoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                ffn_type=config.ffn_type,
                norm_type=config.norm_type,
                use_rope=config.use_rope,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalisation
        NormClass = RMSNorm if config.norm_type == "rms" else LayerNorm
        self.final_norm = NormClass(config.d_model)

        # ── LM head ───────────────────────────────────────────────────────
        # Projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding and LM head share the same weight matrix
        # This reduces parameters by vocab_size × d_model and often improves perplexity
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.embedding.weight

        # Apply init
        initialize_weights(self)
        if not config.tie_weights:
            # If weights are tied, don't double-init the lm_head
            nn.init.normal_(self.lm_head.weight, std=0.02)
        scale_residual_weights(self, num_layers=config.num_layers)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """
        Forward pass for training (full sequence → next-token logits).

        Args:
            input_ids:        (B, T) — input token IDs.
            attention_mask:   (B, T) — 1 for real, 0 for padding.
            output_attentions: If True, return attention weights from all layers.

        Returns:
            Dict with keys:
                "logits":        (B, T, vocab_size) — next-token prediction logits.
                "last_hidden":   (B, T, D) — final hidden states.
                "all_attentions": List[(B, H, T, T)] if output_attentions.
        """
        B, T = input_ids.shape

        # Embed tokens
        x = self.token_emb(input_ids)    # (B, T, D)

        # Add positional embeddings (only if not using RoPE)
        if self.pos_emb is not None:
            x = self.pos_emb(x)

        x = self.emb_dropout(x)

        # Padding mask
        key_padding_mask = attention_mask.bool() if attention_mask is not None else None

        # Decoder layers
        all_attentions: List[Tensor] = []
        for layer in self.layers:
            x, weights = layer(x, key_padding_mask=key_padding_mask)
            if output_attentions:
                all_attentions.append(weights)

        x = self.final_norm(x)    # (B, T, D)

        # Project to vocabulary
        logits = self.lm_head(x)   # (B, T, vocab_size)

        result = {"logits": logits, "last_hidden": x}
        if output_attentions:
            result["all_attentions"] = all_attentions

        return result

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total parameter count."""
        params = self.parameters() if not trainable_only else \
            (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, "
            f"d_model={self.config.d_model}, "
            f"num_layers={self.config.num_layers}, "
            f"params={self.num_parameters():,}"
        )
