"""
InsightSerenity AI Engine — BERT-Style Transformer Encoder
==========================================================
Implements the transformer encoder architecture for bidirectional
language understanding, as introduced in BERT (Devlin et al., 2018).

Architecture of one encoder layer:
    x → LayerNorm → SelfAttention → + (residual) → LayerNorm → FFN → + (residual)

This is the PRE-NORM variant (also called Pre-LN), which is more stable
to train than the original Post-LN. LLaMA and most modern models use pre-norm.

Differences between Pre-LN (this) and Post-LN (original BERT):
    Post-LN: attention → + residual → LayerNorm → FFN → + residual → LayerNorm
    Pre-LN:  LayerNorm → attention → + residual → LayerNorm → FFN → + residual
    Pre-LN is more numerically stable for very deep models.

Full encoder:
    Input token IDs + attention mask
    → TokenEmbedding + PositionalEmbedding
    → Stack of N TransformerEncoderLayers
    → [Optional MLM head for pretraining]
    → Sequence of contextualised token representations

Uses:
    - Text classification (sentence embeddings from [CLS] token)
    - Token classification (NER, POS tagging)
    - Question answering
    - Masked language model pretraining
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.nn.layers.feedforward import FeedForward
from src.nn.normalization.normalization import LayerNorm, RMSNorm
from src.nn.layers.embedding import TokenEmbedding, SinusoidalPositionalEmbedding, LearnedPositionalEmbedding
from src.architectures.transformer.attention.multi_head_attention import SelfAttention
from src.nn.initializers.initializers import initialize_weights, scale_residual_weights
from src.tokenizer.special_tokens import SpecialTokens as ST


@dataclass
class BERTConfig:
    """Configuration for a BERT-style encoder model."""
    vocab_size:      int   = 30_522      # Standard BERT vocab size
    d_model:         int   = 768         # BERT-base hidden size
    num_heads:       int   = 12          # BERT-base attention heads
    num_layers:      int   = 12          # BERT-base transformer layers
    d_ff:            int   = 3072        # 4 × d_model
    max_seq_len:     int   = 512
    dropout:         float = 0.1
    attention_dropout: float = 0.1
    norm_type:       str   = "layer"     # "layer" or "rms"
    pos_emb_type:    str   = "learned"   # "learned" or "sinusoidal"
    pad_id:          int   = ST.PAD_ID
    # MLM head
    mlm_head:        bool  = False       # Add masked LM head for pretraining


class TransformerEncoderLayer(nn.Module):
    """
    One Pre-LN transformer encoder layer.

    Architecture:
        x → RMSNorm/LayerNorm → SelfAttention → + → RMSNorm/LayerNorm → FFN → +

    Args:
        d_model:           Model dimension.
        num_heads:         Number of attention heads.
        d_ff:              FFN hidden dimension.
        dropout:           Residual/FFN dropout.
        attention_dropout: Dropout in attention weights.
        norm_type:         "layer" (LayerNorm) or "rms" (RMSNorm).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        norm_type: str = "layer",
    ) -> None:
        super().__init__()

        NormClass = RMSNorm if norm_type == "rms" else LayerNorm

        self.norm_1   = NormClass(d_model)
        self.norm_2   = NormClass(d_model)
        self.attn     = SelfAttention(d_model, num_heads, dropout=attention_dropout)
        self.ffn      = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop     = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:                (B, T, D) — input representations.
            key_padding_mask: (B, T) — True for real tokens, False for padding.

        Returns:
            Tuple (output (B, T, D), attention_weights (B, H, T, T)).
        """
        # Pre-norm self-attention sublayer
        residual = x
        x        = self.norm_1(x)
        x, weights = self.attn(x, key_padding_mask=key_padding_mask)
        x        = self.drop(x)
        x        = residual + x

        # Pre-norm FFN sublayer
        residual = x
        x        = self.norm_2(x)
        x        = self.ffn(x)
        x        = self.drop(x)
        x        = residual + x

        return x, weights


class BERTEncoder(nn.Module):
    """
    BERT-style bidirectional transformer encoder.

    Can be used as:
        1. A pretrained encoder (with MLM head) — run through BERTForMLM
        2. A feature extractor — extract the [CLS] embedding for classification
        3. A sequence encoder — output all token representations for token tasks

    Args:
        config: BERTConfig dataclass with all hyperparameters.
    """

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.config = config

        # ── Embeddings ──────────────────────────────────────────────────────
        self.token_emb = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            pad_id=config.pad_id,
            scale=False,   # BERT does not scale embeddings
        )

        if config.pos_emb_type == "learned":
            self.pos_emb = LearnedPositionalEmbedding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            self.pos_emb = SinusoidalPositionalEmbedding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )

        self.emb_norm    = LayerNorm(config.d_model)
        self.emb_dropout = nn.Dropout(p=config.dropout)

        # ── Transformer layers ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                norm_type=config.norm_type,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalisation (pre-norm architecture needs norm at output)
        NormClass = RMSNorm if config.norm_type == "rms" else LayerNorm
        self.final_norm = NormClass(config.d_model)

        # ── Optional MLM head ─────────────────────────────────────────────────
        if config.mlm_head:
            self.mlm_head: Optional[nn.Module] = BERTMLMHead(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
            )
        else:
            self.mlm_head = None

        # Apply default init, then scale residual paths
        initialize_weights(self)
        scale_residual_weights(self, num_layers=config.num_layers)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """
        Args:
            input_ids:        (B, T) — integer token IDs.
            attention_mask:   (B, T) — 1 for real tokens, 0 for padding.
                              If None, all positions are attended to.
            output_attentions: If True, include attention weights in output.

        Returns:
            Dict with keys:
                "last_hidden_state": (B, T, D) — final layer representations
                "pooler_output":     (B, D)    — [CLS] token representation
                "all_attentions":    List of (B, H, T, T) if output_attentions
                "mlm_logits":        (B, T, vocab_size) if mlm_head is set
        """
        B, T = input_ids.shape

        # ── Embed ──────────────────────────────────────────────────────────
        x = self.token_emb(input_ids)   # (B, T, D)
        x = self.pos_emb(x)             # (B, T, D)
        x = self.emb_norm(x)

        # Convert attention_mask to key_padding_mask (bool, True=real)
        if attention_mask is not None:
            key_padding_mask = attention_mask.bool()
        else:
            key_padding_mask = None

        # ── Transformer layers ──────────────────────────────────────────────
        all_attentions = []
        for layer in self.layers:
            x, weights = layer(x, key_padding_mask=key_padding_mask)
            if output_attentions:
                all_attentions.append(weights)

        x = self.final_norm(x)   # (B, T, D)

        # Pooler: take [CLS] token (position 0) as sentence representation
        cls_output = x[:, 0, :]  # (B, D)

        result = {
            "last_hidden_state": x,
            "pooler_output":     cls_output,
        }

        if output_attentions:
            result["all_attentions"] = all_attentions

        if self.mlm_head is not None:
            result["mlm_logits"] = self.mlm_head(x)

        return result


class BERTMLMHead(nn.Module):
    """
    Masked Language Modeling prediction head for BERT pretraining.

    Takes the final encoder representations and predicts the vocabulary
    logit for each masked token position.

    Architecture: Linear → GELU → LayerNorm → Linear (to vocab)
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.dense    = nn.Linear(d_model, d_model)
        self.act      = nn.GELU()
        self.norm     = LayerNorm(d_model)
        self.decoder  = nn.Linear(d_model, vocab_size, bias=True)

        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D) — encoder hidden states.

        Returns:
            (B, T, vocab_size) — logits for each position.
        """
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
        return self.decoder(x)
