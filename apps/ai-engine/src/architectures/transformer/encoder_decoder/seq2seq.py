"""
InsightSerenity AI Engine — Seq2Seq Transformer (T5-Style)
==========================================================
The encoder-decoder transformer architecture for sequence-to-sequence tasks.
Used by T5, BART, mT5, and any model that maps one sequence to another:
    - Translation:       English → French
    - Summarisation:     Article → Summary
    - Code generation:   Docstring → Code
    - Question answering: (Question, Context) → Answer

Architecture:
    Encoder: Stack of bidirectional self-attention layers
    Decoder: Stack of causal self-attention + cross-attention layers

The decoder cross-attention layers allow the decoder to "look at" the
encoder's representations of the input when generating each output token.

Cross-attention: decoder query attends over encoder key/value
    output_t = Attention(Q=decoder_state_t, K=encoder_states, V=encoder_states)

This is fundamentally different from decoder-only models (GPT) where there
is no encoder and the model must store everything in the decoder's context.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from src.nn.layers.feedforward import FeedForward, GatedFeedForward
from src.nn.normalization.normalization import RMSNorm, LayerNorm
from src.nn.layers.embedding import TokenEmbedding, SinusoidalPositionalEmbedding
from src.architectures.transformer.attention.multi_head_attention import (
    SelfAttention, CausalSelfAttention, CrossAttention
)
from src.nn.initializers.initializers import initialize_weights
from src.tokenizer.special_tokens import SpecialTokens as ST


@dataclass
class Seq2SeqConfig:
    """Configuration for a T5-style encoder-decoder model."""
    vocab_size:       int   = 32_000
    d_model:          int   = 512
    num_heads:        int   = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff:             Optional[int] = None
    max_seq_len:      int   = 512
    dropout:          float = 0.1
    attention_dropout: float = 0.1
    ffn_type:         str   = "swiglu"
    norm_type:        str   = "rms"
    pad_id:           int   = ST.PAD_ID
    tie_embeddings:   bool  = True   # Share source/target embeddings


class Seq2SeqEncoderLayer(nn.Module):
    """One encoder layer: Pre-LN SelfAttention + Pre-LN FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int],
        dropout: float,
        attention_dropout: float,
        ffn_type: str,
        norm_type: str,
    ) -> None:
        super().__init__()

        NormClass = RMSNorm if norm_type == "rms" else LayerNorm
        self.norm_1 = NormClass(d_model)
        self.norm_2 = NormClass(d_model)
        self.attn   = SelfAttention(d_model, num_heads, dropout=attention_dropout)
        self.ffn    = GatedFeedForward(d_model, d_ff, dropout=dropout) \
                      if ffn_type == "swiglu" else FeedForward(d_model, d_ff, dropout=dropout)
        self.drop   = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:                (B, T, D)
            key_padding_mask: (B, T) — True for real tokens

        Returns:
            (B, T, D)
        """
        residual = x
        x, _    = self.attn(self.norm_1(x), key_padding_mask=key_padding_mask)
        x       = self.drop(x) + residual

        residual = x
        x        = self.ffn(self.norm_2(x))
        x        = self.drop(x) + residual
        return x


class Seq2SeqDecoderLayer(nn.Module):
    """
    One decoder layer: Pre-LN CausalSelfAttention + CrossAttention + FFN.

    The causal self-attention ensures the decoder cannot see future target tokens.
    The cross-attention allows the decoder to query the encoder's output.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int],
        max_seq_len: int,
        dropout: float,
        attention_dropout: float,
        ffn_type: str,
        norm_type: str,
    ) -> None:
        super().__init__()

        NormClass = RMSNorm if norm_type == "rms" else LayerNorm

        self.norm_1 = NormClass(d_model)
        self.norm_2 = NormClass(d_model)
        self.norm_3 = NormClass(d_model)

        # Causal self-attention over decoder targets (no future leakage)
        self.self_attn  = CausalSelfAttention(
            d_model, num_heads, max_seq_len=max_seq_len, dropout=attention_dropout
        )
        # Cross-attention: decoder attends to encoder output
        self.cross_attn = CrossAttention(d_model, num_heads, dropout=attention_dropout)

        self.ffn  = GatedFeedForward(d_model, d_ff, dropout=dropout) \
                    if ffn_type == "swiglu" else FeedForward(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        decoder_padding_mask: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:                    (B, T_dec, D) — decoder inputs (target tokens).
            encoder_output:       (B, T_enc, D) — encoder hidden states.
            decoder_padding_mask: (B, T_dec) — True for real decoder tokens.
            encoder_padding_mask: (B, T_enc) — True for real encoder tokens.

        Returns:
            (B, T_dec, D) — updated decoder representations.
        """
        # Causal self-attention
        residual = x
        x, _    = self.self_attn(self.norm_1(x), key_padding_mask=decoder_padding_mask)
        x       = self.drop(x) + residual

        # Cross-attention: query from decoder, key/value from encoder
        residual = x
        x, _    = self.cross_attn(
            self.norm_2(x),
            encoder_output,
            context_padding_mask=encoder_padding_mask,
        )
        x       = self.drop(x) + residual

        # FFN
        residual = x
        x        = self.ffn(self.norm_3(x))
        x        = self.drop(x) + residual

        return x


class Seq2SeqTransformer(nn.Module):
    """
    Full encoder-decoder transformer for sequence-to-sequence tasks.

    Args:
        config: Seq2SeqConfig dataclass.
    """

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config

        # ── Embeddings ──────────────────────────────────────────────────────
        self.src_embedding = TokenEmbedding(config.vocab_size, config.d_model, pad_id=config.pad_id)

        if config.tie_embeddings:
            # Same embedding table for source and target
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TokenEmbedding(config.vocab_size, config.d_model, pad_id=config.pad_id)

        self.pos_emb = SinusoidalPositionalEmbedding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder_layers = nn.ModuleList([
            Seq2SeqEncoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                ffn_type=config.ffn_type,
                norm_type=config.norm_type,
            )
            for _ in range(config.num_encoder_layers)
        ])

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            Seq2SeqDecoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                ffn_type=config.ffn_type,
                norm_type=config.norm_type,
            )
            for _ in range(config.num_decoder_layers)
        ])

        NormClass = RMSNorm if config.norm_type == "rms" else LayerNorm
        self.encoder_norm = NormClass(config.d_model)
        self.decoder_norm = NormClass(config.d_model)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.src_embedding.embedding.weight

        initialize_weights(self)

    def encode(
        self,
        src_ids: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode source sequence.

        Args:
            src_ids:  (B, T_src) — source token IDs.
            src_mask: (B, T_src) — 1 for real tokens.

        Returns:
            (B, T_src, D) — encoder representations.
        """
        x = self.src_embedding(src_ids)
        x = self.pos_emb(x)
        kpm = src_mask.bool() if src_mask is not None else None
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=kpm)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt_ids: Tensor,
        encoder_output: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode target sequence conditioned on encoder output.

        Args:
            tgt_ids:        (B, T_tgt) — target token IDs.
            encoder_output: (B, T_src, D) — encoder hidden states.
            tgt_mask:       (B, T_tgt) — target padding mask.
            src_mask:       (B, T_src) — source padding mask.

        Returns:
            (B, T_tgt, D) — decoder representations.
        """
        x = self.tgt_embedding(tgt_ids)
        x = self.pos_emb(x)
        dec_kpm = tgt_mask.bool() if tgt_mask is not None else None
        enc_kpm = src_mask.bool() if src_mask is not None else None
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, dec_kpm, enc_kpm)
        return self.decoder_norm(x)

    def forward(
        self,
        src_ids: Tensor,
        tgt_ids: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full forward pass: encode source, decode target, project to vocab.

        Args:
            src_ids:  (B, T_src) — encoder input token IDs.
            tgt_ids:  (B, T_tgt) — decoder input token IDs (shifted right).
            src_mask: (B, T_src) — encoder padding mask.
            tgt_mask: (B, T_tgt) — decoder padding mask.

        Returns:
            (B, T_tgt, vocab_size) — vocabulary logits.
        """
        enc_output = self.encode(src_ids, src_mask)
        dec_output = self.decode(tgt_ids, enc_output, tgt_mask, src_mask)
        return self.lm_head(dec_output)
