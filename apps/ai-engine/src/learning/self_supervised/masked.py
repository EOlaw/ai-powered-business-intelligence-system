"""
InsightSerenity AI Engine — Masked Self-Supervised Learning
============================================================
Trains a model to reconstruct masked (hidden) portions of the input.
This forces the model to learn contextual representations — it must
use the visible parts to predict the masked parts.

Two variants:

1. MaskedAutoencoder (MAE — He et al., 2022):
   Masks a large fraction (75%) of input patches/tokens.
   The encoder only processes the VISIBLE patches (efficient).
   A lightweight decoder reconstructs the masked patches from:
       - Encoded visible tokens
       - Mask tokens (learned embedding for "something is here")
   Loss: MSE on pixel/token values at masked positions only.

   This is the method behind MAE for images and the principle behind
   many self-supervised text models (except the decoder is usually heavier).

2. BERTStyleMLM (Masked Language Model):
   Wraps our existing MLMDataCollator + BERTEncoder to create a
   complete pretraining pipeline. This is Phase 2 (MLM objective) applied
   as a self-supervised learning paradigm.

   Included here as a complete, standalone training setup so you can
   pretrain a BERT-style encoder on any JSONL text corpus with one call.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Masked Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MAEConfig:
    input_dim:       int           # Dimension of each input token/patch
    encoder_dim:     int   = 256   # Encoder hidden dimension
    decoder_dim:     int   = 64    # Decoder hidden dimension (intentionally smaller)
    encoder_depth:   int   = 4     # Encoder transformer layers
    decoder_depth:   int   = 2     # Decoder transformer layers
    num_heads:       int   = 4     # Attention heads
    mask_ratio:      float = 0.75  # Fraction of tokens to mask
    lr:              float = 1e-4
    batch_size:      int   = 64
    epochs:          int   = 50
    device:          str   = "cpu"


class MAEPatchEncoder(nn.Module):
    """
    Lightweight Transformer encoder for visible patches only (MAE-style).

    Only the visible (unmasked) tokens are processed through this encoder,
    making it much more efficient than processing all tokens.
    """

    def __init__(self, input_dim: int, embed_dim: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm        = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.projection.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N_visible, input_dim) — visible token features.

        Returns:
            (B, N_visible, embed_dim) — encoded representations.
        """
        x = self.projection(x)
        x = self.transformer(x)
        return self.norm(x)


class MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder that reconstructs ALL tokens.

    Takes the encoded visible tokens + mask tokens (learnable embeddings
    for masked positions) and reconstructs the original input at every position.

    The decoder is intentionally shallow (2 layers vs encoder's 4+) — the
    encoder does the heavy lifting of learning good representations.
    After pretraining, the decoder is discarded.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        output_dim:  int,
        depth:       int,
        num_heads:   int,
    ) -> None:
        super().__init__()
        # Project from encoder_dim to decoder_dim
        self.projection  = nn.Linear(encoder_dim, decoder_dim)
        # Learnable mask token: used as a stand-in for each masked position
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        decoder_layer    = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads, dim_feedforward=decoder_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm        = nn.LayerNorm(decoder_dim)
        # Final linear to reconstruct the original token values
        self.output_proj = nn.Linear(decoder_dim, output_dim)

    def forward(
        self,
        visible_tokens: Tensor,   # (B, N_visible, encoder_dim) — encoded visible
        mask_indices:   Tensor,   # (B, N_total) — True at masked positions
        N_total:        int,
    ) -> Tensor:
        """
        Reconstruct all tokens from encoded visible + mask tokens.

        Args:
            visible_tokens: (B, N_visible, encoder_dim) — from MAEPatchEncoder.
            mask_indices:   (B, N_total) BoolTensor — True at masked positions.
            N_total:        Total number of tokens (visible + masked).

        Returns:
            (B, N_total, output_dim) — reconstructed token values.
        """
        B = visible_tokens.shape[0]

        # Project visible tokens to decoder dimension
        visible_proj = self.projection(visible_tokens)   # (B, N_visible, decoder_dim)

        # Build the full token sequence: mask tokens where masked, visible elsewhere
        full_tokens = self.mask_token.expand(B, N_total, -1).clone()
        # Place visible tokens at their original positions
        visible_pos = ~mask_indices   # (B, N_total) True where visible
        # Scatter visible projections back
        for b in range(B):
            full_tokens[b, visible_pos[b]] = visible_proj[b]

        # Decode all positions together (context from both visible and masked)
        full_tokens = self.transformer(full_tokens)
        full_tokens = self.norm(full_tokens)
        return self.output_proj(full_tokens)   # (B, N_total, output_dim)


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for learning representations from unlabelled sequences.

    Works on any tokenised sequence (text tokens, image patches, time series).
    The high mask ratio (default 75%) means the model must learn very good
    representations of the visible context to reconstruct the masked parts.

    Args:
        config: MAEConfig.
    """

    def __init__(self, config: MAEConfig) -> None:
        super().__init__()
        self.config  = config

        self.encoder = MAEPatchEncoder(
            input_dim=config.input_dim,
            embed_dim=config.encoder_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
        )
        self.decoder = MAEDecoder(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            output_dim=config.input_dim,
            depth=config.decoder_depth,
            num_heads=config.num_heads,
        )

        self._is_fitted     = False
        self.train_history: List[Dict] = []

    def _random_mask(self, B: int, N: int, device: torch.device) -> Tensor:
        """
        Generate a random boolean mask. True = masked (hidden from encoder).

        Returns:
            (B, N) BoolTensor — True at mask_ratio fraction of positions.
        """
        n_masked  = int(N * self.config.mask_ratio)
        # Noise-based masking (each example gets independently shuffled mask)
        noise     = torch.rand(B, N, device=device)
        ids_sort  = noise.argsort(dim=1)
        mask      = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_sort[:, :n_masked], True)
        return mask   # True = masked

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass: random mask → encode visible → decode all.

        Args:
            x: (B, N, input_dim) — sequence of token/patch features.

        Returns:
            Tuple (reconstruction (B, N, input_dim), mask (B, N), loss scalar).
        """
        B, N, _ = x.shape
        mask    = self._random_mask(B, N, x.device)   # (B, N) True = masked

        # Extract only visible tokens for the encoder
        visible_list = []
        for b in range(B):
            visible_list.append(x[b, ~mask[b]])   # (N_visible_b, input_dim)

        # Batch requires same sequence length — use the minimum visible count
        n_vis = min(v.shape[0] for v in visible_list)
        visible = torch.stack([v[:n_vis] for v in visible_list])   # (B, n_vis, input_dim)

        # Encode only visible tokens (efficient!)
        encoded = self.encoder(visible)   # (B, n_vis, encoder_dim)

        # Decode all positions
        recon = self.decoder(encoded, mask, N)   # (B, N, input_dim)

        # Loss only on MASKED positions (that's what the model is learning to predict)
        loss = F.mse_loss(recon[mask], x[mask])

        return recon, mask, loss

    def fit(self, dataloader: DataLoader) -> "MaskedAutoencoder":
        """
        Train the MAE on unlabelled sequences.

        Args:
            dataloader: Yields batches of shape (B, N, input_dim) tensors.
                        Labels are not needed — yields (X,) or (X, _).
        """
        device = torch.device(self.config.device)
        self.to(device)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )

        for epoch in range(self.config.epochs):
            self.train()
            ep_loss = 0.0

            for batch in dataloader:
                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                X = X.float().to(device)

                _, _, loss = self.forward(X)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()

            scheduler.step()
            avg = ep_loss / max(len(dataloader), 1)
            self.train_history.append({"epoch": epoch, "loss": avg})

            if (epoch + 1) % 10 == 0:
                logger.info("MAE training", epoch=epoch + 1, loss=round(avg, 6))

        self._is_fitted = True
        return self

    def encode(self, x: Tensor, mask_ratio: float = 0.0) -> Tensor:
        """
        Encode sequences using the trained encoder.

        Args:
            x:          (B, N, input_dim).
            mask_ratio: Optional masking during inference for data augmentation.

        Returns:
            (B, N_visible, encoder_dim) representations.
        """
        self.encoder.eval()
        B, N, _ = x.shape
        device  = x.device

        if mask_ratio > 0:
            mask    = self._random_mask(B, N, device)
            visible = torch.stack([x[b, ~mask[b]] for b in range(B)])
        else:
            visible = x

        with torch.no_grad():
            return self.encoder(visible)


# ─────────────────────────────────────────────────────────────────────────────
# BERT-style MLM pretraining pipeline
# ─────────────────────────────────────────────────────────────────────────────

class BERTStylePretrainer:
    """
    Complete self-supervised pretraining pipeline for BERT-style encoders.

    This wraps the MLMDataCollator (from Phase 6) and BERTEncoder into a
    single, ready-to-run MLM pretraining pipeline.

    Args:
        model:      BERTEncoder with mlm_head=True.
        tokenizer:  Trained tokenizer.
        mask_prob:  Fraction of tokens to mask. Default 0.15.
        lr:         Learning rate.
        device:     Compute device.
    """

    def __init__(
        self,
        model:     Any,
        tokenizer: Any,
        mask_prob: float = 0.15,
        lr:        float = 1e-4,
        device:    str   = "cpu",
    ) -> None:
        from src.llm.pretraining.mlm_objective import MLMDataCollator
        self.model     = model.to(torch.device(device))
        self.tokenizer = tokenizer
        self.device    = torch.device(device)
        self.collator  = MLMDataCollator(tokenizer=tokenizer, mask_prob=mask_prob)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.train_history: List[Dict] = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Run one epoch of MLM pretraining.

        Args:
            dataloader: Yields batches of text token IDs.

        Returns:
            Average MLM loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            # Apply MLM masking
            masked_batch = self.collator([batch] if not isinstance(batch, list) else batch)

            input_ids      = masked_batch["input_ids"].to(self.device)
            labels         = masked_batch["labels"].to(self.device)
            attention_mask = masked_batch["attention_mask"].to(self.device)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output["mlm_logits"]   # (B, T, vocab_size)

            B, T, V = logits.shape
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.reshape(B * T, V),
                labels.reshape(B * T),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        self.train_history.append({"loss": avg})
        return avg
