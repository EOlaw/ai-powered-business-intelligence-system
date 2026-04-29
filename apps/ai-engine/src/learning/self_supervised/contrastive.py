"""
InsightSerenity AI Engine — Contrastive Self-Supervised Learning
================================================================
Contrastive learning trains an encoder to produce similar representations
for augmented views of the same data (positive pairs) and dissimilar
representations for different data (negative pairs) — without any labels.

After contrastive pretraining, the encoder produces rich general-purpose
representations that transfer well to downstream tasks with few labelled examples.

SimCLR (Chen et al., 2020):
    1. Take a data sample x
    2. Apply two random augmentations → x_i and x_j (a positive pair)
    3. Encode both: z_i = g(f(x_i)), z_j = g(f(x_j))
       where f = encoder, g = projection head (2-layer MLP)
    4. NT-Xent loss: maximise agreement between z_i and z_j,
       while pushing apart all other pairs in the batch (negatives)

    NT-Xent (Normalised Temperature-scaled Cross Entropy):
        L = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )
        τ = temperature (default 0.07)

CLIPStyle:
    Dual-encoder contrastive learning for matching two modalities
    (e.g. text descriptions ↔ features). Learns a shared embedding space
    where matched pairs have high cosine similarity.

Usage:
    # SimCLR pretraining
    simclr = SimCLR(encoder=my_encoder, projection_dim=128)
    simclr.fit(unlabelled_loader)
    features = simclr.encode(X)  # Use for downstream tasks
"""

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent Loss
# ─────────────────────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy loss for SimCLR.

    Given a batch of 2N representations (N positive pairs),
    treats the matching view as the positive and all 2(N-1) other
    views in the batch as negatives.

    L = -1/(2N) Σ_i [log exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ)]

    The temperature τ controls the concentration of the distribution:
        Low τ  → sharper distribution, harder negatives
        High τ → softer distribution, easier problem

    Args:
        temperature: τ in the NT-Xent formula. Default 0.07 (SimCLR default).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z_i: (N, D) L2-normalised projections from view i.
            z_j: (N, D) L2-normalised projections from view j.

        Returns:
            Scalar NT-Xent loss.
        """
        N      = z_i.shape[0]
        device = z_i.device

        # Concatenate: shape (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarity matrix (2N, 2N)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        sim = sim / self.temperature

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices:
        # For view i, position i: positive is at position i + N
        # For view j, position i + N: positive is at position i
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Projection head
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head for SimCLR.

    Projects encoder representations into the contrastive space.
    The projection head is ONLY used during contrastive training — it is
    discarded when transferring the encoder to downstream tasks.

    The encoder's hidden representations (before projection) transfer better
    than the projection head outputs (Chen et al., 2020).

    Architecture: Linear → BatchNorm → ReLU → Linear
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)   # L2 normalise the output


# ─────────────────────────────────────────────────────────────────────────────
# Text augmentation for contrastive learning on text features
# ─────────────────────────────────────────────────────────────────────────────

def feature_augment(x: Tensor, noise_std: float = 0.1, dropout_p: float = 0.1) -> Tensor:
    """
    Augment a feature vector for contrastive learning.
    Applies Gaussian noise + random feature dropout.

    For image inputs, use torchvision.transforms instead.
    For text tokens, use token deletion / shuffling.
    """
    noisy   = x + torch.randn_like(x) * noise_std
    dropout = torch.bernoulli(torch.full_like(noisy, 1.0 - dropout_p))
    return noisy * dropout


# ─────────────────────────────────────────────────────────────────────────────
# SimCLR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimCLRConfig:
    encoder_output_dim: int           # Output dimension of the backbone encoder
    projection_dim:     int   = 128   # Dimension of the contrastive embedding space
    temperature:        float = 0.07
    lr:                 float = 3e-4
    weight_decay:       float = 1e-4
    epochs:             int   = 100
    device:             str   = "cpu"
    noise_std:          float = 0.1   # Feature augmentation noise
    dropout_p:          float = 0.1   # Feature dropout rate


class SimCLR:
    """
    SimCLR self-supervised contrastive learning.

    Takes an encoder (any nn.Module that produces a feature vector) and trains
    it to produce view-invariant representations using the NT-Xent objective.

    The encoder is trained on UNLABELLED data. After training, the encoder
    can be fine-tuned on a small labelled dataset for any downstream task.

    Args:
        encoder: A PyTorch module. Must map input → (B, encoder_output_dim).
        config:  SimCLRConfig.
    """

    def __init__(self, encoder: nn.Module, config: SimCLRConfig) -> None:
        self.config  = config
        self.device  = torch.device(config.device)

        self.encoder = encoder.to(self.device)
        self.proj    = ProjectionHead(
            input_dim=config.encoder_output_dim,
            hidden_dim=config.encoder_output_dim * 2,
            output_dim=config.projection_dim,
        ).to(self.device)

        self.criterion     = NTXentLoss(temperature=config.temperature)
        self._is_fitted    = False
        self.train_history: List[Dict] = []

    def fit(
        self,
        dataloader: DataLoader,
        augment_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> "SimCLR":
        """
        Train SimCLR on unlabelled data.

        Args:
            dataloader: DataLoader yielding batches of raw inputs (no labels needed).
                        Yields (X,) or (X, _) — labels are ignored.
            augment_fn: Augmentation function applied twice to each batch.
                        Default: Gaussian noise + feature dropout (for embeddings).
                        For images, pass a torchvision.transforms pipeline.

        Returns:
            self.
        """
        augment = augment_fn or (
            lambda x: feature_augment(x, self.config.noise_std, self.config.dropout_p)
        )

        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.proj.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )

        for epoch in range(self.config.epochs):
            self.encoder.train()
            self.proj.train()
            ep_loss = 0.0

            for batch in dataloader:
                # Support (X,) and (X, label) loaders — labels are ignored
                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                X = X.float().to(self.device)

                # Create two augmented views
                x_i = augment(X)
                x_j = augment(X)

                # Encode both views
                h_i = self.encoder(x_i)
                h_j = self.encoder(x_j)

                # Project to contrastive space
                z_i = self.proj(h_i)
                z_j = self.proj(h_j)

                loss = self.criterion(z_i, z_j)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.proj.parameters()), 1.0
                )
                optimizer.step()
                ep_loss += loss.item()

            scheduler.step()
            avg = ep_loss / max(len(dataloader), 1)
            self.train_history.append({"epoch": epoch, "loss": avg})

            if (epoch + 1) % 10 == 0:
                logger.info("SimCLR training", epoch=epoch + 1, loss=round(avg, 4))

        self._is_fitted = True
        return self

    def encode(self, X: Tensor) -> np.ndarray:
        """
        Encode inputs using the trained encoder (without projection head).

        Args:
            X: Input tensor.

        Returns:
            (N, encoder_output_dim) feature array.
        """
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(X.float().to(self.device))
        return features.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# CLIP-style dual encoder
# ─────────────────────────────────────────────────────────────────────────────

class CLIPStyleModel(nn.Module):
    """
    Dual-encoder contrastive model for cross-modal alignment.

    Learns a shared embedding space where matched (a_i, b_i) pairs have
    high cosine similarity and unmatched pairs have low similarity.

    Applications:
        - Text ↔ image matching
        - Query ↔ document retrieval
        - Cross-lingual sentence alignment
        - Any two-view or two-modality contrastive task

    The symmetric loss aligns both directions:
        L = 0.5 × NCE(A→B) + 0.5 × NCE(B→A)

    Args:
        encoder_a:   Encoder for modality A (e.g. text encoder).
        encoder_b:   Encoder for modality B (e.g. image encoder).
        embed_dim:   Shared embedding dimension after projection.
        temperature: Logit scaling temperature. Initialised to log(1/0.07).
    """

    def __init__(
        self,
        encoder_a: nn.Module,
        encoder_b: nn.Module,
        embed_dim: int,
        temperature: float = math.log(1.0 / 0.07),
    ) -> None:
        super().__init__()
        import math
        self.encoder_a   = encoder_a
        self.encoder_b   = encoder_b
        # Learnable temperature (log scale — ensures positivity)
        self.log_temp    = nn.Parameter(torch.tensor(temperature))

    def encode_a(self, x: Tensor) -> Tensor:
        """Encode modality A and L2-normalise."""
        return F.normalize(self.encoder_a(x), dim=-1)

    def encode_b(self, x: Tensor) -> Tensor:
        """Encode modality B and L2-normalise."""
        return F.normalize(self.encoder_b(x), dim=-1)

    def forward(
        self,
        x_a: Tensor,
        x_b: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute embeddings and similarity logits.

        Args:
            x_a: (N, ...) modality A inputs.
            x_b: (N, ...) modality B inputs.

        Returns:
            Tuple (logits_a_to_b, logits_b_to_a) — shape (N, N) each.
        """
        z_a = self.encode_a(x_a)   # (N, D)
        z_b = self.encode_b(x_b)   # (N, D)

        temp         = self.log_temp.exp()
        logits_a2b   = temp * z_a @ z_b.T   # (N, N)
        logits_b2a   = logits_a2b.T         # (N, N)

        return logits_a2b, logits_b2a

    def loss(self, x_a: Tensor, x_b: Tensor) -> Tensor:
        """
        Symmetric cross-modal contrastive loss.
        Labels: i-th pair is matched (positive = diagonal).
        """
        logits_a2b, logits_b2a = self.forward(x_a, x_b)
        N      = x_a.shape[0]
        labels = torch.arange(N, device=x_a.device)
        loss   = 0.5 * (
            F.cross_entropy(logits_a2b, labels)
            + F.cross_entropy(logits_b2a, labels)
        )
        return loss


# Import math at module level for CLIPStyleModel
import math
