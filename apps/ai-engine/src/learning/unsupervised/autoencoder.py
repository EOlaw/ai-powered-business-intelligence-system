"""
InsightSerenity AI Engine — Autoencoders
=========================================
Autoencoders learn compressed representations (encodings) of data by training
to reconstruct the input through a bottleneck:

    Encoder:  x → z           (compress to latent space)
    Decoder:  z → x̂           (reconstruct from latent space)
    Loss:     ||x - x̂||²      (reconstruction error)

Because the bottleneck forces the model to keep only the most important
information, the latent representation z captures the essential structure.

Two variants:

Autoencoder           — Deterministic encoder/decoder. Simple, fast.
                        Use for: dimensionality reduction, anomaly detection,
                        feature extraction.

DenoisingAutoencoder  — Adds noise to the input before encoding, trains to
                        reconstruct the CLEAN original. Forces the model to
                        learn robust features that ignore noise.
                        Use for: denoising images/text, learning noise-invariant
                        representations.

Both share the same architecture — the only difference is data corruption
applied in the DAE's training loop.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.architectures.feedforward.mlp import MLP
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AutoencoderConfig:
    input_dim:   int
    latent_dim:  int
    hidden_dims: List[int]  = field(default_factory=lambda: [256, 128])
    activation:  str        = "relu"
    dropout:     float      = 0.1
    lr:          float      = 1e-3
    batch_size:  int        = 64
    epochs:      int        = 50
    device:      str        = "cpu"
    # Denoising-specific
    noise_type:  str        = "gaussian"   # "gaussian" | "dropout" | "salt_pepper"
    noise_factor: float     = 0.1


class Autoencoder(nn.Module):
    """
    Standard deterministic autoencoder.

    Architecture:
        Encoder: input_dim → hidden_dims → latent_dim  (with ReLU activations)
        Decoder: latent_dim → reversed(hidden_dims) → input_dim  (sigmoid output)

    The decoder mirrors the encoder architecture with reversed hidden dimensions.

    Args:
        config: AutoencoderConfig.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder: input → hidden → latent
        self.encoder = MLP(
            in_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.latent_dim,
            activation=config.activation,
            dropout=config.dropout,
        )

        # Decoder: latent → reversed hidden → input
        self.decoder = MLP(
            in_dim=config.latent_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
            out_dim=config.input_dim,
            activation=config.activation,
            dropout=config.dropout,
        )

        self._is_fitted = False
        self.train_history: List[Dict] = []

    # ── Forward ────────────────────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        """
        Map input to the latent space.

        Args:
            x: (B, input_dim) input features.

        Returns:
            (B, latent_dim) latent representation.
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """
        Reconstruct input from latent code.

        Args:
            z: (B, latent_dim) latent code.

        Returns:
            (B, input_dim) reconstructed input.
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode then decode.

        Returns:
            Tuple (reconstruction, latent_code).
        """
        z    = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruct(self, x: Tensor) -> Tensor:
        """Convenience: forward pass returning only the reconstruction."""
        x_hat, _ = self.forward(x)
        return x_hat

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X: Union[np.ndarray, Tensor],
        corrupted_X: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> "Autoencoder":
        """
        Train the autoencoder to minimise reconstruction loss.

        Args:
            X:           (N, input_dim) training data (clean targets).
            corrupted_X: Optional (N, input_dim) corrupted inputs.
                         If provided, trains on corrupted → clean (DAE mode).
                         If None, trains on clean → clean (standard AE).
        """
        device = torch.device(self.config.device)
        self.to(device)

        X_t     = self._to_float(X).to(device)
        input_t = self._to_float(corrupted_X).to(device) if corrupted_X is not None else X_t

        loader = DataLoader(
            TensorDataset(input_t, X_t),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        for epoch in range(self.config.epochs):
            self.train()
            ep_loss = 0.0

            for x_in, x_target in loader:
                x_hat, _ = self.forward(x_in)
                loss      = F.mse_loss(x_hat, x_target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

            avg = ep_loss / max(len(loader), 1)
            self.train_history.append({"epoch": epoch, "loss": avg})

            if (epoch + 1) % 10 == 0:
                logger.info("Autoencoder training", epoch=epoch + 1, loss=round(avg, 6))

        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Encode X into the latent space. Returns numpy array."""
        self._check_fitted()
        device = torch.device(self.config.device)
        X_t    = self._to_float(X).to(device)
        self.eval()
        with torch.no_grad():
            z = self.encode(X_t)
        return z.cpu().numpy()

    def reconstruction_error(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Per-sample reconstruction error (MSE). Useful for anomaly detection:
        anomalies have higher reconstruction error than normal samples.
        """
        self._check_fitted()
        device = torch.device(self.config.device)
        X_t    = self._to_float(X).to(device)
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(X_t)
            errors    = F.mse_loss(x_hat, X_t, reduction="none").mean(dim=-1)
        return errors.cpu().numpy()

    @staticmethod
    def _to_float(arr) -> Tensor:
        if isinstance(arr, Tensor):
            return arr.float()
        return torch.tensor(arr, dtype=torch.float32)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")


class DenoisingAutoencoder(Autoencoder):
    """
    Denoising Autoencoder: learns to reconstruct clean input from corrupted input.

    The corruption forces the model to learn representations that are robust
    to noise — it can't just copy the input through the bottleneck.

    Three noise types:
        gaussian:    Add N(0, σ²) noise to each feature
        dropout:     Zero out random features with probability p
        salt_pepper: Set random features to min or max values

    Args:
        config: AutoencoderConfig (noise_type and noise_factor used here).
    """

    def corrupt(self, x: Tensor) -> Tensor:
        """
        Apply noise corruption to the input tensor.

        Args:
            x: (B, input_dim) clean input.

        Returns:
            (B, input_dim) corrupted input.
        """
        if self.config.noise_type == "gaussian":
            noise = torch.randn_like(x) * self.config.noise_factor
            return x + noise

        elif self.config.noise_type == "dropout":
            mask = torch.bernoulli(
                torch.full_like(x, 1.0 - self.config.noise_factor)
            )
            return x * mask

        elif self.config.noise_type == "salt_pepper":
            mask_salt   = torch.bernoulli(torch.full_like(x, self.config.noise_factor / 2))
            mask_pepper = torch.bernoulli(torch.full_like(x, self.config.noise_factor / 2))
            x           = x.clone()
            x[mask_salt.bool()]   = x.max()
            x[mask_pepper.bool()] = x.min()
            return x

        return x

    def fit(self, X: Union[np.ndarray, Tensor], corrupted_X=None) -> "DenoisingAutoencoder":
        """
        Train with automatic corruption. Clean X is provided; corruption
        is generated internally on each batch for fresh noise every epoch.
        """
        device = torch.device(self.config.device)
        self.to(device)

        X_t  = self._to_float(X).to(device)
        loader = DataLoader(
            TensorDataset(X_t),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        for epoch in range(self.config.epochs):
            self.train()
            ep_loss = 0.0

            for (x_clean,) in loader:
                # Fresh corruption each batch
                x_noisy   = self.corrupt(x_clean)
                x_hat, _  = self.forward(x_noisy)
                loss      = F.mse_loss(x_hat, x_clean)   # Reconstruct CLEAN

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

            avg = ep_loss / max(len(loader), 1)
            self.train_history.append({"epoch": epoch, "loss": avg})

            if (epoch + 1) % 10 == 0:
                logger.info("DAE training", epoch=epoch + 1, loss=round(avg, 6))

        self._is_fitted = True
        return self
