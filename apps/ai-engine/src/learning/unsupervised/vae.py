"""
InsightSerenity AI Engine — Variational Autoencoder
====================================================
The VAE extends the standard autoencoder by making the latent space
probabilistic. Instead of encoding x to a single point z, the encoder
outputs parameters of a distribution: μ(x) and σ(x). We then sample z ~ N(μ, σ²).

Why probabilistic?
    Standard AE latent spaces have no structure — points cluster arbitrarily
    with gaps between them. Sampling from those gaps produces gibberish.

    VAE forces the latent space to be a smooth, continuous N(0, I) distribution
    via the KL divergence term. Every point in latent space decodes to something
    meaningful. This enables:
        - Interpolation: lerp between two latent codes = smooth transition
        - Sampling: sample z ~ N(0,1) and decode to generate new examples
        - Structured representation: close latent codes = similar inputs

Training objective (ELBO — Evidence Lower BOund):
    L = E[log p(x|z)] - β × KL(q(z|x) || p(z))
    = reconstruction_loss - β × KL_divergence

    β=1 is the standard VAE (Kingma & Welling, 2013).
    β>1 is β-VAE (Higgins et al., 2017) — stronger disentanglement.

Reparameterization trick:
    z = μ + ε × σ  where ε ~ N(0, I)
    This makes sampling differentiable — gradients flow through z to μ and σ.

Usage:
    vae = VAE(VAEConfig(input_dim=784, latent_dim=32))
    vae.fit(X_train)
    z = vae.encode_sample(X_test)   # Probabilistic encode
    x_hat = vae.decode(z)           # Reconstruct
    samples = vae.sample(n=100)     # Generate 100 new examples
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
class VAEConfig:
    input_dim:   int
    latent_dim:  int
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    activation:  str       = "relu"
    dropout:     float     = 0.1
    beta:        float     = 1.0    # KL weight — β-VAE generalisation
    lr:          float     = 1e-3
    batch_size:  int       = 64
    epochs:      int       = 50
    device:      str       = "cpu"


class VAEEncoder(nn.Module):
    """
    Probabilistic encoder: x → (μ, log σ²).

    Outputs the mean and log-variance of the approximate posterior q(z|x).
    Log-variance rather than variance is used for numerical stability
    (log can be any real value; variance must be positive).
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.shared = MLP(
            in_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.hidden_dims[-1],   # Output of the shared backbone
            activation=config.activation,
            dropout=config.dropout,
        )
        # Two separate linear heads for μ and log σ²
        self.fc_mu      = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.fc_log_var = nn.Linear(config.hidden_dims[-1], config.latent_dim)

        nn.init.normal_(self.fc_mu.weight,      std=0.01)
        nn.init.normal_(self.fc_log_var.weight, std=0.01)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_log_var.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute (μ, log σ²) from input x.

        Args:
            x: (B, input_dim)

        Returns:
            Tuple (mu (B, latent_dim), log_var (B, latent_dim))
        """
        h       = self.shared(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class VAE(nn.Module):
    """
    Variational Autoencoder with reparameterization trick.

    Args:
        config: VAEConfig with all hyperparameters.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = VAEEncoder(config)

        # Decoder: latent → reversed hidden → input
        self.decoder = MLP(
            in_dim=config.latent_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
            out_dim=config.input_dim,
            activation=config.activation,
            dropout=config.dropout,
        )

        self._is_fitted     = False
        self.train_history: List[Dict] = []

    # ── Core operations ────────────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to (mu, log_var) — parameters of q(z|x).

        Args:
            x: (B, input_dim)

        Returns:
            (mu (B, latent_dim), log_var (B, latent_dim))
        """
        return self.encoder(x)

    def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Sample from q(z|x) using the reparameterization trick.

        z = μ + ε × exp(0.5 × log σ²)   where ε ~ N(0, I)

        This makes the sampling operation differentiable so gradients can
        flow from the reconstruction loss back to the encoder parameters.

        Args:
            mu:      (B, latent_dim) — mean of q(z|x)
            log_var: (B, latent_dim) — log variance of q(z|x)

        Returns:
            (B, latent_dim) — sampled latent codes
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # At inference, use the mean for deterministic output
            return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent code z to reconstruction.

        Args:
            z: (B, latent_dim)

        Returns:
            (B, input_dim) — reconstruction x̂
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass: encode → reparameterise → decode.

        Returns:
            Tuple (reconstruction, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decode(z)
        return x_hat, mu, log_var

    # ── Loss ───────────────────────────────────────────────────────────────────

    def loss(self, x: Tensor, x_hat: Tensor, mu: Tensor, log_var: Tensor) -> Dict[str, Tensor]:
        """
        ELBO loss: reconstruction + β × KL divergence.

        Reconstruction: MSE between input and reconstruction.
        KL:             KL(N(μ, σ²) || N(0, 1)) = -0.5 × sum(1 + log σ² - μ² - σ²)

        The KL term pushes the posterior q(z|x) towards the prior N(0, I),
        keeping the latent space compact and well-organised.

        Returns:
            Dict with "total", "reconstruction", "kl" loss tensors.
        """
        recon_loss = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
        kl_loss    = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
        total      = recon_loss + self.config.beta * kl_loss

        return {"total": total, "reconstruction": recon_loss, "kl": kl_loss}

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X: Union[np.ndarray, Tensor]) -> "VAE":
        """
        Train the VAE on data X.

        Args:
            X: (N, input_dim) training data.

        Returns:
            self.
        """
        device = torch.device(self.config.device)
        self.to(device)

        X_t    = self._to_float(X).to(device)
        loader = DataLoader(
            TensorDataset(X_t),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        for epoch in range(self.config.epochs):
            self.train()
            ep_total = ep_recon = ep_kl = 0.0

            for (x_batch,) in loader:
                x_hat, mu, log_var = self.forward(x_batch)
                losses = self.loss(x_batch, x_hat, mu, log_var)

                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                # Clip gradients — VAE training can be unstable early on
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                ep_total += losses["total"].item()
                ep_recon += losses["reconstruction"].item()
                ep_kl    += losses["kl"].item()

            n = max(len(loader), 1)
            record = {
                "epoch":          epoch,
                "total_loss":     ep_total / n,
                "recon_loss":     ep_recon / n,
                "kl_loss":        ep_kl    / n,
            }
            self.train_history.append(record)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "VAE training",
                    epoch=epoch + 1,
                    total=round(record["total_loss"], 4),
                    recon=round(record["recon_loss"], 4),
                    kl=round(record["kl_loss"], 4),
                )

        self._is_fitted = True
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def encode_sample(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Probabilistically encode X (sample from q(z|x)).
        At eval time uses the mean — call with self.training = True for samples.
        """
        self._check_fitted()
        device = torch.device(self.config.device)
        X_t    = self._to_float(X).to(device)
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encode(X_t)
            z = self.reparameterise(mu, log_var)
        return z.cpu().numpy()

    def encode_mean(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Deterministically encode X (returns just the mean μ — no sampling)."""
        self._check_fitted()
        device = torch.device(self.config.device)
        X_t    = self._to_float(X).to(device)
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(X_t)
        return mu.cpu().numpy()

    def reconstruct(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Encode then decode — returns reconstruction of X."""
        self._check_fitted()
        device = torch.device(self.config.device)
        X_t    = self._to_float(X).to(device)
        self.eval()
        with torch.no_grad():
            x_hat, _, _ = self.forward(X_t)
        return x_hat.cpu().numpy()

    def sample(self, n: int = 1) -> np.ndarray:
        """
        Generate n new samples by sampling z ~ N(0, I) and decoding.

        This is the generative capability of the VAE — pure generation
        without any input.

        Args:
            n: Number of samples to generate.

        Returns:
            (n, input_dim) numpy array of generated samples.
        """
        self._check_fitted()
        device = torch.device(self.config.device)
        self.eval()
        with torch.no_grad():
            z     = torch.randn(n, self.config.latent_dim, device=device)
            x_hat = self.decode(z)
        return x_hat.cpu().numpy()

    def interpolate(
        self,
        x_a: Union[np.ndarray, Tensor],
        x_b: Union[np.ndarray, Tensor],
        steps: int = 10,
    ) -> np.ndarray:
        """
        Linearly interpolate in the latent space between two inputs.

        Produces `steps` intermediate reconstructions showing the smooth
        transition from x_a to x_b.

        Args:
            x_a:   First input (1, input_dim).
            x_b:   Second input (1, input_dim).
            steps: Number of interpolation steps.

        Returns:
            (steps, input_dim) array of intermediate reconstructions.
        """
        self._check_fitted()
        device = torch.device(self.config.device)
        xa_t   = self._to_float(x_a).to(device)
        xb_t   = self._to_float(x_b).to(device)

        self.eval()
        with torch.no_grad():
            z_a, _ = self.encode(xa_t.unsqueeze(0) if xa_t.dim() == 1 else xa_t)
            z_b, _ = self.encode(xb_t.unsqueeze(0) if xb_t.dim() == 1 else xb_t)

            interps = []
            for alpha in torch.linspace(0, 1, steps, device=device):
                z     = (1 - alpha) * z_a + alpha * z_b
                x_hat = self.decode(z)
                interps.append(x_hat)

        return torch.cat(interps, dim=0).cpu().numpy()

    @staticmethod
    def _to_float(arr) -> Tensor:
        if isinstance(arr, Tensor):
            return arr.float()
        return torch.tensor(arr, dtype=torch.float32)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
