"""
InsightSerenity AI Engine — Dimensionality Reduction
=====================================================
Compress high-dimensional data into low-dimensional representations that
preserve the most important structure for visualisation and downstream tasks.

PCA   — Principal Component Analysis.
        Linear method. Finds orthogonal axes of maximum variance.
        Fast, exact, deterministic. Preserves global structure.
        Use for: preprocessing before clustering, visualisation of small-D data.

TSNE  — t-Distributed Stochastic Neighbor Embedding (van der Maaten & Hinton, 2008).
        Non-linear. Optimises a layout that preserves local neighbourhoods.
        Excellent for 2D/3D visualisation of high-dimensional clusters.
        Slow (O(N²)), stochastic, not invertible.

UMAP  — Uniform Manifold Approximation and Projection (McInnes et al., 2018).
        Non-linear. Preserves both local and global structure better than t-SNE.
        Faster than t-SNE, can handle millions of points.
        Requires umap-learn (optional dependency).
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PCA — closed-form via SVD
# ─────────────────────────────────────────────────────────────────────────────

class PCA:
    """
    Principal Component Analysis via Singular Value Decomposition.

    Finds the n_components directions of maximum variance in the data.
    The transformation is linear and invertible (approximately).

    Training: SVD of the centred data matrix X - mean(X).
    The right singular vectors (columns of V^T) are the principal components.

    Properties:
        - Deterministic and exact
        - Preserves global variance structure
        - Components are orthonormal
        - Invertible: inverse_transform recovers approximate original data

    Args:
        n_components: Number of principal components to keep.
        whiten:       If True, scale each component to unit variance.
                      Useful for preprocessing before distance-based methods.
    """

    def __init__(
        self,
        n_components: int,
        whiten: bool = False,
    ) -> None:
        self.n_components = n_components
        self.whiten       = whiten

        # Learned attributes (set during fit)
        self.components_:       Optional[np.ndarray] = None   # (n_components, D)
        self.mean_:             Optional[np.ndarray] = None   # (D,)
        self.explained_variance_: Optional[np.ndarray] = None
        self.singular_values_:  Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, Tensor]) -> "PCA":
        """
        Compute principal components from data X.

        Args:
            X: (N, D) data matrix.

        Returns:
            self.
        """
        X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X, dtype=np.float64)
        N, D = X_np.shape

        # Centre the data
        self.mean_ = X_np.mean(axis=0)
        X_centred  = X_np - self.mean_

        # SVD: X = U S V^T
        # U: (N, N), S: (min(N,D),), V: (D, D)
        U, s, Vt = np.linalg.svd(X_centred, full_matrices=False)

        # Principal components are the rows of Vt (right singular vectors)
        self.components_        = Vt[:self.n_components]                  # (n_components, D)
        self.singular_values_   = s[:self.n_components]
        self.explained_variance_ = (s[:self.n_components] ** 2) / (N - 1)

        total_var = (s ** 2).sum() / (N - 1)
        logger.info(
            "PCA fit",
            n_components=self.n_components,
            explained_variance_ratio=round(self.explained_variance_.sum() / total_var, 4),
        )
        return self

    def transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Project X onto the principal components.

        Args:
            X: (N, D) data matrix.

        Returns:
            (N, n_components) projected data.
        """
        self._check_fitted()
        X_np      = X.numpy() if isinstance(X, Tensor) else np.asarray(X, dtype=np.float64)
        X_centred = X_np - self.mean_
        Z         = X_centred @ self.components_.T   # (N, n_components)

        if self.whiten:
            Z /= (self.singular_values_ + 1e-10)

        return Z

    def fit_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Fit PCA and transform X in one call."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Reconstruct approximate original data from low-dimensional codes.

        Args:
            Z: (N, n_components) projected codes.

        Returns:
            (N, D) approximate reconstruction.
        """
        self._check_fitted()
        Z_np = Z.numpy() if isinstance(Z, Tensor) else np.asarray(Z, dtype=np.float64)

        if self.whiten:
            Z_np = Z_np * self.singular_values_

        return Z_np @ self.components_ + self.mean_

    @property
    def explained_variance_ratio_(self) -> Optional[np.ndarray]:
        """Fraction of variance explained by each component."""
        if self.explained_variance_ is None:
            return None
        return self.explained_variance_ / self.explained_variance_.sum()

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError("Call fit() first")


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────────────────────────────────────

class TSNE:
    """
    t-Distributed Stochastic Neighbor Embedding.

    Non-linear dimensionality reduction designed for 2D/3D visualisation.
    Preserves local neighbourhood structure — nearby points in high-D
    stay nearby in the embedding.

    This is a simplified gradient-based implementation. For production use
    on large datasets, prefer sklearn's implementation which has Barnes-Hut
    approximation for O(N log N) complexity.

    Args:
        n_components:  Embedding dimensions (typically 2 or 3).
        perplexity:    Effective number of local neighbours. Typical: 5–50.
        learning_rate: Gradient step size. Default 200.
        n_iter:        Number of optimisation iterations. Default 1000.
        random_state:  Seed for reproducibility.
    """

    def __init__(
        self,
        n_components:  int   = 2,
        perplexity:    float = 30.0,
        learning_rate: float = 200.0,
        n_iter:        int   = 1000,
        random_state:  Optional[int] = 42,
    ) -> None:
        self.n_components  = n_components
        self.perplexity    = perplexity
        self.learning_rate = learning_rate
        self.n_iter        = n_iter
        self.random_state  = random_state
        self.embedding_:   Optional[np.ndarray] = None

    def fit_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Compute the t-SNE embedding.

        Args:
            X: (N, D) data matrix.

        Returns:
            (N, n_components) embedding coordinates.
        """
        X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X, dtype=np.float64)
        N    = X_np.shape[0]
        rng  = np.random.RandomState(self.random_state)

        # Step 1: Compute pairwise Gaussian affinities P in high-D
        P = self._compute_joint_probabilities(X_np)

        # Step 2: Initialise embedding Y randomly
        Y     = rng.randn(N, self.n_components) * 1e-4
        Y_old = Y.copy()
        gains = np.ones_like(Y)

        for iteration in range(self.n_iter):
            # Compute pairwise Student-t affinities Q in low-D
            Q, Q_sum, dists_sq = self._compute_q_and_grad_terms(Y)

            # Gradient of KL divergence
            grad  = np.zeros_like(Y)
            for i in range(N):
                diff   = Y[i] - Y              # (N, n_components)
                pq_diff = (P[i] - Q[i])[:, None]
                q_inv   = (1.0 / (1.0 + dists_sq[i]))[:, None]
                grad[i] = 4.0 * (pq_diff * q_inv * diff).sum(axis=0)

            # Momentum update with adaptive gains
            gains = np.where(np.sign(grad) == np.sign(Y - Y_old), gains * 0.8, gains + 0.2)
            gains = np.clip(gains, 0.01, np.inf)

            Y_new  = Y - self.learning_rate * gains * grad
            Y_old  = Y.copy()
            Y      = Y_new

            # Early exaggeration (helps clusters separate in early iterations)
            if iteration == 250:
                P /= 4.0

            if (iteration + 1) % 100 == 0:
                kl = self._kl_divergence(P, Q, Q_sum)
                logger.debug("t-SNE iteration", iter=iteration + 1, kl=round(kl, 4))

        self.embedding_ = Y
        return Y

    def _compute_joint_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Compute symmetrised Gaussian affinities P_{ij}."""
        N       = X.shape[0]
        sq_dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)   # (N, N)

        # Find per-point bandwidths via binary search for target perplexity
        P_cond = np.zeros((N, N))
        target_entropy = np.log(self.perplexity)

        for i in range(N):
            dists_i = sq_dists[i].copy()
            dists_i[i] = np.inf

            # Binary search for sigma that gives the target entropy
            beta      = 1.0   # beta = 1/(2*sigma²)
            beta_min  = -np.inf
            beta_max  = np.inf

            for _ in range(50):
                exp_d = np.exp(-beta * dists_i)
                exp_d[i] = 0
                sum_exp   = exp_d.sum() + 1e-12
                H         = np.log(sum_exp) + beta * (dists_i * exp_d).sum() / sum_exp
                Hdiff     = H - target_entropy

                if abs(Hdiff) < 1e-5:
                    break
                if Hdiff > 0:
                    beta_min = beta
                    beta = (beta + beta_max) / 2 if beta_max != np.inf else beta * 2
                else:
                    beta_max = beta
                    beta = (beta + beta_min) / 2 if beta_min != -np.inf else beta / 2

            P_cond[i] = exp_d / sum_exp

        # Symmetrise and normalise
        P = (P_cond + P_cond.T) / (2 * N)
        P = np.maximum(P, 1e-12)
        return P

    def _compute_q_and_grad_terms(
        self, Y: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Compute Student-t affinities Q and squared distances."""
        N        = Y.shape[0]
        dists_sq = np.sum((Y[:, None] - Y[None, :]) ** 2, axis=-1)   # (N, N)
        q_unnorm = 1.0 / (1.0 + dists_sq)
        np.fill_diagonal(q_unnorm, 0)
        q_sum    = q_unnorm.sum()
        Q        = np.maximum(q_unnorm / q_sum, 1e-12)
        return Q, q_sum, dists_sq

    def _kl_divergence(self, P: np.ndarray, Q: np.ndarray, Q_sum: float) -> float:
        mask = P > 1e-12
        return float((P[mask] * np.log(P[mask] / Q[mask])).sum())


# ─────────────────────────────────────────────────────────────────────────────
# UMAP wrapper
# ─────────────────────────────────────────────────────────────────────────────

class UMAP:
    """
    Uniform Manifold Approximation and Projection.

    Wraps the umap-learn library behind a consistent interface.
    Install with: pip install umap-learn

    Better than t-SNE for:
        - Preserving global structure
        - Large datasets (much faster)
        - Supervised/semi-supervised dimensionality reduction

    Args:
        n_components:  Target dimensions.
        n_neighbors:   Size of local neighbourhood. Default 15.
        min_dist:      Minimum distance between embedded points. Default 0.1.
        metric:        Distance metric. Default "euclidean".
        random_state:  Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int   = 2,
        n_neighbors:  int   = 15,
        min_dist:     float = 0.1,
        metric:       str   = "euclidean",
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors  = n_neighbors
        self.min_dist     = min_dist
        self.metric       = metric
        self.random_state = random_state
        self._reducer     = None
        self.embedding_:  Optional[np.ndarray] = None

    def fit_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Fit UMAP and return the embedding."""
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP. Install with: pip install umap-learn"
            )

        X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X)

        self._reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )

        self.embedding_ = self._reducer.fit_transform(X_np)
        return self.embedding_

    def transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Project new data using the fitted UMAP model."""
        if self._reducer is None:
            raise RuntimeError("Call fit_transform() first")
        X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X)
        return self._reducer.transform(X_np)
