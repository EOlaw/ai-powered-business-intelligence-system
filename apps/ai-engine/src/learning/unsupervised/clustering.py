"""
InsightSerenity AI Engine — Clustering Algorithms
===================================================
Unsupervised grouping of data points by similarity without predefined labels.

Three algorithms with distinct assumptions and use cases:

KMeans          — partitional, assumes spherical clusters of similar size.
                  Best when you know K and clusters are roughly convex.
                  Time: O(N × K × D × iterations).

DBSCAN          — density-based, finds arbitrarily shaped clusters.
                  Identifies noise points (label=-1) automatically.
                  Best for spatial data with varying cluster shapes.
                  Does not require specifying K upfront.

AgglomerativeClustering — hierarchical, builds a tree of merges (dendrogram).
                  Cuts the tree at a chosen level to get flat clusters.
                  Best for understanding cluster structure at multiple scales.

All follow: fit(X) → predict(X) → fit_predict(X)
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# K-Means
# ─────────────────────────────────────────────────────────────────────────────

class KMeans:
    """
    K-Means clustering with k-means++ initialisation.

    k-means++ selects initial centroids that are spread far apart,
    avoiding the bad convergence that random initialisation can cause.

    Algorithm (Lloyd's):
        1. Initialise K centroids via k-means++
        2. Assign each point to nearest centroid (E-step)
        3. Update centroids as mean of assigned points (M-step)
        4. Repeat 2-3 until convergence or max_iter

    Args:
        n_clusters: K — number of clusters.
        max_iter:   Maximum iterations. Default 300.
        tol:        Convergence tolerance (centroid shift norm). Default 1e-4.
        n_init:     Number of random restarts — keeps the best result. Default 10.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters:   int,
        max_iter:     int   = 300,
        tol:          float = 1e-4,
        n_init:       int   = 10,
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.tol          = tol
        self.n_init       = n_init
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_:          Optional[np.ndarray] = None
        self.inertia_:         float                 = float("inf")
        self.n_iter_:          int                   = 0

    def fit(self, X: Union[np.ndarray, Tensor]) -> "KMeans":
        """
        Fit K-Means on the data matrix X.

        Runs n_init restarts and keeps the one with lowest inertia
        (sum of squared distances to nearest centroid).
        """
        X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X, dtype=np.float64)
        rng  = np.random.RandomState(self.random_state)

        best_centers  = None
        best_labels   = None
        best_inertia  = float("inf")
        best_n_iter   = 0

        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._run_once(X_np, rng)
            if inertia < best_inertia:
                best_inertia  = inertia
                best_centers  = centers
                best_labels   = labels
                best_n_iter   = n_iter

        self.cluster_centers_ = best_centers
        self.labels_          = best_labels
        self.inertia_         = best_inertia
        self.n_iter_          = best_n_iter
        return self

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Assign each sample in X to the nearest cluster centroid."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit() first")
        X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X, dtype=np.float64)
        return self._assign(X_np, self.cluster_centers_)

    def fit_predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    def _run_once(self, X: np.ndarray, rng: np.random.RandomState) -> Tuple:
        """One full K-Means run (init + Lloyd iterations)."""
        N, D    = X.shape
        centers = self._kmeanspp_init(X, rng)

        for iteration in range(self.max_iter):
            labels = self._assign(X, centers)

            # Update centroids
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # Empty cluster: re-initialise to a random point
                    new_centers[k] = X[rng.randint(N)]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if shift < self.tol:
                break

        labels   = self._assign(X, centers)
        inertia  = sum(
            ((X[labels == k] - centers[k]) ** 2).sum()
            for k in range(self.n_clusters)
        )
        return centers, labels, inertia, iteration + 1

    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """K-Means++ centroid initialisation."""
        N = X.shape[0]
        # First centroid: uniform random
        centers = [X[rng.randint(N)]]

        for _ in range(self.n_clusters - 1):
            # Compute distance from each point to nearest existing centroid
            dists = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centers)
                for x in X
            ])
            # Sample next centroid proportionally to distance squared
            probs = dists / dists.sum()
            idx   = rng.choice(N, p=probs)
            centers.append(X[idx])

        return np.array(centers)

    def _assign(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid by Euclidean distance."""
        # (N, K) distance matrix
        diffs = X[:, None, :] - centers[None, :, :]   # (N, K, D)
        dists = (diffs ** 2).sum(axis=-1)              # (N, K)
        return dists.argmin(axis=-1)                   # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Finds clusters as dense regions separated by low-density regions.
    Points in sparse regions are labelled as noise (label = -1).

    Key parameters:
        eps:       Neighbourhood radius. Points within eps of a core point
                   are in its neighbourhood.
        min_samples: Minimum neighbours for a point to be a core point.

    Algorithm:
        1. For each unvisited point, find all points within eps.
        2. If >= min_samples neighbours → core point → start new cluster.
        3. Expand the cluster by recursively adding reachable core points.
        4. Non-core points reachable from a cluster join it as border points.
        5. Points reachable from no core point are noise (label = -1).

    Args:
        eps:         Neighbourhood radius.
        min_samples: Minimum neighbours to be a core point.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        self.eps         = eps
        self.min_samples = min_samples
        self.labels_:    Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None

    def fit_predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Fit and return cluster labels (-1 = noise)."""
        X_np     = X.numpy() if isinstance(X, Tensor) else np.asarray(X, dtype=np.float64)
        N        = X_np.shape[0]
        labels   = np.full(N, -1, dtype=int)
        visited  = np.zeros(N, dtype=bool)
        cluster_id = 0

        # Precompute pairwise distances (O(N²) — fine for N < 10k)
        dists = np.sum((X_np[:, None] - X_np[None, :]) ** 2, axis=-1) ** 0.5

        for i in range(N):
            if visited[i]:
                continue
            visited[i] = True

            neighbours = np.where(dists[i] <= self.eps)[0]

            if len(neighbours) < self.min_samples:
                labels[i] = -1   # Noise
            else:
                self._expand_cluster(i, neighbours, labels, visited, dists, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array([
            i for i in range(N)
            if (dists[i] <= self.eps).sum() >= self.min_samples
        ])
        return self.labels_

    def fit(self, X: Union[np.ndarray, Tensor]) -> "DBSCAN":
        self.fit_predict(X)
        return self

    def _expand_cluster(
        self, point: int, neighbours: np.ndarray,
        labels: np.ndarray, visited: np.ndarray,
        dists: np.ndarray, cluster_id: int,
    ) -> None:
        labels[point] = cluster_id
        seed_set      = list(neighbours)
        idx           = 0

        while idx < len(seed_set):
            q = seed_set[idx]
            if not visited[q]:
                visited[q] = True
                q_neighbours = np.where(dists[q] <= self.eps)[0]
                if len(q_neighbours) >= self.min_samples:
                    seed_set.extend(q_neighbours.tolist())

            if labels[q] == -1:
                labels[q] = cluster_id   # Border point
            idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# Agglomerative Clustering
# ─────────────────────────────────────────────────────────────────────────────

class AgglomerativeClustering:
    """
    Bottom-up hierarchical clustering via agglomeration.

    Starts with N clusters (one per point) and iteratively merges the two
    most similar clusters until only `n_clusters` remain.

    Linkage criteria determine cluster similarity:
        single:   min distance between any pair of points (chaining effect)
        complete: max distance between any pair (compact clusters)
        average:  mean distance between all pairs (balanced)
        ward:     minimise within-cluster variance increase (default, best for most tasks)

    Args:
        n_clusters: Target number of clusters.
        linkage:    "single" | "complete" | "average" | "ward". Default "ward".
    """

    def __init__(self, n_clusters: int = 2, linkage: str = "ward") -> None:
        if linkage not in ("single", "complete", "average", "ward"):
            raise ValueError(f"Invalid linkage '{linkage}'")
        self.n_clusters = n_clusters
        self.linkage    = linkage
        self.labels_:   Optional[np.ndarray] = None

    def fit_predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Fit and return cluster labels."""
        X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X, dtype=np.float64)
        N    = X_np.shape[0]

        # Initial clusters: each point is its own cluster
        clusters: List[List[int]] = [[i] for i in range(N)]

        # Precompute pairwise distances
        dists = np.sum((X_np[:, None] - X_np[None, :]) ** 2, axis=-1) ** 0.5

        while len(clusters) > self.n_clusters:
            # Find the two closest clusters
            best_i, best_j, best_d = 0, 1, float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._cluster_distance(clusters[i], clusters[j], dists, X_np)
                    if d < best_d:
                        best_d  = d
                        best_i  = i
                        best_j  = j

            # Merge best_j into best_i
            clusters[best_i] = clusters[best_i] + clusters[best_j]
            clusters.pop(best_j)

        # Assign integer labels
        labels = np.zeros(N, dtype=int)
        for cluster_id, members in enumerate(clusters):
            for m in members:
                labels[m] = cluster_id

        self.labels_ = labels
        return labels

    def fit(self, X: Union[np.ndarray, Tensor]) -> "AgglomerativeClustering":
        self.fit_predict(X)
        return self

    def _cluster_distance(
        self, ci: List[int], cj: List[int],
        dists: np.ndarray, X: np.ndarray,
    ) -> float:
        pair_dists = dists[np.ix_(ci, cj)].flatten()

        if self.linkage == "single":
            return float(pair_dists.min())
        elif self.linkage == "complete":
            return float(pair_dists.max())
        elif self.linkage == "average":
            return float(pair_dists.mean())
        else:   # Ward: increase in total within-cluster variance
            merged = ci + cj
            combined_mean = X[merged].mean(axis=0)
            var_merged    = ((X[merged] - combined_mean) ** 2).sum()
            var_ci        = ((X[ci] - X[ci].mean(axis=0)) ** 2).sum()
            var_cj        = ((X[cj] - X[cj].mean(axis=0)) ** 2).sum()
            return float(var_merged - var_ci - var_cj)
