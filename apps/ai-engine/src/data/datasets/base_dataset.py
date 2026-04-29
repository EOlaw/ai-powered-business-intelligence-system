"""
InsightSerenity AI Engine — Base Dataset
=========================================
Abstract base class that every dataset in the platform must inherit from.
Defines the contract between the training infrastructure and data sources,
ensuring the Trainer can work with any dataset without knowing its internals.

Inherits from torch.utils.data.Dataset so all datasets are natively
compatible with PyTorch's DataLoader for batching, shuffling, and
parallel loading.

Design principles:
    - Abstract: subclasses must implement __len__ and __getitem__
    - Rich metadata: every dataset exposes its size, column names, and source
    - Split-aware: built-in train/validation/test split support
    - Serialisable: datasets can save/load their state for reproducibility
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset metadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    """
    Metadata about a dataset. Stored alongside the dataset and used by the
    training infrastructure for logging and reproducibility.

    Attributes:
        name:         Short identifier (e.g. "web-corpus-v1").
        description:  Human-readable description of the data source.
        num_examples: Total number of examples.
        features:     Dict mapping feature name → description or dtype.
        source:       Where the data came from (URL, file path, etc.).
        version:      Dataset version string.
        created_at:   ISO timestamp of when the dataset was created.
        splits:       Dict mapping split name → number of examples.
    """
    name:         str             = "unknown"
    description:  str             = ""
    num_examples: int             = 0
    features:     Dict[str, Any]  = field(default_factory=dict)
    source:       str             = ""
    version:      str             = "1.0.0"
    created_at:   str             = ""
    splits:       Dict[str, int]  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all InsightSerenity datasets.

    All concrete datasets must implement:
        __len__()      → total number of examples
        __getitem__()  → single example as a dict of tensors or raw values

    Subclasses may optionally override:
        info()         → DatasetInfo metadata
        column_names() → list of output feature names
    """

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of examples in this dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return a single example by index.

        Returns:
            Dict mapping feature name → value (tensor or primitive).
            The keys must be consistent across all examples.
        """
        ...

    # ── Optional overrides ────────────────────────────────────────────────────

    @property
    def info(self) -> DatasetInfo:
        """Return metadata describing this dataset."""
        return DatasetInfo(num_examples=len(self))

    @property
    def column_names(self) -> List[str]:
        """Return the list of feature names produced by __getitem__."""
        if len(self) == 0:
            return []
        sample = self[0]
        return list(sample.keys())

    # ── Splitting ─────────────────────────────────────────────────────────────

    def split(
        self,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
    ) -> Tuple["SubsetDataset", "SubsetDataset", "SubsetDataset"]:
        """
        Randomly split this dataset into train, validation, and test subsets.

        The ratios must sum to 1.0. This is a static split — the same seed
        always produces the same split, ensuring reproducibility.

        Args:
            train_ratio: Fraction of data for training.
            val_ratio:   Fraction of data for validation.
            test_ratio:  Fraction of data for testing.
            seed:        Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            "Split ratios must sum to 1.0"
        )

        n       = len(self)
        indices = list(range(n))

        rng = random.Random(seed)
        rng.shuffle(indices)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_indices = indices[:n_train]
        val_indices   = indices[n_train:n_train + n_val]
        test_indices  = indices[n_train + n_val:]

        logger.info(
            "Dataset split",
            total=n,
            train=len(train_indices),
            val=len(val_indices),
            test=len(test_indices),
        )

        return (
            SubsetDataset(self, train_indices),
            SubsetDataset(self, val_indices),
            SubsetDataset(self, test_indices),
        )

    def take(self, n: int) -> "SubsetDataset":
        """Return a SubsetDataset with only the first `n` examples."""
        return SubsetDataset(self, list(range(min(n, len(self)))))

    def shuffle(self, seed: int = 42) -> "SubsetDataset":
        """Return a shuffled view of this dataset."""
        indices = list(range(len(self)))
        random.Random(seed).shuffle(indices)
        return SubsetDataset(self, indices)

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Allow iterating over the dataset with a for loop."""
        for i in range(len(self)):
            yield self[i]

    # ── Representation ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_examples={len(self)}, "
            f"columns={self.column_names})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SubsetDataset — index-based view into another dataset
# ─────────────────────────────────────────────────────────────────────────────

class SubsetDataset(BaseDataset):
    """
    A view into a subset of another dataset, defined by a list of indices.

    Used internally by BaseDataset.split() and BaseDataset.take().
    Can also be used directly when you have pre-computed splits.

    Args:
        dataset: The source dataset to index into.
        indices: List of integer indices to include in this subset.
    """

    def __init__(self, dataset: BaseDataset, indices: List[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._dataset[self._indices[index]]

    @property
    def column_names(self) -> List[str]:
        return self._dataset.column_names
