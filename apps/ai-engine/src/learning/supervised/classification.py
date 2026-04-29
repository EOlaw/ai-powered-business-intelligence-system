"""
InsightSerenity AI Engine — Neural Classifiers
================================================
Trainable neural network classifiers for binary, multi-class, and
multi-label classification tasks. All expose a consistent sklearn-style
fit/predict interface so they can be dropped into any evaluation pipeline.

Three modes:
    binary      — sigmoid output, BCEWithLogitsLoss, labels in {0, 1}
    multiclass  — softmax output, CrossEntropyLoss, labels in {0..C-1}
    multilabel  — sigmoid output per class, BCEWithLogitsLoss, labels in {0,1}^C

Architecture: Configurable MLP backbone from architectures.feedforward.mlp
plus a task-specific output head. Optionally a pretrained feature extractor
can be passed as the backbone (backbone → head pattern).

Usage:
    clf = NeuralClassifier(in_dim=128, num_classes=10, mode="multiclass")
    clf.fit(X_train, y_train, epochs=20)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
"""

import math
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
class ClassifierConfig:
    """Configuration for a NeuralClassifier."""
    in_dim:          int
    num_classes:     int
    hidden_dims:     List[int]    = field(default_factory=lambda: [256, 128])
    mode:            str          = "multiclass"    # binary | multiclass | multilabel
    activation:      str          = "relu"
    dropout:         float        = 0.3
    lr:              float        = 1e-3
    weight_decay:    float        = 1e-4
    batch_size:      int          = 64
    epochs:          int          = 50
    patience:        int          = 10              # Early stopping patience
    device:          str          = "cpu"


class NeuralClassifier:
    """
    Neural network classifier with fit/predict/predict_proba interface.

    The classifier is built on top of an MLP backbone. The final linear
    layer maps the last hidden representation to class logits.

    Args:
        config: ClassifierConfig with all hyperparameters.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        self.config   = config
        self.device   = torch.device(config.device)

        # Output dimension: 1 for binary, C for multiclass/multilabel
        out_dim = 1 if config.mode == "binary" else config.num_classes

        self.model = MLP(
            in_dim=config.in_dim,
            hidden_dims=config.hidden_dims,
            out_dim=out_dim,
            activation=config.activation,
            dropout=config.dropout,
        ).to(self.device)

        self._is_fitted = False
        self.train_history: List[Dict[str, float]] = []

    # ── Sklearn-style API ──────────────────────────────────────────────────────

    def fit(
        self,
        X: Union[np.ndarray, Tensor],
        y: Union[np.ndarray, Tensor],
        val_data: Optional[Tuple] = None,
    ) -> "NeuralClassifier":
        """
        Train the classifier on (X, y) pairs.

        Args:
            X:        Feature matrix. Shape: (N, in_dim).
            y:        Labels. Shape: (N,) for binary/multiclass,
                      (N, num_classes) for multilabel.
            val_data: Optional (X_val, y_val) tuple for early stopping.

        Returns:
            self (for chaining).
        """
        X_t, y_t = self._to_tensors(X, y)
        dataset   = TensorDataset(X_t, y_t)
        loader    = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        val_loader = None
        if val_data is not None:
            Xv, yv    = self._to_tensors(*val_data)
            val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=self.config.batch_size)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        loss_fn   = self._get_loss_fn()

        best_val_loss = float("inf")
        patience_ctr  = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss   = loss_fn(logits, self._format_labels(y_batch, logits))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)

            # Validation and early stopping
            val_loss = None
            if val_loader:
                val_loss = self._eval_loss(val_loader, loss_fn)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_ctr  = 0
                else:
                    patience_ctr += 1

                if patience_ctr >= self.config.patience:
                    logger.info("Early stopping", epoch=epoch)
                    break

            self.train_history.append({"epoch": epoch, "loss": avg_loss, "val_loss": val_loss})

            if (epoch + 1) % 10 == 0:
                logger.info("Training", epoch=epoch + 1, loss=round(avg_loss, 4),
                            val_loss=round(val_loss, 4) if val_loss else None)

        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Return predicted class indices (or binary 0/1).

        Args:
            X: (N, in_dim) feature matrix.

        Returns:
            (N,) integer array of predictions.
        """
        proba = self.predict_proba(X)

        if self.config.mode == "binary":
            return (proba >= 0.5).astype(int).squeeze(-1)
        elif self.config.mode == "multilabel":
            return (proba >= 0.5).astype(int)
        else:
            return proba.argmax(axis=-1)

    def predict_proba(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Return class probabilities.

        Returns:
            For binary:     (N, 1) sigmoid probabilities.
            For multiclass: (N, C) softmax probabilities.
            For multilabel: (N, C) per-class sigmoid probabilities.
        """
        self._check_fitted()
        X_t = self._to_tensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            if self.config.mode == "binary":
                proba = torch.sigmoid(logits)
            elif self.config.mode == "multilabel":
                proba = torch.sigmoid(logits)
            else:
                proba = F.softmax(logits, dim=-1)

        return proba.cpu().numpy()

    def score(self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> float:
        """Return accuracy (for multiclass) or mean per-class accuracy (for multilabel)."""
        preds  = self.predict(X)
        labels = y.numpy() if isinstance(y, Tensor) else y
        if self.config.mode == "multilabel":
            return float((preds == labels).all(axis=1).mean())
        return float((preds == labels.squeeze()).mean())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_loss_fn(self):
        if self.config.mode == "binary":
            return nn.BCEWithLogitsLoss()
        elif self.config.mode == "multilabel":
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def _format_labels(self, y: Tensor, logits: Tensor) -> Tensor:
        """Ensure labels are in the correct dtype and shape for the loss."""
        if self.config.mode == "binary":
            return y.float().view_as(logits)
        if self.config.mode == "multilabel":
            return y.float()
        return y.long()

    def _eval_loss(self, loader: DataLoader, loss_fn) -> float:
        self.model.eval()
        total = 0.0
        n     = 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)
                out  = self.model(X_b)
                loss = loss_fn(out, self._format_labels(y_b, out))
                total += loss.item()
                n += 1
        return total / max(n, 1)

    def _to_tensors(self, X, y):
        return self._to_tensor(X), self._to_tensor(y)

    def _to_tensor(self, arr) -> Tensor:
        if isinstance(arr, Tensor):
            return arr.float()
        return torch.tensor(arr, dtype=torch.float32)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors
# ─────────────────────────────────────────────────────────────────────────────

def binary_classifier(in_dim: int, hidden_dims: Optional[List[int]] = None, **kw) -> NeuralClassifier:
    """Create a binary classifier (sigmoid output, BCEWithLogitsLoss)."""
    cfg = ClassifierConfig(in_dim=in_dim, num_classes=1, mode="binary",
                           hidden_dims=hidden_dims or [128, 64], **kw)
    return NeuralClassifier(cfg)


def multiclass_classifier(in_dim: int, num_classes: int, hidden_dims: Optional[List[int]] = None, **kw) -> NeuralClassifier:
    """Create a multi-class classifier (softmax output, CrossEntropyLoss)."""
    cfg = ClassifierConfig(in_dim=in_dim, num_classes=num_classes, mode="multiclass",
                           hidden_dims=hidden_dims or [256, 128], **kw)
    return NeuralClassifier(cfg)


def multilabel_classifier(in_dim: int, num_classes: int, hidden_dims: Optional[List[int]] = None, **kw) -> NeuralClassifier:
    """Create a multi-label classifier (per-class sigmoid, BCEWithLogitsLoss)."""
    cfg = ClassifierConfig(in_dim=in_dim, num_classes=num_classes, mode="multilabel",
                           hidden_dims=hidden_dims or [256, 128], **kw)
    return NeuralClassifier(cfg)
