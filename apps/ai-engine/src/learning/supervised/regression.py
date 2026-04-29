"""
InsightSerenity AI Engine — Neural Regressors
==============================================
Trainable regression models predicting continuous scalar or vector outputs.

Three implementations:
    LinearRegressor       — OLS closed-form solution (no gradient descent).
                            Fast exact solution for well-conditioned problems.
    PolynomialRegressor   — Feature expansion to degree d, then OLS.
                            Fits non-linear relationships without a neural net.
    NeuralRegressor       — Configurable MLP with linear output head.
                            Handles complex non-linear patterns. Fits arbitrary
                            input/output dimensions.

All expose fit(X, y) / predict(X) interface.
NeuralRegressor additionally supports multi-output regression (y is a matrix).
"""

import math
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.architectures.feedforward.mlp import MLP
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Closed-form linear regression (OLS)
# ─────────────────────────────────────────────────────────────────────────────

class LinearRegressor:
    """
    Ordinary Least Squares linear regression solved via the normal equations.

    The closed-form solution: w = (X^T X + λI)^{-1} X^T y
    where λ is L2 regularisation (ridge regression). λ=0 is standard OLS.

    Advantages over gradient descent:
        - Exact solution (no convergence issues)
        - No learning rate tuning
        - Fast for small datasets

    Args:
        alpha: L2 regularisation strength (ridge penalty). Default 1e-4.
               Prevents singular (X^T X) for collinear features.
        fit_intercept: Add a bias column. Default True.
    """

    def __init__(self, alpha: float = 1e-4, fit_intercept: bool = True) -> None:
        self.alpha         = alpha
        self.fit_intercept = fit_intercept
        self.weights_: Optional[np.ndarray] = None
        self.bias_:    Optional[float]      = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        """
        Solve the normal equations.

        Args:
            X: (N, D) feature matrix.
            y: (N,) or (N, K) target values.

        Returns:
            self.
        """
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # (X^T X + λI)^{-1} X^T y
        XtX    = X.T @ X
        reg    = self.alpha * np.eye(XtX.shape[0])
        w      = np.linalg.solve(XtX + reg, X.T @ y)

        if self.fit_intercept:
            self.bias_    = w[0] if y.ndim == 1 else w[0:1]
            self.weights_ = w[1:]
        else:
            self.weights_ = w
            self.bias_    = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values."""
        if self.weights_ is None:
            raise RuntimeError("Call fit() first")
        return X @ self.weights_ + self.bias_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² coefficient of determination."""
        y_hat = self.predict(X)
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / (ss_tot + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial regression
# ─────────────────────────────────────────────────────────────────────────────

class PolynomialRegressor:
    """
    Polynomial regression via feature expansion + LinearRegressor.

    Expands input features to all polynomial terms up to degree d,
    then fits a linear model on the expanded features.

    Example (degree=2, features [x1, x2]):
        Expanded: [1, x1, x2, x1², x1·x2, x2²]

    Warning: feature expansion is O(D^degree) — avoid large D or degree > 3.

    Args:
        degree:        Polynomial degree. Default 2.
        alpha:         L2 regularisation for the linear solver.
        fit_intercept: Add bias column.
    """

    def __init__(self, degree: int = 2, alpha: float = 1e-4, fit_intercept: bool = True) -> None:
        self.degree     = degree
        self._linear    = LinearRegressor(alpha=alpha, fit_intercept=fit_intercept)
        self._n_input_features: Optional[int] = None

    def _expand(self, X: np.ndarray) -> np.ndarray:
        """Expand features to polynomial terms."""
        n, d = X.shape
        cols  = [np.ones((n, 1))]

        for deg in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(d), deg):
                cols.append(X[:, combo].prod(axis=1, keepdims=True))

        return np.hstack(cols[:])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegressor":
        self._n_input_features = X.shape[1]
        X_poly = self._expand(X)
        self._linear.fit_intercept = False   # Intercept already in expansion
        self._linear.fit(X_poly, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._linear.predict(self._expand(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._linear.score(self._expand(X), y)


# ─────────────────────────────────────────────────────────────────────────────
# Neural regressor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressorConfig:
    in_dim:       int
    out_dim:      int            = 1
    hidden_dims:  List[int]      = field(default_factory=lambda: [256, 128])
    activation:   str            = "relu"
    dropout:      float          = 0.2
    loss:         str            = "mse"     # "mse" | "huber" | "mae"
    lr:           float          = 1e-3
    weight_decay: float          = 1e-4
    batch_size:   int            = 64
    epochs:       int            = 100
    patience:     int            = 15
    device:       str            = "cpu"


class NeuralRegressor:
    """
    Neural network regressor with linear output head and MSE/Huber loss.

    Handles single-output (scalar) and multi-output (vector) regression.
    The output layer has no activation — it produces raw predicted values.

    Args:
        config: RegressorConfig with all hyperparameters.
    """

    def __init__(self, config: RegressorConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.model = MLP(
            in_dim=config.in_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.out_dim,
            activation=config.activation,
            dropout=config.dropout,
        ).to(self.device)

        self._is_fitted   = False
        self.train_history: List[Dict] = []

    def fit(
        self,
        X: Union[np.ndarray, Tensor],
        y: Union[np.ndarray, Tensor],
        val_data: Optional[Tuple] = None,
    ) -> "NeuralRegressor":
        """
        Train on (X, y) pairs.

        Args:
            X:        (N, in_dim) features.
            y:        (N,) or (N, out_dim) target values.
            val_data: Optional (X_val, y_val) for early stopping.
        """
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(-1)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader = None
        if val_data is not None:
            Xv = self._to_tensor(val_data[0])
            yv = self._to_tensor(val_data[1])
            if yv.dim() == 1:
                yv = yv.unsqueeze(-1)
            val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=self.config.batch_size)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = self._get_loss_fn()

        best_val = float("inf")
        patience = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            ep_loss = 0.0

            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                pred      = self.model(X_b)
                loss      = loss_fn(pred, y_b)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

            avg = ep_loss / max(len(loader), 1)
            val_loss = None

            if val_loader:
                val_loss = self._eval_loss(val_loader, loss_fn)
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    patience  = 0
                else:
                    patience += 1
                if patience >= self.config.patience:
                    logger.info("Early stopping", epoch=epoch)
                    break

            self.train_history.append({"epoch": epoch, "loss": avg, "val_loss": val_loss})

        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Return predicted values as a numpy array."""
        self._check_fitted()
        X_t = self._to_tensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_t)
        result = out.cpu().numpy()
        return result.squeeze(-1) if result.shape[-1] == 1 else result

    def score(self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> float:
        """Return R² score."""
        y_hat  = self.predict(X)
        y_arr  = y.numpy() if isinstance(y, Tensor) else np.asarray(y)
        ss_res = ((y_arr - y_hat) ** 2).sum()
        ss_tot = ((y_arr - y_arr.mean()) ** 2).sum()
        return float(1 - ss_res / (ss_tot + 1e-10))

    def _get_loss_fn(self):
        if self.config.loss == "huber":
            return nn.HuberLoss()
        elif self.config.loss == "mae":
            return nn.L1Loss()
        return nn.MSELoss()

    def _eval_loss(self, loader, loss_fn):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in loader:
                total += loss_fn(self.model(X_b.to(self.device)), y_b.to(self.device)).item()
                n += 1
        return total / max(n, 1)

    def _to_tensor(self, arr) -> Tensor:
        if isinstance(arr, Tensor):
            return arr.float()
        return torch.tensor(arr, dtype=torch.float32)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
