"""
InsightSerenity AI Engine — Optimizers
=======================================
Optimizers update model parameters using gradients to minimise the loss.
Each optimizer implements a specific update rule with different trade-offs
between convergence speed, memory usage, and generalisation.

Optimizers implemented:

1. SGD (Stochastic Gradient Descent)
   θ ← θ - lr * (∇L + weight_decay * θ)    [with optional momentum]
   The simplest optimizer. With momentum and proper tuning it can match
   Adam on many tasks. Baseline for all comparisons.

2. Adam (Kingma & Ba, 2014)
   Maintains per-parameter first moment (m) and second moment (v)
   estimates of the gradient. Adapts the learning rate per-parameter.
   θ ← θ - lr * m̂ / (sqrt(v̂) + ε)
   The default choice for most deep learning. Works well out of the box
   but can overfit more than AdamW.

3. AdamW (Loshchilov & Hutter, 2017)
   Adam with DECOUPLED weight decay — weight decay is applied directly
   to parameters, not through the gradient.
   θ ← θ - lr * (m̂ / (sqrt(v̂) + ε) + weight_decay * θ)
   The standard optimizer for transformer training (BERT, GPT-2/3/4).
   Correct implementation of L2 regularisation with adaptive learning rates.

4. Lion (Chen et al., 2023 — Google Brain)
   Uses only the SIGN of the gradient update (no per-parameter second moment).
   m ← β₁ * m + (1 - β₁) * ∇L
   θ ← θ - lr * (sign(m̂) + weight_decay * θ)
   Memory efficient (no v), good for large models. Used in PaLM-2.
   Requires smaller learning rates and larger batch sizes than Adam.

Each optimizer here wraps PyTorch's built-in implementations where they exist
(avoiding reimplementing autograd), but exposes a consistent, documented API
and adds default values that are known to work for LLM training.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
import torch.optim as torch_optim


class SGD(torch_optim.SGD):
    """
    Stochastic Gradient Descent with optional momentum and weight decay.

    Update rule (with momentum):
        v ← momentum * v + ∇L
        θ ← θ - lr * (v + weight_decay * θ)

    Nesterov variant looks ahead:
        v ← momentum * v + ∇L
        θ ← θ - lr * (momentum * v + ∇L + weight_decay * θ)

    Args:
        params:       Model parameters (model.parameters()).
        lr:           Learning rate. No sensible default — must be tuned.
        momentum:     Momentum factor. 0.9 is the standard default.
        weight_decay: L2 regularisation strength. Typical: 1e-4.
        nesterov:     Use Nesterov momentum. Often improves convergence.
        dampening:    Dampening for momentum. Usually 0.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        dampening: float = 0.0,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dampening=dampening,
        )


class Adam(torch_optim.Adam):
    """
    Adam optimizer (Kingma & Ba, 2014).

    Maintains exponentially decaying averages of:
        m_t = β₁ * m_{t-1} + (1-β₁) * g_t          [first moment, mean]
        v_t = β₂ * v_{t-1} + (1-β₂) * g_t²         [second moment, variance]

    Bias-corrected estimates:
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)

    Update: θ ← θ - lr * m̂_t / (sqrt(v̂_t) + ε)

    Note: Adam's built-in weight_decay couples the L2 penalty with the
    adaptive learning rate, which is theoretically incorrect. Use AdamW
    for proper decoupled weight decay.

    Args:
        params:       Model parameters.
        lr:           Learning rate. Default 3e-4 (Karpathy's constant).
        betas:        (β₁, β₂) for first/second moment decay. Default (0.9, 0.999).
        eps:          Numerical stability. Default 1e-8.
        weight_decay: L2 regularisation (NOT decoupled). Default 0.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )


class AdamW(torch_optim.AdamW):
    """
    AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

    The key difference from Adam: weight decay is applied AFTER the adaptive
    update step, not mixed into the gradient:
        θ ← θ - lr * (m̂_t / (sqrt(v̂_t) + ε) + weight_decay * θ)

    This is the correct way to implement L2 regularisation with adaptive
    learning rates. The standard optimizer for training transformers.

    GPT-3 / LLaMA default config:
        lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1

    Args:
        params:       Model parameters.
        lr:           Learning rate. Default 3e-4.
        betas:        (β₁, β₂). Default (0.9, 0.95) — lower β₂ is better
                      for LLM training (faster adaption to gradient variance).
        eps:          Numerical stability. Default 1e-8.
        weight_decay: Decoupled L2 penalty. Default 0.1 (GPT-3 default).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )


class Lion(torch_optim.Optimizer):
    """
    Lion optimizer — EvoLved Sign Momentum (Chen et al., 2023).

    Uses only the SIGN of the exponentially moving average of gradients.
    This makes every parameter update the same magnitude (lr * weight_decay
    aside), similar to clipped gradient descent.

    Update rule:
        update = sign(β₁ * m_{t-1} + (1-β₁) * g_t)
        m_t    = β₂ * m_{t-1} + (1-β₂) * g_t
        θ ← θ - lr * (update + weight_decay * θ)

    Memory efficient: only stores one moment (m) vs Adam's two (m, v).
    Requires learning rates ~3-10× smaller than Adam.
    Requires larger batch sizes to work well (~several hundred to thousands).

    Args:
        params:       Model parameters.
        lr:           Learning rate. Typical: 1e-4 (much smaller than Adam).
        betas:        (β₁, β₂). Default (0.9, 0.99).
                      β₁: momentum for the sign update (interpolation step)
                      β₂: momentum for the state update (memory step)
        weight_decay: Decoupled L2 penalty. Default 0.0.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta values: {betas}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single Lion optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and
                     returns the loss. Required for some learning rate
                     schedulers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            wd           = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                # Initialise state on first step
                state = self.state[param]
                if len(state) == 0:
                    # Exponential moving average of gradients (first moment)
                    state["exp_avg"] = torch.zeros_like(param.data)

                exp_avg = state["exp_avg"]

                # Interpolate between previous moving average and current gradient
                update = exp_avg * beta1 + grad * (1.0 - beta1)

                # Parameter update: sign of interpolated gradient + weight decay
                param.data.add_(
                    torch.sign(update) + param.data * wd,
                    alpha=-lr,
                )

                # Update exponential moving average (the memory)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_optimizer(
    name: str,
    params: Iterable,
    lr: float,
    **kwargs,
) -> torch_optim.Optimizer:
    """
    Retrieve an optimizer by name.

    Args:
        name:   "sgd", "adam", "adamw", or "lion" (case-insensitive).
        params: Model parameters.
        lr:     Learning rate.
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        An instantiated optimizer.

    Raises:
        ValueError: If the name is not recognised.
    """
    name_lower = name.lower()
    registry = {
        "sgd":   SGD,
        "adam":  Adam,
        "adamw": AdamW,
        "lion":  Lion,
    }
    cls = registry.get(name_lower)
    if cls is None:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Supported: {sorted(registry.keys())}"
        )
    return cls(params=params, lr=lr, **kwargs)
