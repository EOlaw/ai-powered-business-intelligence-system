"""
InsightSerenity AI Engine — Value Networks
==========================================
Value functions estimate how much cumulative reward the agent expects to
receive from a given state (or state-action pair).

Two value functions:

V(s)  — State value: expected discounted return from state s following policy π.
         V^π(s) = E[R_0 + γR_1 + γ²R_2 + ... | s_0 = s, π]

Q(s,a) — Action-value: expected return from taking action a in state s.
         Q^π(s,a) = E[R_0 + γR_1 + ... | s_0=s, a_0=a, π]

The value function is used by:
    PPO/A3C:    Baseline for variance reduction. Advantage = Q(s,a) - V(s)
    DQN:        Directly learns Q(s,a); policy = argmax_a Q(s,a)

Generalised Advantage Estimation (GAE):
    The standard advantage estimate A(s,a) = r + γV(s') - V(s) is noisy.
    GAE uses an exponentially weighted sum of n-step returns:
    A_t^GAE = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t) (TD error)
    λ=0: pure 1-step TD (low variance, high bias)
    λ=1: full Monte Carlo (high variance, low bias)
    λ=0.95: the standard GAE default (balance)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.architectures.feedforward.mlp import MLP


class ValueNetwork(nn.Module):
    """
    State value function V(s): maps a state to a scalar expected return.

    Used as the critic in Actor-Critic methods (PPO, A3C).
    The critic provides a baseline that reduces the variance of
    policy gradient estimates without introducing bias.

    Args:
        obs_dim:     Observation (state) dimension.
        hidden_dims: MLP hidden layer sizes.
        activation:  Activation function name.
    """

    def __init__(
        self,
        obs_dim:     int,
        hidden_dims: List[int] = None,
        activation:  str       = "tanh",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]

        self.net = MLP(
            in_dim=obs_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            activation=activation,
            dropout=0.0,
        )
        # Initialise with small weights for stable early training
        for name, param in self.net.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, state: Tensor) -> Tensor:
        """
        Compute V(s) for a state or batch of states.

        Args:
            state: (B, obs_dim) or (obs_dim,) observation.

        Returns:
            (B, 1) or (1,) scalar value estimate.
        """
        return self.net(state)


class QNetwork(nn.Module):
    """
    Action-value function Q(s, a) for discrete action spaces.

    Maps a state to Q-values for EVERY action simultaneously.
    The policy selects the action with the highest Q-value:
        π(s) = argmax_a Q(s, a)

    This is the core of DQN and its variants.

    Args:
        obs_dim:     State dimension.
        n_actions:   Number of discrete actions.
        hidden_dims: MLP hidden layer sizes.
    """

    def __init__(
        self,
        obs_dim:     int,
        n_actions:   int,
        hidden_dims: List[int] = None,
        activation:  str       = "relu",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]

        self.net = MLP(
            in_dim=obs_dim,
            hidden_dims=hidden_dims,
            out_dim=n_actions,
            activation=activation,
            dropout=0.0,
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        Compute Q(s, a) for all actions.

        Args:
            state: (B, obs_dim).

        Returns:
            (B, n_actions) Q-values.
        """
        return self.net(state)


# ─────────────────────────────────────────────────────────────────────────────
# GAE — Generalised Advantage Estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards:     Tensor,   # (T,) rewards collected during a rollout
    values:      Tensor,   # (T,) value estimates V(s_t)
    next_values: Tensor,   # (T,) value estimates V(s_{t+1})
    dones:       Tensor,   # (T,) 1.0 if episode ended at step t, else 0.0
    gamma:       float = 0.99,
    lam:         float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """
    Compute GAE advantage estimates and returns.

    The advantage A_t^GAE(λ) is the exponentially weighted average of
    n-step advantage estimates. This reduces the variance of PPO/A3C
    gradient estimates without introducing much bias.

    A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...
    where δ_t = r_t + γ(1-done_t) V(s_{t+1}) - V(s_t)

    Args:
        rewards:     (T,) rewards from the rollout.
        values:      (T,) value estimates V(s_t).
        next_values: (T,) value estimates V(s_{t+1}). For terminal states,
                     use 0.0 (the episode ends, no future value).
        dones:       (T,) episode termination flags.
        gamma:       Discount factor. Default 0.99.
        lam:         GAE lambda. Default 0.95.

    Returns:
        Tuple (advantages (T,), returns (T,)):
            advantages: Normalised advantage estimates for policy gradient.
            returns:    Discounted returns (used as targets for value function).
    """
    T          = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0

    # Compute GAE backwards (from last step to first)
    for t in reversed(range(T)):
        # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        # (1 - done_t) ensures no bootstrapping after terminal states
        delta    = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae

    # Returns = advantages + values (used as targets for critic training)
    returns = advantages + values

    # Normalise advantages for stable training (zero mean, unit std)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns
