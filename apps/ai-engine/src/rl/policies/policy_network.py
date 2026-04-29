"""
InsightSerenity AI Engine — Policy Network
===========================================
The policy network maps a state observation to a probability distribution
over actions. This is what the agent uses to decide what to do.

Two types of action spaces:
    Discrete:   finite set of actions (e.g. token selection in text generation,
                left/right in CartPole). Output: categorical distribution.
    Continuous: real-valued actions (e.g. robot joint torques, control signals).
                Output: diagonal Gaussian distribution (mean + log_std).

The policy network is the "brain" of every model-free RL agent:
    - PPO uses it directly: collect rollouts → compute policy gradient
    - A3C shares it across workers: async gradient updates
    - DQN replaces it with a Q-network (value-based, not policy-based)

Policy gradient theorem (REINFORCE):
    ∇J(θ) = E[∇ log π(a|s; θ) × A(s, a)]
    Gradient points in the direction that increases log-probability of
    actions with positive advantage (good actions) and decreases for
    negative advantage (bad actions).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal

from src.architectures.feedforward.mlp import MLP
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for a policy network."""
    obs_dim:     int            # Observation (state) dimension
    action_dim:  int            # Number of discrete actions OR continuous action dim
    hidden_dims: list           = None    # MLP hidden layer sizes
    activation:  str            = "tanh"  # tanh is standard for RL policies
    dropout:     float          = 0.0
    action_type: str            = "discrete"   # "discrete" or "continuous"
    log_std_min: float          = -20.0   # Clamp log_std for numerical stability
    log_std_max: float          = 2.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class DiscretePolicy(nn.Module):
    """
    Policy network for discrete action spaces.

    Maps state → logits → softmax → Categorical distribution.
    Used in: CartPole, text token selection, any game with finite actions.

    Args:
        config: PolicyConfig with action_type="discrete".
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config

        self.net = MLP(
            in_dim=config.obs_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.action_dim,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(self, state: Tensor) -> Categorical:
        """
        Compute the action distribution for a state.

        Args:
            state: (B, obs_dim) or (obs_dim,) state observation.

        Returns:
            Categorical distribution over actions.
        """
        logits = self.net(state)
        return Categorical(logits=logits)

    def sample_action(self, state: Tensor) -> Tuple[int, Tensor]:
        """
        Sample an action and return its log-probability.

        Args:
            state: (obs_dim,) single state observation.

        Returns:
            Tuple (action_int, log_prob_tensor).
        """
        dist       = self.forward(state.unsqueeze(0) if state.dim() == 1 else state)
        action     = dist.sample()
        log_prob   = dist.log_prob(action)
        return action.item(), log_prob

    def greedy_action(self, state: Tensor) -> int:
        """Return the most likely action (argmax) without sampling."""
        with torch.no_grad():
            logits = self.net(state.unsqueeze(0) if state.dim() == 1 else state)
            return logits.argmax(dim=-1).item()

    def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """Compute log π(a|s) for a batch of (state, action) pairs."""
        dist = self.forward(state)
        return dist.log_prob(action)

    def entropy(self, state: Tensor) -> Tensor:
        """Compute the entropy of the policy distribution H(π(·|s))."""
        return self.forward(state).entropy()


class ContinuousPolicy(nn.Module):
    """
    Policy network for continuous action spaces.

    Maps state → (mean, log_std) → Normal distribution.
    Actions are sampled via the reparameterization trick for differentiable
    log-probabilities.

    Uses a diagonal Gaussian (independent per action dimension) — full
    covariance would be too expensive for high-dimensional action spaces.

    Args:
        config: PolicyConfig with action_type="continuous".
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config

        # Shared backbone for mean and log_std
        self.net = MLP(
            in_dim=config.obs_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.hidden_dims[-1],   # Output of shared backbone
            activation=config.activation,
            dropout=config.dropout,
        )
        self.mean_head    = nn.Linear(config.hidden_dims[-1], config.action_dim)
        self.log_std_head = nn.Linear(config.hidden_dims[-1], config.action_dim)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -0.5)   # Start with small std

    def forward(self, state: Tensor) -> Normal:
        """
        Compute action distribution parameters for a state.

        Args:
            state: (B, obs_dim) observation.

        Returns:
            Normal distribution N(mean, std).
        """
        h       = self.net(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.config.log_std_min, self.config.log_std_max)
        std     = log_std.exp()
        return Normal(mean, std)

    def sample_action(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample action using rsample (reparameterization trick)."""
        dist     = self.forward(state.unsqueeze(0) if state.dim() == 1 else state)
        action   = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)   # Sum over action dims
        return action.squeeze(0), log_prob

    def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """Compute log π(a|s) summed over action dimensions."""
        dist = self.forward(state)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, state: Tensor) -> Tensor:
        """Policy entropy summed over action dimensions."""
        return self.forward(state).entropy().sum(dim=-1)


def make_policy(config: PolicyConfig) -> nn.Module:
    """Factory: create the appropriate policy for the action type."""
    if config.action_type == "discrete":
        return DiscretePolicy(config)
    elif config.action_type == "continuous":
        return ContinuousPolicy(config)
    raise ValueError(f"Unknown action_type '{config.action_type}'. Expected: discrete | continuous")
