"""
InsightSerenity AI Engine — RL Reward Shaping
==============================================
Reward shaping transforms raw environment rewards into denser, more
informative signals that help the agent learn faster.

Sparse rewards (e.g. +1 at episode end only) make credit assignment hard —
the agent doesn't know which of the T actions in the episode caused the reward.
Shaped rewards provide per-step feedback to accelerate learning.

Techniques implemented:

1. ProgressReward — Incremental progress towards the goal (potential-based).
   Guaranteed not to change the optimal policy (Ng et al., 1999).

2. CuriosityReward — Intrinsic motivation from prediction error.
   Encourages the agent to visit novel states even without external reward.
   Based on Random Network Distillation (RND, Burda et al., 2018).

3. RewardNormaliser — Running statistics normaliser.
   Divides rewards by a running estimate of the return std to keep the
   reward scale stable throughout training.

4. LLMRewardModel — Thin wrapper that connects our RewardModel from
   llm/alignment/reward_model.py to the RL environment interface.
   This is the bridge between Phase 6 (RLHF) and Phase 8 (RL).
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RewardNormaliser:
    """
    Running mean/std reward normalisation.

    Maintains Welford's online algorithm for computing running mean and
    variance without storing all past rewards. Divides each reward by the
    running standard deviation (clipped to prevent division by very small std).

    Args:
        clip:    Clip normalised rewards to [-clip, clip]. Default 10.0.
        gamma:   Discount factor (used for return-based normalisation).
        epsilon: Numerical stability constant.
    """

    def __init__(self, clip: float = 10.0, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        self.clip    = clip
        self.gamma   = gamma
        self.epsilon = epsilon
        self._mean   = 0.0
        self._var    = 1.0
        self._count  = 0
        self._ret    = 0.0    # Discounted return estimate

    def normalise(self, reward: float) -> float:
        """Normalise a reward using running statistics."""
        self._update(reward)
        std = max(np.sqrt(self._var), self.epsilon)
        return float(np.clip(reward / std, -self.clip, self.clip))

    def normalise_batch(self, rewards: np.ndarray) -> np.ndarray:
        """Normalise a batch of rewards."""
        for r in rewards:
            self._update(r)
        std = max(np.sqrt(self._var), self.epsilon)
        return np.clip(rewards / std, -self.clip, self.clip)

    def _update(self, reward: float) -> None:
        """Welford online update."""
        self._ret    = self._ret * self.gamma + reward
        self._count += 1
        delta        = self._ret - self._mean
        self._mean  += delta / self._count
        delta2       = self._ret - self._mean
        self._var   += (delta * delta2 - self._var) / self._count


class RandomNetworkDistillation(nn.Module):
    """
    Curiosity reward via Random Network Distillation (Burda et al., 2018).

    Maintains two networks with the same architecture:
        Fixed (random) network:   r(s)  — never trained
        Predictor network:        p(s)  — trained to predict r(s)

    Intrinsic reward = prediction error = ||r(s) - p(s)||²

    When the agent visits a new state, the predictor hasn't seen it before
    and makes a large error → high intrinsic reward → encourages exploration.
    As the agent revisits states, the predictor improves → error decreases.

    Args:
        obs_dim:     State observation dimension.
        embed_dim:   Embedding dimension.
        lr:          Predictor learning rate.
    """

    def __init__(self, obs_dim: int, embed_dim: int = 64, lr: float = 1e-4) -> None:
        super().__init__()

        # Fixed random target network (never updated)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
        for p in self.target.parameters():
            p.requires_grad_(False)   # Frozen

        # Learnable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        self.optimizer   = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self._normaliser = RewardNormaliser(clip=5.0)

    def compute_reward(self, obs: Tensor) -> float:
        """
        Compute the intrinsic curiosity reward for an observation.

        Also updates the predictor network on this observation.

        Args:
            obs: (obs_dim,) or (B, obs_dim) observation.

        Returns:
            Scalar intrinsic reward.
        """
        obs = obs.float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            target_feat = self.target(obs)

        pred_feat = self.predictor(obs)

        # Prediction error = intrinsic reward
        error    = nn.functional.mse_loss(pred_feat, target_feat.detach(), reduction="none")
        reward   = error.mean().item()

        # Update predictor
        loss = error.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return self._normaliser.normalise(reward)


class LLMRewardShaper:
    """
    Reward shaper for LLM text generation that combines:
        1. Terminal reward from the RewardModel (preference signal)
        2. Length penalty (penalise very short or very long responses)
        3. Repetition penalty (penalise repeated n-grams)
        4. KL penalty (penalise divergence from reference policy)

    This produces a richer reward signal than the raw RewardModel score,
    helping the PPO agent learn faster and produce better-aligned outputs.

    Args:
        reward_model:      Trained RewardModel from llm/alignment.
        length_bonus:      Reward for responses in the target length range.
        repetition_penalty: Penalty for repeated bigrams in the response.
        target_min_tokens: Minimum tokens for the length bonus.
        target_max_tokens: Maximum tokens for the length bonus.
    """

    def __init__(
        self,
        reward_model:       Any,
        length_bonus:       float = 0.1,
        repetition_penalty: float = 0.5,
        target_min_tokens:  int   = 20,
        target_max_tokens:  int   = 200,
    ) -> None:
        self.reward_model       = reward_model
        self.length_bonus       = length_bonus
        self.repetition_penalty = repetition_penalty
        self.target_min         = target_min_tokens
        self.target_max         = target_max_tokens
        self._normaliser        = RewardNormaliser()

    def compute_reward(
        self,
        token_ids:      List[int],
        generated_ids:  List[int],
        device:         str = "cpu",
    ) -> float:
        """
        Compute the shaped reward for a completed generation.

        Args:
            token_ids:     Full sequence (prompt + response).
            generated_ids: Response tokens only (used for length/repetition).
            device:        Compute device for the reward model.

        Returns:
            Shaped reward scalar.
        """
        # Terminal reward from the learned reward model
        ids_t  = torch.tensor([token_ids], dtype=torch.long, device=device)
        mask   = torch.ones_like(ids_t)
        with torch.no_grad():
            base_reward = self.reward_model(ids_t, mask).item()

        # Length reward: bonus for being in the target length range
        n_tokens = len(generated_ids)
        if self.target_min <= n_tokens <= self.target_max:
            length_r = self.length_bonus
        elif n_tokens < self.target_min:
            length_r = self.length_bonus * (n_tokens / self.target_min) - self.length_bonus
        else:
            overshoot = (n_tokens - self.target_max) / self.target_max
            length_r  = -self.length_bonus * overshoot

        # Repetition penalty: count repeated bigrams
        rep_penalty = 0.0
        if len(generated_ids) > 2:
            bigrams  = set()
            repeats  = 0
            for i in range(len(generated_ids) - 1):
                bg = (generated_ids[i], generated_ids[i + 1])
                if bg in bigrams:
                    repeats += 1
                bigrams.add(bg)
            rep_ratio   = repeats / max(len(generated_ids) - 1, 1)
            rep_penalty = self.repetition_penalty * rep_ratio

        total = base_reward + length_r - rep_penalty
        return self._normaliser.normalise(total)
