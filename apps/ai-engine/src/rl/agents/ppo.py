"""
InsightSerenity AI Engine — Proximal Policy Optimization Agent
===============================================================
PPO (Schulman et al., 2017) is the gold standard for on-policy deep RL.
It is also the algorithm used in the RLHF fine-tuning pipeline (Phase 6).

PPO collects rollouts (trajectories) from the current policy, then updates
the policy multiple times on the same rollout data using a clipped objective
that prevents too large policy updates.

Clipped objective:
    L^CLIP(θ) = E_t [min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (importance ratio)

Why clip? When r_t is large (new policy differs a lot from old), the gradient
could blow up. Clipping prevents updates that move the policy too far.

Actor-Critic formulation:
    Actor:  Policy π(a|s; θ_π) — learns WHAT to do
    Critic: Value V(s; θ_V) — learns HOW GOOD a state is

Loss = -L^CLIP + c₁ × VF_loss - c₂ × Entropy

    VF_loss: Critic is trained to predict GAE returns
    Entropy: Bonus encourages exploration (prevents premature convergence)

PPO Rollout → Update cycle:
    1. Collect T steps using current policy (rollout phase)
    2. Compute GAE advantages and returns
    3. Run K epochs of mini-batch gradient updates (PPO phase)
    4. Discard rollout data and repeat from step 1

This is used directly by RLHFTrainer (Phase 6) — that trainer is a specialised
version of PPO adapted for language model fine-tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal

from src.rl.environments.base_env import BaseEnvironment
from src.rl.policies.policy_network import DiscretePolicy, ContinuousPolicy, PolicyConfig, make_policy
from src.rl.value.value_network import ValueNetwork, compute_gae
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PPOConfig:
    obs_dim:          int
    action_dim:       int
    action_type:      str       = "discrete"  # "discrete" or "continuous"
    hidden_dims:      List[int] = field(default_factory=lambda: [64, 64])
    gamma:            float     = 0.99    # Discount factor
    lam:              float     = 0.95    # GAE lambda
    lr_actor:         float     = 3e-4
    lr_critic:        float     = 1e-3
    clip_eps:         float     = 0.2    # PPO clip range
    value_coeff:      float     = 0.5    # Critic loss coefficient
    entropy_coeff:    float     = 0.01   # Entropy bonus coefficient
    max_grad_norm:    float     = 0.5    # Gradient clipping norm
    rollout_steps:    int       = 2048   # Steps per rollout batch
    ppo_epochs:       int       = 10     # Epochs per rollout
    ppo_batch_size:   int       = 64     # Mini-batch size
    device:           str       = "cpu"


class RolloutBuffer:
    """
    Stores one PPO rollout: states, actions, rewards, values, log_probs, dones.
    Cleared after each PPO update cycle.
    """

    def __init__(self) -> None:
        self.states:    List[np.ndarray] = []
        self.actions:   List             = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.log_probs: List[float]      = []
        self.dones:     List[float]      = []
        self.advantages: Optional[Tensor] = None
        self.returns:    Optional[Tensor] = None

    def push(
        self,
        state:    np.ndarray,
        action:   int,
        reward:   float,
        value:    float,
        log_prob: float,
        done:     bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(float(done))

    def compute_advantages(
        self, last_value: float, gamma: float, lam: float, device: torch.device
    ) -> None:
        """Compute GAE advantages and returns from the stored rollout."""
        rewards     = torch.tensor(self.rewards,   dtype=torch.float32, device=device)
        values      = torch.tensor(self.values,    dtype=torch.float32, device=device)
        dones       = torch.tensor(self.dones,     dtype=torch.float32, device=device)

        # Build next_values: V(s_{t+1}) for each step
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[-1]  = last_value

        self.advantages, self.returns = compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
            gamma=gamma,
            lam=lam,
        )

    def get_batches(self, batch_size: int, device: torch.device):
        """
        Yield random mini-batches from the rollout data.
        Shuffling ensures diverse, uncorrelated mini-batches.
        """
        N       = len(self.states)
        indices = np.random.permutation(N)
        states_t  = torch.tensor(np.array(self.states),    dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(self.actions),   dtype=torch.long,    device=device)
        log_p_t   = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)

        for start in range(0, N, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                states_t[idx],
                actions_t[idx],
                log_p_t[idx],
                self.advantages[idx],
                self.returns[idx],
            )

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    PPO Actor-Critic agent for discrete and continuous action spaces.

    The actor (policy) and critic (value function) share no weights but
    are trained jointly in the same update step.

    Args:
        config: PPOConfig.
    """

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Actor: policy network
        policy_cfg = PolicyConfig(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            action_type=config.action_type,
            activation="tanh",
        )
        self.actor  = make_policy(policy_cfg).to(self.device)
        self.critic = ValueNetwork(
            obs_dim=config.obs_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self.buffer  = RolloutBuffer()
        self.episode_rewards: List[float] = []
        self._step = 0

    # ── Core API ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action_and_value(
        self, state: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Sample action, log-probability, and value estimate for a state.

        Args:
            state: (obs_dim,) observation.

        Returns:
            Tuple (action, log_prob, value).
        """
        state_t  = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist     = self.actor(state_t.unsqueeze(0))
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        value    = self.critic(state_t.unsqueeze(0)).squeeze(-1)

        return action.item(), log_prob.item(), value.item()

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Run PPO update on the collected rollout.

        Args:
            last_value: V(s_last) — bootstrapped value for the final state.
                        0.0 if the episode terminated.

        Returns:
            Dict with training metrics.
        """
        self.buffer.compute_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            lam=self.config.lam,
            device=self.device,
        )

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for epoch in range(self.config.ppo_epochs):
            for states, actions, old_log_probs, advantages, returns in self.buffer.get_batches(
                self.config.ppo_batch_size, self.device
            ):
                # ── Actor (policy) loss ──────────────────────────────────────
                dist         = self.actor(states)
                new_log_prob = dist.log_prob(actions)
                entropy      = dist.entropy().mean()

                # Importance ratio r_t = π_new(a|s) / π_old(a|s)
                ratio        = (new_log_prob - old_log_probs).exp()

                # PPO clipped objective
                unclipped    = ratio * advantages
                clipped      = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
                policy_loss  = -torch.min(unclipped, clipped).mean()

                # ── Critic (value) loss ──────────────────────────────────────
                values       = self.critic(states).squeeze(-1)
                value_loss   = F.mse_loss(values, returns)

                # ── Combined loss ─────────────────────────────────────────────
                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    - self.config.entropy_coeff * entropy
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                n_updates         += 1

        self.buffer.clear()
        n = max(n_updates, 1)

        return {
            "policy_loss": total_policy_loss / n,
            "value_loss":  total_value_loss  / n,
            "entropy":     total_entropy     / n,
        }

    def train(
        self,
        env:           BaseEnvironment,
        total_steps:   int,
        log_interval:  int = 10,
    ) -> Dict[str, float]:
        """
        Run a complete PPO training run.

        Args:
            env:          The environment to train on.
            total_steps:  Total environment steps to collect.
            log_interval: Log statistics every N episodes.

        Returns:
            Final training statistics.
        """
        obs, _ = env.reset()
        ep_reward   = 0.0
        n_episodes  = 0
        all_metrics: List[Dict] = []

        for step in range(total_steps):
            action, log_prob, value = self.get_action_and_value(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.push(obs, action, reward, value, log_prob, done)
            ep_reward += reward
            obs         = next_obs
            self._step += 1

            if done:
                self.episode_rewards.append(ep_reward)
                ep_reward  = 0.0
                n_episodes += 1
                obs, _     = env.reset()

            # Update after collecting rollout_steps
            if len(self.buffer) >= self.config.rollout_steps:
                _, _, last_value = self.get_action_and_value(obs)
                last_val  = 0.0 if done else last_value
                metrics   = self.update(last_value=last_val)
                all_metrics.append(metrics)

                if n_episodes % log_interval == 0 and self.episode_rewards:
                    avg = np.mean(self.episode_rewards[-log_interval:])
                    logger.info(
                        "PPO training",
                        step=self._step,
                        episodes=n_episodes,
                        avg_reward=round(avg, 2),
                        policy_loss=round(metrics["policy_loss"], 4),
                    )

        avg_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
        return {"avg_reward": avg_reward, "n_episodes": n_episodes}
