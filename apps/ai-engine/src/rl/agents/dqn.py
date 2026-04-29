"""
InsightSerenity AI Engine — Deep Q-Network Agent
==================================================
DQN (Mnih et al., 2015) was the first algorithm to solve complex RL problems
(Atari games) using deep neural networks. It combines:

    1. Q-Network: A neural net that approximates Q(s,a) for all actions.
    2. Experience Replay: Stores (s, a, r, s', done) and samples random
       mini-batches. Breaks correlations in sequential data.
    3. Target Network: A slowly-updated copy of Q-network. Used to compute
       stable TD targets. Prevents oscillation.

TD Target (Bellman equation):
    y_t = r_t + γ × max_{a'} Q_target(s_{t+1}, a')   (if not done)
    y_t = r_t                                           (if done)

Loss: MSE(Q(s,a; θ) - y_t) — update θ to make Q closer to y.

ε-greedy exploration:
    With probability ε: select random action (explore)
    With probability 1-ε: select argmax Q(s,a) (exploit)
    ε decays linearly from 1.0 to 0.05 over training.

Double DQN improvement (van Hasselt et al., 2016):
    Standard DQN overestimates Q-values (max operation introduces positive bias).
    Double DQN uses the online network to SELECT the action but the target
    network to EVALUATE it:
    y_t = r_t + γ × Q_target(s', argmax_{a'} Q_online(s', a'))
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.rl.environments.base_env import BaseEnvironment
from src.rl.memory.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.rl.value.value_network import QNetwork
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DQNConfig:
    obs_dim:          int
    n_actions:        int
    hidden_dims:      List[int] = None
    gamma:            float     = 0.99    # Discount factor
    lr:               float     = 1e-4   # Adam learning rate
    batch_size:       int       = 64
    buffer_size:      int       = 100_000
    learning_starts:  int       = 1_000  # Steps before first gradient update
    train_freq:       int       = 4       # Update every N environment steps
    target_update:    int       = 1_000  # Copy online → target every N steps
    eps_start:        float     = 1.0
    eps_end:          float     = 0.05
    eps_decay_steps:  int       = 50_000
    double_dqn:       bool      = True    # Use Double DQN target
    prioritized:      bool      = False   # Use PER instead of uniform replay
    device:           str       = "cpu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.

    Optionally uses:
        - Double DQN for less biased Q-value estimates
        - Prioritised Experience Replay for faster learning

    Args:
        config: DQNConfig.
    """

    def __init__(self, config: DQNConfig) -> None:
        self.config  = config
        self.device  = torch.device(config.device)

        # Online network: updated at every gradient step
        self.q_online = QNetwork(
            obs_dim=config.obs_dim,
            n_actions=config.n_actions,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        # Target network: slowly updated copy for stable TD targets
        self.q_target = QNetwork(
            obs_dim=config.obs_dim,
            n_actions=config.n_actions,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        # Initialise target with same weights as online
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()   # Target network is never trained directly

        # Replay buffer
        obs_shape = (config.obs_dim,)
        if config.prioritized:
            self.buffer = PrioritizedReplayBuffer(
                capacity=config.buffer_size,
                obs_shape=obs_shape,
                device=config.device,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=config.buffer_size,
                obs_shape=obs_shape,
                device=config.device,
            )

        self.optimizer = torch.optim.Adam(
            self.q_online.parameters(), lr=config.lr
        )

        self._step        = 0      # Total environment steps
        self._updates     = 0      # Total gradient updates
        self.episode_rewards: List[float] = []
        self.losses:          List[float] = []

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection.

        Exploration rate decays linearly from eps_start to eps_end over
        eps_decay_steps environment steps.

        Args:
            state: Current observation as a numpy array.

        Returns:
            Integer action index.
        """
        eps = self._get_epsilon()

        if np.random.random() < eps:
            # Explore: uniform random action
            return np.random.randint(0, self.config.n_actions)

        # Exploit: greedy action from Q-network
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals  = self.q_online(state_t)
            return q_vals.argmax(dim=1).item()

    # ── Learning ───────────────────────────────────────────────────────────────

    def observe(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> Optional[Dict[str, float]]:
        """
        Store a transition and optionally run a gradient update.

        Args:
            state, action, reward, next_state, done: Transition tuple.

        Returns:
            Dict with training metrics if an update was performed, else None.
        """
        self.buffer.push(state, action, reward, next_state, done)
        self._step += 1

        # Don't train until we have enough data
        if self._step < self.config.learning_starts:
            return None

        # Train every `train_freq` steps
        if self._step % self.config.train_freq != 0:
            return None

        if not self.buffer.is_ready(self.config.batch_size):
            return None

        return self._update()

    def _update(self) -> Dict[str, float]:
        """Run one gradient descent step on a mini-batch."""
        batch = self.buffer.sample(self.config.batch_size)

        states      = batch.states.float()
        actions     = batch.actions
        rewards     = batch.rewards
        next_states = batch.next_states.float()
        dones       = batch.dones

        # Current Q-values: Q(s, a) for the taken actions
        q_values = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD target
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: online selects action, target evaluates it
                next_actions  = self.q_online(next_states).argmax(dim=1)
                next_q_values = self.q_target(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q_values = self.q_target(next_states).max(dim=1).values

            targets = rewards + self.config.gamma * next_q_values * (1.0 - dones)

        # TD error
        td_errors = (q_values - targets).detach()

        # Loss (weighted by IS weights for PER, uniform otherwise)
        if batch.weights is not None:
            loss = (batch.weights * F.smooth_l1_loss(q_values, targets, reduction="none")).mean()
            # Update priorities in PER buffer
            if hasattr(self.buffer, "update_priorities"):
                self.buffer.update_priorities(
                    batch.indices, td_errors.abs().cpu().numpy()
                )
        else:
            loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients for stability
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optimizer.step()
        self._updates += 1

        # Update target network periodically
        if self._updates % self.config.target_update == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        self.losses.append(loss.item())
        return {"loss": loss.item(), "epsilon": self._get_epsilon(), "q_mean": q_values.mean().item()}

    def train(self, env: BaseEnvironment, total_steps: int) -> Dict[str, float]:
        """
        Run a complete DQN training run.

        Args:
            env:         The environment to train on.
            total_steps: Total environment interaction steps.

        Returns:
            Dict with final training statistics.
        """
        obs, _ = env.reset()
        episode_reward = 0.0
        n_episodes     = 0

        for step in range(total_steps):
            action        = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done          = terminated or truncated

            self.observe(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs             = next_obs

            if done:
                self.episode_rewards.append(episode_reward)
                episode_reward = 0.0
                n_episodes    += 1
                obs, _         = env.reset()

                if n_episodes % 20 == 0:
                    avg_r = np.mean(self.episode_rewards[-20:])
                    logger.info(
                        "DQN training",
                        step=step,
                        episodes=n_episodes,
                        avg_reward=round(avg_r, 2),
                        epsilon=round(self._get_epsilon(), 3),
                    )

        avg_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
        avg_loss   = float(np.mean(self.losses[-1000:])) if self.losses else 0.0
        return {"avg_reward": avg_reward, "avg_loss": avg_loss, "n_episodes": n_episodes}

    def _get_epsilon(self) -> float:
        """Linearly decay epsilon from eps_start to eps_end."""
        progress = min(self._step / self.config.eps_decay_steps, 1.0)
        return self.config.eps_start + progress * (self.config.eps_end - self.config.eps_start)
