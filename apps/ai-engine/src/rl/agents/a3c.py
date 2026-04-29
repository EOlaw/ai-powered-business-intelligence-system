"""
InsightSerenity AI Engine — A3C Agent (Asynchronous Advantage Actor-Critic)
===========================================================================
A3C (Mnih et al., 2016) extends Actor-Critic by running multiple worker
processes in parallel, each interacting with its own environment copy.
Workers compute gradients locally and apply them to a shared global network.

This parallelism provides two benefits:
    1. Speed: Multiple environments are explored simultaneously
    2. Diversity: Different workers explore different parts of the state space,
       providing more decorrelated, diverse gradient updates than a single worker

A3C vs PPO:
    A3C:  Asynchronous, on-policy, multi-process, no replay buffer
    PPO:  Synchronous, on-policy, single/multi-process, no replay buffer
    DQN:  Off-policy, single-process, replay buffer

A3C Update (per worker):
    1. Interact with environment for t_max steps
    2. Compute returns: R_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
    3. Compute advantages: A_t = R_t - V(s_t)
    4. Policy gradient: ∇ log π(a_t|s_t) × A_t
    5. Value gradient: (R_t - V(s_t))²
    6. Apply gradients to global network

This implementation runs A3C in a single process with sequential pseudo-workers
for clarity and compatibility with systems without multiprocessing.
For true async multi-process training, use torch.multiprocessing.

The key idea — shared global parameters updated by multiple independent
gradient computations — is preserved in the single-process version.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.rl.environments.base_env import BaseEnvironment
from src.rl.policies.policy_network import PolicyConfig, make_policy
from src.rl.value.value_network import ValueNetwork
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class A3CConfig:
    obs_dim:          int
    action_dim:       int
    action_type:      str       = "discrete"
    hidden_dims:      List[int] = field(default_factory=lambda: [128, 128])
    gamma:            float     = 0.99
    lr:               float     = 7e-4
    entropy_coeff:    float     = 0.01
    value_coeff:      float     = 0.5
    t_max:            int       = 20        # Steps per worker update
    max_grad_norm:    float     = 40.0      # A3C uses a larger clip than PPO
    n_workers:        int       = 4         # Pseudo-workers (sequential in single-process)
    device:           str       = "cpu"


class A3CWorker:
    """
    Single A3C worker. Each worker has:
        - A LOCAL copy of the global network (periodically synced)
        - An environment instance
        - A t_max-step rollout buffer

    Workers compute gradients on their local network and return them for
    application to the global network. This is the async nature of A3C.
    """

    def __init__(
        self,
        worker_id:    int,
        config:       A3CConfig,
        global_actor:  nn.Module,
        global_critic: nn.Module,
        env:          BaseEnvironment,
    ) -> None:
        self.worker_id     = worker_id
        self.config        = config
        self.device        = torch.device(config.device)
        self.global_actor  = global_actor
        self.global_critic = global_critic
        self.env           = env

        # Local network (starts as copy of global)
        self.local_actor   = copy.deepcopy(global_actor).to(self.device)
        self.local_critic  = copy.deepcopy(global_critic).to(self.device)

        self._state, _ = env.reset(seed=worker_id)

    def sync_with_global(self) -> None:
        """Copy global network weights to local network."""
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    def rollout(self) -> Tuple[List, List, List, bool, np.ndarray]:
        """
        Collect t_max steps of experience using the local policy.

        Returns:
            Tuple (states, actions, rewards, done_at_end, final_state).
        """
        states, actions, rewards = [], [], []
        done = False

        for _ in range(self.config.t_max):
            state_t = torch.tensor(self._state, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                dist   = self.local_actor(state_t.unsqueeze(0))
                action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            states.append(self._state.copy())
            actions.append(action)
            rewards.append(reward)

            self._state = next_state

            if done:
                self._state, _ = self.env.reset()
                break

        return states, actions, rewards, done, self._state.copy()

    def compute_gradients(self) -> Tuple[Dict, Dict, float]:
        """
        Collect a rollout and compute Actor-Critic gradients.

        Returns:
            Tuple (actor_grads, critic_grads, episode_return).
        """
        # Sync local with global before each rollout
        self.sync_with_global()

        states, actions, rewards, done, final_state = self.rollout()

        # Bootstrap return from final state
        if done:
            R = 0.0
        else:
            final_t = torch.tensor(final_state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                R = self.local_critic(final_t.unsqueeze(0)).item()

        # Compute returns backwards
        returns = []
        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        states_t  = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)

        # Forward pass through local networks
        values      = self.local_critic(states_t).squeeze(-1)
        dists       = self.local_actor(states_t)
        log_probs   = dists.log_prob(actions_t)
        entropy     = dists.entropy().mean()

        advantages  = (returns_t - values).detach()

        # Actor loss: policy gradient with advantage baseline
        actor_loss  = -(log_probs * advantages).mean() - self.config.entropy_coeff * entropy

        # Critic loss: MSE on returns
        critic_loss = self.config.value_coeff * F.mse_loss(values, returns_t)

        # Compute gradients on LOCAL networks
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.config.max_grad_norm)

        # Collect gradients to apply to global
        actor_grads  = {name: p.grad.clone() for name, p in self.local_actor.named_parameters()
                        if p.grad is not None}
        critic_grads = {name: p.grad.clone() for name, p in self.local_critic.named_parameters()
                        if p.grad is not None}

        # Zero local gradients
        self.local_actor.zero_grad()
        self.local_critic.zero_grad()

        episode_return = sum(rewards)
        return actor_grads, critic_grads, episode_return


class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic agent.

    Maintains global actor and critic networks. Spawns N pseudo-workers
    that each collect rollouts, compute gradients, and update the global
    networks asynchronously (sequentially in this single-process version).

    Args:
        config:    A3CConfig.
        env_fn:    Callable that creates a new environment instance.
                   Called once per worker.
    """

    def __init__(self, config: A3CConfig, env_fn) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Global networks — shared across all workers
        policy_cfg = PolicyConfig(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            action_type=config.action_type,
            activation="relu",
        )
        self.global_actor  = make_policy(policy_cfg).to(self.device)
        self.global_critic = ValueNetwork(config.obs_dim, config.hidden_dims).to(self.device)

        # Shared optimizers for the global networks
        self.actor_optimizer  = torch.optim.RMSprop(self.global_actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.RMSprop(self.global_critic.parameters(), lr=config.lr)

        # Create workers
        self.workers = [
            A3CWorker(
                worker_id=i,
                config=config,
                global_actor=self.global_actor,
                global_critic=self.global_critic,
                env=env_fn(),
            )
            for i in range(config.n_workers)
        ]

        self.episode_returns: List[float] = []

    def train(self, total_updates: int) -> Dict[str, float]:
        """
        Run A3C training.

        Each update: one worker computes gradients → applied to global network.
        Workers are selected round-robin.

        Args:
            total_updates: Total gradient update steps.

        Returns:
            Final statistics dict.
        """
        for update in range(total_updates):
            # Select worker (round-robin)
            worker = self.workers[update % self.config.n_workers]

            # Worker computes gradients on its local network
            actor_grads, critic_grads, ep_return = worker.compute_gradients()
            self.episode_returns.append(ep_return)

            # Apply gradients to global network
            self._apply_gradients(self.global_actor, self.actor_optimizer, actor_grads)
            self._apply_gradients(self.global_critic, self.critic_optimizer, critic_grads)

            if (update + 1) % 100 == 0:
                avg = np.mean(self.episode_returns[-100:])
                logger.info(
                    "A3C training",
                    update=update + 1,
                    avg_return=round(avg, 2),
                )

        avg_return = float(np.mean(self.episode_returns[-100:])) if self.episode_returns else 0.0
        return {"avg_return": avg_return, "n_updates": total_updates}

    def _apply_gradients(
        self,
        global_net: nn.Module,
        optimizer:  torch.optim.Optimizer,
        grads:      Dict[str, Tensor],
    ) -> None:
        """Apply computed gradients to the global network."""
        optimizer.zero_grad(set_to_none=True)
        for name, param in global_net.named_parameters():
            if name in grads:
                param.grad = grads[name].to(self.device)
        optimizer.step()

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        """Greedy action from the global policy (for evaluation)."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist    = self.global_actor(state_t.unsqueeze(0))
        return dist.sample().item()
