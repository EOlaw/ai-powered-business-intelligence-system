"""
InsightSerenity AI Engine — Reinforcement Learning Package
===========================================================
Complete RL stack: environments, agents, policies, value networks, memory.

Environments:
    from src.rl import BaseEnvironment, CartPoleEnv, TextGenerationEnv, Space

Agents:
    from src.rl import DQNAgent, DQNConfig
    from src.rl import PPOAgent, PPOConfig
    from src.rl import A3CAgent, A3CConfig

Policies:
    from src.rl import DiscretePolicy, ContinuousPolicy, PolicyConfig, make_policy

Value:
    from src.rl import ValueNetwork, QNetwork, compute_gae

Memory:
    from src.rl import ReplayBuffer, PrioritizedReplayBuffer, SumTree

Reward:
    from src.rl import RewardNormaliser, RandomNetworkDistillation, LLMRewardShaper
"""

from src.rl.environments.base_env import BaseEnvironment, CartPoleEnv, Space
from src.rl.environments.text_env import TextGenerationEnv
from src.rl.policies.policy_network import (
    DiscretePolicy, ContinuousPolicy, PolicyConfig, make_policy,
)
from src.rl.value.value_network import ValueNetwork, QNetwork, compute_gae
from src.rl.memory.replay_buffer import (
    ReplayBuffer, PrioritizedReplayBuffer, SumTree,
    Transition, TransitionBatch,
)
from src.rl.agents.dqn import DQNAgent, DQNConfig
from src.rl.agents.ppo import PPOAgent, PPOConfig, RolloutBuffer
from src.rl.agents.a3c import A3CAgent, A3CConfig, A3CWorker
from src.rl.reward.reward_shaping import (
    RewardNormaliser, RandomNetworkDistillation, LLMRewardShaper,
)

__all__ = [
    # Environments
    "BaseEnvironment", "CartPoleEnv", "TextGenerationEnv", "Space",
    # Policies
    "DiscretePolicy", "ContinuousPolicy", "PolicyConfig", "make_policy",
    # Value
    "ValueNetwork", "QNetwork", "compute_gae",
    # Memory
    "ReplayBuffer", "PrioritizedReplayBuffer", "SumTree",
    "Transition", "TransitionBatch",
    # Agents
    "DQNAgent", "DQNConfig",
    "PPOAgent", "PPOConfig", "RolloutBuffer",
    "A3CAgent", "A3CConfig", "A3CWorker",
    # Reward
    "RewardNormaliser", "RandomNetworkDistillation", "LLMRewardShaper",
]
