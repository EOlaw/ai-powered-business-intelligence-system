"""
InsightSerenity AI Engine — Base RL Environment
=================================================
Abstract base class defining the Gym-compatible environment interface.
All InsightSerenity RL environments inherit from this class.

The environment represents the world the agent interacts with:
    - state (observation): what the agent sees
    - action: what the agent can do
    - reward: the scalar feedback signal
    - done: whether the episode has ended

Gym-compatible means any standard RL library (StableBaselines3, RLlib, etc.)
can use these environments directly.

Episode lifecycle:
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

Space definitions:
    observation_space: Box (continuous) or Discrete (integer)
    action_space:      Box (continuous) or Discrete (integer)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Space:
    """Describes the shape and bounds of an observation or action space."""
    shape:    Tuple[int, ...]
    dtype:    np.dtype
    low:      Optional[np.ndarray] = None
    high:     Optional[np.ndarray] = None
    n:        Optional[int]        = None    # For discrete spaces: number of actions

    @property
    def is_discrete(self) -> bool:
        return self.n is not None

    @property
    def is_continuous(self) -> bool:
        return self.n is None

    def sample(self) -> np.ndarray:
        """Sample a random element from this space."""
        if self.is_discrete:
            return np.array(np.random.randint(0, self.n), dtype=self.dtype)
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        """Return True if x is a valid element of this space."""
        if self.is_discrete:
            return 0 <= int(x) < self.n
        return np.all(x >= self.low) and np.all(x <= self.high)

    @classmethod
    def discrete(cls, n: int) -> "Space":
        """Create a discrete space with n actions (integers 0..n-1)."""
        return cls(shape=(1,), dtype=np.int64, n=n)

    @classmethod
    def box(cls, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32) -> "Space":
        """Create a continuous box space."""
        return cls(
            shape=shape,
            dtype=dtype,
            low=np.full(shape, low, dtype=dtype),
            high=np.full(shape, high, dtype=dtype),
        )


class BaseEnvironment(ABC):
    """
    Abstract Gym-compatible environment interface.

    All RL environments in InsightSerenity inherit from this.
    The interface matches OpenAI Gym's v0.26+ API.

    Concrete environments must implement:
        reset()  → initial observation
        step()   → (next_obs, reward, terminated, truncated, info)

    And declare:
        observation_space: Space
        action_space:      Space
    """

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Define the observation (state) space."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Define the action space."""
        ...

    @abstractmethod
    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to an initial state.

        Args:
            seed:    Random seed for reproducibility.
            options: Optional dict of reset options.

        Returns:
            Tuple (observation, info):
                observation: Initial state array.
                info:        Auxiliary diagnostic information.
        """
        ...

    @abstractmethod
    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply an action and advance the environment one step.

        Args:
            action: The action selected by the agent.

        Returns:
            Tuple (observation, reward, terminated, truncated, info):
                observation:  Next state after the action.
                reward:       Scalar reward signal.
                terminated:   True if the episode ended naturally (goal/failure).
                truncated:    True if the episode ended due to a time limit.
                info:         Auxiliary diagnostic information.
        """
        ...

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment (optional). Returns an RGB array or None.
        Override in environments that support visual rendering.
        """
        return None

    def close(self) -> None:
        """Clean up environment resources. Override if needed."""

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed. Override for deterministic environments."""
        np.random.seed(seed)

    # ── Compatibility shim for Gym v0.21 style (terminated only, no truncated) ──

    def step_compat(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Gym v0.21 compatible step — returns (obs, reward, done, info).
        Combines terminated and truncated into a single `done` flag.
        """
        obs, reward, terminated, truncated, info = self.step(action)
        return obs, reward, terminated or truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Simple CartPole-like test environment (no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class CartPoleEnv(BaseEnvironment):
    """
    Simplified CartPole environment for testing RL agents.

    The cart-pole problem: balance a pole on a moving cart by applying
    left or right force. The episode ends when the pole falls too far
    or the cart moves out of bounds.

    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Action: 0 (push left) or 1 (push right)
    Reward: +1 for every step the pole stays balanced
    Done: pole angle > 12° or cart position > 2.4

    This is a physics simulation with simplified equations of motion.
    """

    # Physics constants
    GRAVITY     = 9.8
    MASS_CART   = 1.0
    MASS_POLE   = 0.1
    POLE_HALF_L = 0.5   # Half pole length
    FORCE_MAG   = 10.0
    TAU         = 0.02  # Time step (seconds)
    MAX_STEPS   = 500

    def __init__(self) -> None:
        self._state:     Optional[np.ndarray] = None
        self._n_steps:   int                   = 0
        self._rng = np.random.RandomState(42)

    @property
    def observation_space(self) -> Space:
        return Space.box(-4.8, 4.8, shape=(4,))

    @property
    def action_space(self) -> Space:
        return Space.discrete(n=2)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._state   = self._rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self._n_steps = 0
        return self._state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        x, x_dot, theta, theta_dot = self._state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        total_mass   = self.MASS_CART + self.MASS_POLE
        pole_mass_l  = self.MASS_POLE * self.POLE_HALF_L

        cos_theta    = np.cos(theta)
        sin_theta    = np.sin(theta)

        temp         = (force + pole_mass_l * theta_dot ** 2 * sin_theta) / total_mass
        theta_acc    = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.POLE_HALF_L * (4.0 / 3.0 - self.MASS_POLE * cos_theta ** 2 / total_mass)
        )
        x_acc        = temp - pole_mass_l * theta_acc * cos_theta / total_mass

        # Euler integration
        x        += self.TAU * x_dot
        x_dot    += self.TAU * x_acc
        theta    += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._n_steps += 1

        terminated = bool(
            abs(x)     > 2.4 or
            abs(theta) > 12 * np.pi / 180
        )
        truncated  = self._n_steps >= self.MAX_STEPS
        reward     = 1.0 if not terminated else 0.0

        return self._state.copy(), reward, terminated, truncated, {}
