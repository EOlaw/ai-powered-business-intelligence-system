"""
InsightSerenity AI Engine — Experience Replay Buffers
======================================================
Experience replay stores transitions (s, a, r, s', done) collected by the
agent during environment interaction. The DQN agent samples mini-batches
from this buffer to break the temporal correlations in sequential data.

Without replay: sequential data → correlated gradients → unstable training.
With replay:    random sampling → decorrelated gradients → stable training.

Two implementations:

ReplayBuffer (Uniform):
    Simple circular queue. Each transition has equal sampling probability.
    O(1) push, O(batch_size) sample. The standard DQN replay buffer.

PrioritizedReplayBuffer (PER):
    Prioritised Experience Replay (Schaul et al., 2015).
    Transitions are sampled with probability proportional to their TD error:
        p_i ∝ |δ_i|^α + ε
    High-error transitions are sampled more often — the agent learns faster
    from surprising experiences.
    Uses a Sum Tree for O(log N) push and O(log N) sample.
    Importance sampling weights correct for the non-uniform sampling bias.

Reference: "Prioritized Experience Replay", Schaul et al., 2015.
"""

import random
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Transition namedtuple
# ─────────────────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    """A single (s, a, r, s', done) transition."""
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class TransitionBatch(NamedTuple):
    """A batch of transitions converted to PyTorch tensors."""
    states:      Tensor
    actions:     Tensor
    rewards:     Tensor
    next_states: Tensor
    dones:       Tensor
    weights:     Optional[Tensor] = None   # Importance sampling weights (PER only)
    indices:     Optional[List[int]] = None  # Buffer indices (for PER priority update)


# ─────────────────────────────────────────────────────────────────────────────
# Uniform Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular experience replay buffer with uniform sampling.

    The buffer is a fixed-size deque. When full, the oldest transitions
    are overwritten first (FIFO eviction).

    Args:
        capacity:     Maximum number of transitions to store.
        obs_shape:    Shape of the observation (state) array.
        device:       Target device for sampled tensors.
    """

    def __init__(
        self,
        capacity:  int,
        obs_shape: Tuple[int, ...],
        device:    str = "cpu",
    ) -> None:
        self.capacity  = capacity
        self.obs_shape = obs_shape
        self.device    = torch.device(device)

        # Pre-allocate numpy arrays for memory efficiency
        self._states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.bool_)

        self._idx  = 0     # Write pointer (wraps around at capacity)
        self._size = 0     # Current number of stored transitions

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Add a transition to the buffer. Overwrites oldest if full.

        Args:
            state:      Current observation.
            action:     Action taken.
            reward:     Reward received.
            next_state: Next observation.
            done:       Whether the episode ended.
        """
        self._states[self._idx]      = state
        self._actions[self._idx]     = action
        self._rewards[self._idx]     = reward
        self._next_states[self._idx] = next_state
        self._dones[self._idx]       = done

        self._idx  = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            TransitionBatch with tensors ready for the DQN update.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} transitions, need {batch_size}"
            )

        indices = np.random.randint(0, self._size, size=batch_size)
        return self._make_batch(indices)

    def _make_batch(self, indices: np.ndarray) -> TransitionBatch:
        """Convert selected indices to a TransitionBatch of tensors."""
        return TransitionBatch(
            states=      torch.tensor(self._states[indices],      device=self.device),
            actions=     torch.tensor(self._actions[indices],     device=self.device),
            rewards=     torch.tensor(self._rewards[indices],     device=self.device),
            next_states= torch.tensor(self._next_states[indices], device=self.device),
            dones=       torch.tensor(self._dones[indices].astype(np.float32), device=self.device),
        )

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        """True if the buffer has enough transitions to sample a full batch."""
        return self._size >= batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Sum Tree for PER
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary sum tree for O(log N) priority sampling.

    Stores priorities in a binary tree where each internal node equals
    the sum of its children. Leaves hold the individual priorities.

    Sampling is O(log N): descend from root, at each node choose left
    or right child with probability proportional to its sub-sum.

    Args:
        capacity: Number of leaf nodes (= buffer capacity).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree    = np.zeros(2 * capacity)   # Internal + leaf nodes
        self._data    = np.zeros(capacity, dtype=object)
        self._idx     = 0

    def push(self, priority: float, data: object) -> None:
        """Add data with given priority."""
        leaf_idx = self._idx + self.capacity
        self._data[self._idx] = data
        self._update(leaf_idx, priority)
        self._idx = (self._idx + 1) % self.capacity

    def _update(self, leaf_idx: int, priority: float) -> None:
        """Update priority at leaf and propagate sum up to root."""
        delta             = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority

        # Propagate up
        idx = leaf_idx
        while idx > 1:
            idx //= 2
            self._tree[idx] += delta

    def sample(self, value: float) -> Tuple[int, float, object]:
        """
        Sample a leaf proportional to its priority.

        Args:
            value: Uniform sample in [0, total_priority].

        Returns:
            Tuple (leaf_idx, priority, data).
        """
        idx = 1   # Start at root
        while idx < self.capacity:
            left  = 2 * idx
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx    = right

        data_idx = idx - self.capacity
        return data_idx, self._tree[idx], self._data[data_idx]

    def update_priority(self, data_idx: int, priority: float) -> None:
        """Update priority for a specific data index after a TD error update."""
        self._update(data_idx + self.capacity, priority)

    @property
    def total_priority(self) -> float:
        """Total sum of all priorities (stored at root)."""
        return float(self._tree[1])


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Experience Replay
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Prioritised Experience Replay buffer (Schaul et al., 2015).

    Samples transitions proportionally to |TD error|^α.
    High TD error = surprising transition = more learning signal.

    Importance sampling weights correct for the sampling bias:
        w_i = (1/(N × p_i))^β
        β starts low (0.4) and anneals to 1 over training.

    Args:
        capacity:    Maximum number of transitions.
        obs_shape:   Shape of observation.
        alpha:       Priority exponent. 0=uniform, 1=full prioritisation.
                     Default 0.6.
        beta_start:  Initial IS weight exponent. Default 0.4.
        beta_end:    Final IS weight exponent (anneal towards 1). Default 1.0.
        beta_steps:  Steps over which to anneal beta. Default 100000.
        eps:         Small constant added to priorities for stability. Default 1e-6.
        device:      Target device for sampled tensors.
    """

    def __init__(
        self,
        capacity:    int,
        obs_shape:   Tuple[int, ...],
        alpha:       float = 0.6,
        beta_start:  float = 0.4,
        beta_end:    float = 1.0,
        beta_steps:  int   = 100_000,
        eps:         float = 1e-6,
        device:      str   = "cpu",
    ) -> None:
        self.capacity    = capacity
        self.obs_shape   = obs_shape
        self.alpha       = alpha
        self.beta        = beta_start
        self.beta_end    = beta_end
        self.beta_incr   = (beta_end - beta_start) / max(beta_steps, 1)
        self.eps         = eps
        self.device      = torch.device(device)

        self._tree       = SumTree(capacity)
        self._max_prio   = 1.0   # Maximum priority seen so far
        self._size       = 0

        # Pre-allocate storage
        self._states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)
        self._write_idx   = 0

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Add transition with maximum current priority (optimistic initialisation)."""
        idx = self._write_idx
        self._states[idx]      = state
        self._actions[idx]     = action
        self._rewards[idx]     = reward
        self._next_states[idx] = next_state
        self._dones[idx]       = float(done)

        # New transitions get maximum priority so they are sampled at least once
        self._tree.push(self._max_prio ** self.alpha, idx)

        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size      = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a batch proportional to priority.

        Divides the total priority into `batch_size` equal segments and
        samples uniformly from each segment (stratified sampling).
        This ensures diverse coverage even with skewed priorities.

        Returns:
            TransitionBatch including importance sampling weights and indices.
        """
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} transitions, need {batch_size}")

        total     = self._tree.total_priority
        segment   = total / batch_size
        indices   = []
        weights   = np.zeros(batch_size, dtype=np.float32)
        data_idxs = []

        for i in range(batch_size):
            lo     = segment * i
            hi     = segment * (i + 1)
            value  = np.random.uniform(lo, hi)
            data_idx, priority, _ = self._tree.sample(value)
            data_idxs.append(data_idx)
            indices.append(data_idx)

            # Importance sampling weight
            prob       = priority / total
            weights[i] = (self._size * prob + self.eps) ** (-self.beta)

        # Normalise weights by maximum weight
        weights /= weights.max()

        # Anneal beta
        self.beta = min(self.beta_end, self.beta + self.beta_incr)

        idxs = np.array(indices)
        return TransitionBatch(
            states=      torch.tensor(self._states[idxs],      device=self.device),
            actions=     torch.tensor(self._actions[idxs],     device=self.device),
            rewards=     torch.tensor(self._rewards[idxs],     device=self.device),
            next_states= torch.tensor(self._next_states[idxs], device=self.device),
            dones=       torch.tensor(self._dones[idxs],       device=self.device),
            weights=     torch.tensor(weights,                  device=self.device),
            indices=     indices,
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities after a learning step using new TD errors.

        Args:
            indices:   Buffer indices returned by sample().
            td_errors: Absolute TD errors |r + γV(s') - V(s)| per transition.
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(float(error)) + self.eps) ** self.alpha
            self._max_prio = max(self._max_prio, priority)
            self._tree.update_priority(idx, priority)

    def __len__(self) -> int:
        return self._size
