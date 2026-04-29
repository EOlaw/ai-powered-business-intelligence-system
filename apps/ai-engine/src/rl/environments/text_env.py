"""
InsightSerenity AI Engine — Text Generation RL Environment
===========================================================
Frames language model text generation as a Markov Decision Process (MDP),
enabling RL algorithms (PPO, A3C) to train language models directly from
reward signals rather than cross-entropy supervised learning.

MDP formulation:
    State:    The sequence of tokens generated so far (a vector of IDs)
    Action:   The next token to generate (integer in {0..vocab_size-1})
    Reward:   Assigned by the reward model at the end of the episode
              (terminal reward), or at each step (dense reward)
    Done:     When EOS token is generated, or max_steps is reached

Episode lifecycle:
    1. Reset: present a prompt to the agent
    2. Step:  agent selects next token (action)
    3. Append the selected token to the current sequence
    4. If EOS or max_steps: compute reward from reward model
    5. Return (new_state, reward, done, info)

The state representation: the most recent `context_length` token IDs.
Older tokens are dropped (sliding window). This bounds state size.

Reward shaping:
    terminal_only: Reward is 0 at every step except the last (sparse).
                   Simple but makes credit assignment harder for long sequences.
    step_reward:   A small per-step reward can be added (e.g. avoid repetition).
    kl_penalty:    KL divergence from reference policy subtracted from reward.
                   This prevents the policy from drifting too far.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.rl.environments.base_env import BaseEnvironment, Space
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextGenerationEnv(BaseEnvironment):
    """
    Text generation as a reinforcement learning environment.

    The agent (a language model) generates one token per step.
    A reward model scores the complete generation at the end of the episode.

    This environment is used in the RLHF pipeline (Phase 6) to align the
    language model to human preferences via PPO.

    Args:
        tokenizer:       Trained tokenizer (for encoding prompts).
        reward_model:    RewardModel that scores completed sequences.
        prompts:         List of training prompts. A random one is selected per episode.
        max_steps:       Maximum tokens to generate per episode.
        context_length:  Observation window size (last N token IDs).
        vocab_size:      Total vocabulary size (= action space size).
        kl_coeff:        KL penalty coefficient (0 = no penalty).
        ref_model:       Reference policy for KL computation. If None, no KL penalty.
        device:          Compute device for model forward passes.
    """

    def __init__(
        self,
        tokenizer:      Any,
        reward_model:   Any,
        prompts:        List[str],
        max_steps:      int   = 256,
        context_length: int   = 64,
        vocab_size:     int   = 32_000,
        kl_coeff:       float = 0.04,
        ref_model:      Optional[Any] = None,
        device:         str   = "cpu",
    ) -> None:
        self.tokenizer      = tokenizer
        self.reward_model   = reward_model
        self.prompts        = prompts
        self.max_steps      = max_steps
        self.context_length = context_length
        self.vocab_size     = vocab_size
        self.kl_coeff       = kl_coeff
        self.ref_model      = ref_model
        self.device         = torch.device(device)

        # Episode state
        self._token_ids:     List[int] = []
        self._prompt_ids:    List[int] = []
        self._current_step:  int       = 0
        self._rng = np.random.RandomState(42)

    @property
    def observation_space(self) -> Space:
        """Observation: last context_length token IDs."""
        return Space.box(0, self.vocab_size - 1, shape=(self.context_length,),
                         dtype=np.int64)

    @property
    def action_space(self) -> Space:
        """Action: select the next token ID."""
        return Space.discrete(n=self.vocab_size)

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Start a new episode with a randomly selected prompt.

        Returns:
            (observation, info) where observation is the prompt token IDs
            padded/truncated to context_length.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Select a random prompt
        prompt = self._rng.choice(self.prompts)

        # Encode the prompt (no BOS/EOS — the formatter adds role tokens)
        self._prompt_ids   = self.tokenizer.encode(prompt, add_special_tokens=False)
        self._token_ids    = list(self._prompt_ids)   # Start with prompt
        self._current_step = 0

        return self._get_observation(), {"prompt": prompt}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Append the selected token and check stopping conditions.

        Args:
            action: Integer token ID to generate.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._token_ids.append(int(action))
        self._current_step += 1

        terminated = int(action) == ST.EOS_ID
        truncated  = self._current_step >= self.max_steps

        done    = terminated or truncated
        reward  = 0.0
        kl_pen  = 0.0

        if done:
            # Compute terminal reward from the reward model
            reward = self._compute_reward()

            # KL penalty to prevent drifting from reference policy
            if self.ref_model is not None and self.kl_coeff > 0:
                kl_pen = self._compute_kl_penalty()
                reward -= self.kl_coeff * kl_pen

        info = {
            "step":       self._current_step,
            "token_id":   action,
            "kl_penalty": kl_pen,
            "n_generated": self._current_step,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def get_generated_text(self) -> str:
        """Decode the generated tokens (excluding prompt) to text."""
        generated_ids = self._token_ids[len(self._prompt_ids):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def get_full_text(self) -> str:
        """Decode the full sequence (prompt + generation) to text."""
        return self.tokenizer.decode(self._token_ids, skip_special_tokens=True)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Return the last context_length token IDs as the observation."""
        ids      = self._token_ids[-self.context_length:]
        # Pad with PAD tokens at the beginning if shorter than context_length
        padded   = [ST.PAD_ID] * (self.context_length - len(ids)) + ids
        return np.array(padded, dtype=np.int64)

    def _compute_reward(self) -> float:
        """Score the full generated sequence with the reward model."""
        try:
            ids    = torch.tensor([self._token_ids], dtype=torch.long, device=self.device)
            mask   = torch.ones_like(ids)
            with torch.no_grad():
                reward = self.reward_model(ids, mask).item()
            return float(reward)
        except Exception as e:
            logger.warning("Reward model scoring failed", error=str(e))
            return 0.0

    def _compute_kl_penalty(self) -> float:
        """
        Estimate KL divergence from reference policy at the last step.
        KL(policy || ref) = log(p_policy / p_ref) per token.
        Returns the mean per-token KL over the generated portion.
        """
        try:
            import torch.nn.functional as F
            ids       = torch.tensor([self._token_ids], dtype=torch.long, device=self.device)
            gen_start = len(self._prompt_ids)

            with torch.no_grad():
                policy_out = self.reward_model.backbone(ids)
                ref_out    = self.ref_model(ids)

            policy_log = F.log_softmax(policy_out["logits"][0, gen_start:], dim=-1)
            ref_log    = F.log_softmax(ref_out["logits"][0, gen_start:], dim=-1)

            kl = (policy_log.exp() * (policy_log - ref_log)).sum(dim=-1).mean()
            return max(0.0, kl.item())
        except Exception:
            return 0.0
