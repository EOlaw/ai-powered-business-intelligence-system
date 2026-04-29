"""
InsightSerenity AI Engine — RLHF Trainer (PPO)
================================================
Reinforcement Learning from Human Feedback using Proximal Policy Optimization.

RLHF Pipeline:
    1. Start with a pretrained + SFT model (the "policy" model)
    2. Use the trained RewardModel to score policy's completions
    3. Use PPO to update the policy to maximise reward
       while staying close to the SFT model (via KL penalty)

The KL penalty is critical. Without it, the policy collapses to reward
hacking — generating text that scores highly on the reward model but is
gibberish or repetitive. The KL term anchors the policy to the SFT model.

PPO objective (simplified for LLMs):
    L = E[min(r * A, clip(r, 1-ε, 1+ε) * A)] - β * KL(policy || ref)

where:
    r = ratio of new policy probability to old policy probability
    A = advantage = reward - value_function(state)
    β = KL penalty coefficient (typically 0.01-0.1)
    ε = PPO clip range (typically 0.2)

Value function: a separate linear head on the policy's hidden states
that predicts the expected reward for the current state.

Note: Full distributed RLHF (as used by ChatGPT) requires running
4 models simultaneously: policy, reference, reward model, and value model.
This implementation runs them on a single GPU for clarity and correctness.
Scaling to multiple GPUs requires model parallelism (handled in Phase 14).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.llm.alignment.reward_model import RewardModel
from src.llm.inference.generator import TextGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for the RLHF training loop."""
    # PPO hyperparameters
    ppo_clip_eps:    float = 0.2    # PPO clip range ε
    value_coeff:     float = 0.5    # Value loss coefficient
    entropy_coeff:   float = 0.01   # Entropy bonus coefficient (encourage exploration)
    kl_coeff:        float = 0.04   # KL divergence penalty coefficient β
    kl_target:       float = 0.02   # Target KL divergence (for adaptive β)
    adaptive_kl:     bool  = True   # Adjust β automatically to hit kl_target

    # Training
    lr:              float = 1e-6   # Policy learning rate (much smaller than SFT)
    max_epochs:      int   = 1      # PPO epochs per rollout
    rollout_batch:   int   = 16     # Prompts per rollout batch
    ppo_batch:       int   = 4      # Mini-batch size for PPO update
    max_new_tokens:  int   = 256    # Max tokens generated per prompt
    device:          str   = "cpu"


class PPOBuffer:
    """
    Rollout buffer for PPO. Stores generated sequences and their rewards,
    old log probabilities, and value estimates.

    One buffer collects a full rollout batch, then PPO updates the policy
    over multiple epochs of mini-batches from the buffer.
    """

    def __init__(self) -> None:
        self.sequences:      List[Tensor] = []
        self.attention_masks: List[Tensor] = []
        self.rewards:        List[float]  = []
        self.old_log_probs:  List[Tensor] = []
        self.values:         List[float]  = []
        self.advantages:     List[float]  = []

    def add(
        self,
        sequence:      Tensor,
        attention_mask: Tensor,
        reward:        float,
        old_log_prob:  Tensor,
        value:         float,
    ) -> None:
        self.sequences.append(sequence)
        self.attention_masks.append(attention_mask)
        self.rewards.append(reward)
        self.old_log_probs.append(old_log_prob)
        self.values.append(value)

    def compute_advantages(self) -> None:
        """Compute advantage estimates: A = reward - value_baseline."""
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        values  = torch.tensor(self.values,  dtype=torch.float)
        adv     = rewards - values
        # Normalise advantages for stable training
        self.advantages = ((adv - adv.mean()) / (adv.std() + 1e-8)).tolist()

    def clear(self) -> None:
        """Clear the buffer after a PPO update cycle."""
        self.__init__()

    def __len__(self) -> int:
        return len(self.sequences)


class ValueHead(nn.Module):
    """
    Value function head: predicts expected future reward from hidden state.
    Added on top of the policy model to estimate the baseline for advantage.

    Args:
        d_model: Policy model hidden dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model // 2)
        self.value = nn.Linear(d_model // 2, 1)
        nn.init.normal_(self.dense.weight, std=0.01)
        nn.init.normal_(self.value.weight, std=0.01)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Args:
            hidden: (B, D) — last hidden state of the sequence.

        Returns:
            (B, 1) — scalar value estimate.
        """
        return self.value(F.tanh(self.dense(hidden)))


class RLHFTrainer:
    """
    Full RLHF training loop using PPO.

    Manages four models:
        policy:     The model being trained (SFT-initialised GPTDecoder)
        ref_policy: Frozen copy of the SFT model (KL penalty reference)
        reward:     Trained RewardModel
        value_head: Separate value function head added to policy

    Training cycle:
        1. Sample prompts from the training set
        2. Generate completions from the policy
        3. Score completions with the reward model
        4. Estimate value function
        5. Compute advantages = reward - value
        6. PPO update: optimise policy to increase advantage
        7. Penalise KL divergence from reference policy

    Args:
        policy:        The policy model (SFT-initialised GPTDecoder).
        ref_policy:    Frozen reference model (copy of SFT model).
        reward_model:  Trained RewardModel for scoring completions.
        tokenizer:     Tokenizer for encoding/decoding.
        config:        RLHFConfig.
    """

    def __init__(
        self,
        policy:       nn.Module,
        ref_policy:   nn.Module,
        reward_model: RewardModel,
        tokenizer:    Any,
        config:       RLHFConfig,
    ) -> None:
        self.config       = config
        self.tokenizer    = tokenizer
        self.device       = torch.device(config.device)

        self.policy       = policy.to(self.device)
        self.ref_policy   = ref_policy.to(self.device)
        self.reward_model = reward_model.to(self.device)

        # Freeze the reference policy and reward model
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)
        for param in self.reward_model.parameters():
            param.requires_grad_(False)

        # Value head — attached to but not part of the policy model
        d_model          = getattr(policy, "config", type("c", (), {"d_model": 768})()).d_model
        self.value_head  = ValueHead(d_model).to(self.device)

        # Separate optimizers for policy and value function
        self.policy_opt = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=config.lr,
            weight_decay=0.01,
        )

        self._kl_coeff = config.kl_coeff
        self.buffer    = PPOBuffer()

        # Generator for rollout generation
        self.generator = TextGenerator(
            model=self.policy,
            tokenizer=tokenizer,
            device=str(self.device),
        )

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Run one RLHF training step: generate rollouts, compute rewards, PPO update.

        Args:
            prompts: List of text prompts for rollout generation.

        Returns:
            Dict of training metrics.
        """
        self.buffer.clear()

        # ── Phase 1: Generate rollouts ──────────────────────────────────────
        self.policy.eval()
        with torch.no_grad():
            for prompt in prompts:
                completion = self.generator.generate(
                    prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    strategy="top_p",
                    top_p=0.9,
                    temperature=1.0,
                )

                full_text  = prompt + completion
                ids        = self.tokenizer.encode(full_text, add_special_tokens=False)
                ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
                mask       = torch.ones_like(ids_tensor)

                # Reward score
                reward     = self.reward_model(ids_tensor, mask).item()

                # Old log probability (for importance sampling ratio)
                output     = self.policy(ids_tensor, attention_mask=mask)
                logits     = output["logits"][0]   # (T, V)
                log_probs  = F.log_softmax(logits, dim=-1)
                # Sum log probs over completion tokens
                old_lp     = log_probs[len(prompt.split()):].gather(
                    1,
                    ids_tensor[0, len(prompt.split()):].unsqueeze(-1),
                ).sum()

                # Value estimate
                hidden     = output["last_hidden"][0, -1, :]   # Last hidden state
                value      = self.value_head(hidden.unsqueeze(0)).item()

                self.buffer.add(ids_tensor, mask, reward, old_lp, value)

        self.buffer.compute_advantages()

        # ── Phase 2: PPO update ─────────────────────────────────────────────
        self.policy.train()
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_kl          = 0.0
        n_updates         = 0

        for epoch in range(self.config.max_epochs):
            for i in range(0, len(self.buffer), self.config.ppo_batch):
                batch_seqs   = self.buffer.sequences[i:i + self.config.ppo_batch]
                batch_masks  = self.buffer.attention_masks[i:i + self.config.ppo_batch]
                batch_adv    = self.buffer.advantages[i:i + self.config.ppo_batch]
                batch_rewards = self.buffer.rewards[i:i + self.config.ppo_batch]
                batch_old_lp  = self.buffer.old_log_probs[i:i + self.config.ppo_batch]

                adv_tensor = torch.tensor(batch_adv, device=self.device)
                rew_tensor = torch.tensor(batch_rewards, device=self.device)

                policy_loss = torch.tensor(0.0, device=self.device)
                value_loss  = torch.tensor(0.0, device=self.device)
                kl_divs     = []

                for seq, mask, adv, old_lp, reward in zip(
                    batch_seqs, batch_masks, adv_tensor, batch_old_lp, rew_tensor
                ):
                    # New policy log probabilities
                    output   = self.policy(seq, attention_mask=mask)
                    logits   = output["logits"][0]
                    new_lp   = F.log_softmax(logits, dim=-1).gather(
                        1, seq[0].unsqueeze(-1)
                    ).sum()

                    # Reference policy log probs (for KL)
                    with torch.no_grad():
                        ref_output = self.ref_policy(seq, attention_mask=mask)
                        ref_logits = ref_output["logits"][0]
                    ref_lp = F.log_softmax(ref_logits, dim=-1).gather(
                        1, seq[0].unsqueeze(-1)
                    ).sum()

                    # PPO importance ratio
                    ratio = torch.exp(new_lp - old_lp)

                    # Clipped PPO objective
                    clip_ratio   = ratio.clamp(
                        1 - self.config.ppo_clip_eps,
                        1 + self.config.ppo_clip_eps,
                    )
                    policy_loss  = policy_loss - torch.min(ratio * adv, clip_ratio * adv)

                    # KL divergence: KL(policy || ref) = new_lp - ref_lp (per token)
                    kl = (new_lp - ref_lp).mean()
                    kl_divs.append(kl.item())

                    # Value loss
                    hidden_last = output["last_hidden"][0, -1, :]
                    value_pred  = self.value_head(hidden_last.unsqueeze(0))
                    value_loss  = value_loss + F.mse_loss(value_pred.squeeze(), reward)

                policy_loss /= len(batch_seqs)
                value_loss  /= len(batch_seqs)
                mean_kl      = sum(kl_divs) / max(len(kl_divs), 1)

                # Total loss with KL penalty
                total_loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    + self._kl_coeff * mean_kl
                )

                self.policy_opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_opt.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_kl          += mean_kl
                n_updates         += 1

        # Adaptive KL coefficient
        if self.config.adaptive_kl and n_updates > 0:
            mean_kl_all = total_kl / n_updates
            if mean_kl_all > 2 * self.config.kl_target:
                self._kl_coeff *= 1.5   # Too much divergence — increase penalty
            elif mean_kl_all < 0.5 * self.config.kl_target:
                self._kl_coeff /= 1.5   # Too little divergence — decrease penalty

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss":  total_value_loss  / max(n_updates, 1),
            "kl_div":      total_kl          / max(n_updates, 1),
            "kl_coeff":    self._kl_coeff,
            "mean_reward": sum(self.buffer.rewards) / max(len(self.buffer.rewards), 1),
        }
