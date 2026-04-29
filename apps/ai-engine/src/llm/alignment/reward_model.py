"""
InsightSerenity AI Engine — Reward Model
=========================================
The reward model is the core of RLHF (Reinforcement Learning from Human
Feedback). It learns to predict human preference: given a prompt and two
possible completions, which one would a human prefer?

Architecture:
    - Base: GPTDecoder (our pretrained LLM) — shares all transformer layers
    - Head: Linear(d_model, 1) — scalar reward output

Training objective: Bradley-Terry preference model.
Given (prompt, chosen_response, rejected_response):
    reward_chosen   = reward_model(prompt + chosen)
    reward_rejected = reward_model(prompt + rejected)
    loss = -log(sigmoid(reward_chosen - reward_rejected))

This loss pushes reward_chosen > reward_rejected.

The reward model takes the reward at the LAST non-padding token position
(the end-of-sequence position), which aggregates information about the
entire completion.

Why not just use RLHF off-the-shelf?
    - We own the reward model, not a cloud API
    - We control what "good" means for our platform
    - We can fine-tune the reward signal over time with user feedback
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from src.architectures.transformer.decoder.gpt_decoder import GPTDecoder
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.file_io import iter_jsonl
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RewardModel(nn.Module):
    """
    Scalar reward model built on top of a GPTDecoder backbone.

    Takes a sequence of token IDs and returns a single scalar reward value
    for the entire sequence. The reward is extracted from the hidden state
    at the last non-padding token (the final token before EOS/padding).

    Args:
        backbone:  Pretrained GPTDecoder. Its weights provide a strong
                   language representation to build the reward on.
        freeze_layers: Number of transformer layers to freeze during reward
                       model training. Freezing early layers speeds training
                       and prevents catastrophic forgetting of language knowledge.
                       Default 0 (fine-tune everything).
    """

    def __init__(
        self,
        backbone:      GPTDecoder,
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.d_model  = backbone.config.d_model

        # Scalar reward head: one value per sequence
        self.reward_head = nn.Linear(self.d_model, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, std=0.02)

        # Remove the LM head from the backbone — we only need the hidden states
        # We keep it in memory but detach it from the computational graph
        if hasattr(self.backbone, "lm_head"):
            self.backbone.lm_head.requires_grad_(False)

        # Optionally freeze early transformer layers
        if freeze_layers > 0:
            self._freeze_early_layers(freeze_layers)

        logger.info(
            "RewardModel initialised",
            d_model=self.d_model,
            frozen_layers=freeze_layers,
        )

    def forward(
        self,
        input_ids:      Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute scalar reward for each sequence in the batch.

        Args:
            input_ids:      (B, T) — token IDs of prompt + completion.
            attention_mask: (B, T) — 1 for real tokens, 0 for padding.

        Returns:
            (B, 1) — scalar reward for each sequence. Higher = better.
        """
        # Get hidden states from the backbone (no LM head)
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = output["last_hidden"]   # (B, T, D)

        # Extract reward at the last NON-PADDING position for each sequence
        # This aggregates the model's understanding of the full sequence
        if attention_mask is not None:
            # Find the index of the last real token in each sequence
            last_token_idx = attention_mask.sum(dim=1) - 1   # (B,)
        else:
            last_token_idx = torch.full(
                (hidden.size(0),),
                hidden.size(1) - 1,
                device=hidden.device,
            )

        # Gather the hidden state at the last token position
        # last_token_idx: (B,) → (B, 1, D) for gather
        idx_expanded = last_token_idx.view(-1, 1, 1).expand(-1, 1, self.d_model)
        last_hidden  = hidden.gather(1, idx_expanded).squeeze(1)   # (B, D)

        # Scalar reward
        reward = self.reward_head(last_hidden)   # (B, 1)
        return reward

    def _freeze_early_layers(self, n_layers: int) -> None:
        """Freeze the first `n_layers` transformer layers."""
        for i, layer in enumerate(self.backbone.layers):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad_(False)
        logger.info("Froze early layers", n=n_layers)


class PreferenceDataset(Dataset):
    """
    Dataset of human preference pairs for reward model training.

    Each example is a (prompt, chosen, rejected) triple where:
        - prompt:   The input question or context
        - chosen:   The preferred completion
        - rejected: The less preferred completion

    JSONL format (one record per line):
        {
            "prompt":   "What is 2+2?",
            "chosen":   "2+2 equals 4.",
            "rejected": "I don't know math."
        }

    Args:
        data_path:   Path to JSONL preference data.
        tokenizer:   Tokenizer with encode() method.
        max_seq_len: Maximum token length for prompt + response.
    """

    def __init__(
        self,
        data_path:   str,
        tokenizer:   Any,
        max_seq_len: int = 512,
    ) -> None:
        self._tokenizer  = tokenizer
        self._max_seq_len = max_seq_len

        logger.info("Loading preference dataset", path=data_path)
        self._pairs: List[Dict[str, Tensor]] = []
        skipped = 0

        for record in iter_jsonl(data_path):
            prompt   = record.get("prompt", "")
            chosen   = record.get("chosen", "")
            rejected = record.get("rejected", "")

            if not (prompt and chosen and rejected):
                skipped += 1
                continue

            chosen_ids   = self._encode_pair(prompt, chosen)
            rejected_ids = self._encode_pair(prompt, rejected)

            if chosen_ids is None or rejected_ids is None:
                skipped += 1
                continue

            self._pairs.append({
                "chosen_input_ids":      chosen_ids["input_ids"],
                "chosen_attention_mask": chosen_ids["attention_mask"],
                "rejected_input_ids":    rejected_ids["input_ids"],
                "rejected_attention_mask": rejected_ids["attention_mask"],
            })

        logger.info(
            "Preference dataset ready",
            pairs=len(self._pairs),
            skipped=skipped,
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return self._pairs[idx]

    def _encode_pair(
        self, prompt: str, response: str
    ) -> Optional[Dict[str, Tensor]]:
        """Tokenise and pad a (prompt, response) pair."""
        full_text = prompt.strip() + "\n" + response.strip() + ST.EOS
        ids = self._tokenizer.encode(full_text, add_special_tokens=False)

        if len(ids) < 2:
            return None

        ids     = ids[:self._max_seq_len]
        pad_len = self._max_seq_len - len(ids)
        padded  = ids + [ST.PAD_ID] * pad_len
        mask    = [1] * len(ids) + [0] * pad_len

        return {
            "input_ids":      torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask,   dtype=torch.long),
        }


class RewardModelTrainer:
    """
    Trains the reward model on human preference pairs using the
    Bradley-Terry pairwise ranking loss.

    Loss = -log(sigmoid(r_chosen - r_rejected))

    This loss is:
        - Minimised when r_chosen >> r_rejected (clear preference)
        - Maximised when r_chosen ≈ r_rejected  (model is uncertain)

    Args:
        reward_model:  The RewardModel instance.
        train_loader:  DataLoader of PreferenceDataset.
        lr:            Learning rate. Default 1e-5.
        device:        Compute device.
    """

    def __init__(
        self,
        reward_model: RewardModel,
        train_loader: DataLoader,
        lr:           float = 1e-5,
        device:       str   = "cpu",
    ) -> None:
        self.model       = reward_model
        self.loader      = train_loader
        self.device      = torch.device(device)
        self.model.to(self.device)

        # Only train parameters that have gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of reward model training."""
        self.model.train()
        total_loss  = 0.0
        total_acc   = 0.0
        n_batches   = 0

        for batch in self.loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Score chosen and rejected completions
            r_chosen   = self.model(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
            ).squeeze(-1)   # (B,)

            r_rejected = self.model(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
            ).squeeze(-1)   # (B,)

            # Bradley-Terry loss: -log(σ(r_chosen - r_rejected))
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            # Accuracy: fraction of pairs where chosen > rejected
            accuracy = (r_chosen > r_rejected).float().mean().item()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy
            n_batches  += 1

        return {
            "reward_loss":     total_loss / max(n_batches, 1),
            "reward_accuracy": total_acc  / max(n_batches, 1),
        }
