"""
InsightSerenity AI Engine — Masked Language Modeling Objective
==============================================================
Implements the data preparation for Masked Language Model (MLM) pretraining.
MLM is the training objective used by BERT, RoBERTa, and other bidirectional
encoder language models.

Objective: Randomly mask 15% of input tokens, then train the model to
predict the original token at each masked position.

BERT masking procedure (applied to each selected position):
    - 80% of the time: replace with [MASK] token
    - 10% of the time: replace with a random token from the vocabulary
    - 10% of the time: keep the original token unchanged

The 10%/10% trick prevents the model from learning that [MASK] means "something
important is here" — it forces it to maintain useful representations at every
position even when the token is unchanged.

Only masked positions contribute to the loss — non-masked positions get
label = -100 (ignored by CrossEntropyLoss). This means the model only
gets gradient signal at the 15% masked positions.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.file_io import iter_jsonl
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLMDataCollator:
    """
    Applies BERT-style random masking to a batch of token sequences.

    Called by the DataLoader as the collate_fn. Each batch is independently
    masked with fresh random masks — the same token gets a different mask
    in each epoch, exposing the model to more training signal.

    Args:
        tokenizer:       Tokenizer instance (for vocabulary size and special IDs).
        mask_prob:       Fraction of tokens to mask. Default 0.15 (BERT standard).
        mask_token_id:   Token ID for [MASK]. Defaults to ST.MASK_ID.
        ignore_index:    Label at non-masked positions. Default -100.
        whole_word_mask: If True, mask whole words rather than sub-words.
                         Improves MLM quality for longer span recovery.
    """

    def __init__(
        self,
        tokenizer:       Any,
        mask_prob:       float = 0.15,
        mask_token_id:   Optional[int] = None,
        ignore_index:    int = -100,
        whole_word_mask: bool = False,
    ) -> None:
        self.vocab_size      = tokenizer.vocab_size
        self.mask_prob       = mask_prob
        self.mask_token_id   = mask_token_id or ST.MASK_ID
        self.ignore_index    = ignore_index
        self.whole_word_mask = whole_word_mask

        # IDs of special tokens — these are never masked
        self.special_ids = set(ST.all_ids())

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Apply masking to a batch of examples.

        Args:
            examples: List of dicts, each with "input_ids" tensor.

        Returns:
            Dict with:
                input_ids:      (B, T) — inputs with masks applied
                labels:         (B, T) — original tokens at masked positions, -100 elsewhere
                attention_mask: (B, T) — 1 for real tokens, 0 for padding
        """
        input_ids_batch      = []
        labels_batch         = []
        attention_mask_batch = []

        for example in examples:
            input_ids      = example["input_ids"]
            attention_mask = example.get("attention_mask")

            if isinstance(input_ids, Tensor):
                input_ids = input_ids.tolist()
            if isinstance(attention_mask, Tensor):
                attention_mask = attention_mask.tolist()
            else:
                attention_mask = [1] * len(input_ids)

            masked_ids, labels = self._mask_sequence(input_ids)

            input_ids_batch.append(masked_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        return {
            "input_ids":      torch.tensor(input_ids_batch,      dtype=torch.long),
            "labels":         torch.tensor(labels_batch,         dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        }

    def _mask_sequence(
        self, input_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply BERT masking procedure to a single sequence.

        Args:
            input_ids: List of integer token IDs.

        Returns:
            Tuple (masked_ids, labels):
                masked_ids: Input IDs with some replaced by [MASK] or random tokens
                labels:     Original IDs at masked positions, -100 elsewhere
        """
        masked_ids = input_ids.copy()
        labels     = [self.ignore_index] * len(input_ids)

        # Candidate positions: non-special, non-padding tokens
        candidate_positions = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in self.special_ids
        ]

        if not candidate_positions:
            return masked_ids, labels

        # Sample mask_prob fraction of candidates
        n_to_mask = max(1, int(len(candidate_positions) * self.mask_prob))
        masked_positions = random.sample(candidate_positions, min(n_to_mask, len(candidate_positions)))

        for pos in masked_positions:
            original_id    = input_ids[pos]
            labels[pos]    = original_id   # Target: predict the original token

            # Apply BERT masking rule
            r = random.random()
            if r < 0.80:
                # 80%: replace with [MASK]
                masked_ids[pos] = self.mask_token_id
            elif r < 0.90:
                # 10%: replace with a random token
                masked_ids[pos] = random.randint(
                    len(ST.all_ids()),   # Skip special tokens
                    self.vocab_size - 1,
                )
            # else: 10% keep original — label is set, but input is unchanged

        return masked_ids, labels


class MLMDataset(Dataset):
    """
    In-memory MLM dataset from a JSONL corpus.

    Pre-tokenises and stores all sequences. Masking is applied dynamically
    at batch time (by MLMDataCollator) so each epoch sees different masks.

    Args:
        corpus_path:  Path to JSONL file.
        tokenizer:    Tokenizer with .encode(text) → List[int].
        max_length:   Maximum sequence length. Sequences are truncated/padded.
        text_field:   Key in JSONL records.
        max_docs:     Maximum documents to load.
    """

    def __init__(
        self,
        corpus_path: str,
        tokenizer:   Any,
        max_length:  int = 512,
        text_field:  str = "text",
        max_docs:    Optional[int] = None,
    ) -> None:
        self.max_length  = max_length
        self._pad_id     = ST.PAD_ID

        logger.info("Building MLM dataset", corpus=corpus_path, max_length=max_length)

        self._examples: List[Dict[str, Tensor]] = []
        docs_loaded = 0

        for record in iter_jsonl(corpus_path):
            text = record.get(text_field, "")
            if not text:
                continue

            # Encode with BOS/EOS
            ids = tokenizer.encode(text, add_special_tokens=True)

            # Truncate to max_length
            ids = ids[:max_length]
            seq_len = len(ids)

            # Pad to max_length
            padding    = max_length - seq_len
            padded_ids = ids + [self._pad_id] * padding
            attn_mask  = [1] * seq_len + [0] * padding

            self._examples.append({
                "input_ids":      torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
            })

            docs_loaded += 1
            if max_docs is not None and docs_loaded >= max_docs:
                break

        logger.info("MLM dataset ready", examples=len(self._examples), docs=docs_loaded)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return self._examples[idx]
