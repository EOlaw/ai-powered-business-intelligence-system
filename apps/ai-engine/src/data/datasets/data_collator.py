"""
InsightSerenity AI Engine — Data Collators
===========================================
Collators are the bridge between individual dataset examples and model-ready
batches. PyTorch's DataLoader calls the collator on a list of examples to
produce a single batched dict of tensors.

Three collators are provided:
    1. DefaultDataCollator      — simple stack of pre-padded tensors
    2. DynamicPaddingCollator   — pads each batch to the longest example
                                  in that batch (reduces wasted computation)
    3. LanguageModelCollator    — creates shifted labels for causal LM training

The DynamicPaddingCollator is recommended for fine-tuning where sequence
lengths vary greatly. For pretraining with fixed-length examples, the
DefaultDataCollator is faster.

Usage:
    from src.data.datasets.data_collator import DynamicPaddingCollator
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=DynamicPaddingCollator(pad_token_id=0),
    )
"""

from typing import Any, Dict, List, Optional

import torch
from torch import LongTensor, Tensor


class DefaultDataCollator:
    """
    Stack a list of pre-padded examples into a batch without any modification.

    Use when all examples already have the same sequence length (e.g. when
    the Dataset pads to a fixed max_length). This is the fastest collator.

    Args:
        keys_to_stack: If provided, only these keys are included in the batch.
                       Defaults to all keys in the first example.
    """

    def __init__(self, keys_to_stack: Optional[List[str]] = None) -> None:
        self._keys = keys_to_stack

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Stack a list of example dicts into a single batched dict.

        Args:
            examples: List of dicts, each mapping feature name → tensor or value.

        Returns:
            Dict mapping feature name → batched tensor of shape (batch, *feature_shape).
        """
        if not examples:
            return {}

        keys = self._keys or list(examples[0].keys())
        batch: Dict[str, Tensor] = {}

        for key in keys:
            values = [ex[key] for ex in examples]
            if isinstance(values[0], Tensor):
                batch[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], int):
                batch[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], float):
                batch[key] = torch.tensor(values, dtype=torch.float)
            else:
                batch[key] = values   # type: ignore[assignment]  # keep as list

        return batch


class DynamicPaddingCollator:
    """
    Collate examples by padding each batch to the maximum sequence length
    within that specific batch (rather than a fixed global max_length).

    Benefits:
        - Eliminates wasted computation on padding tokens in short batches
        - Reduces memory usage during fine-tuning with variable-length data
        - Particularly effective for instruction-following datasets where
          prompts and responses vary widely in length

    Args:
        pad_token_id:   Token ID used for padding input_ids and labels.
        label_pad_id:   Token ID for label padding (default -100 so
                        CrossEntropyLoss ignores these positions).
        padding_side:   "right" (default) or "left". GPT models require
                        right padding; some architectures need left.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_id: int = -100,
        padding_side: str = "right",
    ) -> None:
        self._pad_token_id  = pad_token_id
        self._label_pad_id  = label_pad_id
        self._padding_side  = padding_side

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Pad a batch of variable-length examples to the batch's maximum length.

        Handles these keys automatically:
            input_ids       → padded with pad_token_id
            attention_mask  → padded with 0
            labels          → padded with label_pad_id (-100)
            token_type_ids  → padded with 0

        Other tensor keys are stacked without modification.
        """
        if not examples:
            return {}

        batch: Dict[str, Tensor] = {}
        keys = list(examples[0].keys())

        for key in keys:
            values = [ex[key] for ex in examples]

            if not isinstance(values[0], Tensor):
                batch[key] = values   # type: ignore[assignment]
                continue

            if values[0].dim() == 0:
                # Scalar tensor — just stack
                batch[key] = torch.stack(values)
                continue

            # 1D sequence tensor — needs padding
            max_len = max(v.shape[0] for v in values)
            padded  = self._pad_sequence(values, max_len, key)
            batch[key] = torch.stack(padded)

        return batch

    def _pad_sequence(
        self, sequences: List[Tensor], max_len: int, key: str
    ) -> List[Tensor]:
        """Pad a list of 1D tensors to `max_len`."""
        pad_value = self._get_pad_value(key)
        padded = []

        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if pad_len == 0:
                padded.append(seq)
                continue

            pad_tensor = torch.full((pad_len,), pad_value, dtype=seq.dtype)

            if self._padding_side == "right":
                padded.append(torch.cat([seq, pad_tensor]))
            else:
                padded.append(torch.cat([pad_tensor, seq]))

        return padded

    def _get_pad_value(self, key: str) -> int:
        """Return the appropriate padding value for each key type."""
        if key == "labels":
            return self._label_pad_id
        if key in ("input_ids",):
            return self._pad_token_id
        return 0   # Default: pad with 0 (works for attention_mask, token_type_ids)


class LanguageModelCollator:
    """
    Specialised collator for causal language model pretraining.

    Handles the "packed sequence" approach: instead of padding short sequences,
    multiple short documents are concatenated into a single full-length
    sequence. This maximises GPU utilisation — no computation is wasted on
    padding tokens.

    Documents are separated by an EOS token. The loss is computed on all
    tokens (including across document boundaries), which is standard practice
    for large-scale LLM pretraining.

    Args:
        block_size:   Target sequence length. Examples are concatenated and
                      chunked into blocks of exactly this size.
        eos_token_id: Token ID inserted between concatenated documents.
        pad_token_id: Fallback padding if the last chunk is shorter than block_size.
    """

    def __init__(
        self,
        block_size: int = 1024,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
    ) -> None:
        self._block_size    = block_size
        self._eos_token_id  = eos_token_id
        self._pad_token_id  = pad_token_id

        # Buffer to accumulate tokens across batches
        self._token_buffer: List[int] = []

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Concatenate examples and chunk into fixed-size blocks.

        Args:
            examples: List of example dicts with "input_ids" tensor key.

        Returns:
            Batch dict where each example has exactly block_size tokens.
        """
        # Collect all token IDs from this mini-batch
        all_tokens: List[int] = list(self._token_buffer)

        for ex in examples:
            ids = ex["input_ids"].tolist()
            # Strip padding tokens before concatenation
            ids = [t for t in ids if t != self._pad_token_id]
            all_tokens.extend(ids)
            all_tokens.append(self._eos_token_id)   # Document separator

        # Chunk into blocks of exactly block_size
        chunks: List[List[int]] = []
        for i in range(0, len(all_tokens) - self._block_size + 1, self._block_size):
            chunks.append(all_tokens[i:i + self._block_size])

        # Keep leftover tokens for the next call
        used = len(chunks) * self._block_size
        self._token_buffer = all_tokens[used:]

        if not chunks:
            # Not enough tokens to form a block — return an empty batch
            return {}

        input_ids  = torch.tensor(chunks, dtype=torch.long)
        labels     = input_ids.clone()
        attn_mask  = torch.ones_like(input_ids)

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
        }
