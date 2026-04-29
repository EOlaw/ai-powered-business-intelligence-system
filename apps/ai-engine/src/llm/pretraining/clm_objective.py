"""
InsightSerenity AI Engine — Causal Language Modeling Objective
===============================================================
Implements the data preparation for Causal Language Model (CLM) pretraining.
CLM is the training objective used by GPT-2, GPT-3, LLaMA, and all
autoregressive language models.

Objective: Given tokens [x_1, x_2, ..., x_n], predict the next token at
each position:
    Loss = -1/n * sum_{t=1}^{n} log P(x_t | x_1, ..., x_{t-1})

Data format: The dataset is a sequence of token IDs. We split it into
fixed-length blocks (e.g. 1024 tokens). For each block:
    input_ids:  [BOS, x_1, x_2, ..., x_{n-1}]
    labels:     [x_1, x_2, ..., x_{n-1}, EOS]  (shifted left by 1)

Padding positions have label = -100 → CrossEntropyLoss ignores them.

Packing strategy: Instead of padding every short sequence to block_size,
we concatenate multiple documents and split at block boundaries. This
maximises GPU utilisation — every token in the batch contributes to the loss.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.file_io import iter_jsonl
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CLMExample:
    """One packed block of tokens for CLM training."""
    input_ids:      List[int]
    labels:         List[int]
    attention_mask: List[int]


class CLMDataCollator:
    """
    Collates raw token sequences into CLM training batches.

    The collator:
    1. Takes a batch of examples (each a dict with "input_ids")
    2. Packs multiple short documents end-to-end with EOS separators
    3. Splits into blocks of exactly block_size tokens
    4. Creates shifted labels (input shifted left by 1 = next token target)
    5. Sets labels to -100 at padding positions

    This collator implements the "packing" strategy used by GPT-3 and LLaMA
    for maximum training efficiency.

    Args:
        block_size:    Number of tokens per training block.
        eos_token_id:  Token ID inserted between documents.
        pad_token_id:  Token ID used for padding (last block may be shorter).
        ignore_index:  Label value at padding positions. Default -100.
    """

    def __init__(
        self,
        block_size:   int,
        eos_token_id: int = ST.EOS_ID,
        pad_token_id: int = ST.PAD_ID,
        ignore_index: int = -100,
    ) -> None:
        self.block_size   = block_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self._buffer: List[int] = []

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Process a batch of examples into model-ready tensors.

        Args:
            examples: List of dicts with "input_ids" key (list of int).

        Returns:
            Dict with "input_ids", "labels", "attention_mask" tensors.
        """
        # Accumulate all tokens from the batch into the buffer
        for example in examples:
            ids = example.get("input_ids", [])
            if isinstance(ids, Tensor):
                ids = ids.tolist()
            # Strip padding tokens before concatenating
            ids = [t for t in ids if t != self.pad_token_id]
            self._buffer.extend(ids)
            self._buffer.append(self.eos_token_id)   # Document separator

        # Split buffer into fixed-size blocks
        blocks: List[List[int]] = []
        while len(self._buffer) >= self.block_size + 1:
            block = self._buffer[:self.block_size + 1]
            blocks.append(block)
            self._buffer = self._buffer[self.block_size:]

        if not blocks:
            # Not enough tokens yet — return empty (batch will be skipped)
            return {}

        # Build input_ids and labels (shifted by 1)
        input_ids_list = []
        labels_list    = []
        attn_mask_list = []

        for block in blocks:
            input_ids = block[:self.block_size]
            labels    = block[1:self.block_size + 1]

            # Pad if needed (last block)
            pad_len = self.block_size - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels    = labels    + [self.ignore_index] * pad_len

            attn_mask = [1] * (self.block_size - pad_len) + [0] * pad_len

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attn_mask_list.append(attn_mask)

        return {
            "input_ids":      torch.tensor(input_ids_list, dtype=torch.long),
            "labels":         torch.tensor(labels_list,    dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long),
        }


class CLMDataset(Dataset):
    """
    In-memory CLM dataset from a JSONL corpus.

    Pre-tokenises the entire corpus and packs it into non-overlapping
    blocks of `block_size` tokens. Suitable for medium-sized corpora
    (corpora that fit in RAM). For very large corpora, use StreamingTextDataset.

    Args:
        corpus_path:  Path to JSONL file with {"text": "..."} records.
        tokenizer:    Tokenizer with .encode(text) → List[int].
        block_size:   Tokens per training block.
        text_field:   Key in JSONL records. Default "text".
        max_docs:     Maximum documents to load. Default: all.
    """

    def __init__(
        self,
        corpus_path: str,
        tokenizer:   Any,
        block_size:  int   = 1024,
        text_field:  str   = "text",
        max_docs:    Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self._tokenizer = tokenizer

        logger.info("Building CLM dataset", corpus=corpus_path, block_size=block_size)

        # Tokenise and pack all documents into a flat token stream
        all_tokens: List[int] = []
        docs_loaded = 0

        for record in iter_jsonl(corpus_path):
            text = record.get(text_field, "")
            if not text:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)
            all_tokens.append(ST.EOS_ID)   # Document boundary marker

            docs_loaded += 1
            if max_docs is not None and docs_loaded >= max_docs:
                break

        # Split into non-overlapping blocks
        self._blocks: List[List[int]] = []
        for i in range(0, len(all_tokens) - block_size, block_size):
            self._blocks.append(all_tokens[i : i + block_size + 1])

        logger.info(
            "CLM dataset ready",
            docs=docs_loaded,
            total_tokens=len(all_tokens),
            blocks=len(self._blocks),
        )

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Return one block as input_ids and labels (shifted by 1).

        Returns:
            Dict with:
                input_ids: LongTensor (block_size,)
                labels:    LongTensor (block_size,) — shifted, -100 at EOS
        """
        block     = self._blocks[idx]
        input_ids = torch.tensor(block[:self.block_size], dtype=torch.long)
        labels    = torch.tensor(block[1:self.block_size + 1], dtype=torch.long)

        # Mark EOS positions in labels as -100 so loss ignores them
        # (We don't want to train the model to predict post-EOS tokens)
        eos_positions  = labels == ST.EOS_ID
        labels_masked  = labels.clone()
        labels_masked[eos_positions] = -100

        return {
            "input_ids":      input_ids,
            "labels":         labels_masked,
            "attention_mask": torch.ones(self.block_size, dtype=torch.long),
        }
