"""
InsightSerenity AI Engine — Text Dataset
=========================================
Concrete Dataset implementations for plain text and JSONL corpora.

Two implementations:
    TextDataset       — loads the full corpus into RAM (for small/medium corpora)
    JSONLTextDataset  — wraps a JSONL corpus file loaded into memory

Both produce tokenised examples ready for language model training:
    { "input_ids": LongTensor, "attention_mask": LongTensor, "labels": LongTensor }

The `labels` tensor is identical to `input_ids` for autoregressive
(causal language model) training — the model predicts the next token.

Usage:
    from src.data.datasets.text_dataset import JSONLTextDataset

    dataset = JSONLTextDataset(
        path="storage/datasets/corpus.jsonl",
        tokenizer=bpe_tokenizer,
        max_length=1024,
        text_field="text",
    )
    sample = dataset[0]
    # { "input_ids": tensor([...]), "attention_mask": tensor([...]), "labels": tensor([...]) }
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import LongTensor

from src.data.datasets.base_dataset import BaseDataset, DatasetInfo
from src.utils.file_io import read_jsonl, read_lines
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextDataset(BaseDataset):
    """
    In-memory dataset from a list of raw text strings.

    Suitable for:
      - Small corpora (< a few hundred MB of text)
      - Fine-tuning datasets where the full data fits in RAM
      - Unit testing with synthetic data

    Args:
        texts:      List of raw text strings.
        tokenizer:  Any tokenizer with an `.encode(text)` method that returns
                    a list of integer token IDs.
        max_length: Maximum number of tokens per example. Longer texts are
                    truncated; shorter texts are padded to this length.
        name:       Optional dataset name for metadata.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 1024,
        name: str = "text_dataset",
    ) -> None:
        self._tokenizer  = tokenizer
        self._max_length = max_length
        self._name       = name

        logger.info(
            "Tokenising dataset",
            num_texts=len(texts),
            max_length=max_length,
        )

        # Tokenise and cache all examples at construction time.
        # This is a one-time cost; training iterations become O(1) lookups.
        self._examples: List[Dict[str, LongTensor]] = []
        skipped = 0

        for text in texts:
            example = self._process(text)
            if example is not None:
                self._examples.append(example)
            else:
                skipped += 1

        logger.info(
            "Dataset ready",
            name=name,
            examples=len(self._examples),
            skipped=skipped,
        )

    # ── BaseDataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Dict[str, LongTensor]:
        return self._examples[index]

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self._name,
            num_examples=len(self),
            features={
                "input_ids":      "LongTensor of token ids",
                "attention_mask": "LongTensor of 1s and 0s (1=real, 0=pad)",
                "labels":         "LongTensor (same as input_ids for CLM)",
            },
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _process(self, text: str) -> Optional[Dict[str, LongTensor]]:
        """
        Tokenise a single text and build the model input tensors.

        Truncation: texts longer than max_length are truncated on the right.
        Padding:    texts shorter than max_length are right-padded with 0s
                    and the attention mask is set to 0 for pad positions.

        Returns None for texts that produce zero tokens after tokenisation.
        """
        token_ids = self._tokenizer.encode(text)

        if not token_ids:
            return None

        # Truncate to max_length
        token_ids = token_ids[:self._max_length]
        seq_len   = len(token_ids)

        # Pad to max_length
        padding         = self._max_length - seq_len
        padded_ids      = token_ids + [0] * padding
        attention_mask  = [1] * seq_len + [0] * padding

        input_ids       = torch.tensor(padded_ids,     dtype=torch.long)
        attention_mask_ = torch.tensor(attention_mask, dtype=torch.long)

        # For autoregressive LM training: labels == input_ids
        # Padded positions are set to -100 so CrossEntropyLoss ignores them
        labels = input_ids.clone()
        labels[labels == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask_,
            "labels":         labels,
        }


class JSONLTextDataset(BaseDataset):
    """
    Dataset that loads a JSONL corpus file into memory.

    JSONL format (one record per line):
        {"text": "The article content...", "url": "https://...", ...}

    Only the `text_field` is used for training; other fields are ignored.
    This is the primary dataset class for pretraining from the crawled corpus.

    Args:
        path:       Path to the JSONL corpus file.
        tokenizer:  Tokenizer with an `.encode(text) → List[int]` method.
        max_length: Maximum tokens per example.
        text_field: Key in each JSONL record containing the document text.
        max_examples: Cap on number of examples to load (for debugging/testing).
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer: Any,
        max_length: int = 1024,
        text_field: str = "text",
        max_examples: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        self._path       = Path(path)
        self._max_length = max_length
        self._text_field = text_field
        self._name       = name or self._path.stem
        self._tokenizer  = tokenizer

        logger.info(
            "Loading JSONL dataset",
            path=str(self._path),
            max_length=max_length,
            max_examples=max_examples,
        )

        records = read_jsonl(str(self._path))

        if max_examples is not None:
            records = records[:max_examples]

        self._texts = [r.get(text_field, "") for r in records if r.get(text_field)]

        # Build underlying TextDataset for tokenisation
        self._inner = TextDataset(
            texts=self._texts,
            tokenizer=tokenizer,
            max_length=max_length,
            name=self._name,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> Dict[str, LongTensor]:
        return self._inner[index]

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self._name,
            source=str(self._path),
            num_examples=len(self),
            features=self._inner.info.features,
        )
