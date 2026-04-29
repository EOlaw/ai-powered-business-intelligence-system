"""
InsightSerenity AI Engine — Streaming Dataset
==============================================
Memory-efficient dataset that reads a JSONL corpus file lazily, one record
at a time, without loading the full file into RAM.

This is the correct approach for pretraining on large corpora (10GB+ of text)
where loading everything into memory would be infeasible. It sacrifices random
access (no O(1) __getitem__ by index) in exchange for constant memory usage.

How it works:
    - The file is read with a line-by-line generator
    - PyTorch's IterableDataset interface is used (yields examples in order)
    - Each worker in a multi-process DataLoader reads a non-overlapping shard
      of the file (via worker_init_fn) to avoid duplicate examples
    - Shuffling is done with a fixed-size shuffle buffer: the buffer fills up,
      is shuffled in memory, then yielded example-by-example

Limitation: The dataset length is not known until the file is fully scanned.
We pre-scan the file at init time to compute the length (this is a one-time
cost and is cached).

Usage:
    from src.data.datasets.streaming_dataset import StreamingTextDataset
    from torch.utils.data import DataLoader

    dataset = StreamingTextDataset(
        path="storage/datasets/corpus.jsonl",
        tokenizer=bpe_tokenizer,
        max_length=1024,
        shuffle_buffer_size=10_000,
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    for batch in loader:
        # batch["input_ids"] shape: (32, 1024)
        train_step(batch)
"""

import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import IterableDataset

from src.utils.file_io import iter_jsonl, count_lines
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large JSONL corpora.

    Implements PyTorch's IterableDataset protocol: instead of random-access
    __getitem__, it yields examples one at a time via __iter__.

    Args:
        path:               Path to the JSONL corpus file.
        tokenizer:          Tokenizer with `.encode(text) → List[int]`.
        max_length:         Maximum tokens per example.
        text_field:         JSON key containing document text.
        shuffle_buffer_size: Number of examples to buffer for in-memory shuffle.
                             Set to 1 to disable shuffling (e.g. for evaluation).
        seed:               Random seed for shuffle reproducibility.
        max_examples:       Cap on total examples yielded (for debugging).
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer: Any,
        max_length: int = 1024,
        text_field: str = "text",
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        max_examples: Optional[int] = None,
    ) -> None:
        self._path                = Path(path)
        self._tokenizer           = tokenizer
        self._max_length          = max_length
        self._text_field          = text_field
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed                = seed
        self._max_examples        = max_examples

        # Pre-scan to count lines (cached, done once at init)
        logger.info("Counting lines in corpus", path=str(self._path))
        self._num_lines: int = count_lines(self._path)
        logger.info("Corpus line count", path=str(self._path), lines=self._num_lines)

    def __len__(self) -> int:
        """Approximate length (may differ slightly if some records are skipped)."""
        return self._num_lines

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield tokenised examples one at a time.

        In a multi-worker DataLoader, each worker automatically receives a
        different shard via PyTorch's worker_info mechanism.
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process mode: yield all examples
            yield from self._stream_shard(worker_id=0, num_workers=1)
        else:
            # Multi-process mode: each worker handles a disjoint subset
            yield from self._stream_shard(
                worker_id=worker_info.id,
                num_workers=worker_info.num_workers,
            )

    def _stream_shard(
        self, worker_id: int, num_workers: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield examples from this worker's shard of the file.

        Sharding strategy: worker i yields examples at positions
        {i, i+num_workers, i+2*num_workers, ...}.
        This ensures no two workers yield the same example.
        """
        shuffle_buffer: List[Dict[str, torch.Tensor]] = []
        rng    = random.Random(self._seed + worker_id)
        count  = 0

        for line_num, record in enumerate(iter_jsonl(str(self._path))):
            # This worker's shard: skip lines not assigned to this worker
            if line_num % num_workers != worker_id:
                continue

            text = record.get(self._text_field, "")
            if not text:
                continue

            example = self._process(text)
            if example is None:
                continue

            # Fill the shuffle buffer
            shuffle_buffer.append(example)

            if len(shuffle_buffer) >= self._shuffle_buffer_size:
                rng.shuffle(shuffle_buffer)
                # Yield all but the last element (keep some for mixing next batch)
                yield_count = len(shuffle_buffer) - self._shuffle_buffer_size // 10
                for example in shuffle_buffer[:yield_count]:
                    count += 1
                    yield example
                    if self._max_examples and count >= self._max_examples:
                        return
                shuffle_buffer = shuffle_buffer[yield_count:]

        # Yield remaining examples in buffer
        rng.shuffle(shuffle_buffer)
        for example in shuffle_buffer:
            count += 1
            yield example
            if self._max_examples and count >= self._max_examples:
                return

    def _process(self, text: str) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenise a text and build model input tensors. Returns None for empty texts."""
        token_ids = self._tokenizer.encode(text)
        if not token_ids:
            return None

        token_ids = token_ids[:self._max_length]
        seq_len   = len(token_ids)
        padding   = self._max_length - seq_len

        padded_ids     = token_ids + [0] * padding
        attention_mask = [1] * seq_len + [0] * padding

        input_ids      = torch.tensor(padded_ids,     dtype=torch.long)
        attn_mask      = torch.tensor(attention_mask, dtype=torch.long)
        labels         = input_ids.clone()
        labels[labels == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
        }
