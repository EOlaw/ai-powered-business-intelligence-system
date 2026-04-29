"""
InsightSerenity AI Engine — Abstract Base Tokenizer
====================================================
Defines the interface that every tokenizer implementation must satisfy.
By programming against this abstract class, the rest of the platform
(training, inference, serving) is decoupled from any specific tokenizer
algorithm. You can swap BPE for WordPiece without changing training code.

Contract (what subclasses must implement):
    encode(text)  → list of integer token IDs
    decode(ids)   → reconstructed string
    train(corpus) → learn vocabulary and merge rules from a text corpus
    save(dir)     → persist tokenizer state to disk
    load(dir)     → restore tokenizer state from disk

The base class provides default implementations of the richer API
(batch encode/decode, tokenize, convert_tokens_to_ids, etc.) built
on top of the abstract encode/decode primitives.

Usage (always interact through this interface):
    from src.tokenizer.base.base_tokenizer import BaseTokenizer

    def train_model(tokenizer: BaseTokenizer, dataset):
        for text in dataset:
            ids = tokenizer.encode(text)  # All tokenizers look the same here
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseTokenizer(ABC):
    """
    Abstract base tokenizer. All concrete implementations inherit from this.

    The tokenizer wraps a Vocabulary and provides:
      - encode / decode (the essential pair)
      - tokenize (text → string tokens, useful for debugging)
      - batch versions of the above
      - special token handling (BOS/EOS prepend/append)
      - convert helpers (tokens → ids, ids → tokens)
      - padding and truncation
      - save and load for persistence

    Subclasses MUST implement: encode, decode, _tokenize, train, save, load.
    Subclasses MAY override: any public method for performance or customisation.
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        max_length: int = 2048,
    ) -> None:
        self._vocab         = vocab or Vocabulary()
        self._add_bos       = add_bos_token
        self._add_eos       = add_eos_token
        self._max_length    = max_length

    # ── Abstract methods — must be implemented by subclasses ──────────────────

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        """
        Split a text string into a list of sub-word token strings.

        This is the core algorithm that differs between BPE, WordPiece, etc.
        It does NOT add special tokens — that is handled by encode().

        Args:
            text: Pre-processed input text.

        Returns:
            List of string tokens (e.g. ["Hello", "Ġworld", "!"]).
        """
        ...

    @abstractmethod
    def train(self, corpus_path: str, vocab_size: int, **kwargs) -> None:
        """
        Train the tokenizer on a text corpus.

        After training, the internal vocabulary is populated and merge rules
        (for BPE) or sub-word models (for WordPiece) are learned.

        Args:
            corpus_path: Path to the text corpus (plain text or JSONL).
            vocab_size:  Target vocabulary size.
            **kwargs:    Algorithm-specific hyperparameters.
        """
        ...

    @abstractmethod
    def save(self, directory: Union[str, Path]) -> None:
        """
        Persist all tokenizer state to `directory`.

        Must create the directory if it does not exist. Must save at minimum:
          - vocab.json (the vocabulary)
          - Any algorithm-specific files (e.g. merges.txt for BPE)
          - tokenizer_config.json (class name, max_length, etc.)
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Union[str, Path]) -> "BaseTokenizer":
        """
        Restore a tokenizer from files saved by save().

        Args:
            directory: Directory containing the tokenizer files.

        Returns:
            An initialised tokenizer ready to encode/decode.
        """
        ...

    # ── Core encode / decode ──────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncate: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Convert a text string to a list of integer token IDs.

        Steps:
          1. Tokenize text into string tokens (_tokenize)
          2. Convert tokens to IDs via the vocabulary
          3. Prepend BOS and/or append EOS if configured
          4. Truncate to max_length if needed

        Args:
            text:               Input text.
            add_special_tokens: Whether to add BOS/EOS. Default True.
            truncate:           Truncate to max_length. Default True.
            max_length:         Override the instance max_length.

        Returns:
            List of integer token IDs.
        """
        if not text:
            return []

        # Tokenize into string tokens
        tokens = self._tokenize(text)

        # Convert to IDs (unknown tokens → UNK_ID)
        ids = [self._vocab.token_to_id(t) for t in tokens]

        # Add special tokens
        if add_special_tokens:
            if self._add_bos:
                ids = [ST.BOS_ID] + ids
            if self._add_eos:
                ids = ids + [ST.EOS_ID]

        # Truncate
        limit = max_length or self._max_length
        if truncate and len(ids) > limit:
            # Always keep EOS at the end if we truncated
            if add_special_tokens and self._add_eos:
                ids = ids[:limit - 1] + [ST.EOS_ID]
            else:
                ids = ids[:limit]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Convert a list of integer token IDs back to a text string.

        Args:
            ids:                     List of integer token IDs.
            skip_special_tokens:     If True, remove BOS/EOS/PAD/MASK tokens
                                     from the output. Default True.
            clean_up_tokenization_spaces: Remove spaces before punctuation.

        Returns:
            Reconstructed text string.
        """
        special_ids = set(ST.all_ids()) if skip_special_tokens else set()

        tokens = [
            self._vocab.id_to_token(i)
            for i in ids
            if i not in special_ids
        ]

        text = self._tokens_to_string(tokens)

        if clean_up_tokenization_spaces:
            text = self._clean_spaces(text)

        return text

    # ── Tokenize / convert helpers ─────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """
        Convert text to a list of string tokens WITHOUT adding special tokens.
        Useful for inspecting the tokenizer's segmentation decisions.

        Example:
            tokenizer.tokenize("Hello, world!")
            → ["Hello", ",", "Ġworld", "!"]
        """
        return self._tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Map a list of token strings to their integer IDs."""
        return [self._vocab.token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Map a list of integer IDs to their token strings."""
        return [self._vocab.id_to_token(i) for i in ids]

    # ── Batch operations ───────────────────────────────────────────────────────

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a list of texts.

        Args:
            texts:              List of input strings.
            add_special_tokens: Whether to add BOS/EOS.
            padding:            If True, pad all sequences to the same length
                                (the longest in the batch).
            truncation:         Truncate to max_length.
            max_length:         Override max_length.

        Returns:
            Dict with keys:
                "input_ids":      List[List[int]] — token IDs per example
                "attention_mask": List[List[int]] — 1s for real tokens, 0s for padding
        """
        all_ids = [
            self.encode(text, add_special_tokens, truncation, max_length)
            for text in texts
        ]

        if padding and all_ids:
            max_len = max(len(ids) for ids in all_ids)
            attention_masks = []
            padded_ids      = []

            for ids in all_ids:
                pad_len = max_len - len(ids)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
                padded_ids.append(ids + [ST.PAD_ID] * pad_len)

            return {
                "input_ids":      padded_ids,
                "attention_mask": attention_masks,
            }

        return {
            "input_ids":      all_ids,
            "attention_mask": [[1] * len(ids) for ids in all_ids],
        }

    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID sequences to text strings."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    # ── Vocabulary passthrough properties ─────────────────────────────────────

    @property
    def vocab(self) -> Vocabulary:
        """The underlying Vocabulary object."""
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Total number of tokens including special tokens."""
        return len(self._vocab)

    @property
    def pad_token_id(self) -> int:
        return ST.PAD_ID

    @property
    def bos_token_id(self) -> int:
        return ST.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return ST.EOS_ID

    @property
    def unk_token_id(self) -> int:
        return ST.UNK_ID

    @property
    def mask_token_id(self) -> int:
        return ST.MASK_ID

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _tokens_to_string(self, tokens: List[str]) -> str:
        """
        Merge a list of tokens back into a single string.

        Default implementation: join with spaces. BPE overrides this to
        handle the Ġ (space prefix) convention.
        """
        return " ".join(tokens)

    @staticmethod
    def _clean_spaces(text: str) -> str:
        """
        Remove spaces before punctuation that should not have a preceding space.
        "Hello , world !" → "Hello, world!"
        """
        import re
        text = re.sub(r'\s+([,\.!?;:\)\]\}])', r'\1', text)
        text = re.sub(r'([\(\[\{])\s+', r'\1', text)
        return text

    # ── Representation ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"max_length={self._max_length})"
        )
