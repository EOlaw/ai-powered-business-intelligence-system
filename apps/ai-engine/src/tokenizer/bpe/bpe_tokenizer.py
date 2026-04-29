"""
InsightSerenity AI Engine — BPE Tokenizer
==========================================
Encodes and decodes text using a trained BPE vocabulary and merge rules.
This is the runtime tokenizer used during model training and inference.

The BPETokenizer:
    1. Pre-tokenizes text (whitespace + GPT-2 regex → word units)
    2. For each word, applies BPE merges greedily in learned order
    3. Returns integer token IDs

Decoding:
    1. Maps IDs back to token strings
    2. Removes the Ġ space-prefix convention and reconstructs original text

The class follows the BaseTokenizer interface so it can be used anywhere
a BaseTokenizer is expected, including the training infrastructure.

Usage:
    from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer

    # Train from corpus
    tokenizer = BPETokenizer()
    tokenizer.train("storage/datasets/corpus.jsonl", vocab_size=32_000)
    tokenizer.save("storage/tokenizers/bpe-32k/")

    # Or load pre-trained
    tokenizer = BPETokenizer.load("storage/tokenizers/bpe-32k/")

    ids  = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
    # text == "Hello, world!"
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from src.tokenizer.base.base_tokenizer import BaseTokenizer
from src.tokenizer.bpe.bpe_trainer import BPETrainer, SPACE_PREFIX
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.file_io import write_json, read_json, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BPETokenizer(BaseTokenizer):
    """
    BPE tokenizer: trained on our corpus, owned by us, no external keys.

    After training or loading, encodes any string to token IDs and decodes
    any token ID sequence back to text with full round-trip fidelity.

    Args:
        vocab:          Pre-built Vocabulary (populated by training).
        merge_rules:    Ordered list of (a, b) merge pairs. Order matters —
                        earlier rules have higher priority.
        add_bos_token:  Prepend <bos> on encode. Default True.
        add_eos_token:  Append <eos> on encode. Default True.
        max_length:     Hard maximum token length. Default 2048.
    """

    # GPT-2 pre-tokenisation regex (same as BPETrainer)
    _PRETOKENIZE_RE = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        re.IGNORECASE,
    )

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        merge_rules: Optional[List[Tuple[str, str]]] = None,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        max_length: int = 2048,
    ) -> None:
        super().__init__(
            vocab=vocab,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            max_length=max_length,
        )

        self._merge_rules: List[Tuple[str, str]] = merge_rules or []

        # Build a priority dict for fast merge lookup:
        # (a, b) → merge_index (lower = higher priority)
        self._merge_priority: Dict[Tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(self._merge_rules)
        }

        # Word-level encode cache: { "word_string": [token_ids, ...] }
        # Caches encoding results for frequently seen words
        self._encode_cache: Dict[str, List[str]] = {}

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        corpus_path: str,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        text_field: str = "text",
        **kwargs,
    ) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        Delegates to BPETrainer and populates this instance's
        vocabulary and merge rules.

        Args:
            corpus_path:  Path to a plain text or JSONL corpus.
            vocab_size:   Target vocabulary size.
            min_frequency: Minimum word frequency to include in training.
            text_field:   For JSONL: key containing the document text.
        """
        trainer = BPETrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

        merge_rules, vocab = trainer.train(
            corpus_path=corpus_path,
            text_field=text_field,
        )

        self._vocab          = vocab
        self._merge_rules    = merge_rules
        self._merge_priority = {pair: idx for idx, pair in enumerate(merge_rules)}
        self._encode_cache   = {}

        logger.info(
            "BPETokenizer training complete",
            vocab_size=self.vocab_size,
            merge_rules=len(merge_rules),
        )

    # ── Tokenization ───────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using BPE.

        Steps:
          1. Pre-tokenize into word units (GPT-2 regex)
          2. Apply BPE merges to each word
          3. Return flat list of subword tokens

        Uses the encode cache to avoid re-computing BPE for words seen before.
        """
        tokens: List[str] = []

        for word in self._pretokenize(text):
            if not word:
                continue

            if word in self._encode_cache:
                tokens.extend(self._encode_cache[word])
            else:
                word_tokens = self._bpe_encode_word(word)
                self._encode_cache[word] = word_tokens
                tokens.extend(word_tokens)

        return tokens

    def _bpe_encode_word(self, word: str) -> List[str]:
        """
        Apply BPE merge rules to encode a single word.

        The algorithm:
          1. Start with characters (+ Ġ prefix for word-initial space)
          2. Repeatedly find the highest-priority merge applicable to
             the current sequence
          3. Apply it (merge the pair into one token)
          4. Repeat until no more applicable merges exist

        This greedy algorithm is O(len(word)^2 × log(merges)) per word.
        The cache in _tokenize() amortises this cost in practice.

        Args:
            word: A single pre-tokenized word string (may start with Ġ).

        Returns:
            List of sub-word token strings.
        """
        # Initial character sequence
        if word.startswith(SPACE_PREFIX):
            chars: List[str] = [SPACE_PREFIX] + list(word[len(SPACE_PREFIX):])
        else:
            chars = list(word)

        if len(chars) == 1:
            return chars

        while True:
            # Find the highest-priority merge applicable to the current sequence
            best_pair:     Optional[Tuple[str, str]] = None
            best_priority: int                        = len(self._merge_rules)   # ∞

            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                priority = self._merge_priority.get(pair, len(self._merge_rules))
                if priority < best_priority:
                    best_priority = priority
                    best_pair     = pair

            if best_pair is None:
                # No more applicable merges
                break

            # Apply the merge: replace all occurrences of best_pair
            merged     = best_pair[0] + best_pair[1]
            new_chars: List[str] = []
            i          = 0

            while i < len(chars):
                if (
                    i < len(chars) - 1
                    and chars[i]   == best_pair[0]
                    and chars[i+1] == best_pair[1]
                ):
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1

            chars = new_chars

            if len(chars) == 1:
                break

        return chars

    def _pretokenize(self, text: str) -> List[str]:
        """Split text into word units using the GPT-2 pre-tokenisation regex."""
        words = self._PRETOKENIZE_RE.findall(text)
        result = []
        for word in words:
            if word.startswith(" ") and word.strip():
                result.append(SPACE_PREFIX + word.strip())
            elif word.strip():
                result.append(word.strip())
        return result

    # ── Decoding ───────────────────────────────────────────────────────────────

    def _tokens_to_string(self, tokens: List[str]) -> str:
        """
        Reconstruct original text from BPE tokens.

        The Ġ convention encodes spaces as a prefix on the following word.
        "Hello", "Ġworld", "!" → "Hello world!"
        """
        text = "".join(
            (" " + t[len(SPACE_PREFIX):]) if t.startswith(SPACE_PREFIX) else t
            for t in tokens
        )
        return text.lstrip()   # Remove leading space from the first word

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save the tokenizer to disk.

        Files created:
            vocab.json            — token → ID mapping
            merges.txt            — one merge rule per line: "a b\n"
            tokenizer_config.json — class name and config metadata
        """
        directory = ensure_dir(directory)

        # Save vocabulary
        self._vocab.save(directory / "vocab.json")

        # Save merge rules
        with open(directory / "merges.txt", "w", encoding="utf-8") as f:
            f.write("#version: 1.0 — InsightSerenity BPE\n")
            for a, b in self._merge_rules:
                f.write(f"{a} {b}\n")

        # Save config
        write_json(directory / "tokenizer_config.json", {
            "tokenizer_class":  "BPETokenizer",
            "vocab_size":       self.vocab_size,
            "num_merges":       len(self._merge_rules),
            "add_bos_token":    self._add_bos,
            "add_eos_token":    self._add_eos,
            "max_length":       self._max_length,
            "bos_token":        ST.BOS,
            "eos_token":        ST.EOS,
            "unk_token":        ST.UNK,
            "pad_token":        ST.PAD,
            "mask_token":       ST.MASK,
            "space_prefix":     SPACE_PREFIX,
        })

        logger.info(
            "BPETokenizer saved",
            directory=str(directory),
            vocab_size=self.vocab_size,
            merges=len(self._merge_rules),
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "BPETokenizer":
        """
        Load a BPETokenizer from a directory previously saved by save().

        Args:
            directory: Path to the tokenizer directory.

        Returns:
            Initialised BPETokenizer ready for use.
        """
        directory = Path(directory)

        # Load config
        config = read_json(directory / "tokenizer_config.json")

        # Load vocabulary
        vocab = Vocabulary.load(directory / "vocab.json")

        # Load merge rules
        merge_rules: List[Tuple[str, str]] = []
        with open(directory / "merges.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    merge_rules.append((parts[0], parts[1]))

        tokenizer = cls(
            vocab=vocab,
            merge_rules=merge_rules,
            add_bos_token=config.get("add_bos_token", True),
            add_eos_token=config.get("add_eos_token", True),
            max_length=config.get("max_length", 2048),
        )

        logger.info(
            "BPETokenizer loaded",
            directory=str(directory),
            vocab_size=tokenizer.vocab_size,
            merges=len(merge_rules),
        )
        return tokenizer

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str,
        vocab_size: int = 32_000,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> "BPETokenizer":
        """
        Convenience method: train a BPETokenizer on a corpus and optionally save it.

        Args:
            corpus_path: Path to the training corpus (JSONL or plain text).
            vocab_size:  Target vocabulary size.
            save_dir:    If provided, save the tokenizer here after training.
            **kwargs:    Forwarded to train().

        Returns:
            Trained BPETokenizer instance.
        """
        tokenizer = cls()
        tokenizer.train(corpus_path, vocab_size=vocab_size, **kwargs)

        if save_dir:
            tokenizer.save(save_dir)

        return tokenizer
