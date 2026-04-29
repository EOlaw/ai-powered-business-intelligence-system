"""
InsightSerenity AI Engine — Tokenizer Package
==============================================
Public API for the tokenizer package. Import from here, not from sub-modules,
so refactoring internals never breaks external callers.

Quick reference:
    BPETokenizer       — GPT-style, space-prefix convention (Ġ)
    WordPieceTokenizer — BERT-style, continuation-prefix convention (##)
    SentencePieceWrapper — Language-agnostic, multilingual, byte-fallback
    Vocabulary         — Token ↔ ID mapping shared by all tokenizers
    SpecialTokens      — PAD, UNK, BOS, EOS, MASK and chat role tokens
    BaseTokenizer      — Abstract interface (use for type hints)

Factory function:
    load_tokenizer(directory) — auto-detects class from tokenizer_config.json

Usage:
    from src.tokenizer import BPETokenizer, load_tokenizer

    # Train
    tok = BPETokenizer()
    tok.train("storage/datasets/corpus.jsonl", vocab_size=32_000)
    tok.save("storage/tokenizers/bpe-32k/")

    # Load (auto-detects class)
    tok = load_tokenizer("storage/tokenizers/bpe-32k/")

    # Use
    ids  = tok.encode("Hello, world!")
    text = tok.decode(ids)
"""

from src.tokenizer.base.base_tokenizer import BaseTokenizer
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.special_tokens import SpecialTokens, ALL_SPECIAL_TOKENS, SPECIAL_TOKENS
from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer
from src.tokenizer.bpe.bpe_trainer import BPETrainer
from src.tokenizer.wordpiece.wordpiece_tokenizer import WordPieceTokenizer, WordPieceTrainer
from src.tokenizer.sentencepiece.sp_wrapper import SentencePieceWrapper

import json
from pathlib import Path
from typing import Union


def load_tokenizer(directory: Union[str, Path]) -> BaseTokenizer:
    """
    Load a tokenizer from a directory, automatically detecting its class
    from the `tokenizer_class` field in tokenizer_config.json.

    This is the recommended way to load a tokenizer at runtime — it decouples
    callers from knowing which specific class was used during training.

    Args:
        directory: Path to the directory containing tokenizer files.

    Returns:
        An initialised tokenizer instance ready for encoding/decoding.

    Raises:
        FileNotFoundError: If tokenizer_config.json does not exist.
        ValueError:        If the tokenizer_class in config is unknown.
    """
    config_path = Path(directory) / "tokenizer_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"tokenizer_config.json not found in {directory}. "
            f"Make sure the tokenizer was saved with tokenizer.save(directory)."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_class = config.get("tokenizer_class", "")

    _CLASS_MAP = {
        "BPETokenizer":         BPETokenizer,
        "WordPieceTokenizer":   WordPieceTokenizer,
        "SentencePieceWrapper": SentencePieceWrapper,
    }

    cls = _CLASS_MAP.get(tokenizer_class)
    if cls is None:
        raise ValueError(
            f"Unknown tokenizer_class '{tokenizer_class}' in {config_path}. "
            f"Expected one of: {list(_CLASS_MAP.keys())}"
        )

    return cls.load(directory)


__all__ = [
    "BaseTokenizer",
    "Vocabulary",
    "SpecialTokens",
    "ALL_SPECIAL_TOKENS",
    "SPECIAL_TOKENS",
    "BPETokenizer",
    "BPETrainer",
    "WordPieceTokenizer",
    "WordPieceTrainer",
    "SentencePieceWrapper",
    "load_tokenizer",
]
