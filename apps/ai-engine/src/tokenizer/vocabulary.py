"""
InsightSerenity AI Engine — Vocabulary
=======================================
Manages the bijective mapping between string tokens and integer token IDs.
Every tokenizer implementation uses a Vocabulary instance as its core
lookup structure.

The vocabulary is the single most important file produced by tokenizer
training. It determines:
  - What the model can express (out-of-vocabulary tokens → <unk>)
  - The embedding table size (vocab_size × embedding_dim parameters)
  - Token ID assignments (which must remain stable across all uses)

File format: We save two files:
  vocab.json   — { "token": id, ... }  (token → id mapping)
  Special tokens always occupy IDs 0 through num_special_tokens-1.

Usage:
    from src.tokenizer.vocabulary import Vocabulary
    from src.tokenizer.special_tokens import SpecialTokens as ST

    vocab = Vocabulary()
    vocab.add_token("hello")
    id    = vocab["hello"]      # 12 (after special tokens)
    token = vocab[12]           # "hello"
    vocab.save("storage/tokenizers/bpe-32k/vocab.json")
    vocab = Vocabulary.load("storage/tokenizers/bpe-32k/vocab.json")
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

from src.tokenizer.special_tokens import ALL_SPECIAL_TOKENS, SpecialTokens as ST
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Vocabulary:
    """
    Bidirectional token ↔ ID mapping with special-token support.

    Special tokens are always pre-populated at construction with their
    reserved IDs. Regular tokens are added starting from the next available
    ID after the special tokens block.

    Args:
        unk_token:  The token to return when a string is not in the vocabulary.
                    Defaults to "<unk>".
    """

    def __init__(self, unk_token: str = ST.UNK) -> None:
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._unk_token   = unk_token

        # Pre-populate special tokens with their reserved IDs
        for special in ALL_SPECIAL_TOKENS:
            self._add(special.token, special.token_id)

        # Next available ID for regular tokens
        self._next_id: int = max(st.token_id for st in ALL_SPECIAL_TOKENS) + 1

    # ── Core interface ─────────────────────────────────────────────────────────

    def add_token(self, token: str) -> int:
        """
        Add a new token to the vocabulary and return its assigned ID.
        If the token already exists, return its existing ID without modification.

        Args:
            token: The string to add (e.g. "hello", "Ġworld", "##ing").

        Returns:
            The integer ID assigned to this token.
        """
        if token in self._token_to_id:
            return self._token_to_id[token]

        token_id = self._next_id
        self._add(token, token_id)
        self._next_id += 1
        return token_id

    def add_tokens(self, tokens: Iterable[str]) -> List[int]:
        """Add multiple tokens and return their IDs."""
        return [self.add_token(t) for t in tokens]

    def token_to_id(self, token: str) -> int:
        """
        Look up the ID for a token string.
        Returns the UNK token ID if the token is not in the vocabulary.
        """
        return self._token_to_id.get(token, self._token_to_id[self._unk_token])

    def id_to_token(self, token_id: int) -> str:
        """
        Look up the token string for an ID.
        Returns the UNK token if the ID is not in the vocabulary.
        """
        return self._id_to_token.get(token_id, self._unk_token)

    def __getitem__(self, key: Union[str, int]) -> Union[int, str]:
        """
        Bidirectional lookup:
            vocab["hello"]  → integer ID
            vocab[12]       → token string
        """
        if isinstance(key, str):
            return self.token_to_id(key)
        return self.id_to_token(key)

    def __contains__(self, token: str) -> bool:
        """True if the token string is in the vocabulary."""
        return token in self._token_to_id

    def __len__(self) -> int:
        """Total vocabulary size (special tokens + regular tokens)."""
        return len(self._token_to_id)

    def __iter__(self) -> Iterator[str]:
        """Iterate over all token strings in ID order."""
        for token_id in sorted(self._id_to_token):
            yield self._id_to_token[token_id]

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, unk='{self._unk_token}')"

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Alias for len(vocab)."""
        return len(self)

    @property
    def unk_token_id(self) -> int:
        return self._token_to_id[self._unk_token]

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
    def mask_token_id(self) -> int:
        return ST.MASK_ID

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the vocabulary to a JSON file.

        Format: { "token": id, "token": id, ... }
        Sorted by token ID for human readability.

        Args:
            path: Destination file path. The parent directory is created
                  if it does not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sort by ID for deterministic output
        ordered = dict(
            sorted(self._token_to_id.items(), key=lambda x: x[1])
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)

        logger.info("Vocabulary saved", path=str(path), size=len(self))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """
        Load a vocabulary from a JSON file.

        Args:
            path: Path to the vocab.json file.

        Returns:
            A new Vocabulary instance with all tokens loaded.
        """
        with open(path, "r", encoding="utf-8") as f:
            token_to_id: Dict[str, int] = json.load(f)

        vocab = cls.__new__(cls)
        vocab._token_to_id = {}
        vocab._id_to_token = {}
        vocab._unk_token   = ST.UNK

        for token, token_id in token_to_id.items():
            vocab._add(token, token_id)

        vocab._next_id = max(token_to_id.values()) + 1 if token_to_id else 0

        logger.info("Vocabulary loaded", path=str(path), size=len(vocab))
        return vocab

    # ── Utilities ──────────────────────────────────────────────────────────────

    def most_common(self, n: Optional[int] = None) -> List[str]:
        """
        Return the first n tokens in ID order (lowest ID first).
        Special tokens appear first by construction.
        """
        sorted_tokens = [
            self._id_to_token[i]
            for i in sorted(self._id_to_token.keys())
        ]
        return sorted_tokens[:n] if n else sorted_tokens

    def tokens(self) -> List[str]:
        """Return all token strings in ID order."""
        return self.most_common()

    def ids(self) -> List[int]:
        """Return all token IDs in sorted order."""
        return sorted(self._id_to_token.keys())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _add(self, token: str, token_id: int) -> None:
        """Direct insertion into both dicts. Does not update _next_id."""
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
