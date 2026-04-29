"""
InsightSerenity AI Engine — WordPiece Tokenizer
================================================
Implements the WordPiece tokenizer algorithm from scratch.
WordPiece is the algorithm used by BERT, DistilBERT, and ALBERT.
It differs from BPE in how merges are selected during training:

    BPE:       greedily merges the most FREQUENT pair
    WordPiece: greedily merges the pair that maximises the LIKELIHOOD of
               the training data — i.e. the pair (a, b) where
               score(a, b) = freq(ab) / (freq(a) × freq(b))

This score function penalises merging very common individual tokens,
preferring merges that create informative compound units. The result is a
tokenizer that tends to produce linguistically motivated sub-words.

Encoding differences from BPE:
    - Continuation tokens are marked with "##" prefix (NOT Ġ space-prefix)
    - "playing" → ["play", "##ing"]
    - BPE uses: "playing" → ["play", "ing"]  (Ġ on word-start tokens only)
    - WordPiece uses longest-match-first decoding (not iterative merges)

Training algorithm:
    1. Build initial character vocabulary from the corpus
    2. Score every possible pair: score = freq(pair) / (freq(left) × freq(right))
    3. Merge the highest-scoring pair
    4. Repeat until vocab_size is reached

Encoding algorithm (inference-time, not training):
    Uses a greedy longest-first match (NOT the training merge order).
    For each word, try to match the longest prefix present in the vocabulary.
    Remaining suffix gets "##" prepended, then repeat.
    If no match is found at any position → whole word becomes [UNK].

Usage:
    from src.tokenizer.wordpiece.wordpiece_tokenizer import WordPieceTokenizer

    tokenizer = WordPieceTokenizer()
    tokenizer.train("storage/datasets/corpus.jsonl", vocab_size=30_522)
    tokenizer.save("storage/tokenizers/wordpiece-30k/")

    tokenizer = WordPieceTokenizer.load("storage/tokenizers/wordpiece-30k/")
    ids  = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from src.tokenizer.base.base_tokenizer import BaseTokenizer
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.file_io import iter_jsonl, write_json, read_json, ensure_dir
from src.utils.logger import get_logger, LogTimer

logger = get_logger(__name__)

# WordPiece continuation prefix — marks a token as a word-internal subword
CONTINUATION_PREFIX = "##"


class WordPieceTrainer:
    """
    Trains a WordPiece vocabulary from a text corpus.

    Implements the likelihood-maximising merge strategy described in
    Schuster & Nakamura (2012), the original WordPiece paper.

    Args:
        vocab_size:    Target vocabulary size (including special tokens).
        min_frequency: Minimum word frequency in the corpus to be included.
        lowercase:     Lowercase all text before training (standard for BERT).
        show_progress: Log a status message every N merges.
    """

    # Simple whitespace tokeniser for initial word splitting
    _WHITESPACE_RE = re.compile(r"\s+")

    # Punctuation characters that should become their own tokens
    _PUNCT_RE = re.compile(r"([^\w\s])", re.UNICODE)

    def __init__(
        self,
        vocab_size: int = 30_522,
        min_frequency: int = 2,
        lowercase: bool = True,
        show_progress: int = 1000,
    ) -> None:
        self._vocab_size    = vocab_size
        self._min_frequency = min_frequency
        self._lowercase     = lowercase
        self._show_progress = show_progress

    def train(
        self,
        corpus_path: str,
        text_field: str = "text",
    ) -> Tuple[Vocabulary, Dict[str, int]]:
        """
        Run WordPiece training on a corpus.

        Args:
            corpus_path: Path to a plain text or JSONL corpus file.
            text_field:  For JSONL: key containing the document text.

        Returns:
            Tuple of:
              - vocabulary: Trained Vocabulary object
              - word_freqs: Final word frequency counts (for diagnostics)
        """
        logger.info(
            "WordPiece training started",
            corpus=corpus_path,
            target_vocab_size=self._vocab_size,
        )

        # Step 1: Count word frequencies
        with LogTimer(logger, "WordPiece word counting"):
            word_freqs = self._count_words(corpus_path, text_field)

        logger.info(
            "Word frequencies counted",
            unique_words=len(word_freqs),
        )

        # Step 2: Build initial character-level vocabulary
        # Characters that begin a word are stored as-is ("h", "e", ...)
        # Characters inside a word get the "##" continuation prefix ("##e", "##l", ...)
        vocab = Vocabulary()
        char_freqs: Counter = Counter()

        for word, freq in word_freqs.items():
            if freq < self._min_frequency:
                continue
            chars = self._word_to_chars(word)
            for char in chars:
                char_freqs[char] += freq

        for char in sorted(char_freqs):
            vocab.add_token(char)

        logger.info(
            "Initial char vocab built",
            char_types=len(char_freqs),
            vocab_size=vocab.size,
        )

        # Step 3: Build working representation: word → list of current subwords
        word_to_subwords: Dict[str, List[str]] = {
            word: self._word_to_chars(word)
            for word, freq in word_freqs.items()
            if freq >= self._min_frequency
        }

        # Step 4: Iteratively merge highest-scoring pairs
        num_merges_needed = self._vocab_size - vocab.size

        with LogTimer(logger, "WordPiece merge learning", target_merges=num_merges_needed):
            for merge_idx in range(num_merges_needed):
                # Compute subword frequencies from current representations
                subword_freqs, pair_freqs = self._compute_freqs(
                    word_to_subwords, word_freqs
                )

                if not pair_freqs:
                    logger.info("No more pairs to merge", merges=merge_idx)
                    break

                # Score each pair: freq(ab) / (freq(a) × freq(b))
                best_pair   = None
                best_score  = -1.0

                for pair, pair_freq in pair_freqs.items():
                    a_freq = subword_freqs.get(pair[0], 1)
                    b_freq = subword_freqs.get(pair[1], 1)
                    score  = pair_freq / (a_freq * b_freq)

                    if score > best_score:
                        best_score = score
                        best_pair  = pair

                if best_pair is None:
                    break

                # Create the merged token
                a, b   = best_pair
                merged = a + b[len(CONTINUATION_PREFIX):]   # "play" + "##ing" → "playing"

                vocab.add_token(merged)

                # Apply the merge to all word representations
                word_to_subwords = self._apply_merge(
                    word_to_subwords, best_pair, merged
                )

                if (merge_idx + 1) % self._show_progress == 0:
                    logger.info(
                        "WordPiece progress",
                        merges=merge_idx + 1,
                        vocab_size=vocab.size,
                        best_pair=f"{a} + {b}",
                        score=round(best_score, 8),
                    )

        logger.info(
            "WordPiece training complete",
            final_vocab_size=vocab.size,
        )
        return vocab, word_freqs

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _count_words(self, corpus_path: str, text_field: str) -> Counter:
        """Count word frequencies with whitespace + punctuation pre-tokenisation."""
        word_freqs: Counter = Counter()
        docs = 0

        if corpus_path.endswith(".jsonl") or corpus_path.endswith(".jsonl.gz"):
            texts: Iterator[str] = (r.get(text_field, "") for r in iter_jsonl(corpus_path))
            for text in texts:
                if not text:
                    continue
                if self._lowercase:
                    text = text.lower()

                # Punctuation splits: "hello,world" → ["hello", ",", "world"]
                text = self._PUNCT_RE.sub(r" \1 ", text)

                for word in self._WHITESPACE_RE.split(text):
                    if word:
                        word_freqs[word] += 1
                docs += 1

            return word_freqs

        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = (line.strip() for line in f)
            for text in lines:
                if not text:
                    continue
                if self._lowercase:
                    text = text.lower()

                # Punctuation splits: "hello,world" → ["hello", ",", "world"]
                text = self._PUNCT_RE.sub(r" \1 ", text)

                for word in self._WHITESPACE_RE.split(text):
                    if word:
                        word_freqs[word] += 1
                docs += 1

        return word_freqs

    def _word_to_chars(self, word: str) -> List[str]:
        """
        Split a word into its initial character sequence using WordPiece convention.
        First character: plain (e.g. "h")
        Subsequent characters: continuation-prefixed (e.g. "##e", "##l")

        "hello" → ["h", "##e", "##l", "##l", "##o"]
        """
        if not word:
            return []
        return [word[0]] + [CONTINUATION_PREFIX + c for c in word[1:]]

    def _compute_freqs(
        self,
        word_to_subwords: Dict[str, List[str]],
        word_freqs: Dict[str, int],
    ) -> Tuple[Counter, Counter]:
        """
        Compute:
          subword_freqs — frequency of each individual subword
          pair_freqs    — frequency of each adjacent subword pair

        Both weighted by the frequency of the containing word.
        """
        subword_freqs: Counter = Counter()
        pair_freqs:    Counter = Counter()

        for word, subwords in word_to_subwords.items():
            freq = word_freqs.get(word, 0)
            if freq == 0:
                continue

            for subword in subwords:
                subword_freqs[subword] += freq

            for i in range(len(subwords) - 1):
                pair = (subwords[i], subwords[i + 1])
                pair_freqs[pair] += freq

        return subword_freqs, pair_freqs

    def _apply_merge(
        self,
        word_to_subwords: Dict[str, List[str]],
        pair: Tuple[str, str],
        merged: str,
    ) -> Dict[str, List[str]]:
        """Apply a merge rule to all word representations."""
        a, b = pair
        updated: Dict[str, List[str]] = {}

        for word, subwords in word_to_subwords.items():
            new_subwords: List[str] = []
            i = 0
            while i < len(subwords):
                if (
                    i < len(subwords) - 1
                    and subwords[i]     == a
                    and subwords[i + 1] == b
                ):
                    new_subwords.append(merged)
                    i += 2
                else:
                    new_subwords.append(subwords[i])
                    i += 1
            updated[word] = new_subwords

        return updated


class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer — the algorithm used by BERT and its descendants.

    Encoding uses the longest-match-first (greedy) algorithm at inference
    time, which is different from the training merge order used in BPE.

    Args:
        vocab:         Pre-built Vocabulary (populated by training or loading).
        lowercase:     Lowercase input text before encoding. Default True
                       (matches BERT-base-uncased behaviour).
        add_bos_token: Prepend <bos> (or [CLS] in BERT terminology) on encode.
        add_eos_token: Append <eos> (or [SEP] in BERT terminology) on encode.
        max_length:    Maximum token sequence length.
        max_word_chars: Maximum characters in a single word. Words longer
                        than this are tokenised as [UNK] directly.
    """

    _WHITESPACE_RE = re.compile(r"\s+")
    _PUNCT_RE      = re.compile(r"([^\w\s])", re.UNICODE)

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        lowercase: bool = True,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        max_length: int = 512,
        max_word_chars: int = 100,
    ) -> None:
        super().__init__(
            vocab=vocab,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            max_length=max_length,
        )
        self._lowercase      = lowercase
        self._max_word_chars = max_word_chars

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        corpus_path: str,
        vocab_size: int = 30_522,
        min_frequency: int = 2,
        text_field: str = "text",
        **kwargs,
    ) -> None:
        """
        Train the WordPiece tokenizer on a text corpus.

        Args:
            corpus_path:   Path to JSONL or plain text corpus.
            vocab_size:    Target vocabulary size.
            min_frequency: Minimum word frequency to include.
            text_field:    For JSONL: key containing the document text.
        """
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            lowercase=self._lowercase,
        )
        vocab, _ = trainer.train(corpus_path=corpus_path, text_field=text_field)
        self._vocab = vocab

        logger.info(
            "WordPieceTokenizer training complete",
            vocab_size=self.vocab_size,
        )

    # ── Tokenisation ───────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the WordPiece longest-match-first algorithm.

        Steps:
          1. Lowercase (if configured)
          2. Split on whitespace and punctuation
          3. For each word, run the WordPiece greedy encoder
          4. Return the flat list of subword tokens
        """
        if self._lowercase:
            text = text.lower()

        # Punctuation isolation: "hello,world" → "hello , world"
        text = self._PUNCT_RE.sub(r" \1 ", text)

        tokens: List[str] = []
        for word in self._WHITESPACE_RE.split(text):
            if not word:
                continue
            word_tokens = self._wordpiece_encode(word)
            tokens.extend(word_tokens)

        return tokens

    def _wordpiece_encode(self, word: str) -> List[str]:
        """
        Encode a single word using the WordPiece longest-match-first algorithm.

        Algorithm:
          1. If the word is too long → return [UNK]
          2. Try to match the longest prefix of the remaining string
             that is present in the vocabulary
          3. If no prefix matches at position 0 → return [UNK] for whole word
          4. Prepend "##" to all tokens after the first one
          5. Repeat for the remaining suffix

        This algorithm is O(len(word)^2) in the worst case, but in practice
        words are short and the vocabulary covers most common sub-words.

        Args:
            word: A single word string (already lowercased if configured).

        Returns:
            List of sub-word token strings, or [ST.UNK] if unrepresentable.
        """
        if len(word) > self._max_word_chars:
            return [ST.UNK]

        tokens: List[str] = []
        start = 0

        while start < len(word):
            end = len(word)
            current_substr = None

            # Try to match the longest possible substring starting at `start`
            while start < end:
                substr = word[start:end]
                # Continuation tokens (not the first sub-word) get "##" prefix
                if start > 0:
                    substr = CONTINUATION_PREFIX + substr

                if substr in self._vocab:
                    current_substr = substr
                    break

                end -= 1

            if current_substr is None:
                # No match found — the whole word becomes [UNK]
                return [ST.UNK]

            tokens.append(current_substr)
            start = end

        return tokens

    # ── Decoding ───────────────────────────────────────────────────────────────

    def _tokens_to_string(self, tokens: List[str]) -> str:
        """
        Reconstruct text from WordPiece tokens.

        Removes "##" continuation prefixes and joins tokens.
        "play", "##ing", "Ġthe" → "playing the"

        The first token in each word group has no prefix; subsequent tokens
        have "##" stripped and are concatenated without a space.
        """
        text = ""
        for token in tokens:
            if token.startswith(CONTINUATION_PREFIX):
                # Continuation: concatenate directly (no space)
                text += token[len(CONTINUATION_PREFIX):]
            else:
                # New word: add space before (except at the very start)
                text = text + (" " if text else "") + token

        return text

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save the tokenizer to disk.

        Files created:
            vocab.json            — token → ID mapping
            tokenizer_config.json — class name and config metadata
        """
        directory = ensure_dir(directory)

        self._vocab.save(directory / "vocab.json")

        write_json(directory / "tokenizer_config.json", {
            "tokenizer_class":  "WordPieceTokenizer",
            "vocab_size":       self.vocab_size,
            "lowercase":        self._lowercase,
            "add_bos_token":    self._add_bos,
            "add_eos_token":    self._add_eos,
            "max_length":       self._max_length,
            "max_word_chars":   self._max_word_chars,
            "continuation_prefix": CONTINUATION_PREFIX,
            "unk_token":        ST.UNK,
            "pad_token":        ST.PAD,
            "bos_token":        ST.BOS,
            "eos_token":        ST.EOS,
            "mask_token":       ST.MASK,
            "sep_token":        ST.SEP,
            "cls_token":        ST.CLS,
        })

        logger.info(
            "WordPieceTokenizer saved",
            directory=str(directory),
            vocab_size=self.vocab_size,
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "WordPieceTokenizer":
        """
        Load a WordPieceTokenizer from disk.

        Args:
            directory: Directory containing vocab.json and tokenizer_config.json.

        Returns:
            Initialised WordPieceTokenizer.
        """
        directory = Path(directory)

        config = read_json(directory / "tokenizer_config.json")
        vocab  = Vocabulary.load(directory / "vocab.json")

        tokenizer = cls(
            vocab=vocab,
            lowercase=config.get("lowercase", True),
            add_bos_token=config.get("add_bos_token", True),
            add_eos_token=config.get("add_eos_token", True),
            max_length=config.get("max_length", 512),
            max_word_chars=config.get("max_word_chars", 100),
        )

        logger.info(
            "WordPieceTokenizer loaded",
            directory=str(directory),
            vocab_size=tokenizer.vocab_size,
        )
        return tokenizer
