"""
InsightSerenity AI Engine — BPE Trainer
=========================================
Implements the Byte Pair Encoding (BPE) tokenizer training algorithm from
scratch. BPE is the tokenizer algorithm used by GPT-2, GPT-3, GPT-4, and
most modern LLMs. We implement it here from first principles — no dependency
on HuggingFace tokenizers.

Algorithm (original Sennrich et al. 2016, adapted for characters):
────────────────────────────────────────────────────────────────────
1. Pre-tokenize: split corpus into words by whitespace
2. Represent each word as a sequence of characters + end-of-word marker
   "hello" → ["h", "e", "l", "l", "o", "</w>"]
3. Count word frequencies across the corpus
4. Initialize vocabulary with all individual characters
5. Repeat until vocab_size is reached:
   a. Find the most frequent pair of adjacent tokens across all words
   b. Merge that pair into a new token (e.g. "h" + "e" → "he")
   c. Update all word representations to use the new merged token
   d. Add the merge rule to the merge table
6. Save: vocab.json + merges.txt

The end-of-word marker (</w> or Ġ in GPT-2 style) allows the tokenizer
to distinguish "est" inside "interest" from "est" at the end of "interest".
We use the Ġ convention (space prefix) matching GPT-2/GPT-NeoX.

Time complexity:  O(vocab_size × corpus_size) — can be slow on large corpora.
For production corpora > 10GB, use the C++ implementation via the
HuggingFace tokenizers library as a drop-in (same vocab format).

Output files (saved by BPETokenizer.save):
    vocab.json  — { "token": id, ... }
    merges.txt  — "token_a token_b\n" for each merge rule (in order)
    tokenizer_config.json — metadata

Usage:
    from src.tokenizer.bpe.bpe_trainer import BPETrainer

    trainer = BPETrainer(vocab_size=32_000)
    merge_rules, vocab = trainer.train("storage/datasets/corpus.jsonl")
"""

import re
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

from src.config.settings import settings
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.file_io import iter_jsonl, read_lines
from src.utils.logger import get_logger, LogTimer

logger = get_logger(__name__)

# The space-prefix convention: Ġ marks that this token starts a word
# (preceded by a space). This lets BPE distinguish intra-word from
# word-boundary subwords without a separate end-of-word symbol.
SPACE_PREFIX = "Ġ"


class BPETrainer:
    """
    Trains a BPE tokenizer on a text corpus.

    Args:
        vocab_size:     Target vocabulary size (including special tokens).
                        Actual vocab may be slightly smaller if the corpus
                        doesn't have enough unique sub-words.
        min_frequency:  Minimum word frequency for a word to be included
                        in training. Words below this threshold are replaced
                        by their character decomposition but their pairs
                        don't influence which merges are chosen.
        pre_tokenize_pattern: Regex for initial word splitting. Default matches
                        GPT-2's pre-tokeniser: contractions, letters, digits,
                        and punctuation as atomic units.
        show_progress:  Log progress every N merges.
    """

    # GPT-2 / GPT-NeoX pre-tokenisation regex
    # This splits on: contractions ("'s", "'t", "'re"), words, numbers, and
    # punctuation/symbols, treating each as an atomic unit before BPE.
    _GPT2_PRETOKENIZE_RE = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        re.IGNORECASE,
    )

    def __init__(
        self,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        pre_tokenize_pattern: Optional[str] = None,
        show_progress: int = 1000,
    ) -> None:
        self._vocab_size    = vocab_size
        self._min_frequency = min_frequency
        self._show_progress = show_progress

        if pre_tokenize_pattern:
            self._pretok_re = re.compile(pre_tokenize_pattern)
        else:
            self._pretok_re = self._GPT2_PRETOKENIZE_RE

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(
        self,
        corpus_path: str,
        text_field: str = "text",
    ) -> Tuple[List[Tuple[str, str]], Vocabulary]:
        """
        Run the full BPE training algorithm on a corpus.

        Args:
            corpus_path: Path to a plain text file or JSONL file.
                         If JSONL, `text_field` is used to extract text.
            text_field:  Key in JSONL records containing the document text.

        Returns:
            Tuple of:
              - merge_rules: Ordered list of (token_a, token_b) merge pairs
              - vocabulary:  Trained Vocabulary object

        The returned merge_rules are in the order they were learned — this
        order is critical and must be preserved for correct encoding.
        """
        logger.info(
            "BPE training started",
            corpus=corpus_path,
            target_vocab_size=self._vocab_size,
        )

        # Step 1: Count word frequencies in the corpus
        with LogTimer(logger, "Word frequency counting"):
            word_freqs = self._count_words(corpus_path, text_field)

        logger.info(
            "Word frequencies computed",
            unique_words=len(word_freqs),
            total_words=sum(word_freqs.values()),
        )

        # Step 2: Build initial character-level vocabulary
        # Each word is split into its characters, with Ġ prepended to mark
        # word starts (i.e. preceded by a space in the original text)
        word_to_chars: Dict[str, List[str]] = {}
        char_vocab: Set[str] = set()

        for word, freq in word_freqs.items():
            if freq < self._min_frequency:
                continue
            chars = self._word_to_char_sequence(word)
            word_to_chars[word] = chars
            char_vocab.update(chars)

        # Step 3: Build Vocabulary starting with special tokens + char vocab
        vocab = Vocabulary()
        for char in sorted(char_vocab):
            vocab.add_token(char)

        logger.info(
            "Initial vocabulary built",
            char_types=len(char_vocab),
            vocab_size=vocab.size,
        )

        # Step 4: Learn BPE merges until we reach the target vocab size
        num_merges_needed = self._vocab_size - vocab.size
        merge_rules: List[Tuple[str, str]] = []

        with LogTimer(logger, "BPE merge learning", target_merges=num_merges_needed):
            for merge_idx in range(num_merges_needed):
                # Find the most frequent adjacent pair
                pair_freqs = self._count_pairs(word_to_chars, word_freqs)

                if not pair_freqs:
                    logger.info("No more pairs to merge", merges_learned=merge_idx)
                    break

                best_pair = max(pair_freqs, key=pair_freqs.__getitem__)
                best_freq = pair_freqs[best_pair]

                if best_freq < self._min_frequency:
                    logger.info("Best pair below min_frequency", freq=best_freq)
                    break

                # Create the merged token
                merged = best_pair[0] + best_pair[1]
                merge_rules.append(best_pair)
                vocab.add_token(merged)

                # Apply the merge to all word representations
                word_to_chars = self._apply_merge(word_to_chars, best_pair, merged)

                if (merge_idx + 1) % self._show_progress == 0:
                    logger.info(
                        "BPE progress",
                        merges=merge_idx + 1,
                        vocab_size=vocab.size,
                        best_pair=f"{best_pair[0]} + {best_pair[1]}",
                        freq=best_freq,
                    )

        logger.info(
            "BPE training complete",
            merges_learned=len(merge_rules),
            final_vocab_size=vocab.size,
        )

        return merge_rules, vocab

    # ── Internal steps ─────────────────────────────────────────────────────────

    def _count_words(
        self, corpus_path: str, text_field: str
    ) -> Counter:
        """
        Count the frequency of each pre-tokenized word across the corpus.

        Pre-tokenization splits text into word-like units using the GPT-2
        regex before applying BPE character-level operations.
        """
        word_freqs: Counter = Counter()
        docs_seen   = 0

        # Support both plain text and JSONL.
        if corpus_path.endswith(".jsonl") or corpus_path.endswith(".jsonl.gz"):
            texts: Iterator[str] = (r.get(text_field, "") for r in iter_jsonl(corpus_path))
            for text in texts:
                if not text:
                    continue
                for word in self._pretokenize(text):
                    if word.strip():
                        word_freqs[word] += 1
                docs_seen += 1

                if docs_seen % 100_000 == 0:
                    logger.debug("Word counting progress", docs=docs_seen)
            return word_freqs

        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = (line.strip() for line in f)
            for text in lines:
                if not text:
                    continue
                for word in self._pretokenize(text):
                    if word.strip():
                        word_freqs[word] += 1
                docs_seen += 1

                if docs_seen % 100_000 == 0:
                    logger.debug("Word counting progress", docs=docs_seen)

        return word_freqs

    def _pretokenize(self, text: str) -> List[str]:
        """
        Split text into word-like units using the pre-tokenisation regex.

        The Ġ prefix convention:
          - Words NOT at the start of the text have Ġ prepended
          - This marks them as "preceded by a space"
          - "hello world" → ["hello", "Ġworld"]
        """
        tokens = self._pretok_re.findall(text)
        result = []
        for i, token in enumerate(tokens):
            if token.startswith(" ") and token.strip():
                result.append(SPACE_PREFIX + token.strip())
            elif token.strip():
                result.append(token.strip())
        return result

    def _word_to_char_sequence(self, word: str) -> List[str]:
        """
        Convert a word string to its initial character sequence.

        "Ġhello" → ["Ġ", "h", "e", "l", "l", "o"]

        The Ġ prefix is kept as a separate character in the initial
        vocabulary so it can be merged with "h" → "Ġh" in later steps.
        """
        if word.startswith(SPACE_PREFIX):
            # Space prefix is a separate character
            return [SPACE_PREFIX] + list(word[len(SPACE_PREFIX):])
        return list(word)

    def _count_pairs(
        self,
        word_to_chars: Dict[str, List[str]],
        word_freqs: Dict[str, int],
    ) -> Counter:
        """
        Count the frequency of every adjacent token pair across all words.

        The frequency of a pair equals the sum of the word frequencies
        for every word that contains that pair at least once.

        This is O(sum of word lengths) per iteration — the most expensive
        step in BPE training.
        """
        pair_freqs: Counter = Counter()

        for word, chars in word_to_chars.items():
            freq = word_freqs.get(word, 0)
            if freq == 0:
                continue
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _apply_merge(
        self,
        word_to_chars: Dict[str, List[str]],
        pair: Tuple[str, str],
        merged: str,
    ) -> Dict[str, List[str]]:
        """
        Apply a single merge rule to all word representations.

        Replaces every occurrence of (pair[0], pair[1]) with `merged`
        in every word's character sequence.

        Args:
            word_to_chars: Current word→char-sequence mapping.
            pair:          The (left, right) pair to merge.
            merged:        The new token formed by merging the pair.

        Returns:
            Updated word→char-sequence mapping.
        """
        a, b = pair
        updated: Dict[str, List[str]] = {}

        for word, chars in word_to_chars.items():
            new_chars: List[str] = []
            i = 0
            while i < len(chars):
                # Check if this position starts the target pair
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            updated[word] = new_chars

        return updated
