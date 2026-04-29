"""
InsightSerenity AI Engine — Text Normaliser
============================================
Final normalization pass applied to cleaned text before it is stored in
the training corpus. Operates at the document level and the sentence level.

This stage makes deliberate, configurable decisions about the text
distribution the model will learn from. Each decision is explained below.

Usage:
    from src.data.preprocessing.normalizer import TextNormalizer

    normalizer = TextNormalizer(lowercase=False)
    normalised = normalizer.normalize(text)
"""

import re
import unicodedata
from typing import List, Optional

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# Common ligatures that should be expanded for tokenizer consistency
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl",
    "æ": "ae",   # æ
    "œ": "oe",   # œ
}

# Smart/curly quotes → straight ASCII equivalents
_QUOTE_MAP = {
    "‘": "'",   # Left single quotation mark
    "’": "'",   # Right single quotation mark
    "‚": "'",   # Single low-9 quotation mark
    "‛": "'",   # Single high-reversed-9 quotation mark
    "“": '"',   # Left double quotation mark
    "”": '"',   # Right double quotation mark
    "„": '"',   # Double low-9 quotation mark
    "‟": '"',   # Double high-reversed-9 quotation mark
    "‹": "'",   # Single left-pointing angle quotation mark
    "›": "'",   # Single right-pointing angle quotation mark
    "«": '"',   # «
    "»": '"',   # »
}

# Typographic dashes and hyphens → ASCII hyphen-minus
_DASH_MAP = {
    "–": "-",   # en dash
    "—": "-",   # em dash
    "―": "-",   # horizontal bar
    "−": "-",   # minus sign
    "﹘": "-",   # small em dash
    "﹣": "-",   # small hyphen-minus
    "－": "-",   # fullwidth hyphen-minus
}

# Ellipsis variants → "..."
_ELLIPSIS_MAP = {
    "…": "...",  # …
    "⋯": "...",  # ⋯
}

# Non-breaking / exotic spaces → regular space
_SPACE_MAP = {
    " ": " ",   # non-breaking space
    " ": " ",   # narrow no-break space
    " ": " ",   # medium mathematical space
    "　": " ",   # ideographic space
    " ": " ",   # figure space
    " ": " ",   # punctuation space
    " ": " ",   # thin space
    " ": " ",   # hair space
}

# All character substitutions merged into a single translation table
_TRANSLATION_TABLE = str.maketrans({
    **_LIGATURES,
    **_QUOTE_MAP,
    **_DASH_MAP,
    **_ELLIPSIS_MAP,
    **_SPACE_MAP,
})

# Sentence splitting: split on ". " or "! " or "? " followed by capital letter
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Leading/trailing punctuation on sentences
_LEADING_PUNCT_RE  = re.compile(r"^[\s\-_=+*#@|\\]+")
_TRAILING_SPACE_RE = re.compile(r"\s+$")


class TextNormalizer:
    """
    Final normalisation pass for training corpus documents.

    Applies typographic normalisation (smart quotes, dashes, ligatures),
    optional lowercasing, sentence boundary detection, and length-based
    filtering of individual sentences.

    Args:
        lowercase:         Convert all text to lowercase. Default False —
                           case information helps the model distinguish proper
                           nouns, acronyms, and sentence starts.
        normalise_quotes:  Replace typographic quotes with ASCII. Default True.
        normalise_dashes:  Replace em/en dashes with ASCII hyphen. Default True.
        expand_ligatures:  Expand fi/fl/ff ligatures. Default True.
        min_sentence_len:  Drop sentences shorter than this many characters.
        max_sentence_len:  Truncate or drop sentences longer than this.
    """

    def __init__(
        self,
        lowercase: bool = False,
        normalise_quotes: bool = True,
        normalise_dashes: bool = True,
        expand_ligatures: bool = True,
        min_sentence_len: int = 10,
        max_sentence_len: int = 5000,
    ) -> None:
        self._lowercase         = lowercase
        self._normalise_quotes  = normalise_quotes
        self._normalise_dashes  = normalise_dashes
        self._expand_ligatures  = expand_ligatures
        self._min_sentence_len  = min_sentence_len
        self._max_sentence_len  = max_sentence_len

    def normalize(self, text: str) -> str:
        """
        Apply all normalization steps to a complete document.

        Args:
            text: Cleaned plain text (output of TextCleaner).

        Returns:
            Normalised text ready for the deduplication and quality-filter stages.
        """
        if not text:
            return ""

        # Typographic character substitutions (one O(n) pass)
        text = text.translate(_TRANSLATION_TABLE)

        # Lowercase (if configured — off by default, see docstring)
        if self._lowercase:
            text = text.lower()

        # Remove leading/trailing whitespace from each line
        lines = [line.rstrip() for line in text.splitlines()]
        text  = "\n".join(lines)

        return text.strip()

    def normalize_sentences(self, text: str) -> List[str]:
        """
        Split a normalised document into individual sentences and
        apply sentence-level length filters.

        Useful for the quality filter's repetition-detection step, which
        operates at sentence level.

        Args:
            text: Normalised document text.

        Returns:
            List of clean, non-trivial sentences.
        """
        # First normalise the full document
        text = self.normalize(text)

        # Naive sentence splitting on end-of-sentence punctuation
        # (not using NLTK/spaCy to avoid heavy dependencies at this stage)
        raw_sentences = _SENTENCE_SPLIT_RE.split(text)

        sentences: List[str] = []
        for raw in raw_sentences:
            sentence = raw.strip()
            sentence = _LEADING_PUNCT_RE.sub("", sentence).strip()

            if len(sentence) < self._min_sentence_len:
                continue
            if len(sentence) > self._max_sentence_len:
                # Truncate rather than drop — the beginning is usually the best part
                sentence = sentence[:self._max_sentence_len]

            sentences.append(sentence)

        return sentences

    def compute_alpha_ratio(self, text: str) -> float:
        """
        Compute the ratio of alphabetic characters to total non-whitespace characters.

        Used by the quality filter to discard documents that are predominantly
        numbers, symbols, or punctuation.

        Returns:
            Float in [0.0, 1.0]. Returns 0.0 for empty strings.
        """
        non_ws = [c for c in text if not c.isspace()]
        if not non_ws:
            return 0.0
        alpha = sum(1 for c in non_ws if c.isalpha())
        return alpha / len(non_ws)
