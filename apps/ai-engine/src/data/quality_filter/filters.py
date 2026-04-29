"""
InsightSerenity AI Engine — Quality Filters
============================================
A suite of document-level quality filters applied after deduplication. The
goal is to reject documents that, while technically not duplicates, would
degrade model quality if included in the training corpus.

Quality signals implemented here:
    1. Length filter    — too short or too long documents
    2. Alpha ratio      — documents dominated by symbols/numbers
    3. Repetition filter — documents with pathological repeated phrases
                          (a common pattern in scraped web content)
    4. Bullet/fragment  — documents consisting mostly of list items,
                          navigation fragments, or short single-line texts
    5. Perplexity filter — (optional) uses a small reference LM to score
                          document naturalness; high perplexity = low quality

The filters are composable: build a QualityFilterPipeline and add whichever
filters are appropriate for your data source.

Usage:
    from src.data.quality_filter.filters import QualityFilterPipeline

    pipeline = QualityFilterPipeline()
    result = pipeline.filter("some text document")
    if result.passed:
        save(result.text)
    else:
        logger.debug(f"Rejected: {result.rejection_reason}")
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.config.settings import settings
from src.data.preprocessing.normalizer import TextNormalizer
from src.data.quality_filter.language_detector import LanguageDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Filter result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """
    Output of running a document through the quality filter pipeline.

    Attributes:
        text:             The (possibly modified) document text.
        passed:           True if the document passes all filters.
        rejection_reason: Human-readable reason if the document was rejected.
        scores:           Dict of per-filter diagnostic scores for analysis.
    """
    text:             str
    passed:           bool
    rejection_reason: Optional[str] = None
    scores:           dict           = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Individual filters
# ─────────────────────────────────────────────────────────────────────────────

class LengthFilter:
    """
    Reject documents that are too short or too long.

    Too-short documents (< min_tokens) don't provide enough context for the
    model to learn sentence structure. Too-long documents (> max_tokens) are
    typically concatenated scraped pages or raw data dumps — not prose.

    Tokens are approximated by whitespace splitting (fast, no tokenizer needed
    at this stage of the pipeline).
    """

    def __init__(
        self,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        cfg = settings.preprocessing
        self._min = min_tokens if min_tokens is not None else cfg.min_tokens
        self._max = max_tokens if max_tokens is not None else cfg.max_tokens

    def check(self, text: str) -> Tuple[bool, Optional[str], dict]:
        """
        Returns (passed, rejection_reason, scores).
        """
        token_count = len(text.split())
        if token_count < self._min:
            return (
                False,
                f"too_short ({token_count} tokens < {self._min})",
                {"token_count": token_count},
            )
        if token_count > self._max:
            return (
                False,
                f"too_long ({token_count} tokens > {self._max})",
                {"token_count": token_count},
            )
        return (True, None, {"token_count": token_count})


class AlphaRatioFilter:
    """
    Reject documents where too few characters are alphabetic.

    A document with alpha ratio < 0.6 is dominated by numbers, punctuation,
    or symbols — typical of scraped tables, code dumps, or spam.
    """

    def __init__(self, min_alpha_ratio: Optional[float] = None) -> None:
        cfg = settings.preprocessing
        self._min = min_alpha_ratio if min_alpha_ratio is not None else cfg.min_alpha_ratio
        self._normalizer = TextNormalizer()

    def check(self, text: str) -> Tuple[bool, Optional[str], dict]:
        alpha_ratio = self._normalizer.compute_alpha_ratio(text)
        if alpha_ratio < self._min:
            return (
                False,
                f"low_alpha_ratio ({alpha_ratio:.3f} < {self._min})",
                {"alpha_ratio": alpha_ratio},
            )
        return (True, None, {"alpha_ratio": alpha_ratio})


class RepetitionFilter:
    """
    Detect documents with pathological text repetition.

    Two repetition signals:
      1. n-gram repetition: fraction of the document covered by the most
         common n-gram. A value > 0.3 suggests the text is mostly repeated.
      2. Sentence repetition: same sentence appears more than 3 times.
         Common in scraped pages with pagination artifacts or injected ads.
    """

    def __init__(
        self,
        max_repetition_ratio: Optional[float] = None,
        max_sentence_repeats: int = 3,
        ngram_size: int = 10,
    ) -> None:
        cfg = settings.quality_filter
        self._max_ratio      = max_repetition_ratio or cfg.max_repetition_ratio
        self._max_repeats    = max_sentence_repeats
        self._ngram_size     = ngram_size

    def check(self, text: str) -> Tuple[bool, Optional[str], dict]:
        # ── n-gram repetition ─────────────────────────────────────────────────
        words   = text.split()
        total   = len(words)

        if total < self._ngram_size * 2:
            return (True, None, {"repetition_ratio": 0.0})

        ngrams: dict = {}
        for i in range(total - self._ngram_size + 1):
            ng = " ".join(words[i:i + self._ngram_size])
            ngrams[ng] = ngrams.get(ng, 0) + 1

        if ngrams:
            most_common_count = max(ngrams.values())
            # How many words are covered by the most common n-gram?
            repetition_ratio = (most_common_count * self._ngram_size) / total
        else:
            repetition_ratio = 0.0

        if repetition_ratio > self._max_ratio:
            return (
                False,
                f"high_ngram_repetition ({repetition_ratio:.3f} > {self._max_ratio})",
                {"repetition_ratio": repetition_ratio},
            )

        # ── Sentence repetition ───────────────────────────────────────────────
        sentences: dict = {}
        for sentence in re.split(r"[.!?\n]+", text):
            s = sentence.strip().lower()
            if len(s) > 20:
                sentences[s] = sentences.get(s, 0) + 1

        if sentences:
            max_repeats = max(sentences.values())
            if max_repeats > self._max_repeats:
                return (
                    False,
                    f"repeated_sentences (sentence repeated {max_repeats} times)",
                    {"max_sentence_repeats": max_repeats, "repetition_ratio": repetition_ratio},
                )

        return (True, None, {"repetition_ratio": repetition_ratio})


class BulletRatioFilter:
    """
    Reject documents that consist primarily of short list fragments.

    Navigation menus, footer links, and tag clouds are common web scraping
    artifacts. These look like:
        Home
        About
        Contact
        Services
        Blog

    They have no training value and inflate the vocabulary with isolated words.
    """

    def __init__(self, max_bullet_ratio: Optional[float] = None) -> None:
        cfg = settings.quality_filter
        self._max_ratio = max_bullet_ratio or cfg.max_bullet_ratio

    def check(self, text: str) -> Tuple[bool, Optional[str], dict]:
        lines      = [l.strip() for l in text.splitlines() if l.strip()]
        total_lines = len(lines)

        if total_lines < 5:
            return (True, None, {"bullet_ratio": 0.0})

        # A "bullet" line is short (< 8 words) and likely navigational
        short_lines = sum(1 for l in lines if len(l.split()) < 8)
        bullet_ratio = short_lines / total_lines

        if bullet_ratio > self._max_ratio:
            return (
                False,
                f"high_bullet_ratio ({bullet_ratio:.3f} > {self._max_ratio})",
                {"bullet_ratio": bullet_ratio},
            )
        return (True, None, {"bullet_ratio": bullet_ratio})


class LanguageFilter:
    """
    Reject documents not in the configured allowed languages.
    Wraps LanguageDetector for use in the filter pipeline.
    """

    def __init__(
        self,
        allowed_languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> None:
        cfg = settings.quality_filter
        self._detector = LanguageDetector(
            allowed_languages=allowed_languages or cfg.allowed_languages,
            min_confidence=min_confidence or cfg.min_language_confidence,
        )

    def check(self, text: str) -> Tuple[bool, Optional[str], dict]:
        lang, confidence = self._detector.detect(text)
        scores = {"language": lang, "lang_confidence": confidence}

        if not self._detector.is_allowed(text):
            return (
                False,
                f"language_rejected (detected={lang}, confidence={confidence:.3f})",
                scores,
            )
        return (True, None, scores)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class QualityFilterPipeline:
    """
    Applies multiple quality filters in sequence, short-circuiting on the
    first failure.

    Filters are applied in the order they are added. Adding expensive filters
    (e.g. language detection) after cheap ones (e.g. length) maximises
    throughput by rejecting bad documents early.

    Usage:
        pipeline = QualityFilterPipeline()
        # Filters are pre-loaded with sensible defaults:
        result = pipeline.filter(text)

        # Or build a custom pipeline:
        pipeline = QualityFilterPipeline(filters=[
            LengthFilter(min_tokens=100),
            AlphaRatioFilter(min_alpha_ratio=0.7),
            LanguageFilter(allowed_languages=["en"]),
        ])
    """

    def __init__(self, filters=None) -> None:
        if filters is not None:
            self._filters = filters
        else:
            # Default pipeline: ordered from cheapest to most expensive
            self._filters = [
                LengthFilter(),
                AlphaRatioFilter(),
                RepetitionFilter(),
                BulletRatioFilter(),
                LanguageFilter(),
            ]

    def filter(self, text: str) -> FilterResult:
        """
        Run the document through all filters.

        Short-circuits on the first failure (does not run remaining filters).
        This is intentional — once a document fails a cheap filter, there is
        no value in running expensive ones.

        Args:
            text: The document text to evaluate.

        Returns:
            FilterResult indicating pass/fail and diagnostic scores.
        """
        all_scores: dict = {}

        for quality_filter in self._filters:
            passed, reason, scores = quality_filter.check(text)
            all_scores.update(scores)

            if not passed:
                return FilterResult(
                    text=text,
                    passed=False,
                    rejection_reason=reason,
                    scores=all_scores,
                )

        return FilterResult(text=text, passed=True, scores=all_scores)

    def filter_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
    ) -> dict:
        """
        Apply the quality filter pipeline to an entire JSONL corpus file.

        Args:
            input_path:  Source JSONL file.
            output_path: Destination JSONL for passing documents.
            text_field:  JSON key containing the document text.

        Returns:
            Statistics dict with counts per rejection reason.
        """
        from src.utils.file_io import iter_jsonl, append_jsonl

        stats: dict = {
            "total": 0, "passed": 0, "rejected": 0, "reasons": {}
        }

        for record in iter_jsonl(input_path):
            stats["total"] += 1
            text = record.get(text_field, "")

            result = self.filter(text)

            if result.passed:
                stats["passed"] += 1
                append_jsonl(output_path, record)
            else:
                stats["rejected"] += 1
                reason = result.rejection_reason or "unknown"
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

            if stats["total"] % 10_000 == 0:
                logger.info(
                    "Quality filter progress",
                    total=stats["total"],
                    passed=stats["passed"],
                    rejected=stats["rejected"],
                )

        logger.info(
            "Quality filter completed",
            total=stats["total"],
            passed=stats["passed"],
            rejected=stats["rejected"],
            pass_rate=round(stats["passed"] / max(stats["total"], 1), 4),
        )
        return stats
