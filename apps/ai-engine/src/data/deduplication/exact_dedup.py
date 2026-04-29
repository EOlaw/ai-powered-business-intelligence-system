"""
InsightSerenity AI Engine — Exact Deduplication
================================================
Detects and removes documents that are character-for-character identical
to another document in the corpus. Exact duplicates arise naturally from
web crawling: the same article is often hosted at multiple URLs (canonical
redirect failures), syndicated to many sites, or scraped from mirrors.

Training on exact duplicates inflates data diversity metrics while actually
teaching the model to memorise repeated passages verbatim — a significant
quality problem.

Approach: MD5 hashing
    - Compute MD5(normalize(text)) for each document
    - MD5 is fast, low memory, and sufficient for deduplication
      (we are not using it for security — collision risk here is negligible)
    - Store seen hashes in an in-memory set (O(1) lookup)
    - Optionally persist the hash set to disk for multi-process or resumable deduplication

Usage:
    from src.data.deduplication.exact_dedup import ExactDeduplicator

    dedup = ExactDeduplicator()
    is_dup = dedup.is_duplicate("some document text")
    dedup.add("some document text")   # mark as seen

    # Or process a full JSONL file:
    stats = dedup.deduplicate_file(
        input_path="raw.jsonl",
        output_path="deduped.jsonl",
        text_field="text",
    )
"""

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Set

from src.utils.file_io import iter_jsonl, append_jsonl, save_pickle, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _normalise_for_hash(text: str) -> str:
    """
    Lightly normalise text before hashing so minor variations in whitespace
    or Unicode encoding do not prevent duplicate detection.

    Steps:
      1. Unicode NFC normalisation
      2. Lowercase
      3. Collapse all whitespace to single spaces
      4. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _hash_text(text: str) -> str:
    """Return the MD5 hex digest of normalised text."""
    normalised = _normalise_for_hash(text)
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """
    In-memory exact deduplicator backed by an MD5 hash set.

    Suitable for corpora up to ~100M documents on a machine with sufficient
    RAM (each 32-char hex hash takes ~120 bytes with Python set overhead,
    so 100M docs ≈ 12 GB RAM). For larger corpora, use MinHash or a
    bloom-filter-backed variant.

    Args:
        hash_store_path: Optional path to persist/restore the hash set.
                         Enables resumable deduplication across multiple runs.
    """

    def __init__(self, hash_store_path: Optional[str] = None) -> None:
        self._seen_hashes: Set[str] = set()
        self._hash_store_path = Path(hash_store_path) if hash_store_path else None

        # Restore previous state if a store file exists
        if self._hash_store_path and self._hash_store_path.exists():
            self._restore()

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_duplicate(self, text: str) -> bool:
        """
        Return True if this text (after normalisation) has been seen before.

        Does NOT add the text to the seen set — call `add()` separately if
        you want to record it after checking.
        """
        return _hash_text(text) in self._seen_hashes

    def add(self, text: str) -> None:
        """Mark a document as seen by storing its hash."""
        self._seen_hashes.add(_hash_text(text))

    def check_and_add(self, text: str) -> bool:
        """
        Atomic check-and-add: return True if the text is a duplicate,
        otherwise record it and return False.

        Use this in tight processing loops instead of separate is_duplicate + add calls.
        """
        h = _hash_text(text)
        if h in self._seen_hashes:
            return True   # Duplicate
        self._seen_hashes.add(h)
        return False      # Not a duplicate

    def deduplicate_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
    ) -> Dict[str, int]:
        """
        Deduplicate an entire JSONL file in one pass.

        Reads the input file record-by-record (streaming, O(1) memory per
        record), checks each document against the seen-hash set, and writes
        unique documents to the output file.

        Args:
            input_path:  Source JSONL file path.
            output_path: Destination JSONL file for unique records.
            text_field:  Key in each JSON record that holds the document text.

        Returns:
            Statistics dict: { "total", "unique", "duplicate" }
        """
        total     = 0
        unique    = 0
        duplicate = 0

        for record in iter_jsonl(input_path):
            total += 1
            text = record.get(text_field, "")

            if not text:
                # Empty text field — skip silently
                continue

            if self.check_and_add(text):
                duplicate += 1
            else:
                unique += 1
                append_jsonl(output_path, record)

            if total % 50_000 == 0:
                logger.info(
                    "Exact dedup progress",
                    total=total,
                    unique=unique,
                    duplicate=duplicate,
                    dup_rate=round(duplicate / total, 4),
                )

        stats = {
            "total":     total,
            "unique":    unique,
            "duplicate": duplicate,
            "dup_rate":  round(duplicate / max(total, 1), 4),
        }

        logger.info("Exact dedup completed", **stats)

        # Persist the updated hash set
        if self._hash_store_path:
            self._save()

        return stats

    @property
    def size(self) -> int:
        """Number of unique documents currently tracked."""
        return len(self._seen_hashes)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist the hash set to disk as a pickle file."""
        try:
            save_pickle(str(self._hash_store_path), self._seen_hashes)
            logger.debug(
                "Hash store saved",
                path=str(self._hash_store_path),
                hashes=len(self._seen_hashes),
            )
        except Exception as e:
            logger.warning("Failed to save hash store", error=str(e))

    def _restore(self) -> None:
        """Reload the hash set from a pickle file."""
        try:
            self._seen_hashes = load_pickle(str(self._hash_store_path))
            logger.info(
                "Hash store restored",
                path=str(self._hash_store_path),
                hashes=len(self._seen_hashes),
            )
        except Exception as e:
            logger.warning("Failed to restore hash store; starting fresh", error=str(e))
            self._seen_hashes = set()
