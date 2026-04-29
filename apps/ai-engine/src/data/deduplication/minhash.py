"""
InsightSerenity AI Engine — MinHash Near-Duplicate Detection
=============================================================
Detects documents that are highly similar but not byte-for-byte identical —
so-called "near-duplicates". Common examples: articles with one sentence
changed, boilerplate with slightly different names inserted, scraped content
with minor editorial tweaks.

Training on near-duplicates has the same harmful effect as exact duplicates:
the model memorises the repeated content instead of learning language.

Algorithm: MinHash + Locality-Sensitive Hashing (LSH)
─────────────────────────────────────────────────────
1. Represent each document as a set of character n-grams (shingles)
2. Apply k independent hash functions to the shingle set
3. For each hash function, record the *minimum* hash value seen
   → This produces a k-dimensional "signature" (the MinHash)
4. The probability that two signatures agree on position i equals
   the Jaccard similarity between the original shingle sets
5. LSH bands trick: group the k signature values into b bands of r rows each.
   Two documents are candidate duplicates if their signatures agree on ALL r
   values within at least one band. This dramatically reduces the number of
   exact comparisons needed.
6. For candidate pairs, compute exact Jaccard similarity to confirm.

Time complexity:
    Signature computation:  O(|shingles| × k)  per document
    LSH insertion:          O(b)                per document
    Candidate retrieval:    O(b)                per query document

Space complexity:  O(n × k × 4 bytes) for n documents with k-dim signatures

Reference: Leskovec, Rajaraman, Ullman — "Mining of Massive Datasets", Ch. 3

Usage:
    from src.data.deduplication.minhash import MinHashDeduplicator

    dedup = MinHashDeduplicator(num_perm=128, threshold=0.85)
    stats = dedup.deduplicate_file("cleaned.jsonl", "deduped.jsonl")
"""

import hashlib
import struct
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.config.settings import settings
from src.utils.file_io import iter_jsonl, append_jsonl, save_pickle, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shingle extraction
# ─────────────────────────────────────────────────────────────────────────────

def _shingle(text: str, n: int = 5) -> Set[str]:
    """
    Build the set of all character n-grams (shingles) from `text`.

    Character n-grams rather than word n-grams are used because they:
      - Handle morphological variation better
      - Are language-agnostic
      - Are more robust to OCR/encoding errors

    Args:
        text: Input text (should be normalised/cleaned beforehand).
        n:    n-gram size. 5 is a standard value for web text deduplication.

    Returns:
        Set of n-gram strings. Empty set for text shorter than n characters.
    """
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _shingle_hashes(text: str, n: int) -> Set[int]:
    """
    Convert character n-grams to integer hashes (32-bit).
    Using integer hashes rather than raw strings is ~4x faster for NumPy
    operations and reduces memory usage.
    """
    shingles = _shingle(text, n)
    result = set()
    for s in shingles:
        # Use first 4 bytes of MD5 as a 32-bit unsigned integer
        digest = hashlib.md5(s.encode("utf-8")).digest()
        result.add(struct.unpack("<I", digest[:4])[0])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MinHash signature
# ─────────────────────────────────────────────────────────────────────────────

class MinHashSignature:
    """
    Computes MinHash signatures using universal hash functions.

    Each hash function is h_i(x) = (a_i * x + b_i) mod p mod 2^32
    where p is a large prime and a_i, b_i are random coefficients.

    This avoids running k separate hash algorithms; instead we compute k
    linear transformations of a single underlying hash value.

    Args:
        num_perm: Number of permutations (k). Higher → more accurate but slower.
                  128 is a good default (gives ~1% error on Jaccard estimates).
        seed:     Random seed for reproducibility.
    """

    # Large prime for the universal hash family
    _MERSENNE_PRIME = (1 << 61) - 1
    _MAX_HASH       = (1 << 32)

    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        self._num_perm = num_perm
        rng = np.random.RandomState(seed)

        # Generate random coefficients for the universal hash family
        self._a = rng.randint(1, self._MERSENNE_PRIME, size=num_perm, dtype=np.int64)
        self._b = rng.randint(0, self._MERSENNE_PRIME, size=num_perm, dtype=np.int64)

    def compute(self, shingle_hashes: Set[int]) -> np.ndarray:
        """
        Compute the MinHash signature for a document.

        Args:
            shingle_hashes: Set of integer shingle hashes for the document.

        Returns:
            1D numpy array of shape (num_perm,) containing the signature.
            All-maximum array (max_hash) for empty documents.
        """
        # Start with the maximum possible value for each permutation
        signature = np.full(self._num_perm, self._MAX_HASH, dtype=np.int64)

        if not shingle_hashes:
            return signature.astype(np.uint32)

        # Convert shingle hash set to numpy array for vectorised operations
        hashes = np.array(list(shingle_hashes), dtype=np.int64)

        # For each permutation, compute h(x) = (a*x + b) mod p mod 2^32
        # and take the minimum across all shingles
        for i in range(self._num_perm):
            perm_hashes = (self._a[i] * hashes + self._b[i]) % self._MERSENNE_PRIME % self._MAX_HASH
            signature[i] = perm_hashes.min()

        return signature.astype(np.uint32)

    def jaccard_estimate(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Estimate the Jaccard similarity between two documents from their signatures.

        Formula: J(A, B) ≈ |{i : sig_a[i] == sig_b[i]}| / num_perm

        Args:
            sig_a, sig_b: Signature arrays of shape (num_perm,).

        Returns:
            Estimated Jaccard similarity in [0.0, 1.0].
        """
        return float(np.sum(sig_a == sig_b)) / self._num_perm


# ─────────────────────────────────────────────────────────────────────────────
# LSH index
# ─────────────────────────────────────────────────────────────────────────────

class LSHIndex:
    """
    Locality-Sensitive Hashing index for fast candidate pair retrieval.

    Divides the k-dimensional signature into b bands of r rows.
    Two documents are candidate duplicates if they hash to the same bucket
    in at least one band.

    Theoretical false-negative rate ≈ (1 - threshold^r)^b at the threshold.
    Choose b and r so that: b * r == num_perm.

    Args:
        num_perm:   Total signature length (must equal MinHashSignature.num_perm).
        threshold:  Jaccard similarity threshold above which documents are
                    considered near-duplicates.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85) -> None:
        self._num_perm  = num_perm
        self._threshold = threshold
        self._b, self._r = self._optimal_params(num_perm, threshold)

        # One dict per band: bucket_hash → list of doc ids
        self._bands: List[Dict[int, List[str]]] = [
            {} for _ in range(self._b)
        ]

        logger.debug(
            "LSH index initialised",
            num_perm=num_perm,
            threshold=threshold,
            bands=self._b,
            rows_per_band=self._r,
        )

    def insert(self, doc_id: str, signature: np.ndarray) -> None:
        """
        Add a document's signature to the index.

        Args:
            doc_id:    Unique identifier for this document (e.g. URL or hash).
            signature: MinHash signature array of shape (num_perm,).
        """
        for band_idx in range(self._b):
            start  = band_idx * self._r
            end    = start + self._r
            band   = signature[start:end]
            bucket = hash(band.tobytes())   # Python int hash of the band bytes

            if bucket not in self._bands[band_idx]:
                self._bands[band_idx][bucket] = []
            self._bands[band_idx][bucket].append(doc_id)

    def query(self, signature: np.ndarray) -> Set[str]:
        """
        Find all documents in the index that are candidate near-duplicates
        of the given signature.

        Returns:
            Set of doc IDs that share at least one band bucket with this signature.
            May include false positives — callers should verify with exact Jaccard.
        """
        candidates: Set[str] = set()

        for band_idx in range(self._b):
            start  = band_idx * self._r
            end    = start + self._r
            band   = signature[start:end]
            bucket = hash(band.tobytes())

            if bucket in self._bands[band_idx]:
                candidates.update(self._bands[band_idx][bucket])

        return candidates

    @staticmethod
    def _optimal_params(num_perm: int, threshold: float) -> Tuple[int, int]:
        """
        Find band count b and row count r that minimise error at `threshold`.

        We search for the (b, r) pair where b * r <= num_perm that minimises
        |threshold - (1/b)^(1/r)|, which is the x-coordinate of the S-curve
        inflection point.

        Returns:
            (b, r) tuple.
        """
        best_b, best_r = 1, num_perm
        best_error     = float("inf")

        for b in range(1, num_perm + 1):
            r = num_perm // b
            if r == 0:
                continue
            # Inflection point of the LSH S-curve at this (b, r)
            inflection = (1.0 / b) ** (1.0 / r)
            error = abs(threshold - inflection)
            if error < best_error:
                best_error = error
                best_b, best_r = b, r

        return best_b, best_r


# ─────────────────────────────────────────────────────────────────────────────
# Full deduplicator
# ─────────────────────────────────────────────────────────────────────────────

class MinHashDeduplicator:
    """
    End-to-end near-duplicate deduplicator using MinHash + LSH.

    Processes a JSONL corpus file and writes unique (non-near-duplicate)
    documents to an output JSONL file. The first document in any duplicate
    cluster is always kept.

    Args:
        num_perm:  Number of MinHash permutations. Default 128.
        threshold: Jaccard similarity threshold. Default 0.85 (very similar).
        ngram_size: Character n-gram size for shingling. Default 5.
        index_path: Optional path to save/restore the LSH index for resumable runs.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        ngram_size: int = 5,
        index_path: Optional[str] = None,
    ) -> None:
        cfg = settings.deduplication

        self._num_perm   = num_perm or cfg.minhash_num_perm
        self._threshold  = threshold or cfg.minhash_threshold
        self._ngram_size = ngram_size or cfg.minhash_ngram_size
        self._index_path = Path(index_path) if index_path else None

        self._hasher = MinHashSignature(num_perm=self._num_perm)
        self._index  = LSHIndex(num_perm=self._num_perm, threshold=self._threshold)

        # Store signatures for exact Jaccard verification of candidates
        # Maps doc_id → signature array
        self._signatures: Dict[str, np.ndarray] = {}

        if self._index_path and self._index_path.exists():
            self._restore()

    def is_near_duplicate(self, text: str, doc_id: str) -> bool:
        """
        Check if `text` is a near-duplicate of any previously indexed document.

        Does NOT add the document to the index.

        Args:
            text:   Document text to check.
            doc_id: Identifier for this document (used for self-exclusion
                    if this doc was previously indexed).

        Returns:
            True if a near-duplicate exists; False otherwise.
        """
        shingles  = _shingle_hashes(text, self._ngram_size)
        signature = self._hasher.compute(shingles)
        candidates = self._index.query(signature)
        candidates.discard(doc_id)   # Don't compare against self

        for cand_id in candidates:
            cand_sig = self._signatures.get(cand_id)
            if cand_sig is None:
                continue
            estimated_j = self._hasher.jaccard_estimate(signature, cand_sig)
            if estimated_j >= self._threshold:
                return True

        return False

    def add(self, text: str, doc_id: str) -> None:
        """Index a document so future calls can detect duplicates of it."""
        shingles  = _shingle_hashes(text, self._ngram_size)
        signature = self._hasher.compute(shingles)
        self._index.insert(doc_id, signature)
        self._signatures[doc_id] = signature

    def check_and_add(self, text: str, doc_id: str) -> bool:
        """
        Atomic: check for near-duplicate, then index regardless.
        Returns True if near-duplicate found, False if unique.
        """
        is_dup = self.is_near_duplicate(text, doc_id)
        self.add(text, doc_id)
        return is_dup

    def deduplicate_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        id_field: Optional[str] = "url",
    ) -> Dict[str, int]:
        """
        Stream-process a JSONL file and write unique documents to output.

        Args:
            input_path:  Source JSONL file.
            output_path: Destination JSONL file for unique records.
            text_field:  JSON key containing the document text.
            id_field:    JSON key containing a unique document ID.
                         If None, a sequential integer is used.

        Returns:
            Statistics dict: { "total", "unique", "duplicate" }
        """
        total     = 0
        unique    = 0
        duplicate = 0

        for record in iter_jsonl(input_path):
            total += 1
            text   = record.get(text_field, "")
            doc_id = str(record.get(id_field, total)) if id_field else str(total)

            if not text:
                continue

            if self.check_and_add(text, doc_id):
                duplicate += 1
            else:
                unique += 1
                append_jsonl(output_path, record)

            if total % 10_000 == 0:
                logger.info(
                    "MinHash dedup progress",
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
        logger.info("MinHash dedup completed", **stats)

        if self._index_path:
            self._save()

        return stats

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        state = {
            "signatures":  self._signatures,
            "index_bands": self._index._bands,
        }
        try:
            save_pickle(str(self._index_path), state)
            logger.debug("MinHash index saved", path=str(self._index_path))
        except Exception as e:
            logger.warning("Failed to save MinHash index", error=str(e))

    def _restore(self) -> None:
        try:
            state = load_pickle(str(self._index_path))
            self._signatures    = state["signatures"]
            self._index._bands  = state["index_bands"]
            logger.info(
                "MinHash index restored",
                path=str(self._index_path),
                docs=len(self._signatures),
            )
        except Exception as e:
            logger.warning("Failed to restore MinHash index; starting fresh", error=str(e))
