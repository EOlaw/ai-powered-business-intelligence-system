"""
InsightSerenity AI Engine — Long-Term (FAISS) Memory
=====================================================
Long-term memory persists across sessions and retrieves relevant past
experiences, facts, and observations when the agent needs them.

Architecture: Vector store backed by FAISS
    - Store: text → embed → add to FAISS index + save to disk
    - Retrieve: query → embed → cosine search → return top-K texts

Why FAISS?
    FAISS (Facebook AI Similarity Search) is a library for efficient
    similarity search on dense vectors. It runs entirely locally (no cloud)
    and scales to billions of vectors. We use IndexFlatIP (inner product)
    with normalised vectors, which is equivalent to cosine similarity.

Memory types:
    EPISODIC  — specific events ("On Monday I searched for X and found Y")
    SEMANTIC  — general facts ("The capital of France is Paris")
    PROCEDURAL — task completions ("To solve Y: do steps A, B, C")

The agent uses long-term memory to:
    1. Avoid repeating searches it has done before
    2. Recall facts from previous sessions
    3. Learn from past successes and failures

Persistence:
    The FAISS index and passages are saved to disk after each write.
    On startup, they are restored so memory persists across restarts.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """
    FAISS-backed long-term semantic memory for agents.

    Stores arbitrary text passages as dense embeddings and retrieves
    the most semantically similar passages for a given query.

    Falls back to keyword search if FAISS is not available or no
    encoder is configured.

    Args:
        embed_fn:    Callable mapping text → np.ndarray of shape (D,).
                     Typically BERTEncoder.encode or a sentence embedding model.
        embed_dim:   Output dimension of embed_fn.
        persist_dir: Directory where the index and passages are saved.
        top_k:       Default number of results to return.
        similarity_threshold: Minimum similarity score to include a result.
    """

    def __init__(
        self,
        embed_fn:             Optional[Any]  = None,
        embed_dim:            int            = 256,
        persist_dir:          Optional[str]  = None,
        top_k:                int            = 5,
        similarity_threshold: float          = 0.3,
    ) -> None:
        self.embed_fn             = embed_fn
        self.embed_dim            = embed_dim
        self.persist_dir          = Path(persist_dir) if persist_dir else None
        self.top_k                = top_k
        self.similarity_threshold = similarity_threshold

        self._passages:   List[str]              = []
        self._metadata:   List[dict]             = []
        self._embeddings: Optional[np.ndarray]   = None
        self._index       = None

        # Try to load FAISS
        self._faiss = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="builtin type (SwigPy.*|swigvarlink) has no __module__ attribute",
                    category=DeprecationWarning,
                )
                import faiss
            self._faiss = faiss
        except ImportError:
            logger.warning("faiss-cpu not installed — using brute-force search")

        # Restore persisted memory
        if self.persist_dir:
            self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(self, text: str, metadata: Optional[dict] = None) -> None:
        """
        Store a text passage in long-term memory.

        Args:
            text:     The passage to remember.
            metadata: Optional dict (e.g. {"source": "web_search", "timestamp": ...}).
        """
        if not text.strip():
            return

        self._passages.append(text)
        self._metadata.append(metadata or {})

        if self.embed_fn is not None:
            emb = self._encode(text)
            if self._embeddings is None:
                self._embeddings = emb.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])
            self._rebuild_index()

        if self.persist_dir:
            self._save()

        logger.debug("Memory stored", n_total=len(self._passages))

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve the most relevant passages for a query.

        Args:
            query: Natural language query.
            top_k: Override the default top_k.

        Returns:
            Formatted string of relevant passages, or empty string if none found.
        """
        k = top_k or self.top_k

        if not self._passages:
            return ""

        if self.embed_fn is None or self._embeddings is None:
            results = self._keyword_search(query, k)
        else:
            results = self._vector_search(query, k)

        if not results:
            return ""

        return "\n---\n".join(results[:k])

    def store_episode(self, task: str, result: str) -> None:
        """
        Convenience: store a completed task and its result as an episodic memory.

        Args:
            task:   The original task.
            result: The final answer or result.
        """
        episode = f"Task: {task}\nResult: {result}"
        self.store(episode, metadata={"type": "episodic"})

    def store_fact(self, fact: str, source: Optional[str] = None) -> None:
        """
        Convenience: store a semantic fact.

        Args:
            fact:   The fact to remember.
            source: Where this fact came from.
        """
        self.store(fact, metadata={"type": "semantic", "source": source or "unknown"})

    def clear(self) -> None:
        """Clear all stored memories."""
        self._passages   = []
        self._metadata   = []
        self._embeddings = None
        self._index      = None
        if self.persist_dir:
            self._save()

    @property
    def size(self) -> int:
        """Number of stored passages."""
        return len(self._passages)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _vector_search(self, query: str, k: int) -> List[str]:
        """FAISS cosine similarity search."""
        query_emb = self._encode(query).reshape(1, -1)
        # Normalise for cosine similarity via inner product
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        if self._index is not None:
            distances, indices = self._index.search(query_norm.astype(np.float32), k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and float(dist) >= self.similarity_threshold:
                    results.append(self._passages[idx])
            return results
        else:
            # Brute force
            norms     = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
            emb_norm  = self._embeddings / norms
            sims      = (emb_norm @ query_norm.T).squeeze()
            if sims.ndim == 0:
                sims = sims.reshape(1)
            top_idx = sims.argsort()[::-1][:k]
            return [
                self._passages[i] for i in top_idx
                if float(sims[i]) >= self.similarity_threshold
            ]

    def _keyword_search(self, query: str, k: int) -> List[str]:
        """Simple keyword overlap fallback when embeddings are unavailable."""
        query_words = set(query.lower().split())
        scored = []
        for passage in self._passages:
            score = len(query_words & set(passage.lower().split()))
            if score > 0:
                scored.append((score, passage))
        scored.sort(reverse=True)
        return [p for _, p in scored[:k]]

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to a float32 embedding vector."""
        emb = self.embed_fn(text)
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        return np.asarray(emb, dtype=np.float32).flatten()

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from scratch after adding new passages."""
        if self._faiss is None or self._embeddings is None:
            return
        D     = self._embeddings.shape[1]
        index = self._faiss.IndexFlatIP(D)
        norms  = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normed = (self._embeddings / norms).astype(np.float32)
        index.add(normed)
        self._index = index

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist passages, metadata, and embeddings to disk."""
        if not self.persist_dir:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Save passages and metadata as JSONL
        with open(self.persist_dir / "passages.jsonl", "w") as f:
            for passage, meta in zip(self._passages, self._metadata):
                f.write(json.dumps({"text": passage, "meta": meta}) + "\n")

        # Save embeddings as numpy
        if self._embeddings is not None:
            np.save(str(self.persist_dir / "embeddings.npy"), self._embeddings)

        logger.debug("Long-term memory saved", dir=str(self.persist_dir), n=len(self._passages))

    def _load(self) -> None:
        """Restore memory from disk."""
        if not self.persist_dir:
            return

        passages_file = self.persist_dir / "passages.jsonl"
        embeddings_file = self.persist_dir / "embeddings.npy"

        if passages_file.exists():
            with open(passages_file) as f:
                for line in f:
                    record = json.loads(line)
                    self._passages.append(record["text"])
                    self._metadata.append(record.get("meta", {}))

        if embeddings_file.exists():
            self._embeddings = np.load(str(embeddings_file))
            self._rebuild_index()

        if self._passages:
            logger.info("Long-term memory restored", n=len(self._passages))

    def __repr__(self) -> str:
        return f"LongTermMemory(size={self.size}, faiss={self._faiss is not None})"
