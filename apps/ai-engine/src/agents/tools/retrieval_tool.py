"""
InsightSerenity AI Engine — Retrieval Tool
==========================================
The retrieval tool gives the agent access to its long-term memory via
FAISS vector similarity search. When the agent needs information it may
have encountered in a previous session or a preloaded knowledge base,
it uses this tool to retrieve the most relevant stored passages.

Architecture:
    1. Agent sends a natural language query
    2. Tool encodes the query using our embedding model
    3. FAISS finds the top-K nearest stored passages by cosine similarity
    4. Tool returns the most relevant passages as context

This is the Retrieval-Augmented Generation (RAG) pattern:
    Knowledge base → FAISS index → relevant passages → LLM context

The tool can be seeded with:
    - Documentation
    - Previous conversation summaries
    - Domain-specific facts
    - Tool execution histories

The embedding model is our own encoder (BERTEncoder) — no external API.
"""

from typing import Any, List, Optional

import numpy as np

from src.agents.tools.tool_registry import BaseTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalTool(BaseTool):
    """
    FAISS-backed document retrieval tool.

    Stores text passages as dense embeddings and retrieves the most
    similar passages for a given query. Uses cosine similarity.

    Args:
        embed_fn:  Callable that maps text → numpy array of shape (D,).
                   Typically the encode method of a BERTEncoder.
        top_k:     Number of passages to return. Default 3.
        embed_dim: Embedding dimension (must match embed_fn output).
    """

    name        = "retrieve_knowledge"
    description = (
        "Retrieves relevant information from the knowledge base. "
        "Use when you need factual information stored from previous interactions "
        "or loaded documents. Input: a natural language query."
    )

    def __init__(
        self,
        embed_fn:  Optional[Any] = None,
        top_k:     int           = 3,
        embed_dim: int           = 256,
    ) -> None:
        super().__init__()
        self.embed_fn  = embed_fn
        self.top_k     = top_k
        self.embed_dim = embed_dim

        # In-memory storage for passages and their embeddings
        self._passages:   List[str]        = []
        self._embeddings: Optional[np.ndarray] = None  # (N, D)

        # Lazy-import FAISS
        self._index = None
        self._faiss_available = False
        try:
            import faiss
            self._faiss_available = True
        except ImportError:
            logger.warning("FAISS not installed — using brute-force search fallback")

    def add_documents(self, texts: List[str]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            texts: List of text passages to store.
        """
        if not texts:
            return

        if self.embed_fn is None:
            # Without an encoder, store texts for exact/keyword search
            self._passages.extend(texts)
            logger.info("Documents added (no encoder — keyword search only)", n=len(texts))
            return

        # Encode each passage
        new_embeddings = []
        for text in texts:
            try:
                emb = self._encode(text)
                new_embeddings.append(emb)
                self._passages.append(text)
            except Exception as e:
                logger.warning("Failed to encode document", error=str(e))

        if not new_embeddings:
            return

        new_emb_arr = np.array(new_embeddings, dtype=np.float32)

        if self._embeddings is None:
            self._embeddings = new_emb_arr
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb_arr])

        # Rebuild FAISS index
        self._build_index()
        logger.info("Knowledge base updated", total=len(self._passages))

    def _run(self, tool_input: str) -> str:
        """
        Retrieve relevant passages for the query.

        Args:
            tool_input: Natural language query string.

        Returns:
            Top-K relevant passages as formatted text.
        """
        query = tool_input.strip()

        if not self._passages:
            return "Knowledge base is empty. No documents have been loaded."

        if self.embed_fn is None or self._embeddings is None:
            # Keyword fallback: find passages containing query terms
            return self._keyword_search(query)

        try:
            return self._vector_search(query)
        except Exception as e:
            logger.warning("Vector search failed, falling back to keyword", error=str(e))
            return self._keyword_search(query)

    def _vector_search(self, query: str) -> str:
        """FAISS cosine similarity search."""
        query_emb = self._encode(query).reshape(1, -1)

        if self._index is not None:
            # FAISS search
            distances, indices = self._index.search(query_emb, min(self.top_k, len(self._passages)))
            results = [self._passages[i] for i in indices[0] if i >= 0]
        else:
            # NumPy brute-force cosine similarity
            # Normalise
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
            emb_norm   = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)
            sims       = (emb_norm @ query_norm.T).squeeze()
            top_idx    = sims.argsort()[::-1][:self.top_k]
            results    = [self._passages[i] for i in top_idx]

        if not results:
            return "No relevant information found."

        return self._format_results(results)

    def _keyword_search(self, query: str) -> str:
        """Simple keyword matching fallback."""
        query_lower = query.lower()
        keywords    = set(query_lower.split())
        scored      = []

        for passage in self._passages:
            passage_lower = passage.lower()
            score = sum(1 for kw in keywords if kw in passage_lower)
            if score > 0:
                scored.append((score, passage))

        if not scored:
            return "No relevant passages found for the query."

        scored.sort(reverse=True)
        results = [p for _, p in scored[:self.top_k]]
        return self._format_results(results)

    def _encode(self, text: str) -> np.ndarray:
        """Encode text using the embedding function."""
        emb = self.embed_fn(text)
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        return emb.flatten().astype(np.float32)

    def _build_index(self) -> None:
        """Build or rebuild the FAISS index from stored embeddings."""
        if not self._faiss_available or self._embeddings is None:
            return
        try:
            import faiss
            D     = self._embeddings.shape[1]
            index = faiss.IndexFlatIP(D)   # Inner product (use normalised embeddings for cosine)
            # Normalise embeddings for cosine similarity
            norms  = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
            normed = (self._embeddings / norms).astype(np.float32)
            index.add(normed)
            self._index = index
        except Exception as e:
            logger.warning("FAISS index build failed", error=str(e))

    @staticmethod
    def _format_results(passages: List[str]) -> str:
        """Format retrieved passages as numbered list."""
        parts = []
        for i, p in enumerate(passages, start=1):
            parts.append(f"[{i}] {p[:500]}")
        return "\n\n".join(parts)
