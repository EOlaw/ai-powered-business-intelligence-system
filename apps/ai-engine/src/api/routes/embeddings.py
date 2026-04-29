"""
InsightSerenity AI Engine — Embeddings Route
=============================================
POST /v1/embeddings — Dense vector embedding endpoint.

Accepts one or more text strings and returns fixed-length floating-point
vectors produced by mean-pooling the encoder's final hidden states.
The output format is OpenAI-compatible; existing clients that call
openai.embeddings.create() work with zero changes.

Dimensionality truncation:
    If the caller passes `dimensions=N`, embeddings are sliced to the first
    N components.  The caller is responsible for re-normalising if needed;
    we do NOT re-normalise after truncation because many downstream uses
    (e.g. FAISS cosine search) normalise themselves.

Encoding format:
    "float"  → List[float] per embedding (default)
    "base64" → base64-encoded IEEE 754 little-endian bytes
               (matches OpenAI's behaviour for bandwidth-sensitive clients)
"""

import base64
import struct
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.middleware.auth import require_internal_auth, AuthContext
from src.api.schemas.requests import EmbeddingRequest
from src.api.schemas.responses import EmbeddingObject, EmbeddingResponse, UsageInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _floats_to_base64(values: List[float]) -> str:
    """
    Encode a list of float32 values as a base64 string.

    Uses little-endian IEEE 754 single precision (4 bytes each), matching
    the format OpenAI uses for its base64 embedding encoding.
    """
    raw = struct.pack(f"<{len(values)}f", *values)
    return base64.b64encode(raw).decode("ascii")


@router.post(
    "/v1/embeddings",
    response_model=None,
    summary="Create text embeddings",
)
async def create_embeddings(
    request:     EmbeddingRequest,
    raw_request: Request,
    auth:        AuthContext = Depends(require_internal_auth),
):
    """
    Embedding endpoint — returns dense vector representations of input text.

    The model encodes each input independently (no cross-input attention).
    Embeddings are L2-normalised before return so cosine similarity equals
    dot product, enabling efficient FAISS IndexFlatIP searches.
    """
    engine = getattr(raw_request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialised",
            headers={"Retry-After": "30"},
        )

    texts      = request.get_texts()
    model_name = getattr(raw_request.app.state, "model_name", "insightserenity-1")

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="input must contain at least one non-empty string",
        )

    # ── Build embedding request and call engine ────────────────────────────────
    from src.serving.inference.inference_engine import EmbeddingRequest as EngineEmbedRequest

    engine_req = EngineEmbedRequest(texts=texts)
    try:
        result = engine.embed(engine_req)
    except Exception as exc:
        logger.error("Embedding inference failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {exc}",
        )

    # ── Format output objects ─────────────────────────────────────────────────
    embedding_objects: List[EmbeddingObject] = []
    for idx, vector in enumerate(result.embeddings):
        if request.dimensions is not None:
            vector = vector[: request.dimensions]
        encoded: object = _floats_to_base64(vector) if request.encoding_format == "base64" else vector
        embedding_objects.append(EmbeddingObject(embedding=encoded, index=idx))

    # ── Exact token count from inference result ───────────────────────────────
    total_prompt_tokens = result.total_prompt_tokens or len(texts)

    # Write to request state so access-log middleware can record it
    raw_request.state.prompt_tokens = total_prompt_tokens

    return EmbeddingResponse(
        data=embedding_objects,
        model=model_name,
        usage=UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=0,
            total_tokens=total_prompt_tokens,
        ),
    )
