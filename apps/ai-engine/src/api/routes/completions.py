"""
InsightSerenity AI Engine — Text Completions Route
===================================================
POST /v1/completions — Legacy text completion endpoint.

Accepts a raw prompt (not a conversation) and returns a continuation.
This is the "instruct" style API — provide an instruction, get the output.

Phase C hardening:
  - Context length validation before touching the GPU (400 if prompt too long)
  - Token counts written to request.state for the access-log middleware
  - Inference latency metric recorded independently of HTTP latency
  - 503 Retry-After header when engine is unavailable
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import require_internal_auth, AuthContext
from src.api.metrics import INFERENCE_LATENCY
from src.api.schemas.requests import CompletionRequest
from src.api.schemas.responses import (
    CompletionResponse, CompletionChoice, UsageInfo,
)
from src.serving.inference.inference_engine import GenerationRequest
from src.serving.streaming.sse_streamer import async_stream_tokens
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/v1/completions",
    response_model=None,
    summary="Create a text completion",
)
async def completions(
    request:     CompletionRequest,
    raw_request: Request,
    auth:        AuthContext = Depends(require_internal_auth),
):
    """
    Text completion endpoint (legacy / instruct style).
    Accepts a prompt string and returns the model's continuation.
    """
    engine = getattr(raw_request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialised",
            headers={"Retry-After": "30"},
        )

    prompt     = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    model_name = getattr(raw_request.app.state, "model_name", "insightserenity-1")

    # ── Context length validation ────────────────────────────────────────────
    max_seq = getattr(getattr(engine.model, "config", None), "max_seq_len", 2048)
    try:
        prompt_len = len(engine.tokenizer.encode(prompt, add_special_tokens=False))
    except Exception:
        prompt_len = len(prompt.split())
    if prompt_len >= max_seq:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Prompt too long: {prompt_len} tokens exceeds model limit of {max_seq}. "
                "Shorten the prompt or use a model with a larger context window."
            ),
        )

    gen_request = GenerationRequest(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        strategy="top_p" if request.temperature > 0 else "greedy",
        stop_sequences=request.stop or [],
        stream=request.stream,
    )

    # ── Streaming mode ───────────────────────────────────────────────────────
    if request.stream:
        raw_request.state.prompt_tokens = prompt_len
        token_iter = engine.stream_generate(gen_request)
        return StreamingResponse(
            async_stream_tokens(token_iter, model=model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming mode ───────────────────────────────────────────────────
    t0     = time.perf_counter()
    result = engine.generate(gen_request)
    INFERENCE_LATENCY.labels(endpoint="/v1/completions", model_version=model_name).observe(
        time.perf_counter() - t0
    )

    raw_request.state.prompt_tokens     = result.prompt_tokens
    raw_request.state.completion_tokens = result.completion_tokens

    text = (prompt + result.text) if request.echo else result.text
    return CompletionResponse(
        model=model_name,
        choices=[CompletionChoice(text=text, index=0, finish_reason=result.finish_reason)],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
