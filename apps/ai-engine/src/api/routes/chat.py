"""
InsightSerenity AI Engine — Chat Completions Route
===================================================
POST /v1/chat/completions — The primary generation endpoint.

Accepts a conversation history and returns the next assistant message,
optionally as a streaming SSE response.

Compatible with the OpenAI Chat Completions API format.

Phase C hardening:
  - Context length validation before GPU allocation
  - Token counts written to request.state
  - Inference latency metric
  - 503 Retry-After header when engine unavailable
  - Streaming timeout: asyncio.timeout wraps the generator (5-minute hard limit)
"""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import require_internal_auth, AuthContext
from src.api.metrics import INFERENCE_LATENCY
from src.api.schemas.requests import ChatCompletionRequest, ChatMessage
from src.api.schemas.responses import (
    ChatCompletionResponse, ChatChoice,
    ChatMessage as RespChatMessage, UsageInfo,
)
from src.serving.inference.inference_engine import GenerationRequest
from src.serving.streaming.sse_streamer import async_stream_tokens
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _build_prompt_from_messages(
    messages:          list,
    system_override:   str = None,
) -> str:
    """
    Convert a list of ChatMessage objects into a single prompt string
    using the ChatML format matching our model's fine-tuning template.
    """
    from src.tokenizer.special_tokens import SpecialTokens as ST

    parts = []
    for msg in messages:
        role    = msg.role
        content = msg.content.strip()

        if role == "system":
            parts.append(f"{ST.SYSTEM}\n{content}\n")
        elif role == "user":
            parts.append(f"{ST.USER}\n{content}\n")
        elif role == "assistant":
            parts.append(f"{ST.ASSISTANT}\n{content}\n{ST.END_TURN}\n")
        elif role == "tool":
            parts.append(f"Tool result: {content}\n")

    # Add system override if provided and no system message in history
    has_system = any(m.role == "system" for m in messages)
    if system_override and not has_system:
        parts.insert(0, f"{ST.SYSTEM}\n{system_override}\n")

    # End with the assistant delimiter to trigger generation
    if not messages or messages[-1].role != "assistant":
        parts.append(f"{ST.ASSISTANT}\n")

    return "".join(parts)


@router.post(
    "/v1/chat/completions",
    response_model=None,
    summary="Create a chat completion",
    description="Generate the next assistant message in a conversation.",
)
async def chat_completions(
    request:     ChatCompletionRequest,
    raw_request: Request,
    auth:        AuthContext = Depends(require_internal_auth),
):
    """
    Chat completions endpoint.

    Supports both streaming (SSE) and non-streaming responses.
    The `stream` field in the request body controls the mode.
    """
    app_state = raw_request.app.state
    engine    = getattr(app_state, "engine", None)

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialised",
            headers={"Retry-After": "30"},
        )

    prompt     = _build_prompt_from_messages(request.messages, system_override=request.is_system_override)
    model_name = getattr(app_state, "model_name", "insightserenity-1")

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
                f"Conversation history too long: {prompt_len} tokens exceeds model limit of "
                f"{max_seq}. Remove earlier messages or summarise the context."
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

    # ── Streaming mode ─────────────────────────────────────────────────────
    if request.stream:
        raw_request.state.prompt_tokens = prompt_len

        async def _timed_stream():
            """Wrap the token iterator with a 5-minute hard timeout."""
            try:
                async with asyncio.timeout(300):
                    async for chunk in async_stream_tokens(engine.stream_generate(gen_request), model=model_name):
                        yield chunk
            except asyncio.TimeoutError:
                # Emit a final [DONE] so the client's SSE parser closes cleanly
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            _timed_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming mode ─────────────────────────────────────────────────
    t0     = time.perf_counter()
    result = engine.generate(gen_request)
    INFERENCE_LATENCY.labels(endpoint="/v1/chat/completions", model_version=model_name).observe(
        time.perf_counter() - t0
    )

    raw_request.state.prompt_tokens     = result.prompt_tokens
    raw_request.state.completion_tokens = result.completion_tokens

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=RespChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
