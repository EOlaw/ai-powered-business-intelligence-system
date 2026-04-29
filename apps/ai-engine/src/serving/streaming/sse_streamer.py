"""
InsightSerenity AI Engine — SSE Token Streamer
===============================================
Server-Sent Events (SSE) streaming for token-by-token response delivery.

Instead of waiting for the entire response before sending it to the client,
SSE streams each token as it is generated. This dramatically improves the
perceived latency of the API — users see the first word in milliseconds
rather than waiting for the full response.

SSE wire format (MIME: text/event-stream):
    data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}\n\n
    data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}\n\n
    data: [DONE]\n\n

This format is compatible with the OpenAI Chat Completions streaming API,
so clients built for OpenAI can use our API with zero changes.

FastAPI integration:
    Each route that supports streaming returns a StreamingResponse with
    content_type="text/event-stream" and yields SSE-formatted data frames.
"""

import json
import time
import uuid
from typing import AsyncIterator, Iterator

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_completion_id() -> str:
    """Generate a unique completion ID (OpenAI compatible format)."""
    return f"chatcmpl-{uuid.uuid4().hex[:16]}"


def format_sse_chunk(
    token:         str,
    completion_id: str,
    model:         str = "insightserenity-1",
    role:          str = "assistant",
) -> str:
    """
    Format a single token as an SSE data frame.

    Compatible with OpenAI's streaming format so existing clients work.

    Args:
        token:         The decoded token string to send.
        completion_id: Unique ID for this completion request.
        model:         Model name to include in the response.
        role:          "assistant" for the first chunk, "" for subsequent.

    Returns:
        SSE-formatted string ready to send over HTTP.
    """
    chunk = {
        "id":      completion_id,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    **({"role": role} if role else {}),
                    "content": token,
                },
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def format_sse_final(
    completion_id: str,
    finish_reason: str = "stop",
    model:         str = "insightserenity-1",
) -> str:
    """
    Format the final SSE frame that signals stream completion.

    Args:
        completion_id: Unique ID for this completion.
        finish_reason: "stop" or "length".
        model:         Model name.

    Returns:
        SSE final chunk string + "[DONE]" sentinel.
    """
    chunk = {
        "id":      completion_id,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"


def stream_tokens(
    token_iterator: Iterator[str],
    model:          str = "insightserenity-1",
) -> Iterator[str]:
    """
    Wrap a token iterator in SSE formatting.

    Takes a plain Iterator[str] of tokens (from InferenceEngine.stream_generate)
    and yields SSE-formatted data frames ready for HTTP streaming.

    Args:
        token_iterator: Iterator yielding decoded token strings.
        model:          Model name for response metadata.

    Yields:
        SSE data frame strings.
    """
    completion_id = _make_completion_id()
    is_first      = True
    n_tokens      = 0

    for token in token_iterator:
        role = "assistant" if is_first else ""
        yield format_sse_chunk(token, completion_id, model=model, role=role)
        is_first = False
        n_tokens += 1

    yield format_sse_final(completion_id, model=model)
    logger.debug("SSE stream complete", tokens=n_tokens, id=completion_id)


async def async_stream_tokens(
    token_iterator: Iterator[str],
    model:          str = "insightserenity-1",
) -> AsyncIterator[str]:
    """
    Async version of stream_tokens for use with FastAPI StreamingResponse.

    Wraps the synchronous token iterator in async yields so FastAPI can
    stream the response without blocking the event loop.

    Usage in FastAPI:
        @router.post("/v1/chat/completions")
        async def chat(request: ChatRequest):
            tokens = engine.stream_generate(gen_request)
            return StreamingResponse(
                async_stream_tokens(tokens),
                media_type="text/event-stream",
            )
    """
    completion_id = _make_completion_id()
    is_first      = True

    for token in token_iterator:
        role = "assistant" if is_first else ""
        yield format_sse_chunk(token, completion_id, model=model, role=role)
        is_first = False

    yield format_sse_final(completion_id, model=model)
