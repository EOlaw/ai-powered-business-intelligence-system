"""
InsightSerenity AI Engine — Request Context Middleware
=======================================================
Single Starlette middleware that handles every production concern that
belongs at the HTTP layer rather than inside individual route handlers:

1. Concurrency gate
   Checks `app.state.concurrency_semaphore` before processing.
   Returns 503 with `Retry-After: 30` if all slots are occupied.

2. Correlation ID
   Reads `X-Request-ID` from the incoming request or generates a new one.
   Attaches it to `request.state.request_id` and echoes it in the response.

3. Response timing
   Records `request.state.start_time` before dispatch; computes latency
   after the response is fully sent.

4. X-Model-Version header
   Reads `app.state.model_name` and sets it on every response so clients
   can always see which model version served them.

5. Prometheus counters
   Increments `REQUEST_COUNT` and observes `REQUEST_LATENCY` after each
   request. Uses `request.state.status_code` written by the route handler,
   or falls back to `response.status_code`.

6. Structured access log
   Appends one JSON line to `storage/logs/access.jsonl` after every request.
   Token counts are read from `request.state.prompt_tokens` and
   `request.state.completion_tokens` — set by route handlers before returning.

Token count convention (used by routes to report usage to the middleware):
    request.state.prompt_tokens     = N   (set after generation)
    request.state.completion_tokens = N   (set after generation)
"""

import json
import time
import uuid
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from src.api.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_REQUESTS, TOKENS_TOTAL,
)
from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger("request-context")

# Access log path — created at first write
_ACCESS_LOG = Path(settings.storage.logs) / "access.jsonl"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Attaches request-scoped state, enforces concurrency limits,
    sets standard response headers, records metrics and access logs.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:

        # ── Skip admin and health routes (no metrics / concurrency needed) ───
        path = request.url.path
        is_internal = path.startswith("/admin") or path in ("/health", "/readiness", "/metrics")

        # ── Correlation ID ───────────────────────────────────────────────────
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id      = req_id
        request.state.start_time      = time.perf_counter()
        request.state.prompt_tokens   = 0
        request.state.completion_tokens = 0

        # ── Concurrency gate ─────────────────────────────────────────────────
        if not is_internal:
            sem = getattr(request.app.state, "concurrency_semaphore", None)
            if sem is not None and sem.locked():
                # All slots occupied — reject immediately
                max_concurrent = getattr(settings.serving, "max_concurrent_requests", 8)
                log.warning("Concurrency limit reached", path=path, limit=max_concurrent)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "code":    "CAPACITY_EXCEEDED",
                            "message": (
                                f"Server is at capacity ({max_concurrent} concurrent requests). "
                                "Retry after 30 seconds."
                            ),
                        }
                    },
                    headers={
                        "Retry-After":    "30",
                        "X-Request-ID":   req_id,
                    },
                )

        # ── Track active requests ────────────────────────────────────────────
        if not is_internal:
            ACTIVE_REQUESTS.inc()

        # ── Dispatch to route handler ─────────────────────────────────────────
        try:
            response = await call_next(request)
        except Exception as exc:
            log.error("Unhandled exception in route", path=path, error=str(exc), exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code":    "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                        "request_id": req_id,
                    }
                },
            )
        finally:
            if not is_internal:
                ACTIVE_REQUESTS.dec()

        # ── Attach standard headers to every response ────────────────────────
        response.headers["X-Request-ID"] = req_id

        model_version = getattr(request.app.state, "model_name", "")
        if model_version:
            response.headers["X-Model-Version"] = model_version

        # ── Record metrics ───────────────────────────────────────────────────
        if not is_internal:
            latency      = time.perf_counter() - request.state.start_time
            status_code  = str(response.status_code)
            method       = request.method
            endpoint     = _normalise_path(path)

            REQUEST_COUNT.labels(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                model_version=model_version,
            ).inc()

            REQUEST_LATENCY.labels(
                endpoint=endpoint,
                method=method,
                model_version=model_version,
            ).observe(latency)

            pt = request.state.prompt_tokens
            ct = request.state.completion_tokens
            if pt > 0:
                TOKENS_TOTAL.labels(type="prompt",     model_version=model_version).inc(pt)
            if ct > 0:
                TOKENS_TOTAL.labels(type="completion", model_version=model_version).inc(ct)

            # ── Structured access log ─────────────────────────────────────────
            _write_access_log({
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id":        req_id,
                "method":            method,
                "endpoint":          path,
                "status_code":       response.status_code,
                "latency_ms":        round(latency * 1000, 1),
                "prompt_tokens":     pt,
                "completion_tokens": ct,
                "total_tokens":      pt + ct,
                "model_version":     model_version,
                "org_id":            request.headers.get("x-org-id", ""),
                "key_id":            request.headers.get("x-key-id", ""),
            })

        return response


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_path(path: str) -> str:
    """
    Collapse dynamic path segments to prevent cardinality explosion.
    e.g. /v1/models/insightserenity-1:v0.0.1 → /v1/models/{model_id}
    """
    import re
    path = re.sub(r"/v1/models/[^/]+", "/v1/models/{model_id}", path)
    return path


def _write_access_log(entry: dict) -> None:
    """Append one JSON line to the access log. Creates the file if needed."""
    try:
        _ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _ACCESS_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Failed to write access log", error=str(exc))
