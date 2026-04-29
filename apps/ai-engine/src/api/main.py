"""
InsightSerenity AI Engine — FastAPI Application
================================================
This is the entry point for the AI engine's HTTP server.

Startup sequence:
    1. FastAPI lifespan function is called (before any requests are served)
    2. ModelRegistry scans storage/models/ for available model directories
    3. The default model (MODEL_NAME env var or first discovered) is loaded
    4. InferenceEngine is instantiated with the loaded model + tokenizer
    5. TextGenerator (thin wrapper over InferenceEngine) is attached to app.state
    6. Server is ready to accept requests

Application state (available in all routes via `raw_request.app.state`):
    app.state.registry       — ModelRegistry (loaded models, LRU cache)
    app.state.engine         — InferenceEngine (forward-pass executor)
    app.state.generator      — TextGenerator (high-level generation API)
    app.state.model_name     — Active model name string
    app.state.tool_registry  — ToolRegistry (populated lazily on first agent call)
    app.state.started_at     — Unix timestamp of startup (for uptime calculation)

Internal auth:
    Every non-health endpoint requires "Authorization: Bearer <SERVING_INTERNAL_API_SECRET>".
    This is the shared secret between the Node.js API gateway and this engine.
    It is NEVER exposed to external clients.

Ports:
    Default: 8001 (set via SERVING_PORT env var or serving.port in settings)
    The Node.js gateway runs on 3000 and proxies AI requests to 8001.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from src.api.routes import chat, completions, embeddings, agents, admin
from src.api.schemas.responses import HealthResponse, ModelsListResponse, ErrorResponse, ErrorDetail
from src.api.middleware.request_context import RequestContextMiddleware
from src.api.metrics import record_active_model
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    app.state.started_at = time.time()

    logger.info(
        "AI Engine starting",
        version=settings.version,
        environment=settings.environment.value,
        device=str(settings.torch_device),
    )

    # ── Model Registry ─────────────────────────────────────────────────────────
    from src.serving.registry.model_registry import ModelRegistry
    registry = ModelRegistry(
        models_dir=str(settings.storage.models),
        device=str(settings.torch_device),
    )
    app.state.registry = registry

    # ── Load the default model ─────────────────────────────────────────────────
    import os
    default_model = os.getenv("MODEL_NAME", "insightserenity-1")
    available     = registry.list_models()

    model      = None
    tokenizer  = None
    model_name = default_model

    if available:
        # Prefer the exact requested name, fall back to first available
        candidate = default_model if default_model in available else available[0]
        try:
            model, tokenizer = registry.get_model(candidate)
            model_name       = candidate
            logger.info("Default model loaded", model=model_name)
        except Exception as exc:
            logger.error("Failed to load default model", model=candidate, error=str(exc))
    else:
        # No trained models on disk yet — create a lightweight stub so the server
        # can still respond to health checks and return sensible 503 errors on
        # generation requests rather than crashing at import time.
        logger.warning(
            "No models found in storage/models/. "
            "Engine will start but generation will fail until a model is trained and saved."
        )

    # ── InferenceEngine ────────────────────────────────────────────────────────
    app.state.model_name = model_name

    if model is not None and tokenizer is not None:
        from src.serving.inference.inference_engine import InferenceEngine
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device=str(settings.torch_device),
            use_amp=settings.training.use_amp,
        )
        app.state.engine = engine

        # ── TextGenerator (agent-facing wrapper) ──────────────────────────────
        from src.llm.inference.generator import TextGenerator
        try:
            generator = TextGenerator(
                model=model,
                tokenizer=tokenizer,
                device=str(settings.torch_device),
            )
            app.state.generator = generator
        except Exception as exc:
            logger.warning("TextGenerator init failed — agents unavailable", error=str(exc))
            app.state.generator = None
    else:
        app.state.engine    = None
        app.state.generator = None

    # ── Tool registry (lazy — populated on first agent request) ───────────────
    app.state.tool_registry = None

    # ── Concurrency semaphore ─────────────────────────────────────────────────
    max_concurrent = getattr(settings.serving, "max_concurrent_requests", 8)
    app.state.concurrency_semaphore = asyncio.Semaphore(max_concurrent)

    # ── Publish model version to Prometheus gauge ──────────────────────────
    if model is not None:
        registry_info = registry.get_info(model_name)
        _param_count  = getattr(registry_info, "param_count", 0) or sum(
            p.numel() for p in model.parameters()
        )
        record_active_model(
            model_name=model_name.split(":")[0],
            version=model_name.split(":")[-1] if ":" in model_name else "unknown",
            promoted_at_unix=getattr(registry_info, "promoted_at_unix", 0.0) or 0.0,
            param_count=_param_count,
        )

    logger.info("AI Engine ready", model=model_name, port=settings.serving.port)

    yield   # ← server is live and handling requests here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("AI Engine shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Returns a fully configured app instance ready for Uvicorn.
    This factory pattern makes the app importable for testing without
    triggering the lifespan startup sequence.
    """
    app = FastAPI(
        title="InsightSerenity AI Engine",
        description=(
            "Internal inference API for the InsightSerenity platform. "
            "Provides text generation, chat completions, embeddings, and agent execution. "
            "This endpoint is NOT public — requests must include the internal service token."
        ),
        version=settings.version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Request context middleware (outermost — runs on every request) ───────
    # Must be added BEFORE CORSMiddleware so X-Request-ID is always set.
    app.add_middleware(RequestContextMiddleware)

    # ── CORS ──────────────────────────────────────────────────────────────────
    allowed_origins = (
        ["*"] if settings.is_development
        else ["http://localhost:3000", "http://api-gateway:3000"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["POST", "GET"],
        allow_headers=["Authorization", "Content-Type",
                       "X-Org-Id", "X-Key-Id", "X-Scopes", "X-Request-ID"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch unhandled exceptions and return a structured error response."""
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="An unexpected error occurred",
                    type="internal_error",
                )
            ).model_dump(),
        )

    # ── Routes — Core inference ───────────────────────────────────────────────
    app.include_router(completions.router, tags=["Completions"])
    app.include_router(chat.router,        tags=["Chat"])
    app.include_router(embeddings.router,  tags=["Embeddings"])
    app.include_router(agents.router,      tags=["Agents"])

    # ── Admin routes — internal worker pipeline only ──────────────────────
    app.include_router(admin.router,       tags=["Admin"])

    # ── Health & utility ─────────────────────────────────────────────────────
    _register_utility_routes(app)

    return app


def _register_utility_routes(app: FastAPI) -> None:
    """Register health, readiness, metrics, and model-listing endpoints."""

    @app.get(
        "/metrics",
        tags=["Observability"],
        summary="Prometheus metrics",
        include_in_schema=False,   # Don't expose in OpenAPI docs
    )
    async def prometheus_metrics():
        """
        Exposes all Prometheus metrics in the text exposition format.
        Scraped by Prometheus every 15 seconds. Not auth-protected — only
        reachable from within the private network.
        """
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Server health check",
    )
    async def health(request: Request):
        """
        Liveness probe — confirms the server process is running.
        Does NOT verify model availability; use /readiness for that.
        """
        uptime = time.time() - getattr(request.app.state, "started_at", time.time())
        return HealthResponse(
            status="ok",
            model=getattr(request.app.state, "model_name", None),
            version=settings.version,
            uptime=round(uptime, 1),
        )

    @app.get(
        "/readiness",
        tags=["Health"],
        summary="Readiness probe — confirms model is loaded",
    )
    async def readiness(request: Request):
        """
        Readiness probe for load-balancer/orchestration use.
        Returns 200 only when the inference engine is loaded and ready.
        Returns 503 if the engine is still loading or failed to load.
        """
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "inference engine not loaded"},
            )
        return {"status": "ready", "model": getattr(request.app.state, "model_name", None)}

    @app.get(
        "/v1/models",
        response_model=None,
        tags=["Models"],
        summary="List available models",
    )
    async def list_models(request: Request):
        """
        Returns all models registered in the model registry.
        Compatible with the OpenAI GET /v1/models response format.
        """
        registry = getattr(request.app.state, "registry", None)
        if registry is None:
            return ModelsListResponse(data=[])

        now    = int(time.time())
        models = []
        for key in registry.list_models():
            info = registry.get_info(key)
            models.append({
                "id":       key,
                "object":   "model",
                "created":  int(info.loaded_at) if info and info.loaded_at else now,
                "owned_by": "insightserenity",
                "capabilities": info.capabilities if info else [],
            })

        return ModelsListResponse(data=models)

    @app.get(
        "/v1/models/{model_id:path}",
        response_model=None,
        tags=["Models"],
        summary="Retrieve a specific model",
    )
    async def get_model(model_id: str, request: Request):
        """
        Returns metadata for a single model by its ID.
        Returns 404 if the model is not found in the registry.
        """
        registry = getattr(request.app.state, "registry", None)
        if registry is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error=ErrorDetail(message=f"Model '{model_id}' not found", type="not_found")
                ).model_dump(),
            )

        info = registry.get_info(model_id)
        if info is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error=ErrorDetail(message=f"Model '{model_id}' not found", type="not_found")
                ).model_dump(),
            )

        return {
            "id":              f"{info.name}:{info.version}",
            "object":          "model",
            "created":         int(info.loaded_at) if info.loaded_at else int(time.time()),
            "owned_by":        "insightserenity",
            "capabilities":    info.capabilities,
            # Architecture
            "vocab_size":      info.vocab_size,
            "d_model":         info.d_model,
            "context_length":  info.context_length,
            "param_count":     info.param_count,
            # Provenance — from metadata.json
            "perplexity":      info.perplexity,
            "val_loss":        info.val_loss,
            "training_run":    info.training_run,
            "promoted_at":     info.promoted_at,
            "checksum_sha256": info.checksum_sha256,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Application instance (used by Uvicorn and tests)
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.serving.host,
        port=settings.serving.port,
        reload=settings.is_development,
        log_level=settings.log_level.value.lower(),
        workers=1,       # Single worker — the engine holds GPU state in memory
        access_log=settings.is_development,
    )
