"""
InsightSerenity AI Engine — Prometheus Metrics
===============================================
All metrics are defined once here and imported wherever they are updated.
Prometheus auto-discovers them via the default CollectorRegistry.

Metric naming follows the Prometheus convention:
    <namespace>_<subsystem>_<name>_<unit>

Namespace: insightserenity
Subsystem: api | inference

Labels shared across metrics:
    endpoint      — request path (/v1/chat/completions, /v1/completions, …)
    model_version — active model name:version string

Usage
-----
    from src.api.metrics import (
        REQUEST_COUNT, REQUEST_LATENCY, INFERENCE_LATENCY,
        TOKENS_TOTAL, ACTIVE_REQUESTS
    )
    REQUEST_COUNT.labels(endpoint="/v1/chat/completions",
                         method="POST",
                         status_code="200",
                         model_version="insightserenity-1:v0.1.0").inc()
"""

from prometheus_client import Counter, Histogram, Gauge, REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
# HTTP layer metrics
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "insightserenity_api_requests_total",
    "Total HTTP requests to the AI engine API",
    labelnames=["endpoint", "method", "status_code", "model_version"],
)

REQUEST_LATENCY = Histogram(
    "insightserenity_api_request_duration_seconds",
    "End-to-end HTTP request latency including model inference",
    labelnames=["endpoint", "method", "model_version"],
    # Fine-grained below 1s (typical range), coarser above
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

ACTIVE_REQUESTS = Gauge(
    "insightserenity_api_active_requests",
    "Number of requests currently being processed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Inference metrics
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_LATENCY = Histogram(
    "insightserenity_inference_duration_seconds",
    "Time spent in the model forward pass (excludes tokenisation and I/O)",
    labelnames=["endpoint", "model_version"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

TOKENS_TOTAL = Counter(
    "insightserenity_tokens_total",
    "Total tokens processed, split by type",
    labelnames=["type", "model_version"],   # type: prompt | completion
)

# ─────────────────────────────────────────────────────────────────────────────
# Model version tracking
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_MODEL_PROMOTED_AT = Gauge(
    "insightserenity_active_model_promoted_at_unix",
    "Unix timestamp of when the currently-serving model was promoted to production. "
    "Used by the stale-model alert: fires if time() - this_value > 30 days.",
    labelnames=["model_name", "model_version"],
)

ACTIVE_MODEL_PARAM_COUNT = Gauge(
    "insightserenity_active_model_param_count",
    "Number of parameters in the currently-serving model",
    labelnames=["model_name", "model_version"],
)


def record_active_model(model_name: str, version: str,
                        promoted_at_unix: float, param_count: int) -> None:
    """
    Update model-tracking gauges after a load or hot-swap.
    Called from main.py on startup and from admin/reload-model on hot-swap.
    """
    ACTIVE_MODEL_PROMOTED_AT.labels(
        model_name=model_name, model_version=version
    ).set(promoted_at_unix)

    if param_count > 0:
        ACTIVE_MODEL_PARAM_COUNT.labels(
            model_name=model_name, model_version=version
        ).set(param_count)
