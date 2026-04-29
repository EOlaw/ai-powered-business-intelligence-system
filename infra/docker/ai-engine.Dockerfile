# InsightSerenity AI Engine — Production Dockerfile
# ===================================================
# Multi-stage build:
#   builder — installs Python deps (including compiled torch extensions)
#   runtime — minimal image with only what's needed to run
#
# GPU support: build with --build-arg CUDA=true to include CUDA drivers.
# CPU-only:    default build — suitable for staging/testing environments.
#
# Usage:
#   docker build -f infra/docker/ai-engine.Dockerfile -t insightserenity/ai-engine:latest .
#   docker run --gpus all -p 8001:8001 --env-file .env insightserenity/ai-engine:latest

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install system libraries needed to build Python packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first — Docker cache layer: only rebuilds when deps change
COPY apps/ai-engine/requirements.txt .

# Install dependencies to a local prefix for clean copy to runtime stage
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1 is required by PyTorch at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r aiengine && useradd -r -g aiengine aiengine

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY apps/ai-engine/src         ./src
COPY apps/ai-engine/pyproject.toml .

# Create storage directories with correct ownership
RUN mkdir -p storage/models storage/tokenizers storage/datasets storage/logs \
 && chown -R aiengine:aiengine /app

USER aiengine

# Environment defaults (override via .env or K8s secrets)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SERVING_HOST=0.0.0.0 \
    SERVING_PORT=8001 \
    ENVIRONMENT=production

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--log-level", "info"]
