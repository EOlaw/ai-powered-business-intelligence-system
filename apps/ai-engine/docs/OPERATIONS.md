# AI Engine — Operations Guide

How to set up, train, serve, and test the AI engine in development and production.

---

## Quick Reference

| Task | Command |
|---|---|
| Start server (dev) | `uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload` |
| Start server (prod) | `uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --workers 4` |
| Run data pipeline | `python -m src.data.pipeline --seeds scripts/training/seed_urls.txt --output storage/datasets/` |
| Train (single GPU) | `python scripts/training/train.py --model gpt-small --tokenizer storage/tokenizers/bpe-32k --data storage/datasets/05_corpus.jsonl --output storage/checkpoints --run-name my-run` |
| Train (multi-GPU) | `torchrun --nproc_per_node=4 scripts/training/train.py --model gpt-large ... --distributed` |
| Resume training | `python scripts/training/train.py --resume storage/checkpoints/my-run/step_00010000 ...` |
| Promote checkpoint | `python scripts/models/promote.py --checkpoint storage/checkpoints/my-run/step_XXXXXX --name gpt-small --version v1.0.0` |
| Health check | `curl http://localhost:8001/health` |
| List models | `curl http://localhost:8001/v1/models` |
| Dev auth token | `change-me-in-production` (used as Bearer token in development only) |

---

## Environment Setup

### 1. Create a virtual environment

```bash
cd apps/ai-engine
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Create a `.env` file

**Development** — copy this exactly:

```env
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

SERVING_HOST=0.0.0.0
SERVING_PORT=8001
SERVING_INTERNAL_API_SECRET=change-me-in-production
SERVING_MAX_CONCURRENT_REQUESTS=32

MODEL_NAME=gpt-small
DEVICE=auto
```

**Production** — required changes:

```env
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

SERVING_HOST=0.0.0.0
SERVING_PORT=8001
SERVING_INTERNAL_API_SECRET=your-long-random-secret-here
SERVING_MAX_CONCURRENT_REQUESTS=32

MODEL_NAME=gpt-small
DEVICE=cuda
```

> The server will refuse to start in production if `SERVING_INTERNAL_API_SECRET` is still `change-me-in-production`.
> Generate a strong secret with: `openssl rand -hex 32`

---

## Full Workflow: First Time Setup

Follow these steps in order the first time you run the system.

### Step 1 — Prepare training data

Run the full data pipeline. It crawls, cleans, deduplicates, and quality-filters text into a JSONL corpus.

```bash
python -m src.data.pipeline \
    --seeds scripts/training/seed_urls.txt \
    --output storage/datasets/
```

If you already have data and want to skip crawling:

```bash
python -m src.data.pipeline \
    --seeds scripts/training/seed_urls.txt \
    --output storage/datasets/ \
    --skip-crawl \
    --start-from quality
```

The final dataset lands at `storage/datasets/05_corpus.jsonl`.

The pipeline is resumable. If it stops mid-way, re-run with `--start-from <stage>`:

| Stage name | What it does |
|---|---|
| `crawl` | Fetch pages from seed URLs |
| `extract` | Pull text from raw HTML |
| `clean` | Normalize and strip noise |
| `dedup` | Remove near-duplicate documents (MinHash) |
| `quality` | Filter by language, perplexity, repetition |
| `finalize` | Write final JSONL corpus |

### Step 2 — Train a model

```bash
python scripts/training/train.py \
    --model gpt-small \
    --tokenizer storage/tokenizers/bpe-32k \
    --data storage/datasets/05_corpus.jsonl \
    --output storage/checkpoints \
    --run-name pretrain-v1 \
    --epochs 3 \
    --batch-size 4 \
    --lr 3e-4
```

Available model sizes:

| Flag | Size | Use case |
|---|---|---|
| `gpt-small` | Small | Development and testing |
| `gpt-medium` | Medium | Mid-tier production |
| `gpt-large` | Large | Full production |
| `bert-base` | Medium | Embeddings only |

Checkpoints are saved to `storage/checkpoints/pretrain-v1/step_XXXXXXXX/` every 1000 steps by default.

**Multi-GPU training:**

```bash
torchrun --nproc_per_node=4 scripts/training/train.py \
    --model gpt-large \
    --tokenizer storage/tokenizers/bpe-32k \
    --data storage/datasets/05_corpus.jsonl \
    --output storage/checkpoints \
    --run-name pretrain-v1 \
    --distributed
```

**Resume a stopped training run:**

```bash
python scripts/training/train.py \
    --resume storage/checkpoints/pretrain-v1/step_00010000 \
    --model gpt-small \
    --tokenizer storage/tokenizers/bpe-32k \
    --data storage/datasets/05_corpus.jsonl \
    --output storage/checkpoints
```

### Step 3 — Promote a checkpoint to a serving model

Training saves raw checkpoints. Before the server can use a model, you promote it. This writes the final `model.pt`, `config.json`, `metadata.json`, and tokenizer into `storage/models/`.

```bash
python scripts/models/promote.py \
    --checkpoint storage/checkpoints/pretrain-v1/step_00010000 \
    --name gpt-small \
    --version v1.0.0
```

The promoted model lands at:

```
storage/models/
  gpt-small/
    v1.0.0/
      model.pt
      config.json
      metadata.json
      tokenizer/
        vocab.json
        merges.txt
        tokenizer_config.json
```

### Step 4 — Start the server

**Development:**

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

**Production:**

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --workers 4
```

The server scans `storage/models/` on startup and loads the model set in `MODEL_NAME`.

---

## Manual Tests

Run these with the server running at `http://localhost:8001`.

### Health and status (no auth required)

```bash
# Check server is alive
curl http://localhost:8001/health

# Check server is ready to serve requests
curl http://localhost:8001/readiness

# List all loaded models
curl http://localhost:8001/v1/models

# Get info on a specific model
curl http://localhost:8001/v1/models/gpt-small

# Prometheus metrics
curl http://localhost:8001/metrics
```

### Inference endpoints (Bearer token required)

In development the token is `change-me-in-production`. In production it is your `SERVING_INTERNAL_API_SECRET` value (but you will never call the engine directly — the gateway does this for you).

**Text completion:**

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Chat completion:**

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 200
  }'
```

**Streaming chat (token-by-token):**

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

**Embeddings:**

```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{
    "input": "Hello world"
  }'
```

### Admin endpoints (Bearer token required)

**Engine status:**

```bash
curl http://localhost:8001/admin/status \
  -H "Authorization: Bearer change-me-in-production"
```

**Hot-swap the active model without restarting the server:**

```bash
curl -X POST http://localhost:8001/admin/reload-model \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{"model_name": "gpt-small:v1.0.0"}'
```

**Trigger a training run via API:**

```bash
curl -X POST http://localhost:8001/admin/train \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{
    "model": "gpt-small",
    "run_name": "api-triggered-run"
  }'
```

**Trigger data preprocessing via API:**

```bash
curl -X POST http://localhost:8001/admin/preprocess \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{}'
```

---

## How API Keys Work in Production

There are two completely separate keys. Never confuse them.

### Key 1 — Customer API Key

This is the key your clients use when calling your platform (e.g. `sk-live-abc123`).

- Clients send it as `Authorization: Bearer sk-live-abc123`
- It is validated by the **Node.js API Gateway**, not the AI engine
- It lives in your database, linked to an organization
- It carries scopes that control what the key is allowed to do

### Key 2 — Internal Service Secret (`SERVING_INTERNAL_API_SECRET`)

This is a shared secret between the Node.js gateway and the AI engine. It never leaves your servers and clients never see it.

- Set on both the gateway server and the AI engine server via `.env`
- The AI engine validates this on every request at `src/api/middleware/auth.py`
- Uses `hmac.compare_digest()` for constant-time comparison (prevents timing attacks)

### Request flow in production

```
Client
  Authorization: Bearer sk-live-abc123          ← customer API key
        ↓
Node.js API Gateway
  - Looks up the key in the database
  - Validates it is active, not expired, has the right scope
  - Strips the customer key
  - Forwards to AI Engine with:
      Authorization: Bearer <SERVING_INTERNAL_API_SECRET>
      X-Org-Id: org_abc
      X-Key-Id: key_123
      X-Scopes: chat:read,completions:write
        ↓
AI Engine
  - Validates SERVING_INTERNAL_API_SECRET
  - Builds auth context with org_id, key_id, scopes
  - Runs inference
  - Returns response
        ↓
Node.js API Gateway
  - Records usage against the organization and key
  - Returns response to client
```

### Scopes

| Scope | Access |
|---|---|
| `chat:read` | Use chat completions endpoint |
| `completions:write` | Use completions endpoint |
| `admin:all` | Access all admin endpoints (train, reload, preprocess) |
| `*` | Everything — used in development only |

### Development auth bypass

In development, if `SERVING_INTERNAL_API_SECRET` is still `change-me-in-production`, the auth middleware skips token validation entirely. You can call the engine without any token. This bypass is disabled the moment you set a real secret.

---

## Storage Layout

```
storage/
  datasets/          Raw and processed training data (JSONL)
  checkpoints/       Training checkpoints (step_XXXXXXXX folders)
  models/            Promoted models ready for serving
    {name}/
      {version}/
        model.pt
        config.json
        metadata.json
        tokenizer/
  tokenizers/        Trained tokenizer artifacts
  logs/              Training and serving logs
```

---

## Key Files

| File | Purpose |
|---|---|
| `src/api/main.py` | FastAPI server entry point, startup, routing |
| `src/api/middleware/auth.py` | Bearer token validation, AuthContext building |
| `src/api/routes/chat.py` | Chat completions endpoint |
| `src/api/routes/completions.py` | Text completions endpoint |
| `src/api/routes/embeddings.py` | Embeddings endpoint |
| `src/api/routes/admin.py` | Admin endpoints (train, reload, preprocess) |
| `src/serving/registry/model_registry.py` | Model discovery, lazy loading, LRU cache |
| `src/serving/inference/inference_engine.py` | Forward pass, sampling, streaming |
| `src/config/settings.py` | All configuration and environment variables |
| `scripts/training/train.py` | Training CLI entry point |
| `scripts/models/promote.py` | Promote checkpoint to serving model |
| `src/data/pipeline.py` | Data pipeline CLI entry point |
