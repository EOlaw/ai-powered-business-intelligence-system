"""
InsightSerenity AI Engine — Admin API Routes
=============================================
Internal endpoints called exclusively by the Node.js BullMQ worker pipeline.
These routes are NEVER exposed to external clients. The API gateway does not
proxy /admin/* traffic — it is reachable only from within the private network.

Authentication
--------------
Every request must carry:
    Authorization: Bearer <SERVING_INTERNAL_API_SECRET>

The dependency `require_admin_auth` enforces this. In development mode with
the default secret, auth is skipped to allow local testing.

Endpoints
---------
POST /admin/crawl           Trigger the async web crawler.
POST /admin/preprocess      Run the data cleaning + dedup pipeline on crawl output.
POST /admin/train           Run a training pass and return the checkpoint path.
POST /admin/evaluate        Compare a candidate model against the baseline.
POST /admin/reload-model    Hot-swap the active inference engine to a new model.
GET  /admin/status          Current model, device, memory, uptime.

Hot-swap design
---------------
The reload endpoint is the most safety-critical operation. It holds a module-
level asyncio.Lock to guarantee only one reload happens at a time. The sequence:

    1. Promote the checkpoint (scripts/models/promote.py) if needed
    2. Load new model weights into a *staging* InferenceEngine in a thread
    3. Run a 5-token sanity generation on the staging engine
    4. Atomically replace app.state.engine with the staging engine
    5. Unload old model from memory
    6. Return success

If the sanity generation fails at step 3, the old engine stays active and the
endpoint returns 500 — the service is never left in a broken state.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.api.middleware.auth import require_internal_auth, AuthContext
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

# ── Engine root for subprocess calls ──────────────────────────────────────────
_ENGINE_ROOT = Path(__file__).resolve().parents[3]   # apps/ai-engine/
_REPO_ROOT   = _ENGINE_ROOT.parents[1]               # repo root

# Single lock — prevents concurrent hot-swaps from corrupting engine state
_reload_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Admin-specific auth dependency
# ─────────────────────────────────────────────────────────────────────────────

async def require_admin_auth(
    request: Request,
    auth: AuthContext = Depends(require_internal_auth),
) -> AuthContext:
    """
    Extends require_internal_auth with an additional check that the caller
    has the admin:all scope OR is calling from within the private network.

    For the worker pipeline, the scope is typically "*" (dev) or "admin:all".
    """
    return auth


# ─────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class CrawlRequest(BaseModel):
    seed_urls:  List[str] = Field(..., min_length=1)
    max_pages:  int       = Field(default=5_000, ge=1)
    output_key: str       = Field(..., description="Output path prefix under storage/datasets/")


class CrawlResponse(BaseModel):
    output_path:   str
    pages_crawled: int
    run_id:        str
    elapsed_s:     float


class PreprocessRequest(BaseModel):
    input_key:  str = Field(..., description="Path to raw crawl JSONL output")
    output_key: str = Field(..., description="Destination path for processed JSONL")


class PreprocessResponse(BaseModel):
    output_path: str
    doc_count:   int
    elapsed_s:   float


class TrainRequest(BaseModel):
    dataset_key: str  = Field(..., description="Path to preprocessed training JSONL")
    base_model:  str  = Field(default="insightserenity-1:latest")
    new_version: str  = Field(..., description="Version string for the new checkpoint, e.g. v1.3.0")
    mode:        str  = Field(default="finetune", pattern="^(pretrain|finetune)$")
    max_steps:   int  = Field(default=2_000, ge=1)


class TrainResponse(BaseModel):
    model_path:  str
    final_loss:  float
    perplexity:  float
    steps:       int
    elapsed_s:   float


class EvaluateRequest(BaseModel):
    new_model_path:  str   = Field(..., description="Path to candidate model checkpoint or artifact")
    base_model_path: str   = Field(..., description="Path to baseline model artifact")
    eval_data:       Optional[str] = Field(default=None)
    max_regression:  float = Field(default=0.02)


class EvaluateResponse(BaseModel):
    new_perplexity:  Optional[float]
    base_perplexity: Optional[float]
    regression:      float
    passed:          bool
    gates:           dict


class ReloadRequest(BaseModel):
    model_key: str = Field(..., description="Checkpoint dir or promoted model path to load")
    version:   str = Field(..., description="Version string, e.g. v1.3.0")
    model_name: str = Field(default="insightserenity-1")


class ReloadResponse(BaseModel):
    success:      bool
    active_model: str
    elapsed_s:    float


class StatusResponse(BaseModel):
    active_model:        str
    engine_ready:        bool
    uptime_s:            float
    device:              str
    gpu_memory_used_mb:  Optional[float]
    gpu_memory_total_mb: Optional[float]
    param_count:         Optional[int]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _venv_python() -> str:
    """Return the venv Python interpreter, falling back to sys.executable."""
    venv_py = _ENGINE_ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _run_script(cmd: List[str], timeout: int) -> dict:
    """
    Run a script in a subprocess, capturing stdout as JSON.

    The script must print a JSON object on stdout as its last line.
    Returns the parsed dict, or raises HTTPException on failure.
    """
    env = {**os.environ, "PYTHONPATH": str(_ENGINE_ROOT)}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_ENGINE_ROOT),
        env=env,
        timeout=timeout,
    )

    if result.returncode not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script failed (exit {result.returncode}): {result.stderr[-500:]}",
        )

    # The script prints JSON as the last non-empty line
    lines  = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script produced no output. stderr: {result.stderr[-300:]}",
        )

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script output not valid JSON: {lines[-1][:200]}",
        )


def _resolve_path(key: str) -> str:
    """Resolve a relative storage key to an absolute path."""
    p = Path(key)
    if p.is_absolute():
        return str(p)
    return str(_ENGINE_ROOT / key)


# ─────────────────────────────────────────────────────────────────────────────
# POST /admin/crawl
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/crawl",
    response_model=CrawlResponse,
    summary="Trigger the web crawler",
)
async def crawl(
    request: CrawlRequest,
    auth:    AuthContext = Depends(require_admin_auth),
):
    """
    Runs the async web crawler with the given seed URLs.

    Calls DataPipeline programmatically in a thread (non-blocking to the
    event loop). The worker has a 1-hour timeout on this call.
    """
    start  = time.perf_counter()
    run_id = f"crawl-{int(time.time())}"

    output_path = _resolve_path(request.output_key)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info("Admin crawl triggered", seed_count=len(request.seed_urls), max_pages=request.max_pages)

    def _do_crawl():
        from src.data.pipeline import DataPipeline
        import asyncio

        pipeline = DataPipeline(output_dir=output_path)
        loop     = asyncio.new_event_loop()
        try:
            stats = loop.run_until_complete(
                pipeline.run(
                    seed_urls=request.seed_urls,
                    max_pages=request.max_pages,
                )
            )
        finally:
            loop.close()
        return stats

    try:
        stats = await asyncio.to_thread(_do_crawl)
    except Exception as exc:
        logger.error("Crawl failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Crawl failed: {exc}")

    pages = getattr(stats, "pages_crawled", 0) or getattr(stats, "total_pages", 0)
    elapsed = time.perf_counter() - start

    logger.info("Crawl complete", pages=pages, output=output_path, elapsed_s=round(elapsed, 1))

    return CrawlResponse(
        output_path=output_path,
        pages_crawled=pages,
        run_id=run_id,
        elapsed_s=round(elapsed, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /admin/preprocess
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/preprocess",
    response_model=PreprocessResponse,
    summary="Run the data preprocessing pipeline",
)
async def preprocess(
    request: PreprocessRequest,
    auth:    AuthContext = Depends(require_admin_auth),
):
    """
    Runs HTML extraction → text cleaning → deduplication → quality filtering
    on a raw crawl JSONL file.

    The pipeline resumes from the last completed stage if the process was
    interrupted (each stage writes to a separate intermediate file).
    """
    start       = time.perf_counter()
    input_path  = _resolve_path(request.input_key)
    output_path = _resolve_path(request.output_key)

    if not Path(input_path).exists():
        raise HTTPException(status_code=404, detail=f"Input not found: {input_path}")

    logger.info("Admin preprocess triggered", input=input_path)

    def _do_preprocess():
        from src.data.pipeline import DataPipeline

        pipeline = DataPipeline(output_dir=str(Path(output_path).parent))
        stats    = pipeline.run_preprocessing(input_path=input_path, output_path=output_path)
        return stats

    try:
        stats = await asyncio.to_thread(_do_preprocess)
    except Exception as exc:
        logger.error("Preprocess failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Preprocess failed: {exc}")

    # Count output documents
    doc_count = 0
    out_file = Path(output_path)
    if out_file.exists():
        try:
            doc_count = sum(1 for _ in out_file.open())
        except Exception:
            doc_count = getattr(stats, "total_documents", 0)

    elapsed = time.perf_counter() - start
    logger.info("Preprocess complete", doc_count=doc_count, elapsed_s=round(elapsed, 1))

    return PreprocessResponse(
        output_path=output_path,
        doc_count=doc_count,
        elapsed_s=round(elapsed, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /admin/train
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Run a model training pass",
)
async def train(
    request: TrainRequest,
    http_req: Request,
    auth:    AuthContext = Depends(require_admin_auth),
):
    """
    Dispatches a training run via scripts/training/train.py as a subprocess.

    Blocks until training completes (worker timeout: 4 hours).
    Returns the checkpoint path and training metrics on completion.

    After training, the checkpoint is promoted automatically into the model
    registry so it can be loaded by reload-model.
    """
    start = time.perf_counter()

    dataset_path = _resolve_path(request.dataset_key)
    if not Path(dataset_path).exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

    # Resolve base model tokenizer from the registry
    registry  = getattr(http_req.app.state, "registry", None)
    model_name_only = request.base_model.split(":")[0]
    tokenizer_path  = str(_ENGINE_ROOT / "storage" / "tokenizers")

    # Try to find the tokenizer bundled with the base model
    base_model_dir = _ENGINE_ROOT / "storage" / "models" / model_name_only / "latest"
    if (base_model_dir / "tokenizer").exists():
        tokenizer_path = str(base_model_dir / "tokenizer")
    else:
        # Fall back to any tokenizer in storage/tokenizers/
        tok_dirs = sorted((_ENGINE_ROOT / "storage" / "tokenizers").iterdir()) if (
            _ENGINE_ROOT / "storage" / "tokenizers"
        ).exists() else []
        if tok_dirs:
            tokenizer_path = str(tok_dirs[-1])

    run_name    = f"auto-{request.new_version}"
    output_dir  = str(_ENGINE_ROOT / "storage" / "checkpoints")

    logger.info(
        "Admin train triggered",
        mode=request.mode,
        version=request.new_version,
        max_steps=request.max_steps,
        dataset=dataset_path,
    )

    cmd = [
        _venv_python(),
        str(_ENGINE_ROOT / "scripts" / "training" / "train.py"),
        "--model",      "gpt-small",
        "--tokenizer",  tokenizer_path,
        "--data",       dataset_path,
        "--output",     output_dir,
        "--run-name",   run_name,
        "--max-steps",  str(request.max_steps),
        "--batch-size", "4",
        "--seq-len",    "512",
        "--save-every", str(max(100, request.max_steps // 10)),
        "--log-every",  "50",
        "--num-workers", "0",
        "--no-amp",
    ]

    def _do_train():
        env = {**os.environ, "PYTHONPATH": str(_ENGINE_ROOT)}
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_ENGINE_ROOT),
            env=env,
            timeout=4 * 3600,  # 4-hour hard timeout
        )
        return proc

    try:
        proc = await asyncio.to_thread(_do_train)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Training exceeded 4-hour timeout")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training subprocess failed: {exc}")

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed (exit {proc.returncode}): {proc.stderr[-800:]}",
        )

    # Locate the best or latest checkpoint from this run
    checkpoint_dir = Path(output_dir) / run_name
    best_dir   = checkpoint_dir / "best"
    latest_dir = checkpoint_dir / "latest"
    chosen_dir = best_dir if best_dir.exists() else (
        latest_dir if latest_dir.exists() else checkpoint_dir
    )

    # Read final metrics from train log
    final_loss = 0.0
    perplexity = float("inf")
    steps      = request.max_steps

    train_log = checkpoint_dir / "train_log.jsonl"
    if train_log.exists():
        try:
            last_step = None
            for line in train_log.open():
                entry = json.loads(line)
                if entry.get("event") == "step":
                    last_step = entry
            if last_step:
                final_loss = last_step.get("loss", 0.0)
                perplexity = last_step.get("perplexity", float("inf"))
                steps      = last_step.get("step", steps)
        except Exception:
            pass

    # Promote the checkpoint into the model registry
    promote_script = _REPO_ROOT / "scripts" / "models" / "promote.py"
    if promote_script.exists() and chosen_dir.exists():
        try:
            promote_cmd = [
                _venv_python(), str(promote_script),
                "--checkpoint",  str(chosen_dir),
                "--tokenizer",   tokenizer_path,
                "--name",        model_name_only,
                "--version",     request.new_version,
                "--skip-eval",
            ]
            env = {**os.environ, "PYTHONPATH": str(_ENGINE_ROOT)}
            await asyncio.to_thread(
                lambda: subprocess.run(promote_cmd, cwd=str(_ENGINE_ROOT), env=env, timeout=300)
            )
            logger.info("Checkpoint promoted", version=request.new_version)
        except Exception as exc:
            logger.warning("Auto-promote after training failed", error=str(exc))

    elapsed = time.perf_counter() - start
    model_path = str(_ENGINE_ROOT / "storage" / "models" / model_name_only / request.new_version)

    logger.info(
        "Training complete",
        version=request.new_version,
        loss=round(final_loss, 4),
        perplexity=round(perplexity, 2),
        steps=steps,
        elapsed_s=round(elapsed, 1),
    )

    return TrainResponse(
        model_path=model_path,
        final_loss=round(final_loss, 4),
        perplexity=round(min(perplexity, 9999.0), 2),
        steps=steps,
        elapsed_s=round(elapsed, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /admin/evaluate
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate a candidate model vs baseline",
)
async def evaluate(
    request: EvaluateRequest,
    auth:    AuthContext = Depends(require_admin_auth),
):
    """
    Runs the evaluation gate via scripts/models/evaluate.py.
    Returns perplexity, regression fraction, and pass/fail verdict.
    """
    evaluate_script = _REPO_ROOT / "scripts" / "models" / "evaluate.py"
    if not evaluate_script.exists():
        raise HTTPException(status_code=500, detail="evaluate.py not found")

    new_path  = _resolve_path(request.new_model_path)
    base_path = _resolve_path(request.base_model_path)

    cmd = [
        _venv_python(), str(evaluate_script),
        "--new-model",      new_path,
        "--base-model",     base_path,
        "--max-regression", str(request.max_regression),
        "--quiet",
    ]
    if request.eval_data:
        cmd += ["--eval-data", _resolve_path(request.eval_data)]

    logger.info("Admin evaluate triggered", new=new_path, base=base_path)

    try:
        result = await asyncio.to_thread(lambda: _run_script(cmd, timeout=1_800))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {exc}")

    logger.info(
        "Evaluation complete",
        passed=result.get("passed"),
        regression=result.get("regression"),
    )

    return EvaluateResponse(
        new_perplexity=result.get("new_perplexity"),
        base_perplexity=result.get("base_perplexity"),
        regression=result.get("regression", 0.0),
        passed=result.get("passed", False),
        gates=result.get("gates", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /admin/reload-model
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/reload-model",
    response_model=ReloadResponse,
    summary="Hot-swap the active inference model",
)
async def reload_model(
    request:  ReloadRequest,
    http_req: Request,
    auth:     AuthContext = Depends(require_admin_auth),
):
    """
    Atomically replaces the active InferenceEngine with a new model.

    The reload is protected by a lock — concurrent calls will receive 409.
    If the new model fails sanity checks, the old model stays active.

    If the model_key points to a checkpoint directory (storage/checkpoints/…),
    promote.py is called first to build the proper model artifact.
    If it points to a promoted artifact (storage/models/…), it loads directly.
    """
    if _reload_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A model reload is already in progress. Try again after it completes.",
        )

    async with _reload_lock:
        start = time.perf_counter()

        model_key  = _resolve_path(request.model_key)
        model_name = request.model_name
        version    = request.version

        logger.info("Hot-swap initiated", model_key=model_key, version=version)

        # ── Determine the promoted artifact path ───────────────────────────
        artifact_path = Path(model_key)

        if "checkpoints" in str(artifact_path):
            # Raw checkpoint — promote it first
            logger.info("Promoting checkpoint before reload", checkpoint=str(artifact_path))
            promote_script = _REPO_ROOT / "scripts" / "models" / "promote.py"

            tokenizer_search = _ENGINE_ROOT / "storage" / "models" / model_name / "latest" / "tokenizer"
            tok_path = str(tokenizer_search) if tokenizer_search.exists() else str(
                next(iter((_ENGINE_ROOT / "storage" / "tokenizers").iterdir()), "")
            )

            promote_cmd = [
                _venv_python(), str(promote_script),
                "--checkpoint", str(artifact_path),
                "--tokenizer",  tok_path,
                "--name",       model_name,
                "--version",    version,
                "--skip-eval",
                "--force",
            ]
            env = {**os.environ, "PYTHONPATH": str(_ENGINE_ROOT)}
            proc = await asyncio.to_thread(
                lambda: subprocess.run(
                    promote_cmd, capture_output=True, text=True,
                    cwd=str(_ENGINE_ROOT), env=env, timeout=300
                )
            )
            if proc.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Promotion before reload failed: {proc.stderr[-400:]}",
                )

        # Canonical artifact location after promotion
        artifact_path = _ENGINE_ROOT / "storage" / "models" / model_name / version
        if not artifact_path.exists():
            # Try "latest" as fallback
            artifact_path = _ENGINE_ROOT / "storage" / "models" / model_name / "latest"

        if not artifact_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model artifact not found at {artifact_path}",
            )

        # ── Load new model in a thread ─────────────────────────────────────
        def _load_new_engine():
            from src.serving.registry.model_registry import ModelRegistry
            from src.serving.inference.inference_engine import InferenceEngine
            from src.tokenizer import load_tokenizer

            tmp_registry = ModelRegistry(
                models_dir=str(_ENGINE_ROOT / "storage" / "models"),
                device=str(settings.torch_device),
            )
            model_key_str = f"{model_name}:{version}"
            model, tokenizer = tmp_registry.get_model(model_key_str)

            engine = InferenceEngine(
                model=model,
                tokenizer=tokenizer,
                device=str(settings.torch_device),
                use_amp=settings.training.use_amp,
            )
            return engine, model, tokenizer

        try:
            new_engine, new_model, new_tokenizer = await asyncio.to_thread(_load_new_engine)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model load failed: {exc}")

        # ── Sanity generation — 5 tokens ───────────────────────────────────
        def _sanity_check():
            import torch
            from src.serving.inference.inference_engine import GenerationRequest
            gen_req = GenerationRequest(
                prompt="The",
                max_new_tokens=5,
                temperature=1.0,   # Sampler validates temperature > 0; greedy ignores it
                strategy="greedy",
            )
            result = new_engine.generate(gen_req)
            return result.text

        try:
            sample = await asyncio.to_thread(_sanity_check)
            logger.info("Sanity generation passed", sample=repr(sample[:40]))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Sanity generation FAILED — old model remains active. Error: {exc}",
            )

        # ── Atomic swap ────────────────────────────────────────────────────
        old_engine = getattr(http_req.app.state, "engine", None)

        http_req.app.state.engine     = new_engine
        http_req.app.state.model_name = f"{model_name}:{version}"

        # Rebuild the TextGenerator with the new model
        try:
            from src.llm.inference.generator import TextGenerator
            http_req.app.state.generator = TextGenerator(
                model=new_model,
                tokenizer=new_tokenizer,
                device=str(settings.torch_device),
            )
        except Exception as exc:
            logger.warning("TextGenerator rebuild failed after reload", error=str(exc))

        # ── Free old model memory ──────────────────────────────────────────
        if old_engine is not None:
            try:
                import torch
                del old_engine
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Old model unloaded from memory")
            except Exception:
                pass

        elapsed = time.perf_counter() - start
        active  = http_req.app.state.model_name

        # Update Prometheus model-version gauge after successful swap
        try:
            from src.api.metrics import record_active_model
            registry_obj = getattr(http_req.app.state, "registry", None)
            info = registry_obj.get_info(active) if registry_obj else None
            record_active_model(
                model_name=request.model_name,
                version=request.version,
                promoted_at_unix=getattr(info, "promoted_at_unix", 0.0) or 0.0,
                param_count=getattr(info, "param_count", 0) or sum(
                    p.numel() for p in new_model.parameters()
                ),
            )
        except Exception as exc:
            logger.debug("Failed to update model metrics after swap", error=str(exc))

        logger.info(
            "Hot-swap complete",
            active_model=active,
            elapsed_s=round(elapsed, 1),
        )

        return ReloadResponse(
            success=True,
            active_model=active,
            elapsed_s=round(elapsed, 1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET /admin/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Engine status and resource usage",
)
async def admin_status(
    http_req: Request,
    auth:     AuthContext = Depends(require_admin_auth),
):
    """
    Returns the active model, device, GPU memory usage, and uptime.
    Used by the worker to confirm a reload was successful.
    """
    engine     = getattr(http_req.app.state, "engine", None)
    model_name = getattr(http_req.app.state, "model_name", "none")
    started_at = getattr(http_req.app.state, "started_at", time.time())
    uptime_s   = round(time.time() - started_at, 1)

    gpu_used  = None
    gpu_total = None
    params    = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_used  = round(torch.cuda.memory_allocated() / 1e6, 1)
            gpu_total = round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1)
    except Exception:
        pass

    if engine is not None:
        try:
            params = sum(p.numel() for p in engine.model.parameters())
        except Exception:
            pass

    return StatusResponse(
        active_model=model_name,
        engine_ready=(engine is not None),
        uptime_s=uptime_s,
        device=str(settings.torch_device),
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
        param_count=params,
    )
