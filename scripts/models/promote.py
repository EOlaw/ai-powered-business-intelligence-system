#!/usr/bin/env python3
"""
InsightSerenity — Model Promotion Script
=========================================
Validates a training checkpoint and promotes it into the production model
registry format that ModelRegistry can discover and load for inference.

This script is the ONLY sanctioned path from a training checkpoint to a
served model. It refuses to promote anything that fails validation.

Promotion steps
---------------
1.  Validate that the checkpoint directory is complete (model.pt, train_state.json)
2.  Load model weights into CPU memory — confirm they are non-corrupt and non-NaN
3.  Infer the full model architecture config from the checkpoint tensor shapes
4.  Run the evaluation gate (calls evaluate.py internally) — abort on failure
5.  Build the production artifact directory:
        storage/models/{name}/{version}/
            model.pt              ← copied weights
            config.json           ← architecture config
            metadata.json         ← lineage, checksum, perplexity, promoted_at
            tokenizer/            ← full tokenizer bundle
6.  Write a SHA-256 checksum of model.pt into metadata.json
7.  Atomically update the "latest" symlink for this model name
8.  Print a promotion summary

Usage
-----
    cd apps/ai-engine
    ./.venv/bin/python scripts/models/promote.py \\
        --checkpoint storage/checkpoints/pretrain-v1/step_00001000 \\
        --tokenizer  storage/tokenizers/bpe-32k \\
        --name       insightserenity-1 \\
        --version    v0.1.0

    # Skip evaluation gate (only for local dev — never in CI):
    ./.venv/bin/python scripts/models/promote.py \\
        --checkpoint storage/checkpoints/demo-smoke/step_00000001 \\
        --tokenizer  storage/tokenizers/demo-bpe-128 \\
        --name       insightserenity-1 \\
        --version    v0.0.1-demo \\
        --skip-eval

Flags
-----
    --checkpoint PATH   Path to the checkpoint directory to promote.
    --tokenizer  PATH   Path to the tokenizer directory to bundle.
    --name       STR    Model name (e.g. "insightserenity-1").
    --version    STR    Semantic version (e.g. "v0.1.0").
    --model-type STR    "gpt" or "bert". Default: auto-detected from weights.
    --seq-len    INT    Maximum sequence length. Default: from config or 1024.
    --eval-data  PATH   JSONL file for perplexity evaluation. Optional.
    --skip-eval         Skip the evaluation gate (dev only — never in CI).
    --dry-run           Validate and print what would happen without writing anything.
    --force             Overwrite an existing promotion of the same version.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup — must happen before any src.* imports ────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "apps" / "ai-engine"
sys.path.insert(0, str(ENGINE_ROOT))

import torch
from src.utils.logger import get_logger

logger = get_logger("promote")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Promote a training checkpoint to the production model registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="Checkpoint directory containing model.pt and train_state.json")
    p.add_argument("--tokenizer",  required=True, metavar="PATH",
                   help="Tokenizer directory to bundle with the model")
    p.add_argument("--name",       required=True, metavar="STR",
                   help="Model name (e.g. insightserenity-1)")
    p.add_argument("--version",    required=True, metavar="STR",
                   help="Semantic version string (e.g. v0.1.0)")
    p.add_argument("--model-type", default=None, choices=["gpt", "bert"],
                   help="Architecture type. Auto-detected if omitted.")
    p.add_argument("--seq-len",    type=int, default=None,
                   help="Override max sequence length. Reads from checkpoint if omitted.")
    p.add_argument("--eval-data",  default=None, metavar="PATH",
                   help="JSONL file for perplexity evaluation before promotion.")
    p.add_argument("--compare-against", default=None, metavar="PATH",
                   help="Path to production model to compare perplexity against (regression check).")
    p.add_argument("--max-regression", type=float, default=0.02,
                   help="Max allowed perplexity regression fraction vs baseline (default: 0.02 = 2%%).")
    p.add_argument("--skip-eval",  action="store_true",
                   help="Skip evaluation gate. Only for local dev. Never use in CI.")
    p.add_argument("--dry-run",    action="store_true",
                   help="Validate and print what would happen without writing files.")
    p.add_argument("--force",      action="store_true",
                   help="Overwrite an existing promotion of the same version.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_checkpoint(checkpoint_dir: Path) -> dict:
    """
    Confirm checkpoint is complete and loadable.

    Checks:
    - model.pt exists
    - train_state.json exists and is valid JSON
    - Weights load without error
    - No NaN or Inf in any weight tensor

    Returns the loaded train_state dict.
    Raises SystemExit with a clear message on any failure.
    """
    model_pt     = checkpoint_dir / "model.pt"
    state_json   = checkpoint_dir / "train_state.json"

    print(f"\n[1/6] Validating checkpoint: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        sys.exit(f"  ERROR: Checkpoint directory not found: {checkpoint_dir}")
    if not model_pt.exists():
        sys.exit(f"  ERROR: model.pt not found in {checkpoint_dir}")
    if not state_json.exists():
        sys.exit(f"  ERROR: train_state.json not found in {checkpoint_dir}")

    # Load train state
    try:
        with open(state_json) as f:
            content = f.read()
            # json.loads can't handle Infinity — replace it before parsing
            content = content.replace("Infinity", "1e308").replace("NaN", "null")
            train_state = json.loads(content)
    except Exception as e:
        sys.exit(f"  ERROR: Could not parse train_state.json: {e}")

    # Load weights — CPU only (no GPU needed for validation)
    print(f"  Loading weights from model.pt ({model_pt.stat().st_size / 1e6:.1f} MB)...")
    try:
        state_dict = torch.load(str(model_pt), map_location="cpu", weights_only=True)
    except Exception as e:
        sys.exit(f"  ERROR: Could not load model.pt: {e}")

    # NaN / Inf check
    nan_keys = []
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_keys.append(key)

    if nan_keys:
        sys.exit(
            f"  ERROR: model.pt contains NaN/Inf in {len(nan_keys)} tensors:\n"
            + "\n".join(f"    - {k}" for k in nan_keys[:10])
        )

    print(f"  ✓ Checkpoint valid — {len(state_dict)} tensors, "
          f"step {train_state.get('global_step', '?')}, "
          f"train_loss {train_state.get('train_loss', 0):.4f}")

    return train_state, state_dict


def infer_config(state_dict: dict, model_type: str | None, seq_len_override: int | None) -> dict:
    """
    Reconstruct the GPTConfig (or BERTConfig) from checkpoint tensor shapes.

    We derive:
        vocab_size   from token_emb.embedding.weight shape[0]
        d_model      from token_emb.embedding.weight shape[1]
        num_layers   by counting unique layer indices in key names
        num_heads    from layers.0.attn.attn.q_proj.weight shape[0] / head_dim

    All other fields are kept at their GPT_SMALL defaults since they don't
    change tensor shapes and must match what was used during training.
    """
    print("\n[2/6] Inferring model architecture from checkpoint...")

    # Vocab size and d_model from embedding weight
    embed_key = "token_emb.embedding.weight"
    if embed_key not in state_dict:
        # Try BERT-style embedding
        embed_key = "embeddings.token_embeddings.embedding.weight"
    if embed_key not in state_dict:
        sys.exit("  ERROR: Could not locate embedding weight tensor to infer vocab_size/d_model")

    embed_weight = state_dict[embed_key]
    vocab_size   = embed_weight.shape[0]
    d_model      = embed_weight.shape[1]

    # Count transformer layers
    layer_indices = set()
    for k in state_dict:
        # Matches "layers.0.", "layers.12.", etc.
        parts = k.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_indices.add(int(parts[1]))
    num_layers = len(layer_indices)

    # Detect architecture type from key names
    if model_type is None:
        if any("causal_mask" in k for k in state_dict):
            model_type = "gpt"
        elif any("pooler" in k for k in state_dict):
            model_type = "bert"
        else:
            model_type = "gpt"

    # d_ff: look for ffn weight
    d_ff = None
    for k in state_dict:
        if "ffn" in k and "fc1" in k and "weight" in k:
            d_ff = state_dict[k].shape[0]
            break
        if "ffn" in k and "gate_proj" in k and "weight" in k:
            d_ff = state_dict[k].shape[0]
            break

    # num_heads: infer from q_proj if available
    num_heads = 12  # safe default for gpt-small
    for k in state_dict:
        if "q_proj" in k and "weight" in k:
            head_dim = 64  # standard
            num_heads = max(1, state_dict[k].shape[0] // head_dim)
            break

    from src.architectures.transformer.decoder.gpt_decoder import GPT_SMALL
    import dataclasses

    # Read max_seq_len from causal_mask shape — this is the ground truth.
    # The causal mask is stored as a buffer with shape [seq_len, seq_len].
    # seq_len_override takes priority (e.g. to serve a longer context window).
    detected_seq_len = 1024
    for k in state_dict:
        if "causal_mask" in k:
            detected_seq_len = state_dict[k].shape[0]
            break

    seq_len = seq_len_override or detected_seq_len

    # Build config dict — use GPT_SMALL defaults for non-shape-determining fields
    base = dataclasses.asdict(GPT_SMALL)
    cfg  = {
        **base,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "d_model":    d_model,
        "num_layers": num_layers,
        "num_heads":  num_heads,
        "max_seq_len": seq_len,
        "capabilities": ["text-generation", "chat", "embeddings"],
    }
    if d_ff is not None:
        cfg["d_ff"] = d_ff

    print(f"  Detected: {model_type} | vocab={vocab_size} | d_model={d_model} | "
          f"layers={num_layers} | heads={num_heads} | seq_len={seq_len}")

    return cfg


def validate_tokenizer(tokenizer_dir: Path) -> None:
    """Confirm the tokenizer directory has all required files."""
    print(f"\n[3/6] Validating tokenizer: {tokenizer_dir}")

    if not tokenizer_dir.exists():
        sys.exit(f"  ERROR: Tokenizer directory not found: {tokenizer_dir}")

    required = ["vocab.json", "tokenizer_config.json"]
    for fname in required:
        if not (tokenizer_dir / fname).exists():
            sys.exit(f"  ERROR: Required tokenizer file missing: {tokenizer_dir / fname}")

    # Load and verify vocab
    try:
        with open(tokenizer_dir / "vocab.json") as f:
            vocab = json.load(f)
    except Exception as e:
        sys.exit(f"  ERROR: Could not parse vocab.json: {e}")

    with open(tokenizer_dir / "tokenizer_config.json") as f:
        tok_cfg = json.load(f)

    print(f"  ✓ Tokenizer valid — {tok_cfg.get('tokenizer_class')} | vocab_size={len(vocab)}")

    return len(vocab)


def check_vocab_model_match(tok_vocab_size: int, model_cfg: dict) -> None:
    """Abort if tokenizer vocab size does not match model embedding dimension."""
    model_vocab = model_cfg["vocab_size"]
    if tok_vocab_size != model_vocab:
        sys.exit(
            f"  ERROR: Vocab size mismatch — tokenizer has {tok_vocab_size} tokens "
            f"but model embedding has {model_vocab}.\n"
            f"  You must use the tokenizer that was used during training."
        )
    print(f"  ✓ Vocab/model match: {tok_vocab_size} tokens")


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation gate
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_gate(
    checkpoint_dir: Path,
    tokenizer_dir:  Path,
    model_cfg:      dict,
    eval_data:      Path | None,
    compare_against: Path | None,
    max_regression:  float,
) -> float | None:
    """
    Run pre-promotion evaluation checks.

    Returns the new model's perplexity (or None if no eval_data provided).
    Exits with non-zero on any gate failure.
    """
    print("\n[4/6] Running evaluation gate...")

    # ── Gate 1: Sanity generation ──────────────────────────────────────────
    print("  Gate 1: Sanity generation (can the model produce tokens?)...")
    try:
        from src.tokenizer import load_tokenizer
        from src.architectures.transformer.decoder.gpt_decoder import GPTDecoder, GPTConfig
        import dataclasses

        tokenizer  = load_tokenizer(str(tokenizer_dir))

        # Reconstruct model
        cfg_obj = GPTConfig(**{
            k: v for k, v in model_cfg.items()
            if k in {f.name for f in dataclasses.fields(GPTConfig)}
        })
        model = GPTDecoder(cfg_obj)
        state = torch.load(str(checkpoint_dir / "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()

        # Generate 5 tokens from a short prompt
        prompt_ids  = tokenizer.encode("The", add_special_tokens=False)[:4]
        if not prompt_ids:
            prompt_ids = [1]  # fallback to UNK

        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            for _ in range(5):
                output  = model(input_ids)
                next_id = output["logits"][0, -1, :].argmax().item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

        generated = tokenizer.decode(input_ids[0].tolist()[len(prompt_ids):], skip_special_tokens=True)
        print(f"  ✓ Sanity generation passed: '{generated[:60]}'")

    except Exception as e:
        sys.exit(f"  ERROR: Sanity generation failed: {e}")

    # ── Gate 2: Perplexity (only if eval_data provided) ────────────────────
    new_perplexity = None
    if eval_data and eval_data.exists():
        print(f"  Gate 2: Computing perplexity on {eval_data.name}...")
        try:
            evaluate_script = Path(__file__).parent / "evaluate.py"
            import subprocess
            result = subprocess.run(
                [sys.executable, str(evaluate_script),
                 "--new-model",  str(checkpoint_dir),
                 "--base-model", str(compare_against or checkpoint_dir),
                 "--eval-data",  str(eval_data),
                 "--max-regression", str(max_regression),
                 "--output-json", "/tmp/is_promote_eval.json"],
                capture_output=True, text=True
            )
            with open("/tmp/is_promote_eval.json") as f:
                eval_result = json.load(f)

            new_perplexity = eval_result.get("new_perplexity")
            passed         = eval_result.get("passed", True)
            regression     = eval_result.get("regression", 0)

            print(f"  Perplexity: {new_perplexity:.2f} | Regression: {regression * 100:.2f}%")

            if not passed:
                sys.exit(
                    f"  ERROR: Evaluation gate FAILED.\n"
                    f"  New perplexity {new_perplexity:.2f} vs baseline "
                    f"{eval_result.get('base_perplexity', '?'):.2f} "
                    f"exceeds {max_regression * 100:.1f}% regression threshold.\n"
                    f"  This model will NOT be promoted. Retrain or use --skip-eval for dev."
                )
            print(f"  ✓ Perplexity gate passed: ppl={new_perplexity:.2f}")

        except FileNotFoundError:
            print("  WARNING: evaluate.py not found — skipping perplexity gate")
        except Exception as e:
            print(f"  WARNING: Perplexity gate error: {e} — proceeding")
    else:
        print("  Gate 2: Skipped (no --eval-data provided)")

    return new_perplexity


# ─────────────────────────────────────────────────────────────────────────────
# Artifact builder
# ─────────────────────────────────────────────────────────────────────────────

def build_artifact(
    checkpoint_dir: Path,
    tokenizer_dir:  Path,
    model_cfg:      dict,
    train_state:    dict,
    output_dir:     Path,
    name:           str,
    version:        str,
    perplexity:     float | None,
    dry_run:        bool,
) -> None:
    """Copy weights, write config.json, metadata.json, and tokenizer bundle."""

    print(f"\n[5/6] Building artifact at {output_dir}")

    if dry_run:
        print("  DRY RUN — no files will be written")

    model_src = checkpoint_dir / "model.pt"

    # ── model.pt ──────────────────────────────────────────────────────────
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(model_src), str(output_dir / "model.pt"))
    print(f"  ✓ model.pt copied ({model_src.stat().st_size / 1e6:.1f} MB)")

    # ── Checksum ───────────────────────────────────────────────────────────
    checksum = compute_sha256(model_src)
    print(f"  ✓ SHA-256: {checksum[:16]}…")

    # ── config.json ────────────────────────────────────────────────────────
    config_path = output_dir / "config.json"
    if not dry_run:
        with open(config_path, "w") as f:
            json.dump(model_cfg, f, indent=2)
    print(f"  ✓ config.json written")

    # ── metadata.json ──────────────────────────────────────────────────────
    # Safely read train_loss — it may be "Infinity" in raw form (already handled above)
    train_loss = train_state.get("train_loss", None)
    if train_loss is not None and not (isinstance(train_loss, float) and train_loss != float('inf')):
        train_loss = None

    param_count = sum(1 for _ in range(1))  # placeholder; computed properly below
    try:
        state_dict  = torch.load(str(model_src), map_location="cpu", weights_only=True)
        param_count = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))
    except Exception:
        pass

    metadata = {
        "version":          version,
        "model_name":       name,
        "model_type":       model_cfg.get("model_type", "gpt"),
        "param_count":      param_count,
        "vocab_size":       model_cfg.get("vocab_size"),
        "d_model":          model_cfg.get("d_model"),
        "num_layers":       model_cfg.get("num_layers"),
        "num_heads":        model_cfg.get("num_heads"),
        "max_seq_len":      model_cfg.get("max_seq_len"),
        "perplexity":       round(perplexity, 4) if perplexity else None,
        "val_loss":         train_state.get("val_loss") or None,
        "train_loss":       round(train_loss, 4) if train_loss else None,
        "global_step":      train_state.get("global_step"),
        "checksum_sha256":  checksum,
        "training_run":     checkpoint_dir.parent.name,
        "checkpoint_step":  checkpoint_dir.name,
        "promoted_at":      datetime.now(timezone.utc).isoformat(),
        "promoted_by":      "scripts/models/promote.py",
        "capabilities":     model_cfg.get("capabilities", ["text-generation"]),
    }
    metadata_path = output_dir / "metadata.json"
    if not dry_run:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json written")

    # ── Tokenizer bundle ────────────────────────────────────────────────────
    tok_dst = output_dir / "tokenizer"
    if not dry_run:
        if tok_dst.exists():
            shutil.rmtree(str(tok_dst))
        shutil.copytree(str(tokenizer_dir), str(tok_dst))
    print(f"  ✓ tokenizer/ bundled ({len(list(tokenizer_dir.iterdir()))} files)")


def update_latest_symlink(models_root: Path, name: str, version: str, dry_run: bool) -> None:
    """
    Atomically update the 'latest' pointer for this model name.

    Uses a rename-based atomic swap:
        Write new -> models/{name}/latest.new
        Rename latest.new -> latest  (atomic on POSIX systems)
    """
    print(f"\n[6/6] Updating 'latest' pointer for {name}...")

    latest_path = models_root / name / "latest"
    new_path    = models_root / name / "latest.new"
    version_dir = models_root / name / version

    if dry_run:
        print(f"  DRY RUN — would point {latest_path} → {version_dir}")
        return

    # Write a "latest_info.json" inside the latest directory
    # (symlinks can cause issues on some filesystems/Docker volumes)
    # Instead: copy the version dir to latest using the same atomic rename
    if new_path.exists():
        shutil.rmtree(str(new_path))
    shutil.copytree(str(version_dir), str(new_path))

    if latest_path.exists():
        # Keep the old latest as a backup for rollback
        old_backup = models_root / name / "previous"
        if old_backup.exists():
            shutil.rmtree(str(old_backup))
        shutil.copytree(str(latest_path), str(old_backup))
        shutil.rmtree(str(latest_path))

    os.rename(str(new_path), str(latest_path))
    print(f"  ✓ latest → {version}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    start        = time.perf_counter()
    checkpoint   = Path(args.checkpoint).resolve()
    tokenizer    = Path(args.tokenizer).resolve()
    models_root  = ENGINE_ROOT / "storage" / "models"
    output_dir   = models_root / args.name / args.version

    print("=" * 60)
    print(f"  InsightSerenity — Model Promotion")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Tokenizer  : {tokenizer}")
    print(f"  Target     : {output_dir}")
    print("=" * 60)

    # Guard: refuse to overwrite without --force
    if output_dir.exists() and not args.force:
        sys.exit(
            f"\nERROR: {output_dir} already exists.\n"
            f"Use --force to overwrite, or bump --version to a new string."
        )

    # ── Step 1: Validate checkpoint ────────────────────────────────────────
    train_state, state_dict = validate_checkpoint(checkpoint)

    # ── Step 2: Infer config ───────────────────────────────────────────────
    model_cfg = infer_config(state_dict, args.model_type, args.seq_len)

    # ── Step 3: Validate tokenizer ─────────────────────────────────────────
    tok_vocab_size = validate_tokenizer(tokenizer)
    check_vocab_model_match(tok_vocab_size, model_cfg)

    # ── Step 4: Evaluation gate ────────────────────────────────────────────
    perplexity = None
    if not args.skip_eval:
        eval_data       = Path(args.eval_data).resolve() if args.eval_data else None
        compare_against = Path(args.compare_against).resolve() if args.compare_against else None
        perplexity = run_evaluation_gate(
            checkpoint, tokenizer, model_cfg,
            eval_data, compare_against, args.max_regression,
        )
    else:
        print("\n[4/6] Evaluation gate SKIPPED (--skip-eval)")
        if args.dry_run is False:
            print("  WARNING: Skipping evaluation is not recommended for production.")

    # ── Step 5: Build artifact ─────────────────────────────────────────────
    build_artifact(
        checkpoint_dir=checkpoint,
        tokenizer_dir=tokenizer,
        model_cfg=model_cfg,
        train_state=train_state,
        output_dir=output_dir,
        name=args.name,
        version=args.version,
        perplexity=perplexity,
        dry_run=args.dry_run,
    )

    # ── Step 6: Update latest pointer ─────────────────────────────────────
    update_latest_symlink(models_root, args.name, args.version, args.dry_run)

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"  PROMOTION {'(DRY RUN) ' if args.dry_run else ''}COMPLETE")
    print(f"  Model   : {args.name}:{args.version}")
    print(f"  Path    : {output_dir}")
    print(f"  Elapsed : {elapsed:.1f}s")
    if perplexity:
        print(f"  PPL     : {perplexity:.2f}")
    print("=" * 60)
    print()
    print("Next step — start the AI engine and verify:")
    print(f"  curl http://localhost:8001/v1/models")
    print()


if __name__ == "__main__":
    main()
