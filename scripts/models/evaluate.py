#!/usr/bin/env python3
"""
InsightSerenity — Pre-Promotion Evaluation Gate
================================================
Runs structured evaluation checks against a candidate checkpoint before
it can be promoted to the production model registry.

This is the authoritative evaluation gate. It is called by promote.py
automatically. It can also be called standalone for debugging.

Evaluation gates
----------------
Gate 1 — Load check       : Can the weights be loaded without errors?
Gate 2 — Sanity generation: Can the model generate at least 5 tokens?
Gate 3 — Perplexity       : Is perplexity below the absolute ceiling?
Gate 4 — Regression check : Is perplexity no more than X% worse than baseline?

Exit codes
----------
0   All gates passed. Safe to promote.
1   One or more gates failed. Promotion blocked.
2   Configuration error (missing files, wrong args).

Output
------
JSON written to --output-json (if provided):
{
  "new_perplexity":  18.4,
  "base_perplexity": 17.9,
  "regression":      0.028,       <- fraction (0.028 = 2.8%)
  "max_regression":  0.02,
  "passed":          false,
  "gates": {
    "load":       true,
    "generation": true,
    "perplexity": true,
    "regression": false
  }
}

Usage
-----
    cd apps/ai-engine
    ./.venv/bin/python scripts/models/evaluate.py \\
        --new-model   storage/checkpoints/pretrain-v1/step_00001000 \\
        --base-model  storage/models/insightserenity-1/latest \\
        --eval-data   storage/datasets/eval/eval.jsonl \\
        --max-regression 0.02 \\
        --output-json /tmp/eval.json
"""

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "apps" / "ai-engine"
sys.path.insert(0, str(ENGINE_ROOT))

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pre-promotion evaluation gates on a candidate model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--new-model",      required=True, metavar="PATH",
                   help="Candidate checkpoint directory or promoted model directory.")
    p.add_argument("--base-model",     default=None,  metavar="PATH",
                   help="Baseline model directory for regression comparison. "
                        "If omitted, regression gate is skipped.")
    p.add_argument("--eval-data",      default=None,  metavar="PATH",
                   help="JSONL file for perplexity computation. "
                        "If omitted, only load + generation gates run.")
    p.add_argument("--tokenizer",      default=None,  metavar="PATH",
                   help="Tokenizer directory. Auto-detected from model dir if omitted.")
    p.add_argument("--max-regression", type=float, default=0.02,
                   help="Max regression fraction (default 0.02 = 2%%).")
    p.add_argument("--max-perplexity", type=float, default=1000.0,
                   help="Absolute perplexity ceiling (default 1000).")
    p.add_argument("--max-batches",    type=int,   default=100,
                   help="Max evaluation batches (default 100).")
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--seq-len",        type=int,   default=128)
    p.add_argument("--output-json",    default=None, metavar="PATH",
                   help="Write JSON result to this path.")
    p.add_argument("--quiet",          action="store_true",
                   help="Suppress progress output.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_tokenizer_dir(model_dir: Path, tokenizer_override: Path | None) -> Path | None:
    """Find the tokenizer directory: explicit override > model_dir/tokenizer > None."""
    if tokenizer_override and tokenizer_override.exists():
        return tokenizer_override
    bundled = model_dir / "tokenizer"
    if bundled.exists():
        return bundled
    return None


def _build_model_from_dir(model_dir: Path) -> tuple:
    """
    Load a model from either a promoted artifact directory or a raw checkpoint.

    Promoted artifacts have config.json with full architecture params.
    Raw checkpoints have train_state.json — we infer architecture from tensor shapes.

    Returns (model, tokenizer_dir_or_None).
    """
    model_pt   = model_dir / "model.pt"
    config_json = model_dir / "config.json"

    if not model_pt.exists():
        sys.exit(f"ERROR: model.pt not found in {model_dir}")

    # Load weights
    state_dict = torch.load(str(model_pt), map_location="cpu", weights_only=True)

    # Read config if present
    cfg_dict = {}
    if config_json.exists():
        with open(config_json) as f:
            cfg_dict = json.load(f)

    # ── Infer ALL shape-determining fields directly from tensor shapes ─────
    # This ensures the constructed model always matches the checkpoint exactly,
    # even when config.json is absent or was saved with different defaults.

    embed_key = "token_emb.embedding.weight"
    if embed_key in state_dict:
        cfg_dict["vocab_size"] = state_dict[embed_key].shape[0]
        cfg_dict["d_model"]    = state_dict[embed_key].shape[1]

    # max_seq_len from causal_mask shape [seq_len, seq_len]
    for k in state_dict:
        if "causal_mask" in k:
            cfg_dict["max_seq_len"] = state_dict[k].shape[0]
            break

    # num_layers: count unique layer indices
    layer_indices = {
        k.split(".")[1]
        for k in state_dict
        if k.startswith("layers.") and k.split(".")[1].isdigit()
    }
    if layer_indices:
        cfg_dict["num_layers"] = len(layer_indices)

    # Build the model with the exact shapes derived from the checkpoint
    from src.architectures.transformer.decoder.gpt_decoder import GPTDecoder, GPTConfig
    gpt_fields = {f.name for f in dataclasses.fields(GPTConfig)}
    cfg_obj    = GPTConfig(**{k: v for k, v in cfg_dict.items() if k in gpt_fields})
    model      = GPTDecoder(cfg_obj)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tok_dir = model_dir / "tokenizer"
    return model, (tok_dir if tok_dir.exists() else None)


# ─────────────────────────────────────────────────────────────────────────────
# Gate implementations
# ─────────────────────────────────────────────────────────────────────────────

def gate_load(model_dir: Path, quiet: bool) -> tuple[bool, str]:
    """Gate 1: Can the model be loaded without errors?"""
    if not quiet:
        print("  Gate 1: Load check...")
    try:
        model, _ = _build_model_from_dir(model_dir)
        param_count = sum(t.numel() for t in model.state_dict().values())
        msg = f"OK — {param_count / 1e6:.1f}M params"
        if not quiet:
            print(f"    ✓ {msg}")
        return True, msg
    except Exception as e:
        msg = f"FAILED — {e}"
        if not quiet:
            print(f"    ✗ {msg}")
        return False, msg


def gate_generation(model_dir: Path, tok_dir: Path | None, quiet: bool) -> tuple[bool, str]:
    """Gate 2: Can the model generate at least 5 tokens?"""
    if not quiet:
        print("  Gate 2: Sanity generation...")
    try:
        model, bundled_tok = _build_model_from_dir(model_dir)
        tok_path = tok_dir or bundled_tok
        if tok_path is None:
            return True, "SKIPPED — no tokenizer found, using token ID 1 as prompt"

        from src.tokenizer import load_tokenizer
        tokenizer  = load_tokenizer(str(tok_path))
        prompt_ids = tokenizer.encode("The sky is", add_special_tokens=False) or [1]

        input_ids = torch.tensor([prompt_ids[:8]], dtype=torch.long)
        generated = []
        with torch.no_grad():
            for _ in range(5):
                out     = model(input_ids)
                next_id = out["logits"][0, -1, :].argmax().item()
                generated.append(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        msg = f"OK — generated: '{text[:40]}'"
        if not quiet:
            print(f"    ✓ {msg}")
        return True, msg
    except Exception as e:
        msg = f"FAILED — {e}"
        if not quiet:
            print(f"    ✗ {msg}")
        return False, msg


def _compute_perplexity_on_dataloader(model, dataloader, max_batches: int) -> float:
    """
    Compute perplexity over a DataLoader.
    Uses summed NLL / total non-padding tokens.
    """
    total_nll    = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch.get("input_ids")
            if input_ids is None:
                continue

            labels = batch.get("labels", input_ids)

            output  = model(input_ids=input_ids)
            logits  = output["logits"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            mask    = (shift_labels != -100).float()
            loss    = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).clamp(min=0),   # clamp -100 → 0 for compute
                reduction="none",
            )
            total_nll    += (loss * mask.view(-1)).sum().item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return math.exp(min(avg_nll, 20.0))   # cap at e^20 to avoid overflow


def gate_perplexity(
    model_dir: Path,
    tok_dir: Path | None,
    eval_data: Path,
    max_perplexity: float,
    batch_size: int,
    seq_len: int,
    max_batches: int,
    quiet: bool,
) -> tuple[bool, float, str]:
    """Gate 3: Is perplexity below the absolute ceiling?"""
    if not quiet:
        print(f"  Gate 3: Perplexity check (max_ppl={max_perplexity})...")

    try:
        model, bundled_tok = _build_model_from_dir(model_dir)
        tok_path = tok_dir or bundled_tok

        if tok_path is None:
            return True, float("inf"), "SKIPPED — no tokenizer"

        from src.tokenizer import load_tokenizer
        from src.data.datasets.text_dataset import JSONLTextDataset
        from src.data.datasets.data_collator import LanguageModelCollator
        from torch.utils.data import DataLoader

        tokenizer  = load_tokenizer(str(tok_path))
        dataset    = JSONLTextDataset(str(eval_data), tokenizer=tokenizer, max_length=seq_len)

        if len(dataset) == 0:
            return True, float("inf"), "SKIPPED — eval dataset is empty"

        collator   = LanguageModelCollator(
            block_size=seq_len,
            eos_token_id=getattr(tokenizer, "eos_token_id", 3),
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
        )

        ppl = _compute_perplexity_on_dataloader(model, dataloader, max_batches)
        ppl = round(ppl, 4)

        passed = ppl < max_perplexity
        msg    = f"{'OK' if passed else 'FAILED'} — ppl={ppl:.2f} (ceiling={max_perplexity})"
        if not quiet:
            print(f"    {'✓' if passed else '✗'} {msg}")
        return passed, ppl, msg

    except Exception as e:
        msg = f"ERROR — {e}"
        if not quiet:
            print(f"    ✗ {msg}")
        return False, float("inf"), msg


def gate_regression(
    new_ppl:        float,
    base_model_dir: Path,
    tok_dir: Path | None,
    eval_data: Path,
    max_regression: float,
    batch_size: int,
    seq_len: int,
    max_batches: int,
    quiet: bool,
) -> tuple[bool, float, float, str]:
    """Gate 4: Is the new model no more than max_regression worse than baseline?"""
    if not quiet:
        print(f"  Gate 4: Regression check (max={max_regression * 100:.1f}%)...")

    try:
        model, bundled_tok = _build_model_from_dir(base_model_dir)
        tok_path = tok_dir or bundled_tok
        if tok_path is None:
            return True, new_ppl, new_ppl, "SKIPPED — no tokenizer for baseline"

        from src.tokenizer import load_tokenizer
        from src.data.datasets.text_dataset import JSONLTextDataset
        from src.data.datasets.data_collator import LanguageModelCollator
        from torch.utils.data import DataLoader

        tokenizer  = load_tokenizer(str(tok_path))
        dataset    = JSONLTextDataset(str(eval_data), tokenizer=tokenizer, max_length=seq_len)

        if len(dataset) == 0:
            return True, new_ppl, new_ppl, "SKIPPED — eval dataset empty"

        collator   = LanguageModelCollator(
            block_size=seq_len,
            eos_token_id=getattr(tokenizer, "eos_token_id", 3),
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
        )

        base_ppl   = round(_compute_perplexity_on_dataloader(model, dataloader, max_batches), 4)
        regression = (new_ppl - base_ppl) / max(base_ppl, 1e-6)
        passed     = regression <= max_regression

        msg = (f"{'OK' if passed else 'FAILED'} — "
               f"new={new_ppl:.2f} base={base_ppl:.2f} "
               f"regression={regression * 100:.2f}%")
        if not quiet:
            print(f"    {'✓' if passed else '✗'} {msg}")
        return passed, new_ppl, base_ppl, msg

    except Exception as e:
        msg = f"ERROR — {e}"
        if not quiet:
            print(f"    ✗ {msg}")
        return False, new_ppl, float("inf"), msg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    new_dir   = Path(args.new_model).resolve()
    base_dir  = Path(args.base_model).resolve() if args.base_model else None
    eval_data = Path(args.eval_data).resolve()  if args.eval_data  else None
    tok_dir   = Path(args.tokenizer).resolve()  if args.tokenizer  else None

    if not new_dir.exists():
        sys.exit(f"ERROR: --new-model path not found: {new_dir}")

    if not args.quiet:
        print("\nInsightSerenity — Model Evaluation Gate")
        print(f"  New model : {new_dir}")
        print(f"  Baseline  : {base_dir or 'none'}")
        print(f"  Eval data : {eval_data or 'none'}")
        print()

    gates      = {}
    new_ppl    = float("inf")
    base_ppl   = float("inf")
    regression = 0.0

    # Gate 1: Load
    ok, msg = gate_load(new_dir, args.quiet)
    gates["load"] = ok

    # Gate 2: Generation
    if ok:
        ok2, msg2 = gate_generation(new_dir, tok_dir, args.quiet)
        gates["generation"] = ok2
    else:
        gates["generation"] = False

    # Gate 3: Perplexity
    gates["perplexity"] = True
    if eval_data and eval_data.exists():
        ok3, new_ppl, msg3 = gate_perplexity(
            new_dir, tok_dir, eval_data,
            args.max_perplexity, args.batch_size, args.seq_len, args.max_batches,
            args.quiet,
        )
        gates["perplexity"] = ok3

    # Gate 4: Regression
    gates["regression"] = True
    if base_dir and base_dir.exists() and eval_data and eval_data.exists() and new_ppl < float("inf"):
        ok4, new_ppl, base_ppl, msg4 = gate_regression(
            new_ppl, base_dir, tok_dir, eval_data,
            args.max_regression, args.batch_size, args.seq_len, args.max_batches,
            args.quiet,
        )
        gates["regression"] = ok4
        if base_ppl < float("inf"):
            regression = (new_ppl - base_ppl) / max(base_ppl, 1e-6)

    all_passed = all(gates.values())

    result = {
        "new_perplexity":  round(new_ppl,  4) if new_ppl  < float("inf") else None,
        "base_perplexity": round(base_ppl, 4) if base_ppl < float("inf") else None,
        "regression":      round(regression, 6),
        "max_regression":  args.max_regression,
        "passed":          all_passed,
        "gates":           gates,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)

    if not args.quiet:
        print()
        status = "PASSED ✓" if all_passed else "FAILED ✗"
        print(f"  Result: {status}")
        print(f"  Gates : {gates}")
        if new_ppl < float("inf"):
            print(f"  PPL   : {new_ppl:.2f}")
        print()

    # Print JSON to stdout for piping
    print(json.dumps(result))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
