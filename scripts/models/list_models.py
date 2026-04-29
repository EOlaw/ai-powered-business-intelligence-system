#!/usr/bin/env python3
"""
InsightSerenity — List Promoted Models
=======================================
Prints a formatted table of all models in the production registry
(storage/models/) with their version, perplexity, parameter count,
promotion date, training run, and serving status.

Usage
-----
    cd apps/ai-engine

    # List all models
    ./.venv/bin/python scripts/models/list_models.py

    # Show a specific model
    ./.venv/bin/python scripts/models/list_models.py --name insightserenity-1

    # Output as JSON (for scripting)
    ./.venv/bin/python scripts/models/list_models.py --json

    # Show which version is currently "latest"
    ./.venv/bin/python scripts/models/list_models.py --name insightserenity-1 --show-latest
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "apps" / "ai-engine"
sys.path.insert(0, str(ENGINE_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="List all promoted models in the production registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--name",        default=None, metavar="STR",
                   help="Filter to a specific model name.")
    p.add_argument("--json",        action="store_true",
                   help="Output as JSON array instead of a table.")
    p.add_argument("--show-latest", action="store_true",
                   help="Show which version the 'latest' pointer resolves to.")
    p.add_argument("--models-dir",  default=None, metavar="PATH",
                   help="Override default storage/models/ path.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Registry scanning
# ─────────────────────────────────────────────────────────────────────────────

def scan_registry(models_root: Path, name_filter: str | None) -> list[dict]:
    """
    Walk storage/models/ and collect metadata for every promoted version.

    Returns a list of dicts, one per (model_name, version) pair,
    sorted by (model_name, promoted_at descending).
    """
    if not models_root.exists():
        return []

    records = []
    reserved = {"latest", "previous", "latest.new"}

    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if name_filter and model_dir.name != name_filter:
            continue

        # Find the version the "latest" pointer points to
        latest_dir   = model_dir / "latest"
        latest_version = None
        if latest_dir.exists():
            meta_path = latest_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    latest_version = json.load(f).get("version")

        for version_dir in sorted(model_dir.iterdir(), reverse=True):
            if not version_dir.is_dir():
                continue
            if version_dir.name in reserved:
                continue

            # Read metadata
            meta_path = version_dir / "metadata.json"
            meta      = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            model_pt_exists   = (version_dir / "model.pt").exists()
            config_json_exists = (version_dir / "config.json").exists()
            tok_exists        = (version_dir / "tokenizer").is_dir()
            complete          = model_pt_exists and config_json_exists and tok_exists

            # Calculate model size on disk
            size_bytes = 0
            model_pt = version_dir / "model.pt"
            if model_pt.exists():
                size_bytes = model_pt.stat().st_size

            records.append({
                "model_name":    model_dir.name,
                "version":       meta.get("version", version_dir.name),
                "model_type":    meta.get("model_type", "?"),
                "param_count":   meta.get("param_count"),
                "vocab_size":    meta.get("vocab_size"),
                "d_model":       meta.get("d_model"),
                "max_seq_len":   meta.get("max_seq_len"),
                "perplexity":    meta.get("perplexity"),
                "train_loss":    meta.get("train_loss"),
                "global_step":   meta.get("global_step"),
                "training_run":  meta.get("training_run"),
                "promoted_at":   meta.get("promoted_at"),
                "checksum_sha256": meta.get("checksum_sha256"),
                "capabilities":  meta.get("capabilities", []),
                "is_latest":     (meta.get("version") == latest_version),
                "artifact_complete": complete,
                "size_mb":       round(size_bytes / 1e6, 1),
                "path":          str(version_dir),
            })

    # Sort: model_name asc, then promoted_at desc (newest first per model)
    records.sort(key=lambda r: (r["model_name"], r.get("promoted_at") or "", r["version"]),
                 reverse=False)
    # Re-sort within each model: newest first
    from itertools import groupby
    final = []
    for name, group in groupby(records, key=lambda r: r["model_name"]):
        group_list = list(group)
        group_list.sort(key=lambda r: r.get("promoted_at") or "", reverse=True)
        final.extend(group_list)

    return final


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_params(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)

def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "?"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16]

def _fmt_ppl(ppl: float | None) -> str:
    if ppl is None:
        return "?"
    return f"{ppl:.2f}"


def print_table(records: list[dict]) -> None:
    """Print a formatted ASCII table."""
    if not records:
        print("  No promoted models found in storage/models/")
        return

    COL_W = {
        "Model":       20,
        "Version":     14,
        "Type":        6,
        "Params":      8,
        "PPL":         8,
        "Size":        8,
        "Latest":      7,
        "Promoted":    17,
        "Run":         18,
    }

    header = (
        f"{'Model':<{COL_W['Model']}}"
        f"{'Version':<{COL_W['Version']}}"
        f"{'Type':<{COL_W['Type']}}"
        f"{'Params':<{COL_W['Params']}}"
        f"{'PPL':<{COL_W['PPL']}}"
        f"{'Size':<{COL_W['Size']}}"
        f"{'Latest':<{COL_W['Latest']}}"
        f"{'Promoted':<{COL_W['Promoted']}}"
        f"{'Run':<{COL_W['Run']}}"
        f"  OK?"
    )
    sep = "─" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for r in records:
        latest_mark = "✓ " if r["is_latest"] else "  "
        ok_mark     = "✓" if r["artifact_complete"] else "✗ INCOMPLETE"
        print(
            f"{r['model_name']:<{COL_W['Model']}}"
            f"{r['version']:<{COL_W['Version']}}"
            f"{r['model_type']:<{COL_W['Type']}}"
            f"{_fmt_params(r['param_count']):<{COL_W['Params']}}"
            f"{_fmt_ppl(r['perplexity']):<{COL_W['PPL']}}"
            f"{str(r['size_mb']) + ' MB':<{COL_W['Size']}}"
            f"{latest_mark:<{COL_W['Latest']}}"
            f"{_fmt_date(r['promoted_at']):<{COL_W['Promoted']}}"
            f"{(r['training_run'] or '?'):<{COL_W['Run']}}"
            f"  {ok_mark}"
        )

    print(sep)
    print(f"\n  {len(records)} version(s) listed.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    models_root = (
        Path(args.models_dir).resolve()
        if args.models_dir
        else ENGINE_ROOT / "storage" / "models"
    )

    records = scan_registry(models_root, args.name)

    if args.json:
        print(json.dumps(records, indent=2))
        return

    print("\nInsightSerenity — Production Model Registry")
    print(f"Registry path : {models_root}")

    print_table(records)

    if args.show_latest and args.name:
        model_dir = models_root / args.name
        latest    = model_dir / "latest" / "metadata.json"
        if latest.exists():
            with open(latest) as f:
                meta = json.load(f)
            print(f"  '{args.name}' latest → {meta.get('version', '?')}")
            if meta.get("perplexity"):
                print(f"  PPL: {meta['perplexity']}")
        else:
            print(f"  No 'latest' pointer found for '{args.name}'")


if __name__ == "__main__":
    main()
