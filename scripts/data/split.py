#!/usr/bin/env python3
"""
InsightSerenity — Corpus Split Script
=======================================
Deterministically splits a JSONL corpus into train / validation / test sets.

This is the standalone split tool. The same split logic is also embedded in
scripts/training/train.py (via --val-split / --test-split) for convenience,
but this script gives you an explicit, auditable split step that can be
re-run independently.

Split strategy
--------------
1. Count all documents in the source corpus.
2. Shuffle with a fixed random seed (deterministic — same seed = same split).
3. Assign the first N_test documents to test, the next N_val to validation,
   the remainder to training.
4. Write three JSONL files and a dataset_card.json alongside them.

The dataset card records every parameter so the split is reproducible:
    dataset_card.json: source, split sizes, ratios, seed, MD5, timestamps.

Usage
-----
    cd apps/ai-engine

    # 90% train / 5% val / 5% test
    ./.venv/bin/python scripts/data/split.py \\
        --corpus    storage/datasets/corpus-v1.jsonl \\
        --val-split  0.05 \\
        --test-split 0.05 \\
        --seed       42

    # 95% train / 5% val, no test split
    ./.venv/bin/python scripts/data/split.py \\
        --corpus    storage/datasets/corpus-v1.jsonl \\
        --val-split 0.05

    # Custom output prefix
    ./.venv/bin/python scripts/data/split.py \\
        --corpus    storage/datasets/corpus-v1.jsonl \\
        --val-split 0.05 \\
        --output-prefix storage/datasets/splits/corpus-v1
"""

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a JSONL corpus into train/val/test sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus",        required=True,
                   help="Source JSONL corpus to split.")
    p.add_argument("--val-split",     type=float, default=0.05,
                   help="Fraction for validation (default: 0.05 = 5%%).")
    p.add_argument("--test-split",    type=float, default=0.0,
                   help="Fraction for test set (default: 0 = no test split).")
    p.add_argument("--seed",          type=int,   default=42,
                   help="Random seed for deterministic shuffle (default: 42).")
    p.add_argument("--output-prefix", default=None,
                   help="Output path prefix. Default: same dir + stem as corpus.")
    p.add_argument("--force",         action="store_true",
                   help="Overwrite existing output files.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Split logic
# ─────────────────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def split_corpus(
    corpus_path: Path,
    val_split:   float,
    test_split:  float,
    seed:        int,
    output_prefix: str | None,
    force:       bool,
) -> dict:
    """
    Read the corpus, shuffle, and write train/val/test splits.
    Returns a dataset card dict.
    """
    if val_split + test_split >= 1.0:
        sys.exit("ERROR: val-split + test-split must be < 1.0")
    if val_split < 0 or test_split < 0:
        sys.exit("ERROR: split fractions must be >= 0")

    # ── Read all lines ──────────────────────────────────────────────────────
    print(f"\n  Reading corpus: {corpus_path.name} ...", end=" ", flush=True)
    with open(corpus_path) as f:
        lines = [l for l in f if l.strip()]
    print(f"{len(lines):,} lines")

    if not lines:
        sys.exit("ERROR: Corpus is empty.")

    # ── Shuffle deterministically ───────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(lines)

    # ── Compute split sizes ─────────────────────────────────────────────────
    n        = len(lines)
    n_test   = math.ceil(n * test_split)  if test_split  > 0 else 0
    n_val    = math.ceil(n * val_split)   if val_split   > 0 else 0
    n_train  = n - n_test - n_val

    if n_train <= 0:
        sys.exit(
            f"ERROR: Not enough documents for the requested splits.\n"
            f"  Total: {n}, train: {n_train}, val: {n_val}, test: {n_test}\n"
            f"  Reduce --val-split / --test-split or use a larger corpus."
        )

    # ── Determine output paths ──────────────────────────────────────────────
    if output_prefix:
        prefix = Path(output_prefix)
    else:
        prefix = corpus_path.parent / corpus_path.stem

    train_path = Path(str(prefix) + ".train.jsonl")
    val_path   = Path(str(prefix) + ".val.jsonl")   if n_val  > 0 else None
    test_path  = Path(str(prefix) + ".test.jsonl")  if n_test > 0 else None

    for p in [train_path, val_path, test_path]:
        if p and p.exists() and not force:
            sys.exit(
                f"ERROR: Output already exists: {p}\n"
                f"Use --force to overwrite."
            )

    # ── Write splits ────────────────────────────────────────────────────────
    test_lines  = lines[:n_test]
    val_lines   = lines[n_test:n_test + n_val]
    train_lines = lines[n_test + n_val:]

    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text("".join(train_lines))
    print(f"  Train : {len(train_lines):>8,} docs → {train_path.name}")

    if val_path:
        val_path.write_text("".join(val_lines))
        print(f"  Val   : {len(val_lines):>8,} docs → {val_path.name}")

    if test_path:
        test_path.write_text("".join(test_lines))
        print(f"  Test  : {len(test_lines):>8,} docs → {test_path.name}")

    # ── Dataset card ────────────────────────────────────────────────────────
    card = {
        "name":            corpus_path.stem,
        "source_file":     str(corpus_path),
        "train_path":      str(train_path),
        "val_path":        str(val_path)   if val_path  else None,
        "test_path":       str(test_path)  if test_path else None,
        "doc_count":       n,
        "train_docs":      len(train_lines),
        "val_docs":        len(val_lines),
        "test_docs":       len(test_lines),
        "val_split":       val_split,
        "test_split":      test_split,
        "train_fraction":  round(len(train_lines) / n, 4),
        "seed":            seed,
        "source_checksum_md5":  _md5(corpus_path),
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }

    card_path = Path(str(prefix) + ".dataset_card.json")
    card_path.write_text(json.dumps(card, indent=2))
    print(f"  Card  : {card_path.name}")

    return card


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"ERROR: Corpus file not found: {args.corpus}")

    print("\nInsightSerenity — Corpus Split")
    print(f"  Corpus     : {args.corpus}")
    print(f"  Val split  : {args.val_split * 100:.1f}%")
    print(f"  Test split : {args.test_split * 100:.1f}%")
    print(f"  Seed       : {args.seed}")

    card = split_corpus(
        corpus_path=corpus_path,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        output_prefix=args.output_prefix,
        force=args.force,
    )

    print(f"\n  Split complete.")
    print(f"  Train: {card['train_docs']:,} | Val: {card['val_docs']:,} | Test: {card['test_docs']:,}")
    print()


if __name__ == "__main__":
    main()
