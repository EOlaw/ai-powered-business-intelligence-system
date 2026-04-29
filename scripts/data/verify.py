#!/usr/bin/env python3
"""
InsightSerenity — Corpus Verification Gate
==========================================
Validates a JSONL corpus against hard quality gates before it is used for
training. This script MUST pass before running train.py on a new corpus.

It is the authoritative pre-training data quality check. Call it from CI
pipelines and the data flywheel worker before triggering a training run.

Quality gates (enforced — script exits 1 if any fail)
------------------------------------------------------
GATE 1  JSONL integrity   Every line must parse as valid JSON.
GATE 2  Text field        Every record must have a non-empty 'text' field.
GATE 3  No BOM            No UTF-8 BOM at the start of the file.
GATE 4  No null bytes     No null bytes in any document text.
GATE 5  Min documents     At least MIN_DOCS documents (default: 1,000).
GATE 6  Min tokens        At least MIN_TOKENS approximate tokens (default: 500,000).
GATE 7  Empty doc rate    No more than MAX_EMPTY_RATE empty/near-empty docs (default: 5%).
GATE 8  Non-ASCII rate    No more than MAX_NON_ASCII_RATE non-ASCII-heavy docs (default: 10%).

Thresholds can be overridden via CLI flags for smaller experimental datasets.

Output
------
Prints a gate-by-gate report with PASS / FAIL for each gate.
Writes an optional JSON result to --output-json.
Exit code 0 = all gates passed. Exit code 1 = one or more failed.

Usage
-----
    cd apps/ai-engine

    # Standard pre-training check
    ./.venv/bin/python scripts/data/verify.py \\
        --corpus storage/datasets/corpus-v1.jsonl

    # CI usage — write result to JSON for downstream scripts
    ./.venv/bin/python scripts/data/verify.py \\
        --corpus storage/datasets/corpus-v1.jsonl \\
        --output-json /tmp/verify_result.json

    # Smaller thresholds for experimental datasets
    ./.venv/bin/python scripts/data/verify.py \\
        --corpus storage/datasets/corpus-v1.jsonl \\
        --min-docs 100 \\
        --min-tokens 10000

    # Fail fast on first error (useful in pipelines)
    ./.venv/bin/python scripts/data/verify.py \\
        --corpus storage/datasets/corpus-v1.jsonl \\
        --fail-fast
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Engine path setup ─────────────────────────────────────────────────────────
_ENGINE_ROOT = Path(__file__).resolve().parents[2] / "apps" / "ai-engine"
sys.path.insert(0, str(_ENGINE_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Default thresholds
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_DOCS        = 1_000
DEFAULT_MIN_TOKENS      = 500_000
DEFAULT_MAX_EMPTY_RATE  = 0.05    # 5%
DEFAULT_MAX_NON_ASCII   = 0.10    # 10%


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a JSONL corpus against pre-training quality gates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus",          required=True,
                   help="Path to the JSONL corpus to verify.")
    p.add_argument("--output-json",     default=None,
                   help="Write JSON result to this path.")
    p.add_argument("--min-docs",        type=int,   default=DEFAULT_MIN_DOCS,
                   help=f"Minimum document count (default: {DEFAULT_MIN_DOCS:,}).")
    p.add_argument("--min-tokens",      type=int,   default=DEFAULT_MIN_TOKENS,
                   help=f"Minimum approximate token count (default: {DEFAULT_MIN_TOKENS:,}).")
    p.add_argument("--max-empty-rate",  type=float, default=DEFAULT_MAX_EMPTY_RATE,
                   help=f"Max fraction of near-empty docs (default: {DEFAULT_MAX_EMPTY_RATE}).")
    p.add_argument("--max-non-ascii",   type=float, default=DEFAULT_MAX_NON_ASCII,
                   help=f"Max fraction of non-ASCII-heavy docs (default: {DEFAULT_MAX_NON_ASCII}).")
    p.add_argument("--fail-fast",       action="store_true",
                   help="Stop checking after the first gate failure.")
    p.add_argument("--quiet",           action="store_true",
                   help="Suppress per-gate output. Only print final result.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Single-pass corpus scan
# ─────────────────────────────────────────────────────────────────────────────

def scan(corpus_path: Path, fail_fast: bool, quiet: bool) -> dict:
    """
    Read the corpus once and collect all statistics needed by the gates.
    Returns a stats dict with all raw counts.
    """
    stats = {
        "total_lines":       0,
        "parse_errors":      [],   # List of (lineno, error_msg)
        "missing_text":      [],   # List of lineno
        "null_byte_docs":    [],   # List of lineno
        "total_docs":        0,
        "total_words":       0,
        "empty_docs":        0,    # len(text.strip()) == 0
        "near_empty_docs":   0,    # len(text.strip()) < 20
        "non_ascii_docs":    0,
        "has_bom":           False,
        "first_parse_error": None,
        "first_missing_text": None,
    }

    with open(corpus_path, "rb") as fb:
        first_bytes = fb.read(3)
        stats["has_bom"] = first_bytes == b"\xef\xbb\xbf"

    with open(corpus_path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            # Strip BOM from first line if present
            if lineno == 1:
                line = line.lstrip("﻿")

            stats["total_lines"] += 1
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Gate 1: JSONL parse
            try:
                obj = json.loads(line_stripped)
            except json.JSONDecodeError as e:
                stats["parse_errors"].append((lineno, str(e)))
                if stats["first_parse_error"] is None:
                    stats["first_parse_error"] = (lineno, str(e))
                if fail_fast and stats["parse_errors"]:
                    break
                continue

            # Gate 2: text field
            text = obj.get("text")
            if text is None or not isinstance(text, str):
                stats["missing_text"].append(lineno)
                if stats["first_missing_text"] is None:
                    stats["first_missing_text"] = lineno
                continue

            stats["total_docs"] += 1
            text_stripped = text.strip()
            text_len      = len(text_stripped)

            # Gate 4: null bytes
            if "\x00" in text:
                stats["null_byte_docs"].append(lineno)

            # Gate 7: empty / near-empty
            if text_len == 0:
                stats["empty_docs"]    += 1
                stats["near_empty_docs"] += 1
            elif text_len < 20:
                stats["near_empty_docs"] += 1

            # Gate 8: non-ASCII rate
            sample = text_stripped[:200]
            if sample:
                ascii_cnt = sum(1 for c in sample if ord(c) < 128)
                if ascii_cnt / len(sample) < 0.5:
                    stats["non_ascii_docs"] += 1

            # Token approximation (whitespace word count × 1.3 for BPE estimate)
            stats["total_words"] += len(text.split())

            if stats["total_docs"] % 10_000 == 0 and not quiet:
                print(f"  Scanning... {stats['total_docs']:,} docs", end="\r")

    if not quiet:
        print(f"  Scanned {stats['total_docs']:,} documents.            ")

    stats["total_tokens_approx"] = int(stats["total_words"] * 1.3)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Gate evaluations
# ─────────────────────────────────────────────────────────────────────────────

def run_gates(stats: dict, args: argparse.Namespace) -> list:
    """
    Evaluate each quality gate and return a list of gate result dicts.
    Each dict has: id, name, passed, message.
    """
    gates = []

    def gate(gate_id: str, name: str, passed: bool, message: str):
        gates.append({
            "id":      gate_id,
            "name":    name,
            "passed":  passed,
            "message": message,
        })
        if not args.quiet:
            icon = "✓" if passed else "✗"
            print(f"  {icon} Gate {gate_id}: {name}")
            if not passed:
                print(f"      FAILED — {message}")
        if not passed and args.fail_fast:
            raise StopIteration

    n   = stats["total_docs"]
    n_e = stats["parse_errors"]
    n_m = stats["missing_text"]

    try:
        # GATE 1: JSONL integrity
        if n_e:
            first_line, first_msg = stats["first_parse_error"]
            gate("1", "JSONL integrity", False,
                 f"{len(n_e)} lines failed to parse. First error: line {first_line}: {first_msg[:80]}")
        else:
            gate("1", "JSONL integrity", True, f"All {stats['total_lines']:,} lines valid JSON")

        # GATE 2: text field present
        if n_m:
            gate("2", "Text field", False,
                 f"{len(n_m)} records missing 'text' field. First: line {stats['first_missing_text']}")
        else:
            gate("2", "Text field", True, f"All {n:,} records have 'text' field")

        # GATE 3: BOM check
        gate("3", "No UTF-8 BOM", not stats["has_bom"],
             "File starts with UTF-8 BOM — strip with: sed -i '1s/^\xef\xbb\xbf//' corpus.jsonl"
             if stats["has_bom"] else "No BOM detected")

        # GATE 4: null bytes
        null_count = len(stats["null_byte_docs"])
        gate("4", "No null bytes", null_count == 0,
             f"{null_count} documents contain null bytes"
             if null_count else "No null bytes found")

        # GATE 5: minimum document count
        gate("5", "Minimum document count", n >= args.min_docs,
             f"{n:,} documents < minimum {args.min_docs:,}"
             if n < args.min_docs
             else f"{n:,} documents ≥ minimum {args.min_docs:,}")

        # GATE 6: minimum token count
        tok = stats["total_tokens_approx"]
        gate("6", "Minimum token count", tok >= args.min_tokens,
             f"~{tok:,} tokens < minimum {args.min_tokens:,} "
             f"(need {(args.min_tokens - tok):,} more words)"
             if tok < args.min_tokens
             else f"~{tok:,} approximate tokens ≥ minimum {args.min_tokens:,}")

        # GATE 7: near-empty rate
        near_rate = stats["near_empty_docs"] / max(n, 1)
        gate("7", "Near-empty document rate", near_rate <= args.max_empty_rate,
             f"{near_rate * 100:.2f}% near-empty docs > threshold {args.max_empty_rate * 100:.1f}%"
             if near_rate > args.max_empty_rate
             else f"{near_rate * 100:.2f}% near-empty ≤ {args.max_empty_rate * 100:.1f}% threshold")

        # GATE 8: non-ASCII rate
        ascii_rate = stats["non_ascii_docs"] / max(n, 1)
        gate("8", "Non-ASCII document rate", ascii_rate <= args.max_non_ascii,
             f"{ascii_rate * 100:.2f}% non-ASCII-heavy docs > threshold {args.max_non_ascii * 100:.1f}%"
             if ascii_rate > args.max_non_ascii
             else f"{ascii_rate * 100:.2f}% non-ASCII ≤ {args.max_non_ascii * 100:.1f}% threshold")

    except StopIteration:
        pass   # fail_fast triggered — remaining gates not evaluated

    return gates


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = parse_args()
    corpus_path = Path(args.corpus)

    if not corpus_path.exists():
        print(f"ERROR: Corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(2)

    if not args.quiet:
        print(f"\nInsightSerenity — Corpus Verification")
        print(f"  Corpus     : {corpus_path.name} ({corpus_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  Min docs   : {args.min_docs:,}")
        print(f"  Min tokens : {args.min_tokens:,}")
        print()

    stats = scan(corpus_path, fail_fast=args.fail_fast, quiet=args.quiet)

    if not args.quiet:
        print()

    gates      = run_gates(stats, args)
    all_passed = all(g["passed"] for g in gates)
    n_failed   = sum(1 for g in gates if not g["passed"])
    n_passed   = sum(1 for g in gates if g["passed"])

    if not args.quiet:
        print()
        if all_passed:
            print(f"  RESULT: ALL {n_passed} GATES PASSED ✓  — corpus is ready for training")
        else:
            print(f"  RESULT: {n_failed} GATE(S) FAILED ✗  — fix issues before training")
        print()

    # JSON output
    result = {
        "corpus":      str(corpus_path),
        "passed":      all_passed,
        "n_passed":    n_passed,
        "n_failed":    n_failed,
        "gates":       gates,
        "stats": {
            "total_docs":         stats["total_docs"],
            "total_tokens_approx": stats["total_tokens_approx"],
            "parse_errors":       len(stats["parse_errors"]),
            "missing_text":       len(stats["missing_text"]),
            "empty_docs":         stats["empty_docs"],
            "near_empty_docs":    stats["near_empty_docs"],
            "non_ascii_docs":     stats["non_ascii_docs"],
            "has_bom":            stats["has_bom"],
            "null_byte_docs":     len(stats["null_byte_docs"]),
        },
        "thresholds": {
            "min_docs":       args.min_docs,
            "min_tokens":     args.min_tokens,
            "max_empty_rate": args.max_empty_rate,
            "max_non_ascii":  args.max_non_ascii,
        },
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }

    print(json.dumps(result))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
