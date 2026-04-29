#!/usr/bin/env python3
"""
InsightSerenity — Corpus Inspection Script
===========================================
Produces a comprehensive statistical report on a JSONL corpus.
Run this before training to understand the data you are training on.

Report sections
---------------
1. Overview         — document count, total characters, total approx tokens
2. Length stats     — min / median / p90 / p99 / max document length (chars)
3. Token estimate   — whitespace-token count distribution
4. Quality signals  — empty-doc rate, near-empty rate, non-ASCII rate
5. Language         — detected language distribution (top-10)
6. Source breakdown — top-10 source_type values and source domain counts
7. Sample docs      — 3 random document excerpts (for eyeballing quality)

Outputs
-------
Always prints a human-readable summary.
With --output-json <path>: also writes a machine-readable JSON report
that can be consumed by CI gates and scripts.

Usage
-----
    cd apps/ai-engine

    # Quick human-readable report
    ./.venv/bin/python scripts/data/inspect.py \\
        --corpus storage/datasets/corpus-v1.jsonl

    # Full report including language detection (slower)
    ./.venv/bin/python scripts/data/inspect.py \\
        --corpus storage/datasets/corpus-v1.jsonl \\
        --detect-language \\
        --output-json storage/datasets/corpus-v1.report.json

    # Sample only 10,000 docs for a quick estimate on large corpora
    ./.venv/bin/python scripts/data/inspect.py \\
        --corpus storage/datasets/corpus-v1.jsonl \\
        --max-docs 10000
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ── Engine path setup ─────────────────────────────────────────────────────────
_ENGINE_ROOT = Path(__file__).resolve().parents[2] / "apps" / "ai-engine"
sys.path.insert(0, str(_ENGINE_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a JSONL corpus and produce a statistical report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus",          required=True, help="Path to the JSONL corpus file.")
    p.add_argument("--output-json",     default=None,  help="Write JSON report to this path.")
    p.add_argument("--detect-language", action="store_true",
                   help="Run language detection on each document (slower).")
    p.add_argument("--max-docs",        type=int, default=None,
                   help="Inspect only the first N documents (for large corpora).")
    p.add_argument("--sample-count",    type=int, default=3,
                   help="Number of random sample documents to show (default: 3).")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def percentile(sorted_list: list, p: float) -> float:
    """Return the p-th percentile (0–100) of a sorted list."""
    if not sorted_list:
        return 0.0
    idx = (len(sorted_list) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_list) - 1)
    frac = idx - lo
    return sorted_list[lo] * (1 - frac) + sorted_list[hi] * frac


def approx_word_count(text: str) -> int:
    """Fast approximate word count via whitespace splitting."""
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Language detection (optional, lightweight)
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect the dominant language of a text snippet."""
    try:
        from src.data.quality_filter.language_detector import LanguageDetector
        detector = LanguageDetector()
        lang, _conf = detector.detect(text[:500])
        return lang or "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Corpus scan
# ─────────────────────────────────────────────────────────────────────────────

def scan_corpus(args: argparse.Namespace) -> dict:
    """
    Single-pass scan of the corpus file, collecting all statistics.
    """
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"ERROR: Corpus not found: {args.corpus}")

    print(f"  Scanning: {corpus_path.name} ({corpus_path.stat().st_size / 1e6:.1f} MB)")

    rng = random.Random(args.seed)

    # Accumulators
    total_docs    = 0
    total_chars   = 0
    total_words   = 0
    empty_docs    = 0       # len(text) == 0
    near_empty    = 0       # len(text) < 20
    non_ascii     = 0       # docs where >50% chars are non-ASCII
    parse_errors  = 0
    missing_text  = 0

    char_lengths  = []      # Document char lengths (sampled for p-tiles)
    word_counts   = []      # Approx word counts (sampled)
    lang_counter  = Counter()
    source_types  = Counter()
    domains       = Counter()
    samples       = []      # Random doc excerpts for display

    # Reservoir sampling for docs (to compute percentiles on large corpora)
    RESERVOIR_SIZE = 50_000

    lang_detector_loaded = args.detect_language

    with open(corpus_path) as f:
        for lineno, line in enumerate(f, 1):
            if args.max_docs and total_docs >= args.max_docs:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            text = obj.get("text")
            if text is None:
                missing_text += 1
                continue

            text = str(text)
            total_docs  += 1
            char_len     = len(text)
            word_cnt     = approx_word_count(text)
            total_chars += char_len
            total_words += word_cnt

            if char_len == 0:
                empty_docs += 1
            if char_len < 20:
                near_empty += 1

            # Non-ASCII detection: sample the first 200 chars
            sample_chars = text[:200]
            ascii_count  = sum(1 for c in sample_chars if ord(c) < 128)
            if len(sample_chars) > 0 and ascii_count / len(sample_chars) < 0.5:
                non_ascii += 1

            # Length reservoir sampling
            if len(char_lengths) < RESERVOIR_SIZE:
                char_lengths.append(char_len)
                word_counts.append(word_cnt)
            elif rng.random() < RESERVOIR_SIZE / total_docs:
                idx = rng.randint(0, RESERVOIR_SIZE - 1)
                char_lengths[idx] = char_len
                word_counts[idx]  = word_cnt

            # Language detection (optional)
            if lang_detector_loaded:
                lang = detect_language(text)
                lang_counter[lang] += 1

            # Source metadata
            source_types[obj.get("source_type", "unknown")] += 1
            src = obj.get("source", "")
            if src.startswith("http"):
                # Extract domain
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(src).netloc
                    if domain:
                        domains[domain] += 1
                except Exception:
                    pass

            # Reservoir sample for display
            if len(samples) < args.sample_count:
                samples.append(text)
            elif rng.random() < args.sample_count / total_docs:
                samples[rng.randint(0, args.sample_count - 1)] = text

            if total_docs % 10_000 == 0:
                print(f"  Scanned {total_docs:,} documents...", end="\r")

    print(f"  Scanned {total_docs:,} documents.      ")

    char_lengths.sort()
    word_counts.sort()

    return {
        "corpus_file":        str(corpus_path),
        "corpus_size_mb":     round(corpus_path.stat().st_size / 1e6, 2),
        "scanned_at":         datetime.now(timezone.utc).isoformat(),
        "total_docs":         total_docs,
        "total_chars":        total_chars,
        "total_words_approx": total_words,
        "total_tokens_approx": int(total_words * 1.3),  # BPE typically ~1.3x word count
        "parse_errors":       parse_errors,
        "missing_text_field": missing_text,
        "empty_docs":         empty_docs,
        "near_empty_docs":    near_empty,
        "non_ascii_docs":     non_ascii,
        "empty_rate":         round(empty_docs  / max(total_docs, 1), 4),
        "near_empty_rate":    round(near_empty  / max(total_docs, 1), 4),
        "non_ascii_rate":     round(non_ascii   / max(total_docs, 1), 4),
        "char_length_stats": {
            "min":    min(char_lengths) if char_lengths else 0,
            "p25":    int(percentile(char_lengths, 25)),
            "median": int(percentile(char_lengths, 50)),
            "p75":    int(percentile(char_lengths, 75)),
            "p90":    int(percentile(char_lengths, 90)),
            "p99":    int(percentile(char_lengths, 99)),
            "max":    max(char_lengths) if char_lengths else 0,
            "mean":   int(total_chars / max(total_docs, 1)),
        },
        "word_count_stats": {
            "min":    min(word_counts) if word_counts else 0,
            "median": int(percentile(word_counts, 50)),
            "p90":    int(percentile(word_counts, 90)),
            "max":    max(word_counts) if word_counts else 0,
            "mean":   int(total_words / max(total_docs, 1)),
        },
        "language_distribution": dict(lang_counter.most_common(10)),
        "source_type_counts":    dict(source_types.most_common()),
        "top_domains":           dict(domains.most_common(10)),
        "samples":               [s[:400] + "..." if len(s) > 400 else s for s in samples],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print report
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 30) -> str:
    filled = int(width * value / max(max_val, 1))
    return "█" * filled + "░" * (width - filled)


def print_report(r: dict) -> None:
    sep = "─" * 60

    print(f"\n{sep}")
    print(f"  CORPUS INSPECTION REPORT")
    print(f"  {r['corpus_file']}")
    print(sep)

    print(f"\n  Overview")
    print(f"    Documents      : {r['total_docs']:>12,}")
    print(f"    Total chars    : {r['total_chars']:>12,}")
    print(f"    Approx words   : {r['total_words_approx']:>12,}")
    print(f"    Approx tokens  : {r['total_tokens_approx']:>12,}  (words × 1.3)")
    print(f"    File size      : {r['corpus_size_mb']:>12.1f} MB")

    cs = r["char_length_stats"]
    print(f"\n  Document length (characters)")
    print(f"    Min            : {cs['min']:>8,}")
    print(f"    Median (p50)   : {cs['median']:>8,}")
    print(f"    p90            : {cs['p90']:>8,}")
    print(f"    p99            : {cs['p99']:>8,}")
    print(f"    Max            : {cs['max']:>8,}")
    print(f"    Mean           : {cs['mean']:>8,}")

    ws = r["word_count_stats"]
    print(f"\n  Document length (words)")
    print(f"    Median         : {ws['median']:>8,}")
    print(f"    p90            : {ws['p90']:>8,}")
    print(f"    Max            : {ws['max']:>8,}")

    print(f"\n  Quality signals")
    empty_pct     = r['empty_rate']     * 100
    near_emp_pct  = r['near_empty_rate'] * 100
    non_ascii_pct = r['non_ascii_rate']  * 100
    flag_empty    = " ⚠ HIGH" if empty_pct     > 1.0  else ""
    flag_near     = " ⚠ HIGH" if near_emp_pct  > 5.0  else ""
    flag_ascii    = " ⚠ HIGH" if non_ascii_pct > 10.0 else ""
    print(f"    Empty docs     : {r['empty_docs']:>8,}  ({empty_pct:.2f}%){flag_empty}")
    print(f"    Near-empty (<20 chars): {r['near_empty_docs']:>5,}  ({near_emp_pct:.2f}%){flag_near}")
    print(f"    Non-ASCII heavy: {r['non_ascii_docs']:>8,}  ({non_ascii_pct:.2f}%){flag_ascii}")
    if r['parse_errors']:
        print(f"    JSONL errors   : {r['parse_errors']:>8,}  ⚠ CHECK THESE")
    if r['missing_text_field']:
        print(f"    Missing 'text' : {r['missing_text_field']:>8,}  ⚠ CHECK THESE")

    if r["language_distribution"]:
        print(f"\n  Language distribution (top-10)")
        total_lang = sum(r["language_distribution"].values())
        for lang, cnt in list(r["language_distribution"].items())[:10]:
            pct = cnt / max(total_lang, 1) * 100
            print(f"    {lang:<6}  {cnt:>8,}  {_bar(cnt, total_lang, 20)}  {pct:.1f}%")

    if r["source_type_counts"]:
        print(f"\n  Source types")
        for st, cnt in r["source_type_counts"].items():
            print(f"    {st:<15} {cnt:>8,}")

    if r["top_domains"]:
        print(f"\n  Top domains")
        for domain, cnt in list(r["top_domains"].items())[:5]:
            print(f"    {domain:<35} {cnt:>6,}")

    if r["samples"]:
        print(f"\n  Sample documents ({len(r['samples'])} random)")
        for i, sample in enumerate(r["samples"], 1):
            preview = sample[:200].replace("\n", " ")
            print(f"\n  [{i}] {preview}...")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    report = scan_corpus(args)

    print_report(report)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Remove samples from JSON output (they can be large)
        json_report = {k: v for k, v in report.items() if k != "samples"}
        out.write_text(json.dumps(json_report, indent=2))
        print(f"  JSON report saved: {args.output_json}")


if __name__ == "__main__":
    main()
