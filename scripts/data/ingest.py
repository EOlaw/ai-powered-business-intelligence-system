#!/usr/bin/env python3
"""
InsightSerenity — Data Ingest Script
======================================
Converts raw text from various sources into the canonical JSONL corpus format
required by the training pipeline.

Each output line is a JSON object:
    {"text": "...", "source": "...", "source_type": "...", "ingested_at": "..."}

Supported source types
-----------------------
file          Plain .txt file — the entire file becomes one document.
dir           Directory of .txt files — each file becomes one document.
jsonl         Already-formatted JSONL — validates and copies, adding provenance.
url           Single URL — fetches with httpx and extracts readable text.
url-list      Text file containing one URL per line — fetches each.
wikipedia     Wikipedia XML dump (articles-dump.xml.bz2) — extracts article text.

Output
------
All sources write to a single output JSONL file. Existing output is NOT
overwritten unless --force is passed. Each run also writes a companion
.ingest_log.json recording source counts, error counts, and elapsed time.

Quality pre-filter
------------------
Documents shorter than --min-chars (default: 100) are silently skipped.
Pass --min-chars 0 to disable. The full quality filter (QualityFilterPipeline)
is NOT applied here — run scripts/data/verify.py afterwards for full validation.

Usage
-----
    cd apps/ai-engine

    # From a directory of .txt files
    ./.venv/bin/python scripts/data/ingest.py \\
        --source-type dir \\
        --source     /path/to/text/files/ \\
        --output     storage/datasets/corpus-v1.jsonl

    # From a list of URLs
    ./.venv/bin/python scripts/data/ingest.py \\
        --source-type url-list \\
        --source     scripts/training/seed_urls.txt \\
        --output     storage/datasets/corpus-v1.jsonl

    # From a Wikipedia dump
    ./.venv/bin/python scripts/data/ingest.py \\
        --source-type wikipedia \\
        --source     /data/enwiki-latest-articles.xml.bz2 \\
        --output     storage/datasets/corpus-v1.jsonl \\
        --max-docs   500000

    # Append to an existing corpus
    ./.venv/bin/python scripts/data/ingest.py \\
        --source-type dir \\
        --source     /new/docs/ \\
        --output     storage/datasets/corpus-v1.jsonl \\
        --append
"""

import argparse
import bz2
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET

# ── Engine path setup ─────────────────────────────────────────────────────────
_ENGINE_ROOT = Path(__file__).resolve().parents[2] / "apps" / "ai-engine"
sys.path.insert(0, str(_ENGINE_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest text from various sources into JSONL corpus format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source-type", required=True,
                   choices=["file", "dir", "jsonl", "url", "url-list", "wikipedia"],
                   help="Type of source to ingest.")
    p.add_argument("--source",  required=True, help="Path or URL to the source.")
    p.add_argument("--output",  required=True, help="Output JSONL corpus file.")
    p.add_argument("--append",  action="store_true",
                   help="Append to an existing output file instead of creating a new one.")
    p.add_argument("--force",   action="store_true",
                   help="Overwrite the output file if it already exists.")
    p.add_argument("--min-chars", type=int, default=100,
                   help="Skip documents shorter than this many characters (default: 100).")
    p.add_argument("--max-docs",  type=int, default=None,
                   help="Stop after ingesting this many documents.")
    p.add_argument("--encoding",  default="utf-8",
                   help="Text encoding for .txt files (default: utf-8).")
    p.add_argument("--request-timeout", type=float, default=30.0,
                   help="HTTP request timeout in seconds (default: 30).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Source readers — each yields (text, source_label) pairs
# ─────────────────────────────────────────────────────────────────────────────

def _read_txt_file(path: Path, encoding: str) -> str:
    """Read a text file and return its contents, stripping null bytes."""
    try:
        text = path.read_text(encoding=encoding, errors="replace")
        return text.replace("\x00", "").strip()
    except Exception as e:
        print(f"  WARN: Could not read {path}: {e}", file=sys.stderr)
        return ""


def read_file(source: str, encoding: str) -> Iterator[tuple]:
    """Yield a single document from a plain text file."""
    path = Path(source)
    text = _read_txt_file(path, encoding)
    if text:
        yield text, str(path)


def read_dir(source: str, encoding: str) -> Iterator[tuple]:
    """Yield one document per .txt file found under a directory."""
    root = Path(source)
    if not root.is_dir():
        sys.exit(f"ERROR: --source must be a directory for source-type=dir: {source}")
    for path in sorted(root.rglob("*.txt")):
        text = _read_txt_file(path, encoding)
        if text:
            yield text, str(path.relative_to(root))


def read_jsonl(source: str) -> Iterator[tuple]:
    """
    Yield documents from an existing JSONL file.
    Validates each line, extracts the 'text' field, adds source provenance.
    """
    path = Path(source)
    if not path.exists():
        sys.exit(f"ERROR: JSONL source not found: {source}")

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARN: Line {lineno} is not valid JSON — skipping", file=sys.stderr)
                continue
            text = obj.get("text", "")
            if text:
                yield str(text), f"{path.name}:L{lineno}"


def read_url(url: str, timeout: float) -> Iterator[tuple]:
    """Fetch a single URL and extract its text content."""
    try:
        import httpx
        from src.data.preprocessing.html_extractor import HTMLExtractor

        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "InsightSerenityBot/1.0"})
        resp.raise_for_status()
        extractor = HTMLExtractor()
        text = extractor.extract(resp.text, url=url)
        if text:
            yield text, url
    except ImportError:
        # Fallback: basic urllib if httpx or HTMLExtractor unavailable
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "InsightSerenityBot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Very basic HTML stripping
        import re
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            yield text, url
    except Exception as e:
        print(f"  WARN: Failed to fetch {url}: {e}", file=sys.stderr)


def read_url_list(source: str, timeout: float) -> Iterator[tuple]:
    """Fetch each URL in a newline-delimited file."""
    path = Path(source)
    if not path.exists():
        sys.exit(f"ERROR: URL list not found: {source}")
    urls = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    print(f"  Fetching {len(urls)} URLs...")
    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] {url[:80]}", end="\r")
        yield from read_url(url, timeout)
    print()


def read_wikipedia(source: str, max_docs: int = None) -> Iterator[tuple]:
    """
    Parse a Wikipedia XML dump and yield article text.

    Supports both plain XML and .bz2-compressed dumps.
    Wikipedia dump format: each <page> contains <title> and <text>.
    Filters redirect pages (text starts with '#REDIRECT').

    Download from: https://dumps.wikimedia.org/enwiki/latest/
    Recommended file: enwiki-latest-pages-articles.xml.bz2
    """
    path = Path(source)
    if not path.exists():
        sys.exit(f"ERROR: Wikipedia dump not found: {source}")

    print(f"  Streaming Wikipedia dump: {path.name}")

    # Handle .bz2 compressed dumps transparently
    opener = bz2.open(path, "rb") if path.suffix == ".bz2" else open(path, "rb")

    ns = "http://www.mediawiki.org/xml/export-0.10/"
    title = ""
    count = 0

    with opener as fh:
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            tag = elem.tag.replace(f"{{{ns}}}", "")

            if event == "end" and tag == "title":
                title = elem.text or ""
                elem.clear()

            elif event == "end" and tag == "text":
                raw = (elem.text or "").strip()
                elem.clear()

                # Skip redirects and empty articles
                if not raw or raw.lower().startswith("#redirect"):
                    continue

                # Strip wiki markup (minimal — full parsing needs wikitextprocessor)
                import re
                # Remove templates {{...}}
                text = re.sub(r"\{\{[^}]*\}\}", "", raw)
                # Remove links [[File:...]] and [[Category:...]]
                text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
                # Convert [[link|display]] → display
                text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
                # Remove headings markup == ... ==
                text = re.sub(r"=+\s*(.+?)\s*=+", r"\1", text)
                # Remove bold/italic markup
                text = re.sub(r"'{2,3}", "", text)
                # Collapse whitespace
                text = re.sub(r"\n{3,}", "\n\n", text).strip()

                if len(text) >= 200:  # Skip very short articles
                    yield text, f"wikipedia:{title}"
                    count += 1
                    if count % 10_000 == 0:
                        print(f"  Parsed {count:,} Wikipedia articles...", end="\r")
                    if max_docs and count >= max_docs:
                        break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_path = Path(args.output)

    if output_path.exists() and not args.append and not args.force:
        sys.exit(
            f"ERROR: Output file already exists: {output_path}\n"
            f"Use --append to add to it or --force to overwrite."
        )
    if not args.append:
        output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nInsightSerenity — Data Ingest")
    print(f"  Source type : {args.source_type}")
    print(f"  Source      : {args.source}")
    print(f"  Output      : {args.output}")
    print(f"  Min chars   : {args.min_chars}")

    start_ts  = datetime.now(timezone.utc).isoformat()
    start_wall = time.perf_counter()
    written   = 0
    skipped   = 0
    errors    = 0

    # Choose source reader
    dispatch = {
        "file":      lambda: read_file(args.source, args.encoding),
        "dir":       lambda: read_dir(args.source, args.encoding),
        "jsonl":     lambda: read_jsonl(args.source),
        "url":       lambda: read_url(args.source, args.request_timeout),
        "url-list":  lambda: read_url_list(args.source, args.request_timeout),
        "wikipedia": lambda: read_wikipedia(args.source, args.max_docs),
    }
    reader = dispatch[args.source_type]()

    ingested_at = datetime.now(timezone.utc).isoformat()

    with open(output_path, "a") as out_f:
        for text, source_label in reader:
            if len(text) < args.min_chars:
                skipped += 1
                continue

            # Normalise line endings and strip null bytes
            text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

            record = {
                "text":         text,
                "source":       source_label,
                "source_type":  args.source_type,
                "ingested_at":  ingested_at,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 1_000 == 0:
                print(f"  Wrote {written:,} documents...", end="\r")

            if args.max_docs and written >= args.max_docs:
                break

    elapsed = time.perf_counter() - start_wall
    print(f"\n  ✓ Written  : {written:,} documents")
    print(f"  ✗ Skipped  : {skipped:,} (below min-chars={args.min_chars})")
    if errors:
        print(f"  ! Errors   : {errors}")
    print(f"  Elapsed    : {elapsed:.1f}s")
    print(f"  Output     : {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Write companion ingest log
    log = {
        "source_type":  args.source_type,
        "source":       str(args.source),
        "output":       str(args.output),
        "documents_written": written,
        "documents_skipped": skipped,
        "errors":       errors,
        "min_chars":    args.min_chars,
        "elapsed_s":    round(elapsed, 2),
        "started_at":   start_ts,
        "finished_at":  datetime.now(timezone.utc).isoformat(),
    }
    log_path = output_path.parent / (output_path.stem + ".ingest_log.json")
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Log        : {log_path}")
    print()


if __name__ == "__main__":
    main()
