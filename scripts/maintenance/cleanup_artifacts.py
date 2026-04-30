#!/usr/bin/env python3
"""Remove old generated artifacts while keeping recent files for debugging."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


DEFAULT_PATTERNS = ("*.pt", "*.pth", "*.ckpt", "*.jsonl", "*.parquet", "*.log")


def should_delete(path: Path, older_than_days: int) -> bool:
    cutoff = time.time() - older_than_days * 24 * 60 * 60
    return path.is_file() and path.stat().st_mtime < cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean old local artifacts.")
    parser.add_argument("--root", default=".", help="Directory to scan.")
    parser.add_argument("--older-than-days", type=int, default=14, help="Minimum file age.")
    parser.add_argument("--dry-run", action="store_true", help="Print files without deleting.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    deleted = 0

    for pattern in DEFAULT_PATTERNS:
        for path in root.rglob(pattern):
            if not should_delete(path, args.older_than_days):
                continue

            print(path)
            if not args.dry_run:
                path.unlink()
            deleted += 1

    print(f"matched={deleted}")


if __name__ == "__main__":
    main()
