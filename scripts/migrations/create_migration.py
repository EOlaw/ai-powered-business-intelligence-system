#!/usr/bin/env python3
"""Create a timestamped migration folder with an empty SQL file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def normalize_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a database migration stub.")
    parser.add_argument("name", help="Human-readable migration name.")
    parser.add_argument("--dir", default="packages/database/prisma/migrations", help="Migration directory.")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    migration_dir = Path(args.dir) / f"{timestamp}_{normalize_name(args.name)}"
    migration_dir.mkdir(parents=True, exist_ok=False)

    sql_file = migration_dir / "migration.sql"
    sql_file.write_text("-- Write migration SQL here.\n", encoding="utf-8")
    print(sql_file)


if __name__ == "__main__":
    main()
