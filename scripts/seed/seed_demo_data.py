#!/usr/bin/env python3
"""Generate demo seed data for local development."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write demo seed data as JSON.")
    parser.add_argument("--output", default="storage/seed/demo-data.json", help="Output JSON path.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "organizations": [
            {
                "name": "Demo Analytics Co",
                "slug": "demo-analytics-co",
                "plan": "TEAM",
                "users": [
                    {"email": "admin@example.com", "role": "OWNER"},
                    {"email": "analyst@example.com", "role": "ANALYST"},
                ],
            }
        ],
        "models": [
            {
                "name": "business-intelligence-ai",
                "version": "v0.1.0",
                "status": "active",
            }
        ],
    }

    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
