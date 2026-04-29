#!/usr/bin/env python3
"""
InsightSerenity — Model Rollback Script
========================================
Atomically rolls back a model's "latest" pointer to a previous version.

When to use
-----------
A newly promoted model starts producing bad outputs or higher error rates.
This script restores the previous version instantly without retraining.

Rollback is immediate — the next request to the AI engine (after a reload)
will use the rolled-back version. No restart of the AI engine process is
needed when using the hot-reload admin endpoint.

How it works
------------
Every time promote.py runs, it saves the previous latest as "previous/"
inside the model's directory. rollback.py swaps latest ↔ previous atomically
using the same rename approach.

For rollback to a specific non-adjacent version, use --to <version>.

After rollback, write a rollback_record.json inside the latest/ directory
so the audit trail shows what happened and why.

Usage
-----
    cd apps/ai-engine

    # Roll back to the automatically saved previous version
    ./.venv/bin/python scripts/models/rollback.py \\
        --name insightserenity-1

    # Roll back to a specific named version
    ./.venv/bin/python scripts/models/rollback.py \\
        --name insightserenity-1 \\
        --to   v0.0.9

    # Preview what would happen
    ./.venv/bin/python scripts/models/rollback.py \\
        --name insightserenity-1 \\
        --dry-run

    # Trigger hot-reload after rollback
    ./.venv/bin/python scripts/models/rollback.py \\
        --name insightserenity-1 \\
        --reload-url http://localhost:8001
"""

import argparse
import json
import os
import shutil
import sys
import time
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
        description="Roll back the serving model to a previous version.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--name",       required=True, metavar="STR",
                   help="Model name (e.g. insightserenity-1).")
    p.add_argument("--to",         default=None,  metavar="VERSION",
                   help="Target version to roll back to. "
                        "If omitted, rolls back to the 'previous' save.")
    p.add_argument("--reason",     default="Manual rollback",
                   help="Reason for rollback (written to audit record).")
    p.add_argument("--reload-url", default=None, metavar="URL",
                   help="If set, call POST <url>/admin/reload-model after rollback.")
    p.add_argument("--internal-secret", default=None,
                   help="SERVING_INTERNAL_API_SECRET for the --reload-url call.")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print what would happen without changing anything.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_versions(model_dir: Path) -> list[str]:
    """
    Return all versions that exist under model_dir, excluding 'latest',
    'previous', and any temp directories.
    """
    reserved = {"latest", "previous", "latest.new"}
    versions = [
        d.name for d in model_dir.iterdir()
        if d.is_dir() and d.name not in reserved
    ]
    return sorted(versions)


def read_metadata(model_dir: Path, version_or_name: str) -> dict:
    """Read metadata.json from a version directory, or return {} if missing."""
    meta_path = model_dir / version_or_name / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def write_rollback_record(latest_dir: Path, from_version: str, to_version: str, reason: str) -> None:
    """Write a rollback_record.json inside the new latest/ directory."""
    record = {
        "event":        "rollback",
        "from_version": from_version,
        "to_version":   to_version,
        "reason":       reason,
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
        "rolled_back_by": "scripts/models/rollback.py",
    }
    record_path = latest_dir / "rollback_record.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Rollback logic
# ─────────────────────────────────────────────────────────────────────────────

def perform_rollback(
    models_root:    Path,
    name:           str,
    target_version: str | None,
    reason:         str,
    dry_run:        bool,
) -> None:
    model_dir   = models_root / name
    latest_dir  = model_dir / "latest"
    previous_dir = model_dir / "previous"
    new_latest  = model_dir / "latest.new"

    if not model_dir.exists():
        sys.exit(f"ERROR: Model '{name}' not found in {models_root}")

    available = list_versions(model_dir)
    if not available:
        sys.exit(f"ERROR: No promoted versions found for model '{name}'")

    # ── Determine current version ─────────────────────────────────────────
    current_meta    = read_metadata(model_dir, "latest")
    current_version = current_meta.get("version", "unknown")
    print(f"  Current version : {current_version}")

    # ── Determine target version ───────────────────────────────────────────
    if target_version:
        target_dir = model_dir / target_version
        if not target_dir.exists():
            sys.exit(
                f"ERROR: Version '{target_version}' not found.\n"
                f"Available versions: {', '.join(available)}"
            )
    elif previous_dir.exists():
        target_dir     = previous_dir
        target_meta    = read_metadata(model_dir, "previous")
        target_version = target_meta.get("version", "previous")
        print(f"  Rolling back to : {target_version} (auto 'previous' save)")
    else:
        # Fall back to the most recent version that is not current
        candidates = [v for v in reversed(available) if v != current_version]
        if not candidates:
            sys.exit(
                "ERROR: Only one version exists and 'previous' save not found. "
                "Cannot roll back."
            )
        target_version = candidates[0]
        target_dir     = model_dir / target_version
        print(f"  Rolling back to : {target_version} (most recent non-current)")

    target_meta = read_metadata(model_dir, target_dir.name)
    print(f"  Target version  : {target_version}")
    if target_meta.get("perplexity"):
        print(f"  Target PPL      : {target_meta['perplexity']}")
    print(f"  Reason          : {reason}")

    if dry_run:
        print("\n  DRY RUN — no changes written.")
        return

    # ── Atomic swap ────────────────────────────────────────────────────────
    # 1. Copy target → latest.new
    if new_latest.exists():
        shutil.rmtree(str(new_latest))
    shutil.copytree(str(target_dir), str(new_latest))

    # 2. Save current latest as new "previous"
    if latest_dir.exists():
        if previous_dir.exists():
            shutil.rmtree(str(previous_dir))
        shutil.copytree(str(latest_dir), str(previous_dir))

    # 3. Remove old latest
    if latest_dir.exists():
        shutil.rmtree(str(latest_dir))

    # 4. Rename latest.new → latest  (atomic on POSIX)
    os.rename(str(new_latest), str(latest_dir))

    # 5. Write rollback record
    write_rollback_record(latest_dir, current_version, target_version, reason)

    print(f"\n  ✓ Rollback complete: {current_version} → {target_version}")


def trigger_hot_reload(reload_url: str, model_name: str, internal_secret: str | None) -> None:
    """Call the AI engine's /admin/reload-model endpoint."""
    import urllib.request
    import urllib.error

    url     = reload_url.rstrip("/") + "/admin/reload-model"
    payload = json.dumps({"model": model_name, "version": "latest"}).encode()
    headers = {"Content-Type": "application/json"}
    if internal_secret:
        headers["Authorization"] = f"Bearer {internal_secret}"

    print(f"\n  Triggering hot-reload at {url}...")
    try:
        req  = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read())
        print(f"  ✓ Hot-reload acknowledged: {body}")
    except urllib.error.HTTPError as e:
        print(f"  WARNING: Hot-reload HTTP {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        print(f"  WARNING: Hot-reload failed: {e}")
        print("  The rollback was applied to disk — restart or reload manually.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = parse_args()
    models_root = ENGINE_ROOT / "storage" / "models"

    print("=" * 60)
    print(f"  InsightSerenity — Model Rollback")
    print(f"  Model   : {args.name}")
    print(f"  Target  : {args.to or 'previous'}")
    print("=" * 60)

    # List available versions first
    model_dir  = models_root / args.name
    if model_dir.exists():
        available = list_versions(model_dir)
        print(f"\n  Available versions: {', '.join(available) or 'none'}")

    perform_rollback(
        models_root    = models_root,
        name           = args.name,
        target_version = args.to,
        reason         = args.reason,
        dry_run        = args.dry_run,
    )

    if not args.dry_run and args.reload_url:
        secret = args.internal_secret or os.environ.get("SERVING_INTERNAL_API_SECRET")
        trigger_hot_reload(args.reload_url, args.name, secret)

    print()


if __name__ == "__main__":
    main()
