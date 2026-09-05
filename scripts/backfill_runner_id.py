"""One-time migration: stamp a runner id onto results produced before the
distributed-sweep coordination existed.

Every prediction row and per-line cache entry now carries ``runner`` (see
``arabic_ocr_bench.coordination``). Rows written before that change have no such
field, so ``scripts/sweep_audit.py`` lumps them under "(unstamped)" and its
duplicate check can misfire across old accumulate re-runs. This script writes a
single default runner id (``legacy`` by default) onto every row / cache entry
that lacks one. Idempotent — already-stamped entries are left alone.

    uv run python scripts/backfill_runner_id.py results --runner legacy
    uv run python scripts/backfill_runner_id.py results --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _patch_jsonl(path: Path, runner: str, dry: bool) -> tuple[int, int]:
    """Return (rows_total, rows_stamped) for one predictions.jsonl."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return (0, 0)
    out: list[str] = []
    total = stamped = 0
    changed = False
    for line in lines:
        s = line.strip()
        if not s:
            out.append(line)
            continue
        total += 1
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if not row.get("runner") and not row.get("served_to"):
            row["runner"] = runner
            stamped += 1
            changed = True
        out.append(json.dumps(row, ensure_ascii=False))
    if changed and not dry:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return (total, stamped)


def _patch_cache(path: Path, runner: str, dry: bool) -> int:
    """Return 1 if this cache file was stamped, else 0."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if row.get("runner") or row.get("served_to"):
        return 0
    row["runner"] = runner
    if not dry:
        path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results", help="results/ directory")
    ap.add_argument("--runner", default="legacy", help="id to stamp (default: legacy)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--skip-cache", action="store_true", help="only touch predictions.jsonl")
    args = ap.parse_args(argv)

    root = Path(args.results)
    preds = sorted(root.glob("*/predictions.jsonl"))
    files_touched = rows_total = rows_stamped = 0
    for p in preds:
        t, s = _patch_jsonl(p, args.runner, args.dry_run)
        rows_total += t
        rows_stamped += s
        if s:
            files_touched += 1
    print(f"predictions: {rows_stamped:,} / {rows_total:,} rows stamped "
          f"across {files_touched} / {len(preds)} run dirs")

    if not args.skip_cache:
        cache = sorted((root / "_cache").glob("*.json"))
        cache_stamped = sum(_patch_cache(p, args.runner, args.dry_run) for p in cache)
        print(f"cache      : {cache_stamped:,} / {len(cache):,} entries stamped")

    if args.dry_run:
        print("\n(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
