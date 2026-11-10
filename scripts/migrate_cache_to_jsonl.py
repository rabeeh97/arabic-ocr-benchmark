"""Fold the legacy per-file resume cache (results/_cache/<key>.json) into the single
append-only results/_cache.jsonl (see src/arabic_ocr_bench/cache.py for why).

    PYTHONPATH=src python scripts/migrate_cache_to_jsonl.py [--src DIR] [--dest RESULTS_DIR]

--src defaults to <dest>/_cache. Re-runnable: entries already in the JSONL are
kept; legacy entries are added only if their key is missing. Verifies every legacy
entry reads back identically before reporting success. Never deletes anything —
removing the legacy files is a separate `git rm -r results/_cache` step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_ocr_bench.cache import ResultCache


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="results", help="results/ dir that gets _cache.jsonl")
    ap.add_argument("--src", default=None, help="legacy dir of <key>.json (default <dest>/_cache)")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    src = Path(args.src) if args.src else dest / "_cache"
    files = sorted(src.glob("*.json"))
    if not files:
        print(f"nothing to migrate in {src}")
        return 0

    cache = ResultCache(dest)
    before = len(cache)
    added = skipped = bad = 0
    for f in files:
        key = f.stem
        try:
            row = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            bad += 1
            continue
        if key in cache._rows:
            skipped += 1
            continue
        cache.put(key, row)
        added += 1

    # verify from a cold reload of the JSONL (legacy dir hidden by using a fresh dest-only view)
    fresh = ResultCache(dest)
    mismatches = 0
    for f in files:
        try:
            want = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if fresh._rows.get(f.stem) != want:
            mismatches += 1

    print(f"legacy files: {len(files):,}  added: {added:,}  already-present: {skipped:,}  "
          f"unreadable: {bad}  (jsonl had {before:,} -> now {len(fresh):,})")
    print(f"verification mismatches: {mismatches}")
    return 1 if mismatches or bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
