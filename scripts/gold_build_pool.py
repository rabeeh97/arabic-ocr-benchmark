#!/usr/bin/env python
"""Rank + de-duplicate every eligible line per dataset (gold/config.yaml) and write gold/pool/<key>.jsonl.

    PYTHONPATH=src python scripts/gold_build_pool.py                 # all datasets (about 10 min the first time)
    PYTHONPATH=src python scripts/gold_build_pool.py --only rasam1   # one dataset (dedup then only sees what is listed:
                                                                     # rebuild everything if priorities matter)

Image signatures are cached in gold/.cache (git-ignored), so a re-run is fast.
"""
import argparse
from pathlib import Path

from arabic_ocr_bench.gold import pool

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "gold" / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "gold" / "pool"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = pool.load_config(a.config)
    print(f"seed {cfg['seed']!r}; {len(cfg['datasets'])} datasets")
    rows, meta = pool.build_pool(cfg, only=a.only, cache_dir=ROOT / "gold" / ".cache")
    pool.write_pool(rows, meta, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
