#!/usr/bin/env python
"""Export tranche N of the gold set (images, tags, toolkit notes) to gold/tranches/tNN/.

    PYTHONPATH=src python scripts/gold_export_tranche.py --tranche 1
    PYTHONPATH=src python scripts/gold_export_tranche.py --tranche 2 --only rasam1 khatt

Run scripts/gold_build_pool.py first. Images and lines.jsonl stay local (dataset licences); manifest.json is tracked.
"""
import argparse
from pathlib import Path

from arabic_ocr_bench.gold import pool, tranche

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", type=int, required=True)
    ap.add_argument("--config", default=str(ROOT / "gold" / "config.yaml"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = pool.load_config(a.config)
    out = ROOT / "gold" / "tranches" / f"t{a.tranche:02d}"
    m = tranche.export_tranche(cfg, ROOT / "gold" / "pool", out, a.tranche, ROOT, only=a.only)
    print(f"tranche {a.tranche}: {m['n_lines']} lines -> {out}")


if __name__ == "__main__":
    main()
