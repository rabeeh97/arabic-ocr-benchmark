#!/usr/bin/env python
"""How closely did two labelers land on the same text? (calibration lines and any other overlap)

    PYTHONPATH=src python scripts/gold_agreement.py --tranche t01 [--labelers rabeeh eyad] [--json out.json]
"""
import argparse
import json
from pathlib import Path

from arabic_ocr_bench.gold import agreement

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", default="t01")
    ap.add_argument("--labels", default=str(ROOT / "gold" / "labels"))
    ap.add_argument("--labelers", nargs="*")
    ap.add_argument("--json")
    a = ap.parse_args()
    rep = agreement.compare(a.labels, a.tranche, a.labelers)
    print(agreement.to_markdown(rep))
    if a.json:
        json.dump(rep, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
