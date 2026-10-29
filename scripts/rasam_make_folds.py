"""Split the RASAM-1/2 and TariMa train + validation pages into two folds per manuscript (cross-fitted readings for data cleaning).

    .venv/bin/python scripts/rasam_make_folds.py [--out results/rasam/clean/folds.json]

Writes {'<manuscript>|<page>': 0 or 1}. Pages, not lines, are split, so a fold model never sees a line from a page it reads
(lines of one page share hand and layout; an error copied across a page should not be learnt and then "confirmed").
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/rasam/clean/folds.json"))
    a = ap.parse_args()
    pages = defaultdict(set)
    for name in ("RASAM-1", "RASAM-2", "tarima"):
        for split in ("train", "validation"):
            for f in glob.glob(f"{ROOT}{name}/snapshots/*/data/{split}-*.parquet"):
                for r in pq.read_table(f, columns=["manuscript_name", "image_name"]).to_pylist():
                    pages[r["manuscript_name"]].add(r["image_name"])
    rng = random.Random(0)
    folds = {}
    for ms in sorted(pages):
        ps = sorted(pages[ms])
        rng.shuffle(ps)
        for j, p in enumerate(ps):
            folds[f"{ms}|{p}"] = j % 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(folds, ensure_ascii=False, indent=0))
    print(len(folds), "pages,", sum(folds.values()), "in fold 1, from", len(pages), "manuscripts ->", a.out)


if __name__ == "__main__":
    main()
