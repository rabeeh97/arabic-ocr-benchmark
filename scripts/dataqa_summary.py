"""Firing rates of every data-QA check across the scans in results/dataqa/.

  PYTHONPATH=src python scripts/dataqa_summary.py [scan names ...]
Prints a Markdown table (% of lines per check) and writes results/dataqa/summary.json.
"""

import json
import sys
from pathlib import Path

from arabic_ocr_bench.dataqa import CHECKS

names = sys.argv[1:] or sorted(p.stem for p in Path("results/dataqa").glob("*.json")
                               if not p.stem.startswith(("overlap_", "summary", "rasam_validation")))
rows = {}
for n in names:
    d = json.load(open(f"results/dataqa/{n}.json"))
    rows[n] = {"n": d["n"], "n_read": d["n_read"], "any": d["any"], "counts": d["counts"]}
json.dump(rows, open("results/dataqa/summary.json", "w"), indent=1)
used = [c for c in CHECKS if any(c in r["counts"] for r in rows.values())]
print("| scan | lines | read | " + " | ".join(used) + " |")
print("|---|---:|---:|" + "---:|" * len(used))
for n, r in rows.items():
    cells = []
    for c in used:
        k = r["counts"].get(c, 0)
        cells.append(f"{k} ({100 * k / r['n']:.1f})" if k else "·")
    print(f"| {n} | {r['n']} | {r['n_read']} | " + " | ".join(cells) + " |")
