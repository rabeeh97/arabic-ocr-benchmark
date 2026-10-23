"""Bootstrap CI (by line) and per-manuscript / per-region CER for a results/rasam/{crnn,replication}/*.json file.

    PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/rasam_result_summary.py results/rasam/crnn/A1_w05_s0_best__r1__test.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

PROFILE = "raw"


def line_counts(l: dict) -> tuple[int, int]:
    c = char_edit_counts(normalize(l["ref"], PROFILE), normalize(l["hyp"], PROFILE))
    return c["sub"] + c["del"] + c["ins"], c["ref_len"]


def cer(pairs) -> float:
    e = sum(p[0] for p in pairs)
    n = sum(p[1] for p in pairs)
    return e / n if n else 0.0


def main() -> None:
    d = json.load(open(sys.argv[1]))
    pairs = [line_counts(l) for l in d["lines"]]
    rng = random.Random(0)
    boots = sorted(cer([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(1000))
    print(f"{d['run'] if 'run' in d else d['model']} {d['dataset']} {d['split']} n={len(pairs)}  micro CER({PROFILE}) = {cer(pairs):.4f}"
          f"  95% CI [{boots[25]:.4f}, {boots[974]:.4f}]")
    for key in ("manuscript", "region"):
        g = defaultdict(list)
        for l, p in zip(d["lines"], pairs):
            g[l[key]].append(p)
        print(f"  by {key}:", ", ".join(f"{k}: {cer(v):.3f} (n={len(v)})" for k, v in sorted(g.items())))
    short = [p for l, p in zip(d["lines"], pairs) if len(l["ref"]) <= 20]
    long_ = [p for l, p in zip(d["lines"], pairs) if len(l["ref"]) > 20]
    print(f"  by length: <=20 chars {cer(short):.3f} (n={len(short)}), >20 chars {cer(long_):.3f} (n={len(long_)})")
    exact = sum(normalize(l["ref"], PROFILE) == normalize(l["hyp"], PROFILE) for l in d["lines"]) / len(d["lines"])
    print(f"  exact-line match {exact:.3f}")


if __name__ == "__main__":
    main()
