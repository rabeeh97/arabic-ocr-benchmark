"""B2: compare a random-split model (A2) with a manuscript-disjoint model (B2) on the SAME lines.

    PYTHONPATH=src python scripts/rasam_msplit_compare.py A2_JSON B2_UNSEEN_JSON B2_SEEN_JSON

A2_JSON         results/rasam/crnn/<A2>__r2__test.json         (A2 trained on RASAM-2 train lines of all 15 manuscripts)
B2_UNSEEN_JSON  results/rasam/crnn/<B2>__r2m__test.json        (B2 never saw these 3 manuscripts)
B2_SEEN_JSON    results/rasam/crnn/<B2>__r2s__test.json        (B2 trained on the other 12 manuscripts' train lines)

Prints micro CER (raw profile) on the lines both files score, split into held-out manuscripts (A2 saw their training lines, B2 did not)
and the other 12 manuscripts (both saw training lines). The first gap is the cost of an unseen manuscript.
"""
from __future__ import annotations

import json
import sys

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize


def counts(l: dict) -> tuple[int, int]:
    c = char_edit_counts(normalize(l["ref"], "raw").strip(), normalize(l["hyp"], "raw").strip())
    return c["sub"] + c["del"] + c["ins"], c["ref_len"]


def cer(ids: set, lines: dict) -> tuple[float, int]:
    e = n = 0
    for i in ids:
        a, b = counts(lines[i])
        e += a
        n += b
    return e / max(1, n), len(ids)


def main() -> None:
    a2 = {l["id"]: l for l in json.load(open(sys.argv[1]))["lines"]}
    unseen = {l["id"]: l for l in json.load(open(sys.argv[2]))["lines"]}
    seen = {l["id"]: l for l in json.load(open(sys.argv[3]))["lines"]}
    common_u = set(a2) & set(unseen)
    common_s = set(a2) & set(seen)
    print(f"held-out manuscripts, lines that are in A2's published test split: n={len(common_u)}")
    for name, lines in (("A2 (random split, saw these manuscripts)", a2), ("B2 (manuscript-disjoint, never saw them)", unseen)):
        c, n = cer(common_u, lines)
        print(f"  {name:48s} CER {c:.4f}")
    print(f"B2 on all lines of the held-out manuscripts (n={len(unseen)}): CER {cer(set(unseen), unseen)[0]:.4f}")
    print(f"the other 12 manuscripts, published test lines: n={len(common_s)}")
    for name, lines in (("A2", a2), ("B2 (12 manuscripts only)", seen)):
        c, n = cer(common_s, lines)
        print(f"  {name:48s} CER {c:.4f}")
    by = {}
    for l in unseen.values():
        e, n = counts(l)
        by.setdefault(l["manuscript"], [0, 0])
        by[l["manuscript"]][0] += e
        by[l["manuscript"]][1] += n
    print("B2 per held-out manuscript:", {k: round(v[0] / max(1, v[1]), 4) for k, v in by.items()})


if __name__ == "__main__":
    main()
