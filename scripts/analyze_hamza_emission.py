"""Is the alif-hamza a reading or a learned convention? (plan 2026-10-13, P1.4, T11 across families)

    .geomvenv/bin/python scripts/analyze_hamza_emission.py [--out results/analysis/B/hamza_emission.json]

Share of alifs written with a hamza (أ إ آ among ا أ إ آ) in each label set and in each system's readings of NAKBA test
and blind (lines a system trained on excluded; ≥ 500 alifs). If the hamza were read from the image, systems would agree
with each other whatever they were trained on; if it is a convention, each system writes it at its training labels' rate.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
HAMZA, ALIF = set("أإآ"), set("اأإآ")


def count(t: str) -> tuple[int, int]:
    a = [c for c in t if c in ALIF]
    return sum(c in HAMZA for c in a), len(a)


def trained_on(run: str) -> str:
    r = run.lower()
    if "nakba" in r and "visft" not in r or "baseer" in r:
        return "NAKBA train (labels with hamza)"
    if r.startswith("crnn") or "suite-v1" in r or "exp3090" in r or "visft" in r:
        return "Muharaf train (labels mostly without)"
    return "not fine-tuned on either"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=W / "results/analysis/B/hamza_emission.json")
    a = ap.parse_args()
    S = pq.read_table(W / "results/store/samples.parquet").to_pylist()
    labels = defaultdict(lambda: [0, 0])
    for s in S:
        h, n = count(s["gt_raw"])
        labels[s["dataset"]][0] += h
        labels[s["dataset"]][1] += n
    R = defaultdict(lambda: [0, 0])
    for r in pq.read_table(W / "results/store/lines.parquet").to_pylist():
        if r["transform"] != "none" or r["dataset"] not in ("nakba_lines", "nakba_blind") or r.get("in_train") == "yes" or r["pack"] not in ("panel-2026-10-13", "bench"):
            continue
        h, n = count(r["hyp"])
        R[(r["run_id"], r["dataset"])][0] += h
        R[(r["run_id"], r["dataset"])][1] += n
    runs = {f"{run}|{ds}": {"hamza_share": v[0] / v[1], "alifs": v[1], "trained_on": trained_on(run)} for (run, ds), v in R.items() if v[1] >= 500}
    res = {"labels": {d: v[0] / max(1, v[1]) for d, v in labels.items()}, "runs": runs}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("labels:", {d: f"{100 * x:.1f}%" for d, x in res["labels"].items()})
    for k, v in sorted(runs.items(), key=lambda t: -t[1]["hamza_share"]):
        print(f"  {k[:62]:62s} {100 * v['hamza_share']:5.1f}%  {v['trained_on']}")


if __name__ == "__main__":
    main()
