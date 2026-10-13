"""Spaces inserted inside words: do they fall at PAW gaps more often than chance? (plan 2026-10-13, P1.5 / V5)

    .geomvenv/bin/python scripts/analyze_paw_gaps.py [--store results/store] [--out results/analysis/B/paw_gaps.json]

Per (system, set): word-internal spaces the system inserted (errors.is_paw_gap_space), the share at a PAW gap (after a
letter that cannot join forward) with a line-bootstrap 95 % interval, and the chance share (PAW gaps among all
word-internal letter boundaries of the same lines, errors.paw_gap_base_rate). V5 stays B+ if the share is above chance
(interval clear of it) in ≥ 80 % of competent pairs (plan decision rule). Lines a system trained on are excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from arabic_ocr_bench.characterize.errors import is_paw_gap_space, paw_gap_base_rate  # noqa: E402
from arabic_ocr_bench.characterize.samples import _letters_view  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/B/paw_gaps.json")
    a = ap.parse_args()
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    base = {k: paw_gap_base_rate(_letters_view(s["gt_raw"])) for k, s in S.items()}
    runs = defaultdict(list)
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        if r.get("in_train") == "yes" or r["transform"] != "none" or r["pack"] not in ("panel-2026-10-13", "bench", "eval_data+heldout"):
            continue
        if any(x in r["run_id"] for x in ("ain", "paddleocr-v5", "trocr")) and "unreversed" not in r["run_id"]:
            continue
        if r["e_no_diacritics"] / max(1, r["n_no_diacritics"]) > 0.8:  # whole-line failures say nothing about spacing
            continue
        runs[(r["run_id"], r["dataset"])].append(r)
    res, rng = {}, np.random.default_rng(0)
    for (run, ds), rows in sorted(runs.items()):
        ins = np.array([is_paw_gap_space(_letters_view(S[r["sample_key"]]["gt_raw"]), _letters_view(r["hyp"])) for r in rows], float)
        if ins[:, 0].sum() < 30:
            continue
        ch = np.array([base[r["sample_key"]] for r in rows], float)
        share = ins[:, 1].sum() / ins[:, 0].sum()
        chance = ch[:, 1].sum() / ch[:, 0].sum()
        b = rng.integers(0, len(rows), (2000, len(rows)))
        bs = ins[b, 1].sum(1) / np.maximum(1, ins[b, 0].sum(1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[f"{run}|{ds}"] = {"lines": len(rows), "spaces_inserted": int(ins[:, 0].sum()), "at_paw_gap": float(share), "ci95": [float(lo), float(hi)],
                              "chance": float(chance), "above_chance": bool(lo > chance), "ratio": float(share / chance)}
    n_above = sum(v["above_chance"] for v in res.values())
    summary = {"pairs": len(res), "above_chance": n_above, "share_above": n_above / max(1, len(res)), "v5_holds": n_above >= 0.8 * len(res)}
    a.out.write_text(json.dumps({"summary": summary, "pairs": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summary)
    for k, v in sorted(res.items(), key=lambda t: -t[1]["ratio"])[:12] + sorted(res.items(), key=lambda t: t[1]["ratio"])[:6]:
        print(f"  {k[:60]:60s} spaces {v['spaces_inserted']:5d}  at PAW gap {100 * v['at_paw_gap']:5.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]  chance {100 * v['chance']:.1f}%  ×{v['ratio']:.2f}")


if __name__ == "__main__":
    main()
