"""Leave-manuscript-out report for RASAM-2: how much does the random line split flatter RASAM-2 numbers?

    PYTHONPATH=src python scripts/rasam_lomo_report.py   -> results/rasam/lomo_report.json + printed tables

Inputs: results/rasam/lm/LOMO<i>_lm__r2ho__test.json (fold models, every line of the 3 held-out manuscripts, greedy and char-LM readings) and the
in-split models of the same recipe (RAW_reptm seeds 0-2, 30 epochs, trained on the random line split) on the official RASAM-2 test lines.
Comparison on the SAME lines: the official test lines of each held-out manuscript, read by (a) the fold model that never saw the manuscript and
(b) the in-split models that saw other lines of the same pages. Micro CER, raw profile.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

LM = Path("results/rasam/lm")


def counts(ref, hyp):
    c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp, "raw").strip())
    return c["sub"] + c["del"] + c["ins"], c["ref_len"]


def cer(pairs):
    e = n = 0
    for r, h in pairs:
        a, b = counts(r, h)
        e += a
        n += b
    return round(e / max(1, n) * 100, 2)


def main() -> None:
    insplit = {}
    for s in ("RAW_reptm", "RAW_reptm_s1", "RAW_reptm_s2"):
        f = LM / f"{s}_lm_r2__r2__test.json"
        if f.exists():
            insplit[s] = {l["id"]: l for l in json.loads(f.read_text())["lines"]}
    per_ms = defaultdict(lambda: defaultdict(list))
    folds_done = []
    for i in range(1, 6):
        f = LM / f"LOMO{i}_lm__r2ho__test.json"
        if not f.exists():
            continue
        folds_done.append(i)
        for l in json.loads(f.read_text())["lines"]:
            m = l["manuscript"]
            per_ms[m]["all_lm"].append((l["ref"], l["hyp"]))
            per_ms[m]["all_greedy"].append((l["ref"], l["greedy"]))
            if "/test/" in l["id"]:
                per_ms[m]["test_lomo"].append((l["ref"], l["hyp"]))
                for s, d in insplit.items():
                    if l["id"] in d:
                        per_ms[m][f"test_{s}"].append((d[l["id"]]["ref"], d[l["id"]]["hyp"]))
    rows = {}
    print(f"folds scored: {folds_done}; in-split seeds: {list(insplit)}")
    print(f"{'manuscript':22s} {'lines':>5s} {'LOMO greedy':>11s} {'LOMO+LM':>8s} | official test lines: {'n':>3s} {'LOMO+LM':>8s} {'in-split+LM (seed mean)':>24s}")
    agg = defaultdict(list)
    for m in sorted(per_ms):
        d = per_ms[m]
        ins = [cer(d[f"test_{s}"]) for s in insplit if d[f"test_{s}"]]
        r = {"lines": len(d["all_lm"]), "lomo_greedy": cer(d["all_greedy"]), "lomo_lm": cer(d["all_lm"]), "test_n": len(d["test_lomo"]),
             "test_lomo_lm": cer(d["test_lomo"]), "test_insplit_lm_mean": round(sum(ins) / len(ins), 2) if ins else None}
        rows[m] = r
        for k in ("all_lm", "all_greedy", "test_lomo"):
            agg[k] += d[k]
        for s in insplit:
            agg[f"test_{s}"] += d[f"test_{s}"]
        print(f"{m:22s} {r['lines']:5d} {r['lomo_greedy']:11.2f} {r['lomo_lm']:8.2f} | {r['test_n']:24d} {r['test_lomo_lm']:8.2f} {r['test_insplit_lm_mean'] or float('nan'):24.2f}")
    ins_all = [cer(agg[f"test_{s}"]) for s in insplit if agg[f"test_{s}"]]
    total = {"lines": len(agg["all_lm"]), "lomo_greedy": cer(agg["all_greedy"]), "lomo_lm": cer(agg["all_lm"]), "test_n": len(agg["test_lomo"]),
             "test_lomo_lm": cer(agg["test_lomo"]), "test_insplit_lm_seeds": ins_all,
             "test_insplit_lm_mean": round(sum(ins_all) / len(ins_all), 2) if ins_all else None}
    print(f"{'ALL':22s} {total['lines']:5d} {total['lomo_greedy']:11.2f} {total['lomo_lm']:8.2f} | {total['test_n']:24d} {total['test_lomo_lm']:8.2f} "
          f"{total['test_insplit_lm_mean'] or float('nan'):24.2f}  (seeds {ins_all})")
    json.dump({"folds": folds_done, "per_manuscript": rows, "total": total}, open("results/rasam/lomo_report.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
