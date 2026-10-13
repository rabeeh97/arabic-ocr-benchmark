"""Analysis A, the sample level (plan 2026-10-13, §3.3), on the per-line store.

    .geomvenv/bin/python scripts/analyze_lines.py [--store results/store] [--pairs configs/analysis/pairs.json] [--out results/analysis/A]

A1  per (run, transform, pack, dataset): micro / macro / median line CER and the share of lines in each accuracy bucket
    (exact, ≤ 2 %, ≤ 5 %, ≤ 10 %, ≤ 25 %, ≤ 50 %, ≤ 100 %, > 100 %), plus empty readings and length-ratio outliers;
    at the crnn_native rung (continuity with the ledger) and the no_diacritics rung (comparable across families).
A2  per pair in pairs.json, on the lines both runs read in the same pack: micro ΔCER with a paired line bootstrap
    interval, the bucket transition matrix, lines better / worse beyond the seed-noise band (90th percentile of the
    replicate pair's |Δ line CER| within the same length tercile), a sign test on those, and where the net change comes
    from (share of the net edit change by the line's starting bucket).
A3  per dataset, a line × run matrix at no_diacritics with labels solved_by_all, hard_core, crnn_only_failure,
    vlm_only_failure, and the Spearman correlation of CRNN line CER with the median VLM line CER. API rows with few lines,
    AIN-7B rows (contamination unverified, CORRECTIONS C6) and reversed-output engines (C5) are left out of the matrix.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
BUCKETS = [("exact", 0.0), ("≤2%", 0.02), ("≤5%", 0.05), ("≤10%", 0.10), ("≤25%", 0.25), ("≤50%", 0.50), ("≤100%", 1.0), (">100%", math.inf)]
RUNG_A, RUNG_X = "crnn_native", "no_diacritics"
EXCLUDE_MATRIX = ("ain", "paddleocr", "trocr")


def bucket(c: float) -> int:
    for i, (_, hi) in enumerate(BUCKETS):
        if c <= hi + 1e-12:
            return i
    return len(BUCKETS) - 1


def lc(r: dict, rung: str) -> float:
    return r[f"e_{rung}"] / max(1, r[f"n_{rung}"])


def summary(rows: list[dict], rung: str) -> dict:
    cs = [lc(r, rung) for r in rows]
    e, n = sum(r[f"e_{rung}"] for r in rows), sum(max(1, r[f"n_{rung}"]) for r in rows)
    b = np.bincount([bucket(c) for c in cs], minlength=len(BUCKETS)) / len(cs)
    ratio = [len(r["hyp"]) / max(1, r[f"n_{rung}"]) for r in rows]
    return {"lines": len(rows), "micro": e / n, "macro": float(np.mean(cs)), "median": float(np.median(cs)),
            "buckets": dict(zip([k for k, _ in BUCKETS], map(float, b))), "empty": sum(not r["hyp"].strip() for r in rows) / len(rows),
            "len_ratio_lt_0.5": sum(x < 0.5 for x in ratio) / len(rows), "len_ratio_gt_1.5": sum(x > 1.5 for x in ratio) / len(rows)}


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    from math import comb
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


class Store:
    def __init__(self, path: Path):
        self.idx: dict[tuple, dict[str, dict]] = defaultdict(dict)  # (run, transform, pack, dataset) -> {key: row}
        for r in pq.read_table(path / "lines.parquet").to_pylist():
            if r.get("in_train") == "yes":  # never score lines a run trained on (CORRECTIONS C4)
                continue
            self.idx[(r["run_id"], r["transform"], r["pack"], r["dataset"])][r["sample_key"]] = r
        self.samples = {s["sample_key"]: s for s in pq.read_table(path / "samples.parquet").to_pylist()}

    def get(self, spec: str, ds: str, packs: list[str], pack: str | None = None):
        run, tr = spec.split("@")
        for p in [pack] if pack else packs:
            d = self.idx.get((run, tr, p, ds))
            if d:
                return p, d
        return None, None


def noise_band(st: Store, spec: dict, ds: str, packs: list[str]) -> dict | None:
    """90th percentile of |Δ line CER| between two identical recipes, per length tercile (boundaries from those lines)."""
    for p in packs:
        pa, A = st.get(spec["A"], ds, packs, p)
        pb, B = st.get(spec["B"], ds, packs, p)
        if A and B:
            keys = sorted(set(A) & set(B))
            n = np.array([max(1, A[k][f"n_{RUNG_A}"]) for k in keys])
            d = np.array([abs(lc(B[k], RUNG_A) - lc(A[k], RUNG_A)) for k in keys])
            t1, t2 = np.percentile(n, [33.3, 66.7])
            band = {}
            for name, m in (("short", n <= t1), ("mid", (n > t1) & (n <= t2)), ("long", n > t2)):
                band[name] = float(np.percentile(d[m], 90)) if m.any() else float(np.percentile(d, 90))
            return {"pack": p, "lines": len(keys), "tercile_edges": [float(t1), float(t2)], "p90": band,
                    "share_changed": float((d > 0).mean())}
    return None


def compare(st: Store, pr: dict, ds: str, packs: list[str], noise: dict | None, reps: int = 4000) -> dict | None:
    for p in packs:
        _, A = st.get(pr["A"], ds, packs, p)
        _, B = st.get(pr["B"], ds, packs, p)
        if A and B:
            break
    else:
        return None
    keys = sorted(set(A) & set(B))
    if len(keys) < 20:
        return None
    out = {"pack": p, "lines": len(keys)}
    for rung in (RUNG_A, RUNG_X):
        ea = np.array([A[k][f"e_{rung}"] for k in keys], float)
        eb = np.array([B[k][f"e_{rung}"] for k in keys], float)
        n = np.array([max(1, A[k][f"n_{rung}"]) for k in keys], float)
        rng = np.random.default_rng(0)
        s = rng.integers(0, len(keys), (reps, len(keys)))
        boots = (eb[s] - ea[s]).sum(1) / n[s].sum(1)
        ca, cb = ea / n, eb / n
        ba, bb = [bucket(x) for x in ca], [bucket(x) for x in cb]
        trans = np.zeros((len(BUCKETS), len(BUCKETS)), int)
        for i, j in zip(ba, bb):
            trans[i, j] += 1
        net = (eb - ea).sum()
        by_start = {BUCKETS[i][0]: float((eb - ea)[np.array(ba) == i].sum() / net) if net else 0.0 for i in range(len(BUCKETS)) if (np.array(ba) == i).any()}
        r = {"cer_A": float(ea.sum() / n.sum()), "cer_B": float(eb.sum() / n.sum()), "delta": float(net / n.sum()),
             "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "transition": trans.tolist(),
             "lines_better": int((cb < ca).sum()), "lines_worse": int((cb > ca).sum()), "net_change_by_start_bucket": by_start}
        if noise and rung == RUNG_A:
            t1, t2 = noise["tercile_edges"]
            thr = np.array([noise["p90"]["short" if x <= t1 else "mid" if x <= t2 else "long"] for x in n])
            bet, wor = int(((ca - cb) > thr).sum()), int(((cb - ca) > thr).sum())
            r["beyond_noise"] = {"better": bet, "worse": wor, "sign_test_p": binom_two_sided(min(bet, wor), bet + wor)}
            r["beyond_noise_ids"] = {"better": [k for k, a_, b_, t in zip(keys, ca, cb, thr) if a_ - b_ > t],
                                     "worse": [k for k, a_, b_, t in zip(keys, ca, cb, thr) if b_ - a_ > t]}
        out[rung] = r
    return out


def matrix(st: Store, ds: str, packs: list[str]) -> dict:
    cols = {}
    for (run, tr, pack, d), m in st.idx.items():
        if d != ds:
            continue
        fam = next(iter(m.values()))["family"]
        if fam.startswith("vlm") and tr == "none" and not any(x in run.lower() for x in EXCLUDE_MATRIX) and len(m) >= 50:
            cols[run] = m
    for spec in ("crnn/base@none", "crnn/base@inkcrop", "crnn/combo@inkcrop", "crnn/rep2@none"):
        _, m = st.get(spec, ds, packs)
        if m:
            cols[spec] = m
    keys = sorted({k for m in cols.values() for k in m})
    crnn_cols = [c for c in cols if c.startswith("crnn")]
    vlm_cols = [c for c in cols if c.startswith("vlm")]
    labels, rows = defaultdict(list), {}
    xs, ys = [], []
    for k in keys:
        v = {c: min(1.0, lc(cols[c][k], RUNG_X)) for c in cols if k in cols[c]}
        if len(v) < 0.8 * len(cols):
            continue
        rows[k] = v
        cr = [v[c] for c in crnn_cols if c in v]
        vl = [v[c] for c in vlm_cols if c in v]
        if all(x <= 0.10 for x in v.values()):
            labels["solved_by_all"].append(k)
        if all(x > 0.25 for x in v.values()):
            labels["hard_core"].append(k)
        if cr and vl and all(x > 0.25 for x in cr) and statistics.median(vl) <= 0.15:
            labels["crnn_only_failure"].append(k)
        if cr and vl and all(x > 0.25 for x in vl) and min(cr) <= 0.15:
            labels["vlm_only_failure"].append(k)
        if cr and vl:
            xs.append(statistics.median(cr))
            ys.append(statistics.median(vl))
    from scipy.stats import spearmanr
    rho = float(spearmanr(xs, ys)[0]) if len(xs) > 10 else None
    return {"dataset": ds, "columns": list(cols), "crnn_columns": crnn_cols, "vlm_columns": vlm_cols, "lines": len(rows),
            "label_counts": {k: len(v) for k, v in labels.items()}, "labels": dict(labels), "spearman_crnn_vs_median_vlm": rho,
            "consensus_difficulty": {k: float(np.median(list(v.values()))) for k, v in rows.items()}, "cells": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--pairs", type=Path, default=W / "configs/analysis/pairs.json")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/A")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.pairs.read_text(encoding="utf-8"))
    packs = cfg["pack_pref"]
    st = Store(a.store)
    datasets = sorted({k[3] for k in st.idx})

    A1 = {}
    for (run, tr, pack, ds), m in sorted(st.idx.items()):
        if len(m) >= 20:
            A1[f"{run}@{tr}|{pack}|{ds}"] = {rung: summary(list(m.values()), rung) for rung in (RUNG_A, RUNG_X)}
    (a.out / "summary.json").write_text(json.dumps(A1, indent=1, ensure_ascii=False), encoding="utf-8")

    A2 = {}
    for ds in datasets:
        bands = {nk: noise_band(st, cfg[nk], ds, packs) for nk in ("noise", "noise_cropped")}
        for pr in cfg["pairs"]:
            r = compare(st, pr, ds, packs, bands.get(pr["noise"]))
            if r:
                A2[f"{pr['name']}|{ds}"] = {"A": pr["A"], "B": pr["B"], "noise_band": bands.get(pr["noise"]), **r}
    (a.out / "pairs.json").write_text(json.dumps(A2, indent=1, ensure_ascii=False), encoding="utf-8")

    for ds in ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2"):
        mx = matrix(st, ds, packs)
        if mx["lines"]:
            (a.out / f"matrix_{ds}.json").write_text(json.dumps(mx, ensure_ascii=False), encoding="utf-8")
            print(f"matrix {ds:14s} lines {mx['lines']:5d}  cols {len(mx['columns']):3d}  {mx['label_counts']}  rho {mx['spearman_crnn_vs_median_vlm']}")

    print(f"\n{'pair':42s} {'dataset':13s} {'lines':>5s} {'A':>6s} {'B':>6s} {'Δ':>6s} {'95% CI':>15s} {'beyond noise b/w':>17s}")
    for k, v in A2.items():
        name, ds = k.split("|")
        r = v[RUNG_A]
        bn = r.get("beyond_noise", {})
        print(f"{name[:42]:42s} {ds:13s} {v['lines']:5d} {100 * r['cer_A']:6.1f} {100 * r['cer_B']:6.1f} {100 * r['delta']:+6.2f} "
              f"[{100 * r['ci95'][0]:+.2f},{100 * r['ci95'][1]:+.2f}] {bn.get('better', '-'):>8}/{bn.get('worse', '-')}")


if __name__ == "__main__":
    main()
