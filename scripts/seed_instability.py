"""Seed and numeric instability: the lines that flip between identical trainings (checklist 1.4, evaluation-standards
item 11, detector D24 of run_detectors.py).

    .geomvenv/bin/python scripts/seed_instability.py [--store results/store] [--out results/analysis/D]

Seed groups (GROUPS): the same recipe trained with different seeds, all read on the panel pack (fp32, one line at a time),
at transforms none and inkcrop. For each line × group × transform (a cell) the seeds' edits at the crnn_native rung. A
cell is unstable when one seed reads the line exactly and another does not (a flip), or the seeds' line CER spans
≥ 0.10 with ≥ 2 edits between them. A line is seed-unstable when at least half of its cells (≥ 2) are unstable.
A line is numeric-unstable when a run reads it differently at fp32-bs1 (pack panel-2026-10-13) and at fp16-bs1 (pack
panel/fp16-bs1), the same image. (geom_transfer_eval's fp32 batches of 32 change ~35 % of readings, but they pad the
image to the batch width, so the input differs: reported as `batch_padding`, not counted.)

seed_instability.json:
  datasets          lines, cells, seed-unstable and numeric-unstable lines, D24 (either), per dataset
  disagreement      in unstable cells, what the seeds disagree on: dots only (equal at the rasm rung), spaces only (equal
                    at no_spaces), other letters
  characterise      seed-unstable vs stable lines: each line feature raw (Mann-Whitney, Fisher for flags) and within
                    quintiles of the line's mean seed CER (stratified rank permutation); one BH family (q = 0.1) each
  comparisons       per seed pair of pairs.json: lines better / worse on all seeds and the share of them that are
                    seed-unstable, the instability recomputed without the comparison's own groups; per single pair of
                    analysis A: the same for its beyond-noise better / worse lines; base rate = share among all its lines
  unstable_ids      seed-unstable, numeric-unstable (read by the gallery)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import fisher_exact, mannwhitneyu, rankdata

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from eval_stats import Q_FDR, bh  # noqa: E402

PANEL, FP16 = "panel-2026-10-13", "panel/fp16-bs1"
RUNG = "crnn_native"
GROUPS = {"base": ["crnn/base", "crnn/rep2", "crnn/base_s3"], "randctl": ["crnn/randctl", "crnn/randctl_s2", "crnn/randctl_s3"],
          "shape": ["crnn/shape", "crnn/shape_s2", "crnn/shape_s3"], "combo": ["crnn/combo", "crnn/combo_s2", "crnn/combo_s3"],
          "std": ["crnn/std_s0", "crnn/std_s1"]}
TRANSFORMS = ("none", "inkcrop")
GROUP_OF = {r: g for g, rs in GROUPS.items() for r in rs}


def cell_unstable(edits: list[int], n: int) -> bool:
    e = np.asarray(edits, float)
    flip = e.min() == 0 and e.max() > 0
    return bool(flip or ((e.max() - e.min()) >= 2 and (e.max() - e.min()) / max(1, n) >= 0.10))


def load(store: Path):
    idx = defaultdict(dict)  # (run, transform, pack) -> {key: row}
    for r in pq.read_table(store / "lines.parquet").to_pylist():
        if r.get("in_train") == "yes":
            continue
        if r["run_id"] in GROUP_OF or r["pack"] == FP16:
            idx[(r["run_id"], r["transform"], r["pack"])][r["sample_key"]] = r
    return idx


def cells(idx) -> dict[str, dict[tuple, dict]]:
    """{key: {(group, transform): {"edits": [...], "n": n, "unstable": bool, "kind": ...}}}."""
    out = defaultdict(dict)
    for g, runs in GROUPS.items():
        for tr in TRANSFORMS:
            ms = [idx.get((r, tr, PANEL)) for r in runs]
            if any(m is None for m in ms):
                continue
            for k in set.intersection(*[set(m) for m in ms]):
                rs = [m[k] for m in ms]
                e = [r[f"e_{RUNG}"] for r in rs]
                n = max(1, rs[0][f"n_{RUNG}"])
                u = cell_unstable(e, n)
                kind = None
                if u:
                    if len({r["e_rasm"] for r in rs}) == 1 and len({r["e_no_diacritics"] for r in rs}) > 1:
                        kind = "dots only"
                    elif len({r["e_no_spaces"] for r in rs}) == 1:
                        kind = "spaces only"
                    else:
                        kind = "letters"
                out[k][(g, tr)] = {"edits": e, "n": n, "unstable": u, "kind": kind}
    return out


def seed_unstable(c: dict, exclude: set[str] = frozenset()) -> bool | None:
    """Unstable in at least half of the line's cells outside the excluded groups; None with fewer than 2 cells."""
    v = [x["unstable"] for (g, _), x in c.items() if g not in exclude]
    if len(v) < 2:
        return None
    return sum(v) >= 0.5 * len(v)


def numeric_unstable(idx) -> tuple[set[str], dict]:
    keys, stats = set(), {}
    for (run, tr, pack), m in idx.items():
        if pack != PANEL:
            continue
        f = idx.get((run, tr, FP16))
        if not f:
            continue
        common = set(m) & set(f)
        diff = {k for k in common if m[k]["hyp"] != f[k]["hyp"]}
        ediff = sum(m[k][f"e_{RUNG}"] != f[k][f"e_{RUNG}"] for k in common)
        stats[f"{run}@{tr}"] = {"lines": len(common), "reading_differs": len(diff), "edits_differ": int(ediff)}
        keys |= diff
    return keys, stats


def instability(idx) -> dict:
    """What run_detectors needs: {"seed": set, "numeric": set, "cells": {...}}."""
    c = cells(idx)
    seed = {k for k, v in c.items() if seed_unstable(v)}
    num, _ = numeric_unstable(idx)
    return {"seed": seed, "numeric": num, "cells": c}


def strat_perm_p(x: np.ndarray, y: np.ndarray, strata: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Two-sided p of the mean rank difference (y True vs False) with labels permuted within strata; and the statistic
    as a rank-biserial-like effect in [-1, 1]."""
    r = rankdata(x) / len(x)
    rng = np.random.default_rng(seed)

    def stat(lab):
        s = 0.0
        for g in np.unique(strata):
            m = strata == g
            if lab[m].any() and (~lab[m]).any():
                s += lab[m].sum() * (r[m][lab[m]].mean() - r[m][~lab[m]].mean())
        return s / max(1, lab.sum())

    obs = stat(y)
    groups = [np.where(strata == g)[0] for g in np.unique(strata)]
    null = np.empty(reps)
    for i in range(reps):
        lab = y.copy()
        for gi in groups:
            lab[gi] = rng.permutation(lab[gi])
        null[i] = stat(lab)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (reps + 1)
    return float(p), float(2 * obs)


def characterise(keys: list[str], unstable: np.ndarray, S: dict, F: dict, I: dict, c: dict) -> list[dict]:
    mean_cer = np.array([np.mean([np.mean(x["edits"]) / x["n"] for x in c[k].values()]) for k in keys])
    strata = np.digitize(mean_cer, np.quantile(mean_cer, [0.2, 0.4, 0.6, 0.8]))
    num = {"letters (GT length)": lambda k: max(x["n"] for x in c[k].values()),
           "mean seed CER": None,
           "ink extent": lambda k: S[k]["ink_extent"],
           "px per char at 64 px": lambda k: F.get(k, {}).get("ink_px_per_char"),
           "|slant|": lambda k: abs(F[k]["slant_deg"]) if F.get(k, {}).get("slant_deg") is not None else None,
           "stroke width at 64 px": lambda k: F.get(k, {}).get("stroke_px64"),
           "dot size at 64 px": lambda k: F.get(k, {}).get("dot_px64"),
           "pen contrast": lambda k: F.get(k, {}).get("pen_contrast"),
           "ink density": lambda k: F.get(k, {}).get("ink_density")}
    flags = {"D4 neighbour-line intrusion": lambda k: I.get(k, {}).get("d4"), "ruled": lambda k: F.get(k, {}).get("ruled"),
             "slant outlier": lambda k: F.get(k, {}).get("slant_outlier"), "low resolution": lambda k: F.get(k, {}).get("low_res"),
             "dropped by the CTC filter": lambda k: not S[k]["ctc_ok"]}
    for ds in sorted({S[k]["dataset"] for k in keys}):
        flags[f"dataset {ds}"] = (lambda d: lambda k: S[k]["dataset"] == d)(ds)
    rows = []
    for name, f in num.items():
        x = mean_cer if f is None else np.array([np.nan if f(k) is None else float(f(k)) for k in keys])
        ok = ~np.isnan(x)
        if ok.sum() < 50 or not unstable[ok].any():
            continue
        xu, xs = x[ok & unstable], x[ok & ~unstable]
        row = {"feature": name, "kind": "numeric", "median_unstable": float(np.median(xu)), "median_stable": float(np.median(xs)),
               "p_raw": float(mannwhitneyu(xu, xs, alternative="two-sided").pvalue)}
        if f is not None:
            row["p_cer_matched"], row["effect_cer_matched"] = strat_perm_p(x[ok], unstable[ok], strata[ok])
        rows.append(row)
    for name, f in flags.items():
        x = np.array([bool(f(k)) for k in keys])
        a, b = int((x & unstable).sum()), int((x & ~unstable).sum())
        if a + b == 0:
            continue
        row = {"feature": name, "kind": "flag", "share_unstable": a / max(1, unstable.sum()), "share_stable": b / max(1, (~unstable).sum()),
               "p_raw": float(fisher_exact([[a, int(unstable.sum()) - a], [b, int((~unstable).sum()) - b]])[1])}
        row["p_cer_matched"], row["effect_cer_matched"] = strat_perm_p(x.astype(float), unstable, strata)
        rows.append(row)
    for col in ("p_raw", "p_cer_matched"):
        t = [r for r in rows if col in r]
        rej, adj = bh([r[col] for r in t], Q_FDR)
        for r, j, q in zip(t, rej, adj):
            r[col.replace("p_", "q_")], r[col.replace("p_", "fdr_")] = float(q), bool(j)
    return rows


def comparisons(idx, c: dict, S: dict, pairs_cfg: dict, a_pairs: dict) -> list[dict]:
    out = []
    for sp in pairs_cfg.get("seed_pairs", []):
        runs = [s.split("@") for s in sp["A"] + sp["B"]]
        excl = {GROUP_OF.get(r) for r, _ in runs}
        for ds in sorted({s["dataset"] for s in S.values()}):
            ms = {side: [idx.get((r, t, PANEL), {}) for r, t in (s.split("@") for s in sp[side])] for side in ("A", "B")}
            ms = {side: [{k: v for k, v in m.items() if S[k]["dataset"] == ds} for m in v] for side, v in ms.items()}
            if any(not m for v in ms.values() for m in v):
                continue
            keys = sorted(set.intersection(*[set(m) for m in ms["A"] + ms["B"]]))
            if sp.get("keep"):
                f, kk = sp["keep"].rsplit(":", 1)
                legacy = set(json.loads((W / f).read_text())[kk])
                keys = [k for k in keys if k in legacy or S[k].get("legacy_id") in legacy]
            if len(keys) < 20:
                continue
            n = np.array([max(1, ms["A"][0][k][f"n_{RUNG}"]) for k in keys], float)
            ca = np.array([[m[k][f"e_{RUNG}"] for k in keys] for m in ms["A"]]) / n
            cb = np.array([[m[k][f"e_{RUNG}"] for k in keys] for m in ms["B"]]) / n
            better = [k for k, x in zip(keys, cb.max(0) < ca.min(0)) if x]
            worse = [k for k, x in zip(keys, cb.min(0) > ca.max(0)) if x]
            out.append(_share(sp["name"], ds, "all seeds", keys, better, worse, c, excl))
    for name, v in a_pairs.items():
        pair, ds = name.split("|")
        r = v.get(RUNG, {})
        ids = r.get("beyond_noise_ids")
        if not ids or v.get("pack") != PANEL:
            continue
        runs = [x.split("@")[0] for x in (v["A"], v["B"])] if isinstance(v.get("A"), str) else []
        excl = {GROUP_OF.get(x) for x in runs}
        keys = [k for k in c if S.get(k, {}).get("dataset") == ds]
        out.append(_share(pair, ds, "beyond noise", keys, ids["better"], ids["worse"], c, excl))
    return out


def _share(name, ds, rule, keys, better, worse, c, excl) -> dict:
    def sh(ks):
        v = [seed_unstable(c[k], excl) for k in ks if k in c]
        v = [x for x in v if x is not None]
        return {"lines": len(v), "unstable": int(sum(v)), "share": sum(v) / len(v) if v else None}
    return {"comparison": name, "dataset": ds, "rule": rule, "groups_left_out": sorted(g for g in excl if g),
            "better": sh(better), "worse": sh(worse), "all": sh(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--pairs", type=Path, default=W / "configs/analysis/pairs.json")
    ap.add_argument("--a-pairs", type=Path, default=W / "results/analysis/A/pairs.json")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/D")
    a = ap.parse_args()
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    F = {f["sample_key"]: f for f in pq.read_table(a.store / "features.parquet").to_pylist()} if (a.store / "features.parquet").exists() else {}
    I = {f["sample_key"]: f for f in pq.read_table(a.store / "intrusion.parquet").to_pylist()} if (a.store / "intrusion.parquet").exists() else {}
    idx = load(a.store)
    c = cells(idx)
    num, num_stats = numeric_unstable(idx)
    keys = sorted(k for k in c if seed_unstable(c[k]) is not None)
    uns = np.array([bool(seed_unstable(c[k])) for k in keys])
    seed = {k for k, u in zip(keys, uns) if u}

    datasets = {}
    for ds in sorted({S[k]["dataset"] for k in keys}):
        ks = [k for k in keys if S[k]["dataset"] == ds]
        cl = [x for k in ks for x in c[k].values()]
        nu = [k for k in ks if k in num]
        datasets[ds] = {"lines": len(ks), "cells": len(cl), "unstable_cells": sum(x["unstable"] for x in cl),
                        "flip_cells": sum(min(x["edits"]) == 0 < max(x["edits"]) for x in cl),
                        "seed_unstable": sum(k in seed for k in ks), "seed_unstable_share": sum(k in seed for k in ks) / len(ks),
                        "numeric_unstable": len(nu), "numeric_and_seed": sum(k in seed for k in nu),
                        "d24": sum(k in seed or k in num for k in ks), "d24_share": sum(k in seed or k in num for k in ks) / len(ks)}
    kinds = Counter(x["kind"] for v in c.values() for x in v.values() if x["unstable"])
    per_group = {f"{g}@{t}": {"cells": sum((g, t) in v for v in c.values()), "unstable": sum(v.get((g, t), {}).get("unstable", False) for v in c.values())}
                 for g in GROUPS for t in TRANSFORMS}
    char = characterise(keys, uns, S, F, I, c)
    comp = comparisons(idx, c, S, json.loads(a.pairs.read_text(encoding="utf-8")),
                       json.loads(a.a_pairs.read_text(encoding="utf-8")) if a.a_pairs.exists() else {})
    out = {"_doc": __doc__.split("seed_instability.json:")[0].strip(), "rung": RUNG, "groups": GROUPS, "datasets": datasets,
           "per_group_cell": per_group, "disagreement": dict(kinds), "numeric": num_stats, "characterise": char, "comparisons": comp,
           "unstable_ids": {"seed": sorted(seed), "numeric": sorted(num)}}
    (a.out / "seed_instability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'dataset':16s} {'lines':>6s} {'seed-unstable':>14s} {'numeric':>8s} {'D24':>6s}")
    for ds, v in datasets.items():
        print(f"{ds:16s} {v['lines']:6d} {v['seed_unstable']:6d} ({100 * v['seed_unstable_share']:4.1f}%) {v['numeric_unstable']:8d} {100 * v['d24_share']:5.1f}%")
    print("disagreement in unstable cells:", dict(kinds))
    print("numeric (fp32-bs1 vs fp16-bs1):", {k: f"{v['reading_differs']}/{v['lines']}" for k, v in num_stats.items()})
    print("\ncharacterisation (seed-unstable vs stable):")
    for r in char:
        a_, b_ = (r["median_unstable"], r["median_stable"]) if r["kind"] == "numeric" else (r["share_unstable"], r["share_stable"])
        print(f"  {r['feature']:30s} {a_:8.3f} vs {b_:8.3f}  q_raw {r['q_raw']:.3g}{'*' if r['fdr_raw'] else ' '}"
              f"  q_cer_matched {r.get('q_cer_matched', float('nan')):.3g}{'*' if r.get('fdr_cer_matched') else ' '}")
    print("\nshare of better / worse lines that are seed-unstable (base rate = all lines):")
    for r in comp:
        f = lambda d: f"{d['unstable']:4d}/{d['lines']:<5d}({100 * d['share']:4.1f}%)" if d["share"] is not None else "   —"  # noqa: E731
        print(f"  {r['comparison'][:44]:44s} {r['dataset']:14s} {r['rule']:12s} better {f(r['better'])} worse {f(r['worse'])} all {f(r['all'])}")


if __name__ == "__main__":
    main()
