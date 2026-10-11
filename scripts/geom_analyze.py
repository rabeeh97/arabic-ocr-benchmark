"""Summaries and tests on the per-line geometry (X-G1, X-G2, X-G4, X-G5, X-G6).

    .geomvenv/bin/python scripts/geom_analyze.py results/geometry/run1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
rows = [r for r in json.loads((run / "per_line.json").read_text()) if r.get("ok")]
H = np.load(run / "hists.npz")
tags_h = H["tag"]
ORDER = ["muharaf_train", "muharaf", "rasam1", "rasam2", "nakba", "nakba_blind", "khatt"]
CER = {"muharaf": 7.6, "rasam1": 10.0, "nakba": 13.8, "nakba_blind": 13.7, "khatt": 26.4}  # CRNN, from the ledger (X-E1)
SCALARS = ["sw_native", "dot_diam_over_sw", "sw_horizontal_stroke", "sw_vertical_stroke", "sw_cv", "asc_over_sw", "xheight_over_sw",
           "desc_over_sw", "asc_over_dot", "ink_density", "turn_mean_rad", "slant_skel_deg", "slant_proj_deg", "slope_proj_deg",
           "slope_rot_deg", "slope_skel_deg", "rule_share", "line_native_h", "n_dots_per_mark"]


def col(tag, key):
    return np.array([r.get(key, np.nan) for r in rows if r["tag"] == tag], float)


for r in rows:
    r["line_native_h"] = r["h0"]
    r["n_dots_per_mark"] = r["n_dots"] / r["exp_marks"] if r["exp_marks"] >= 5 else float("nan")
    hs, hv = r.get("sw_horizontal_stroke"), r.get("sw_vertical_stroke")
    r["pen_contrast"] = (hv / hs) if hs and hv and np.isfinite(hs) and np.isfinite(hv) and hs > 0 else float("nan")
SCALARS.append("pen_contrast")


def q(a):
    a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, [10, 25, 50, 75, 90])] if a.size else [np.nan] * 5


out = {"n": {t: int((tags_h == t).sum()) for t in ORDER}, "scalars": {}, "cer": CER}
for k in SCALARS:
    out["scalars"][k] = {t: q(col(t, k)) for t in ORDER}

# --- validation of the dot detector against the transcription -----------------------------------
val = {}
for t in ORDER:
    m = np.array([[r["n_dots"], r["exp_marks"], r["exp_dots"]] for r in rows if r["tag"] == t and r["exp_marks"] >= 3], float)
    val[t] = {"lines": len(m), "pearson_components_vs_marked_letters": float(np.corrcoef(m[:, 0], m[:, 1])[0, 1]),
              "pearson_components_vs_dots": float(np.corrcoef(m[:, 0], m[:, 2])[0, 1]),
              "median_components_per_marked_letter": float(np.median(m[:, 0] / m[:, 1]))}
out["dot_validation"] = val

# --- agreement between estimators --------------------------------------------------------------
def agree(a, b, tags):
    d = []
    for t in tags:
        x, y = col(t, a), col(t, b)
        ok = np.isfinite(x) & np.isfinite(y)
        d += list(x[ok] - y[ok])
    d = np.array(d)
    return {"n": int(d.size), "median_abs_diff_deg": float(np.median(np.abs(d))), "p90_abs_diff_deg": float(np.percentile(np.abs(d), 90))}


out["agreement"] = {
    "slope proj vs skel": agree("slope_proj_deg", "slope_skel_deg", ORDER),
    "slope proj vs rot": agree("slope_proj_deg", "slope_rot_deg", ORDER),
    "slant proj vs skel": agree("slant_proj_deg", "slant_skel_deg", ORDER),
}
for t in ORDER:
    x, y = col(t, "slant_proj_deg"), col(t, "slant_skel_deg")
    ok = np.isfinite(x) & np.isfinite(y)
    out["agreement"][f"slant corr {t}"] = float(np.corrcoef(x[ok], y[ok])[0, 1])

# --- histograms: per dataset mean, distances ---------------------------------------------------
def mean_hist(key, t):
    A = H[key][tags_h == t]
    if A.ndim == 3:  # loci (n, zone, code): normalise per line and zone
        A = A / np.maximum(A.sum(2, keepdims=True), 1)
    return A.mean(0)


def jsd(p, q_):
    p, q_ = p.ravel() / p.sum(), q_.ravel() / q_.sum()
    m = 0.5 * (p + q_)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q_, m)


HK = {"hinge": "hinge", "direction": "f0", "loci": "loci"}
mh = {name: {t: mean_hist(k, t) for t in ORDER} for name, k in HK.items()}
train_ref = {name: 0.5 * (mh[name]["muharaf_train"] + mh[name]["rasam1"]) for name in HK}  # training pool: Muharaf + RASAM
out["distance_to_training_pool"] = {}
for name in HK:
    out["distance_to_training_pool"][name] = {t: jsd(mh[name][t], train_ref[name]) for t in ORDER}
out["distance_to_muharaf_train"] = {name: {t: jsd(mh[name][t], mh[name]["muharaf_train"]) for t in ORDER} for name in HK}
out["pairwise_jsd_hinge"] = {a: {b: jsd(mh["hinge"][a], mh["hinge"][b]) for b in ORDER} for a in ORDER}
out["pairwise_jsd_loci"] = {a: {b: jsd(mh["loci"][a], mh["loci"][b]) for b in ORDER} for a in ORDER}

# scalar-profile distance: standardised median differences (pooled std across all lines)
SC_USE = ["sw_native", "dot_diam_over_sw", "pen_contrast", "asc_over_sw", "xheight_over_sw", "turn_mean_rad", "slant_skel_deg", "ink_density"]
allv = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in SC_USE}
sd = {k: np.nanstd(allv[k]) for k in SC_USE}
med = {t: {k: np.nanmedian(col(t, k)) for k in SC_USE} for t in ORDER}
out["scalar_distance_to_training_pool"] = {}
for t in ORDER:
    ref = {k: 0.5 * (med["muharaf_train"][k] + med["rasam1"][k]) for k in SC_USE}
    out["scalar_distance_to_training_pool"][t] = {
        "total": float(np.sqrt(np.mean([((med[t][k] - ref[k]) / sd[k]) ** 2 for k in SC_USE]))),
        **{k: float((med[t][k] - ref[k]) / sd[k]) for k in SC_USE}}

# --- loci: grouped shares per zone ---------------------------------------------------------------
def code(n, s, e, w):
    return n * 27 + s * 9 + e * 3 + w

groups = {
    "enclosed (ink on all four sides)": [code(n, s, e, w) for n in (1, 2) for s in (1, 2) for e in (1, 2) for w in (1, 2)],
    "open upward only (no ink above)": [code(0, s, e, w) for s in (1, 2) for e in (1, 2) for w in (1, 2)],
    "open downward only (no ink below)": [code(n, 0, e, w) for n in (1, 2) for e in (1, 2) for w in (1, 2)],
    "free (no ink on any side)": [code(0, 0, 0, 0)],
    "double cover (two or more runs on a side)": [c for c in range(81) if 2 in (c // 27, (c // 9) % 3, (c // 3) % 3, c % 3)],
}
out["loci_groups"] = {}
for t in ORDER:
    L = mean_hist("loci", t)  # (zone, 81), each zone sums to one
    out["loci_groups"][t] = {g: [float(L[z][idx].sum()) for z in range(3)] for g, idx in groups.items()}

# --- style separability: nearest centroid, leave-one-out on lines ------------------------------------
def features(sel):
    idx = np.flatnonzero(sel)
    return np.hstack([H["hinge"][idx], H["f0"][idx]]), idx


def loo_nc(tag_list, feats):
    X = np.vstack([feats[t] for t in tag_list])
    y = np.concatenate([[i] * len(feats[t]) for i, t in enumerate(tag_list)])
    X = np.sqrt(X)  # Hellinger-style, makes Euclidean fit histograms
    cm = np.vstack([X[y == c].mean(0) for c in range(len(tag_list))])
    cnt = np.bincount(y)
    correct = 0
    conf = np.zeros((len(tag_list),) * 2, int)
    for i in range(len(X)):
        c_loo = cm.copy()
        c_loo[y[i]] = (cm[y[i]] * cnt[y[i]] - X[i]) / (cnt[y[i]] - 1)
        p = int(np.argmin(((c_loo - X[i]) ** 2).sum(1)))
        conf[y[i], p] += 1
    return {"accuracy": float(np.trace(conf) / conf.sum()), "labels": tag_list, "confusion": conf.tolist(), "chance": 1 / len(tag_list)}


feats = {t: np.hstack([H["hinge"][tags_h == t], H["f0"][tags_h == t]]) for t in ORDER}
sep_tags = ["muharaf", "rasam1", "nakba", "khatt"]
out["separability_hinge_direction"] = loo_nc(sep_tags, feats)
loci_f = {t: H["loci"][tags_h == t].reshape(-1, 243) for t in ORDER}
loci_f = {t: v / np.maximum(v.sum(1, keepdims=True), 1) for t, v in loci_f.items()}
out["separability_loci"] = loo_nc(sep_tags, loci_f)
# within-dataset control: split nakba test vs blind test (same hand, different pages) should be near chance
out["separability_control_nakba_vs_blind"] = loo_nc(["nakba", "nakba_blind"], feats)
out["separability_control_muharaf_train_vs_test"] = loo_nc(["muharaf_train", "muharaf"], feats)

# --- X-G6: do the distances order the transfer datasets as the CER does? ------------------------------
# Reference = Muharaf train sample (the bulk of the training lines). Bootstrap over lines gives intervals; the
# Muharaf test sample and the NAKBA test/blind pair give the noise floor (same source, different lines).
rng = np.random.default_rng(0)
REF = "muharaf_train"
TR = ["nakba", "nakba_blind", "khatt"]
rowsby = {t: [r for r in rows if r["tag"] == t] for t in ORDER}
idx_by = {t: np.flatnonzero(tags_h == t) for t in ORDER}
# rows and hists line up one-to-one (both skip failed lines in the same order)
def hist_dist(name, key, t, ib, ref_ib):
    A = H[key][idx_by[t][ib]]
    R = H[key][idx_by[REF][ref_ib]]
    if A.ndim == 3:
        A = A / np.maximum(A.sum(2, keepdims=True), 1)
        R = R / np.maximum(R.sum(2, keepdims=True), 1)
    return jsd(A.mean(0), R.mean(0))

def scalar_profile(t, ib):
    return {k: np.nanmedian(np.array([rowsby[t][i].get(k, np.nan) for i in ib], float)) for k in SC_USE}

def boot(fn, n_t, n_ref, reps=200):
    vals = [fn(rng.integers(0, n_t, n_t), rng.integers(0, n_ref, n_ref)) for _ in range(reps)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

tt = {}
for t in ["muharaf"] + TR:
    n_t, n_ref = len(idx_by[t]), len(idx_by[REF])
    ent = {}
    for name, key in HK.items():
        ent[name] = {"jsd": hist_dist(name, key, t, np.arange(n_t), np.arange(n_ref)),
                     "ci95": boot(lambda a, b, key=key, name=name: hist_dist(name, key, t, a, b), n_t, n_ref)}
    refp = scalar_profile(REF, np.arange(len(rowsby[REF])))
    def sprof(a, b, t=t):
        pa_, pr_ = scalar_profile(t, a), scalar_profile(REF, b)
        return float(np.sqrt(np.mean([((pa_[k] - pr_[k]) / sd[k]) ** 2 for k in SC_USE])))
    full = sprof(np.arange(len(rowsby[t])), np.arange(len(rowsby[REF])))
    ent["scalar_profile"] = {"dist": full, "ci95": boot(lambda a, b: sprof(a, b), len(rowsby[t]), len(rowsby[REF]), 100),
                             "per_feature_sd": {k: float((scalar_profile(t, np.arange(len(rowsby[t])))[k] - refp[k]) / sd[k]) for k in SC_USE}}
    tt[t] = ent
out["transfer_tests"] = {"reference": REF, "cer": CER, "datasets": tt}

# --- X-G5: which unit is steadier, the stroke width or the dot? -------------------------------------
unit = {}
for k in ["asc_over_sw", "asc_over_dot", "xheight_over_sw"]:
    meds = {t: float(np.nanmedian(col(t, k))) for t in ["muharaf", "nakba", "khatt"]}
    within = {t: float((np.nanpercentile(col(t, k), 75) - np.nanpercentile(col(t, k), 25)) / np.nanmedian(col(t, k))) for t in ["muharaf", "nakba", "khatt"]}
    unit[k] = {"median_by_dataset": meds, "between_dataset_range_over_mean": float((max(meds.values()) - min(meds.values())) / np.mean(list(meds.values()))),
               "within_dataset_iqr_over_median": within}
out["unit_stability"] = unit

(run / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
np.savez_compressed(run / "mean_hists.npz", **{f"{n}__{t}": mh[n][t] for n in HK for t in ORDER})
print(json.dumps({k: out[k] for k in ["n", "dot_validation", "agreement", "distance_to_training_pool", "scalar_distance_to_training_pool", "separability_hinge_direction", "separability_loci", "separability_control_nakba_vs_blind", "separability_control_muharaf_train_vs_test", "transfer_tests", "unit_stability"]}, indent=1, default=float))
