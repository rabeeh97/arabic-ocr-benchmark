"""Which measured characteristics predict a line's CER, jointly and across systems? (plan 2026-10-13, P1.7 / R7, V2 gate)

    .geomvenv/bin/python scripts/analyze_validity.py [--store results/store] [--out results/analysis/V]

Per line: image features from store_features.py and text features from the GT, standardised; target = line CER at the
no-diacritics rung, capped at 1, for several systems (CRNN base and combo on the panel, the standard-split CRNN when
present, the median fine-tuned VLM and the median base VLM over the runs that read the line). Lines a system trained on
are excluded. For each target:
  joint ridge regression (λ = 1) with a line-bootstrap 95 % interval per standardised coefficient;
  leave-one-dataset-out: fit on the other sets, Spearman ρ of predicted vs actual on the held-out set;
  and, for the VLM targets, the CRNN-fitted model's ρ on the VLM lines (does a characteristic that hurts the CRNN hurt the VLMs?).
A characteristic "survives" when its interval excludes 0 for ≥ 2 system families with the same sign.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from analyze_letters import words_view  # noqa: E402
from arabic_ocr_bench.characterize.text import analyze_letters  # noqa: E402

SETS = ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2")
FEATS = ["loose crop (1 − ink extent)", "slant distance from training (°)", "stroke width at 64 px", "pen contrast distance (|log|)",
         "ruled-line ink share", "pixels per character (log)", "dot diameter at 64 px", "line length (log chars)",
         "unseen-word share", "letter-list line", "PAWs per word", "rare letter forms share", "letters per word"]


def lc(r):
    return min(1.0, r["e_no_diacritics"] / max(1, r["n_no_diacritics"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/V")
    ap.add_argument("--vocab-pkl", type=Path, default=W / "results/panel/aligner_data_standard.pkl")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train_rows = [r for r in pickle.load(open(a.vocab_pkl, "rb"))["rows"] if r["role"] == "train"]
    vocab = {w for r in train_rows for w in words_view(r["label"])}
    form_n = defaultdict(int)
    for r in train_rows[:8000]:
        for li in analyze_letters(r["label"]):
            form_n[(li.letter, li.position)] += 1
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    F = {f["sample_key"]: f for f in pq.read_table(a.store / "features.parquet").to_pylist()}
    ref = json.loads((a.store / "features_reference.json").read_text())
    slant0 = ref["slant_deg"]["p50"]

    X, keys = [], []
    for k, s in S.items():
        f = F.get(k)
        if s["dataset"] not in SETS or not f or not f["ok"]:
            continue
        w = words_view(s["gt_raw"])
        lets = analyze_letters(s["gt_raw"])
        if not w or not lets:
            continue
        v = [1 - f["ink_extent"], abs((f["slant_deg"] if f["slant_deg"] is not None else slant0) - slant0), f["stroke_px64"] or 0,
             abs(math.log(f["pen_contrast"])) if f["pen_contrast"] else 0, f["rule_share"] or 0, math.log(max(1e-3, f["ink_px_per_char"])),
             f["dot_px64"] if f["dot_px64"] is not None else ref["dot_px64"]["p50"], math.log(max(1, len(s["gt_clean"]))),
             sum(x not in vocab for x in w) / len(w), float(sum(len(x) == 1 for x in w) >= 4), (max(li.paw for li in lets) + 1) / len(w),
             sum(form_n[(li.letter, li.position)] < 30 for li in lets) / len(lets), len(lets) / len(w)]
        X.append(v)
        keys.append(k)
    X = np.array(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    row_of = {k: i for i, k in enumerate(keys)}
    ds_of = np.array([S[k]["dataset"] for k in keys])

    lines = defaultdict(dict)  # target -> {key: cer}
    vlm = defaultdict(lambda: defaultdict(list))
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        if r.get("in_train") == "yes" or r["transform"] != "none" or r["sample_key"] not in row_of:
            continue
        run = r["run_id"]
        if r["pack"] == "panel-2026-10-13" and run in ("crnn/base", "crnn/combo", "crnn/std_s0", "crnn/std_s1"):
            lines[run][r["sample_key"]] = lc(r)
        elif run.startswith("vlm/") and not any(x in run for x in ("ain", "paddleocr", "trocr")) and r["family"] in ("vlm-ft", "vlm-base"):
            vlm[r["family"]][r["sample_key"]].append(lc(r))
    for fam, d in vlm.items():
        lines[f"median {fam}"] = {k: statistics.median(v) for k, v in d.items() if len(v) >= 2}

    def fit(idx, y, lam=1.0):
        A = np.c_[np.ones(len(idx)), Z[idx]]
        R = lam * np.eye(A.shape[1])
        R[0, 0] = 0
        return np.linalg.solve(A.T @ A + R, A.T @ y)

    res = {"features": FEATS, "targets": {}}
    rng = np.random.default_rng(0)
    crnn_beta = None
    for tgt, d in lines.items():
        ks = [k for k in d if k in row_of]
        if len(ks) < 150:
            continue
        idx = np.array([row_of[k] for k in ks])
        y = np.array([d[k] for k in ks])
        beta = fit(idx, y)
        boots = np.array([fit(idx[s], y[s]) for s in (rng.integers(0, len(idx), len(idx)) for _ in range(300))])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        lodo = {}
        for ds in SETS:
            m = ds_of[idx] == ds
            if m.sum() >= 50 and (~m).sum() >= 100:
                b = fit(idx[~m], y[~m])
                pred = np.c_[np.ones(m.sum()), Z[idx[m]]] @ b
                lodo[ds] = float(spearmanr(pred, y[m])[0])
        pred_all = np.c_[np.ones(len(idx)), Z[idx]] @ beta
        res["targets"][tgt] = {"lines": len(ks), "datasets": {ds: int((ds_of[idx] == ds).sum()) for ds in SETS},
                               "spearman_in_sample": float(spearmanr(pred_all, y)[0]),
                               "coef": {FEATS[j]: [float(beta[j + 1]), float(lo[j + 1]), float(hi[j + 1])] for j in range(len(FEATS))}, "lodo_spearman": lodo}
        if tgt == "crnn/base":
            crnn_beta = beta
    if crnn_beta is not None:
        for tgt in [t for t in res["targets"] if t.startswith("median")]:
            ks = [k for k in lines[tgt] if k in row_of]
            idx = np.array([row_of[k] for k in ks])
            y = np.array([lines[tgt][k] for k in ks])
            pred = np.c_[np.ones(len(idx)), Z[idx]] @ crnn_beta
            res["targets"][tgt]["crnn_model_spearman"] = float(spearmanr(pred, y)[0])
    surv = {}
    for j, f in enumerate(FEATS):
        signs = [np.sign(v["coef"][f][0]) for v in res["targets"].values() if v["coef"][f][1] > 0 or v["coef"][f][2] < 0]
        fams = [t.split("/")[0].split(" ")[-1] for t, v in res["targets"].items() if v["coef"][f][1] > 0 or v["coef"][f][2] < 0]
        surv[f] = {"significant_in": fams, "same_sign": len(set(signs)) <= 1, "survives": len({x.replace("vlm-ft", "vlm").replace("vlm-base", "vlm") for x in fams}) >= 2 and len(set(signs)) == 1}
    res["survival"] = surv
    (a.out / "validity.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'feature':38s}" + "".join(f"{t[:16]:>18s}" for t in res["targets"]) + "   survives")
    for f in FEATS:
        cells = []
        for t, v in res["targets"].items():
            b, lo_, hi_ = v["coef"][f]
            star = "*" if lo_ > 0 or hi_ < 0 else " "
            cells.append(f"{100 * b:+8.2f}{star}[{100 * lo_:+.1f},{100 * hi_:+.1f}]"[:18].rjust(18))
        print(f"{f[:38]:38s}" + "".join(cells) + f"   {'YES' if surv[f]['survives'] else ''}")
    for t, v in res["targets"].items():
        print(f"{t:22s} lines {v['lines']:5d}  in-sample ρ {v['spearman_in_sample']:.2f}  leave-one-dataset-out ρ "
              + ", ".join(f"{d} {x:.2f}" for d, x in v["lodo_spearman"].items()) + (f"  CRNN model on these lines ρ {v['crnn_model_spearman']:.2f}" if "crnn_model_spearman" in v else ""))


if __name__ == "__main__":
    main()
