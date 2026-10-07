"""Per-sample characteristics over complete model × sample blocks
(characterisation R3 / R4.1 / first pass at R7).

For every dataset with a complete block (>= 5 models scoring the same lines):

1. **Difficulty is a property of the line?** Mean pairwise Spearman between
   models' per-line CER, and split-half reliability of the consensus (median)
   CER over random model halves.
2. **What predicts consensus difficulty beyond length?** Spearman and
   length-partial Spearman of every GT feature (``samples.gt_features``) with
   the consensus CER.
3. **Letter forms.** Per-letter success rate by positional form, by dot
   ambiguity, by letter x form, and by relative position in the line.
4. **Label noise (R4.1).** How well consensus CER ranks the CER-HV
   human-verified flags on ``muharaf_lines`` (AUC), for all flags and for the
   real-noise types only (script-mismatch lines are correct labels the CRNN
   charset could not read). Plus the strict rule: every model fails and the
   models agree.
5. **Language prior.** Share of word substitutions that produce a real word
   (in Muharaf train), and whether that word is more frequent than the GT word.
6. **Shape of difficulty.** A multivariable rank regression of consensus CER
   on the GT features, the oracle gap (best model per line vs best single
   model), consensus CER by length quintile, and what predicts whole-line
   failures.

    PYTHONPATH=src .venv/bin/python scripts/sample_characteristics.py

Writes results/characterize/<stamp>/samples.json.
"""

from __future__ import annotations

import json
import random

import numpy as np
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from arabic_ocr_bench.characterize import samples as S
from arabic_ocr_bench.characterize.runs import load_canonical
from arabic_ocr_bench.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_leaderboard import in_train_datasets  # noqa: E402

MIN_MODELS = 5
SPLITS = 200
NOISE_CER = 0.5      # every model at or above this diac-insensitive CER ...
NOISE_AGREE = 0.35   # ... and the models' outputs within this normalised distance of each other
FAIL_CER = 0.8       # a line this bad is a whole-line failure
REGRESSORS = ("log_chars", "word_len", "paws_per_word", "ambiguous_share", "oov_share",
              "non_arabic_share", "diac_coverage", "lam_alef_per_word", "hamza_share")


def muharaf_train_vocab() -> Counter:
    from datasets import load_dataset

    ds = load_dataset("aamijar/muharaf-public", split="train").select_columns(["text"])
    c: Counter = Counter()
    for t in ds["text"]:
        if t:
            c.update(S.words_of(t))
    return c


def rank_regression(rows: list[dict]) -> dict:
    """OLS of rank-standardised ``y`` on rank-standardised REGRESSORS:
    {feature: {beta, t}} plus R^2. Constant columns are dropped."""
    def z(v):
        r = np.array(S._ranks(v), dtype=float)
        return (r - r.mean()) / r.std() if r.std() else None
    y = z([r["y"] for r in rows])
    names, cols = [], []
    for k in REGRESSORS:
        col = z([r.get(k, 0.0) for r in rows])
        if col is not None:
            names.append(k)
            cols.append(col)
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = max(len(y) - len(names), 1)
    se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.pinv(A.T @ A)))
    return {"r2": float(1 - resid @ resid / (y @ y)),
            "coef": {k: {"beta": float(b), "t": float(b / e) if e else 0.0}
                     for k, b, e in zip(names, beta, se)}}


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def nd(a: str, b: str) -> float:
    a, b = normalize(a, "diac_insensitive"), normalize(b, "diac_insensitive")
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def analyse_block(ds: str, models: list[str], sids: list[str], canon, vocab: Counter) -> dict:
    R = {m: canon[(m, ds)]["readings"] for m in models}
    gt = {s: R[models[0]][s].gt for s in sids}
    cer = {m: [S.line_cer(gt[s], R[m][s].pred) for s in sids] for m in models}
    cons = [S.median([cer[m][i] for m in models]) for i in range(len(sids))]

    # 1. reliability
    pair = [S.spearman(cer[a], cer[b]) for a, b in combinations(models, 2)]
    rng = random.Random(0)
    halves = []
    for _ in range(SPLITS):
        ms = models[:]
        rng.shuffle(ms)
        h1, h2 = ms[: len(ms) // 2], ms[len(ms) // 2:]
        c1 = [S.median([cer[m][i] for m in h1]) for i in range(len(sids))]
        c2 = [S.median([cer[m][i] for m in h2]) for i in range(len(sids))]
        halves.append(S.spearman(c1, c2))
    halves = [h for h in halves if h == h]

    # 2. features
    feats = [S.gt_features(gt[s], vocab) for s in sids]
    names = list(feats[0])
    length = [f["log_chars"] for f in feats]
    feat_stats = {}
    for k in names:
        col = [f[k] for f in feats]
        if len(set(col)) < 3:
            continue
        feat_stats[k] = {
            "rho": S.spearman(col, cons),
            "partial_rho": S.partial_spearman(col, cons, length) if k not in ("chars", "log_chars") else None,
            "mean": statistics.fmean(col),
        }

    # 3. letters
    by_pos, by_amb, by_shape, by_decile = Counter(), Counter(), Counter(), Counter()
    tot_pos, tot_amb, tot_shape, tot_decile = Counter(), Counter(), Counter(), Counter()
    by_pos_amb, tot_pos_amb = Counter(), Counter()
    per_model_pos = {m: [Counter(), Counter()] for m in models}
    for m in models:
        for s in sids:
            for letter, pos, rasm, amb, rel, ok in S.letter_outcomes(gt[s], R[m][s].pred):
                tot_pos[pos] += 1; by_pos[pos] += ok
                tot_amb[amb] += 1; by_amb[amb] += ok
                tot_pos_amb[(pos, amb)] += 1; by_pos_amb[(pos, amb)] += ok
                tot_shape[(letter, pos)] += 1; by_shape[(letter, pos)] += ok
                d = min(int(rel * 10), 9)
                tot_decile[d] += 1; by_decile[d] += ok
                per_model_pos[m][0][pos] += ok; per_model_pos[m][1][pos] += 1
    err = lambda good, tot, k: 1 - good[k] / tot[k] if tot[k] else None  # noqa: E731
    shape_err = {f"{l}|{p}": {"err": err(by_shape, tot_shape, (l, p)), "n": tot_shape[(l, p)]}
                 for (l, p) in tot_shape if tot_shape[(l, p)] >= 30 * len(models)}
    letters = {
        "position_err": {p: err(by_pos, tot_pos, p) for p in tot_pos},
        "ambiguous_err": {str(a): err(by_amb, tot_amb, a) for a in tot_amb},
        "position_ambiguous_err": {f"{p}|{a}": err(by_pos_amb, tot_pos_amb, (p, a)) for (p, a) in tot_pos_amb},
        "decile_err": [err(by_decile, tot_decile, d) for d in range(10)],
        "shape_err": shape_err,
        "models_final_worse_than_medial": sum(
            1 for m in models
            if per_model_pos[m][1]["final"] and per_model_pos[m][1]["medial"]
            and 1 - per_model_pos[m][0]["final"] / per_model_pos[m][1]["final"]
            > 1 - per_model_pos[m][0]["medial"] / per_model_pos[m][1]["medial"]),
    }

    # 4. label noise
    flags = []
    for i, s in enumerate(sids):
        if min(cer[m][i] for m in models) < NOISE_CER:
            continue
        preds = [R[m][s].pred for m in models]
        agree = S.median([nd(a, b) for a, b in combinations(preds, 2)])
        if agree <= NOISE_AGREE:
            flags.append({"sample_id": s, "consensus_cer": cons[i], "agreement": agree,
                          "gt": gt[s], "example_pred": preds[0]})

    # 5. language prior: word substitutions
    subs = real = up = down = 0
    for m in models:
        for s in sids:
            g, p = S.words_of(gt[s]), S.words_of(R[m][s].pred)
            for op in Levenshtein.opcodes(g, p):
                if op.tag != "replace" or op.src_end - op.src_start != op.dest_end - op.dest_start:
                    continue
                for gw, pw in zip(g[op.src_start:op.src_end], p[op.dest_start:op.dest_end]):
                    subs += 1
                    fg, fp = vocab.get(gw, 0), vocab.get(pw, 0)
                    if fp:
                        real += 1
                        if fg:
                            up += fp > fg
                            down += fp < fg

    # 6. shape of difficulty
    regression = rank_regression([feats[i] | {"y": cons[i]} for i in range(len(sids))])
    best1 = min(statistics.fmean(v) for v in cer.values())
    oracle = statistics.fmean(min(cer[m][i] for m in models) for i in range(len(sids)))
    fail = [sum(cer[m][i] >= FAIL_CER for m in models) / len(models) for i in range(len(sids))]
    order = sorted(range(len(sids)), key=lambda i: feats[i]["chars"])
    q = max(len(order) // 5, 1)
    quint = [order[k * q:(k + 1) * q] if k < 4 else order[4 * q:] for k in range(5)]

    return {
        "dataset": ds, "models": models, "n_samples": len(sids),
        "consensus_cer_mean": statistics.fmean(cons),
        "model_mean_cer": {m: statistics.fmean(cer[m]) for m in models},
        "reliability": {"pairwise_rho_median": S.median(pair), "pairwise_rho_min": min(pair),
                        "split_half_rho": statistics.fmean(halves)},
        "features": feat_stats,
        "letters": letters,
        "label_noise": {"flags": flags, "n_flags": len(flags)},
        "word_subs": {"n": subs, "real_word_share": real / subs if subs else None,
                      "both_in_vocab": up + down, "pred_more_frequent": up, "pred_less_frequent": down},
        "regression": regression,
        "oracle": {"best_single_model_cer": best1, "per_line_oracle_cer": oracle},
        "length_quintiles": [{"min_chars": feats[ix[0]]["chars"], "max_chars": feats[ix[-1]]["chars"],
                              "consensus_cer": statistics.fmean(cons[i] for i in ix),
                              "failure_share": statistics.fmean(fail[i] for i in ix)} for ix in quint if ix],
        "failure_rho": {k: S.spearman(fail, [f[k] for f in feats]) for k in ("chars", "non_arabic_share", "oov_share")},
        "per_sample": [{"sample_id": s, "consensus_cer": cons[i], **feats[i]} for i, s in enumerate(sids)],
    }


def main() -> None:
    canon = load_canonical(ROOT)
    in_train = in_train_datasets()
    vocab = muharaf_train_vocab()
    by_ds: dict[str, dict] = defaultdict(dict)
    for (m, ds), c in canon.items():
        by_ds[ds][m] = c["readings"]
    blocks = []
    for ds in sorted(by_ds):
        models, sids = S.complete_block(by_ds[ds], MIN_MODELS)
        if not models:
            continue
        print(f"{ds:24s} {len(models):2d} models x {len(sids):5d} lines", file=sys.stderr)
        b = analyse_block(ds, models, sids, canon, vocab)
        b["in_train_note"] = ds in in_train
        blocks.append(b)

    # label-noise check against CER-HV on muharaf_lines
    try:
        from rescore_muharaf_clean import flagged
        cerhv = flagged("test")
        for b in blocks:
            if b["dataset"] == "muharaf_lines":
                cons = {p["sample_id"]: p["consensus_cer"] for p in b["per_sample"]}
                pool = list(cons)
                real = {s for s, lab in cerhv.items() if not lab.lower().startswith("script mismatch")}
                keep = [s for s in pool if s not in cerhv or s in real]
                by_label: dict[str, list[float]] = defaultdict(list)
                for s in pool:
                    if s in cerhv:
                        by_label[cerhv[s]].append(cons[s])
                b["label_noise"]["cerhv"] = {
                    "block_lines": len(pool),
                    "flagged_in_block": sum(s in cerhv for s in pool),
                    "auc_all_flags": auc([cons[s] for s in pool], [s in cerhv for s in pool]),
                    "real_noise_in_block": sum(s in real for s in keep),
                    "auc_real_noise": auc([cons[s] for s in keep], [s in real for s in keep]),
                    "consensus_cer_by_label": {k: S.median(v) for k, v in by_label.items()},
                    "consensus_cer_unflagged": S.median([cons[s] for s in pool if s not in cerhv]),
                }
    except Exception as e:  # network or cache missing: the rest still stands
        print(f"CER-HV check skipped: {e}", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results/characterize" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.json").write_text(json.dumps({"generated": stamp, "blocks": blocks},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
