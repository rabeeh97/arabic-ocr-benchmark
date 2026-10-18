"""Flag suspect ground-truth labels using the model consensus, from existing predictions.

For each line, the consensus is the MBR medoid of all models' outputs (see
experiment_consensus_transcription.py). A label is suspicious when the ground truth is far from a
consensus that the models themselves agree on.

Two parts:
  1. Validation on injected noise. A fraction of ground-truth lines is corrupted in a known way
     (swap with another line / drop 30% of words / replace 30% of words) and we measure how well
     "distance from the consensus" ranks the corrupted lines first, versus the same distance
     computed from the single best model (what a one-model detector such as CER-based cleaning does).
  2. The real data: list the most suspicious lines among those where the models agree.

    PYTHONPATH=src python scripts/experiment_label_noise_detection.py --dataset kitab_khatt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from arabic_ocr_bench.metrics import score_pair  # noqa: E402
from experiment_confidence_baselines import load_latest  # noqa: E402
from experiment_consensus_transcription import medoid  # noqa: E402
from experiment_gliner2_verifier import normalize_for_model  # noqa: E402

PROFILE = "diac_insensitive"


def line_cer(gt: str, hyp: str) -> float:
    c = score_pair(gt, hyp, profiles=(PROFILE,))["char_counts"][PROFILE]
    return (c["sub"] + c["del"] + c["ins"]) / max(c["ref_len"], 1)


def corrupt(gt: str, kind: str, other_gts: list[str], rng: random.Random) -> str:
    words = gt.split()
    if kind == "swap":
        return rng.choice(other_gts)
    n = max(1, int(round(0.3 * len(words))))
    idx = set(rng.sample(range(len(words)), min(n, len(words))))
    if kind == "drop":
        kept = [w for i, w in enumerate(words) if i not in idx]
        return " ".join(kept) if kept else words[0]
    pool = " ".join(other_gts).split()
    return " ".join(rng.choice(pool) if i in idx else w for i, w in enumerate(words))


def auroc(score: np.ndarray, is_pos: np.ndarray) -> float:
    pos, neg = score[is_pos], score[~is_pos]
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def prec_at_k(score: np.ndarray, is_pos: np.ndarray) -> float:
    k = int(is_pos.sum())
    top = np.argsort(-score, kind="stable")[:k]
    return float(is_pos[top].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-pool", type=int, default=5)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    rows = load_latest(args.dataset)
    per_line: dict[str, list[str]] = defaultdict(list)
    for (m, s) in rows:
        per_line[m and s].append(m)
    lines = sorted(s for s, ms in per_line.items() if len(ms) >= args.min_pool)
    for s in lines:
        per_line[s].sort()
    norm = {k: normalize_for_model(r.get("cleaned_text") or "") for k, r in rows.items()}
    gt = {s: rows[(per_line[s][0], s)]["ground_truth"] for s in lines}

    cons_model, agree, cons_text = {}, {}, {}
    for s in lines:
        m, a = medoid(per_line[s], norm, s)
        cons_model[s], agree[s] = m, a
        cons_text[s] = rows[(m, s)].get("cleaned_text") or ""

    # hindsight-best single model (uses ground truth; a deliberately strong one-model baseline)
    cover = defaultdict(list)
    for s in lines:
        for m in per_line[s]:
            cover[m].append(s)
    cand = [m for m, ls in cover.items() if len(ls) >= 0.5 * len(lines)]
    mean_cer = {m: np.mean([line_cer(gt[s], rows[(m, s)].get("cleaned_text") or "") for s in cover[m]]) for m in cand}
    best = min(mean_cer, key=mean_cer.get)
    print(f"{args.dataset}: {len(lines)} lines; single-model baseline = {best} (mean CER {mean_cer[best]:.3f})")
    single_lines = [s for s in lines if (best, s) in rows]

    d_cons = {s: line_cer(gt[s], cons_text[s]) for s in lines}
    d_single = {s: line_cer(gt[s], rows[(best, s)].get("cleaned_text") or "") for s in single_lines}
    print(f"consensus-vs-GT CER: median {np.median(list(d_cons.values())):.3f}, "
          f"p90 {np.percentile(list(d_cons.values()), 90):.3f}")

    # ---- part 1: injected noise ------------------------------------------------------------
    kinds = ["swap", "drop", "replace"]
    res = {k: {"cons_auc": [], "cons_p": [], "single_auc": [], "single_p": []} for k in kinds}
    ls = single_lines
    for seed in range(args.seeds):
        rng = random.Random(seed)
        n_noisy = max(2, int(round(args.rate * len(ls))))
        noisy = set(rng.sample(ls, n_noisy))
        is_pos = np.array([s in noisy for s in ls])
        for kind in kinds:
            sc_c, sc_s = [], []
            for s in ls:
                g = gt[s]
                if s in noisy:
                    others = [gt[o] for o in ls if o != s]
                    g = corrupt(g, kind, others, rng)
                sc_c.append(line_cer(g, cons_text[s]))
                sc_s.append(line_cer(g, rows[(best, s)].get("cleaned_text") or ""))
            sc_c, sc_s = np.array(sc_c), np.array(sc_s)
            res[kind]["cons_auc"].append(auroc(sc_c, is_pos))
            res[kind]["cons_p"].append(prec_at_k(sc_c, is_pos))
            res[kind]["single_auc"].append(auroc(sc_s, is_pos))
            res[kind]["single_p"].append(prec_at_k(sc_s, is_pos))
    base_p = args.rate
    print(f"\ninjected noise: {args.rate:.0%} of {len(ls)} lines, {args.seeds} seeds "
          f"(random precision@k ~ {base_p:.2f}, AUROC 0.5)")
    print(f"{'noise':10}{'consensus AUROC':>17}{'single AUROC':>15}{'consensus P@k':>16}{'single P@k':>13}")
    for kind in kinds:
        r = res[kind]
        print(f"{kind:10}{np.mean(r['cons_auc']):>11.3f}±{np.std(r['cons_auc']):.3f}"
              f"{np.mean(r['single_auc']):>9.3f}±{np.std(r['single_auc']):.3f}"
              f"{np.mean(r['cons_p']):>12.3f}±{np.std(r['cons_p']):.2f}"
              f"{np.mean(r['single_p']):>8.3f}±{np.std(r['single_p']):.2f}")

    # ---- part 2: real data -----------------------------------------------------------------
    med_agree = float(np.median(list(agree.values())))
    confident = [s for s in lines if agree[s] >= med_agree]
    flagged = sorted(confident, key=lambda s: -d_cons[s])
    run_dir = REPO / "results" / "label_noise" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{args.dataset}"
    run_dir.mkdir(parents=True)
    with (run_dir / "flagged.jsonl").open("w", encoding="utf-8") as f:
        for s in sorted(lines, key=lambda x: -d_cons[x]):
            f.write(json.dumps({"sample_id": s, "consensus_cer_vs_gt": d_cons[s], "agreement": agree[s],
                                "confident": agree[s] >= med_agree, "consensus_model": cons_model[s],
                                "ground_truth": gt[s], "consensus": cons_text[s]}, ensure_ascii=False) + "\n")
    (run_dir / "injected_noise.json").write_text(json.dumps(
        {k: {m: float(np.mean(v)) for m, v in r.items()} for k, r in res.items()}, indent=1))
    print(f"\ntop {args.top} suspect lines among the {len(confident)} where models agree "
          f"(agreement >= {med_agree:.2f}); consensus_cer = CER(GT, consensus)")
    for s in flagged[: args.top]:
        print(f"- {s}  consensus_cer={d_cons[s]:.2f} agreement={agree[s]:.2f} (model {cons_model[s]})")
        print(f"    GT : {gt[s][:110]!r}")
        print(f"    CON: {cons_text[s][:110]!r}")
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
