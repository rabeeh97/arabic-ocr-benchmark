"""Consensus transcription by minimum-Bayes-risk (MBR) selection, from existing predictions.

For every line, take all models' outputs and pick the one with the highest mean similarity to the
others (the "medoid"). No ground truth is used to choose. Then compare the picked output's CER with
each individual model's CER on the same lines.

Variants:
  mbr_all   medoid over every model that has an output for the line
  mbr_top   medoid over the half of the models with the highest overall agreement (ground-truth-free
            ranking, so this is still a deployable rule)
  oracle    best candidate per line by true CER (unreachable ceiling)
  random    a uniformly random candidate per line (floor)

    PYTHONPATH=src python scripts/experiment_consensus_transcription.py --dataset kitab_khatt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from arabic_ocr_bench.metrics import score_pair  # noqa: E402
from experiment_confidence_baselines import load_latest  # noqa: E402
from experiment_gliner2_verifier import normalize_for_model  # noqa: E402

PROFILES = ("raw", "diac_insensitive")


def errs_and_len(gt: str, text: str) -> dict[str, tuple[int, int]]:
    ps = score_pair(gt, text, profiles=PROFILES)
    out = {}
    for p in PROFILES:
        c = ps["char_counts"][p]
        out[p] = (c["sub"] + c["del"] + c["ins"], c["ref_len"])
    return out


def medoid(pool: list[str], norm: dict[tuple[str, str], str], s: str) -> tuple[str, float]:
    """Return (chosen model, its mean similarity to the other pool members)."""
    best, best_score = pool[0], -1.0
    for m in pool:
        others = [o for o in pool if o != m]
        score = float(np.mean([Levenshtein.normalized_similarity(norm[(m, s)], norm[(o, s)]) for o in others]))
        if score > best_score:
            best, best_score = m, score
    return best, best_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-pool", type=int, default=5, help="min models per line to form a consensus")
    ap.add_argument("--min-lines", type=int, default=100, help="models compared individually")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--committee", nargs="+", default=None,
                    help="restrict the pool to exactly these models (>= 3), e.g. to test a small ensemble")
    args = ap.parse_args()

    rows = load_latest(args.dataset)
    per_line: dict[str, list[str]] = defaultdict(list)
    for (m, s) in rows:
        per_line[s].append(m)
    for s in per_line:
        per_line[s].sort()
    if args.committee:
        if len(args.committee) < 3:
            raise SystemExit("--committee needs at least 3 models (a 2-model medoid is a coin flip)")
        for s in per_line:
            per_line[s] = [m for m in per_line[s] if m in args.committee]
        args.min_pool = len(args.committee)
    lines = sorted(s for s, ms in per_line.items() if len(ms) >= args.min_pool)
    norm = {k: normalize_for_model(r.get("cleaned_text") or "") for k, r in rows.items()}
    print(f"{args.dataset}: {len(lines)} lines with >= {args.min_pool} models")

    # per-candidate error counts (needed for the oracle and for scoring the picks)
    counts = {(m, s): errs_and_len(rows[(m, s)]["ground_truth"], rows[(m, s)].get("cleaned_text") or "")
              for s in lines for m in per_line[s]}

    # ground-truth-free model ranking by overall agreement, used for mbr_top
    agree_sum: dict[str, list[float]] = defaultdict(list)
    for s in lines:
        for m in per_line[s]:
            others = [o for o in per_line[s] if o != m]
            agree_sum[m].append(float(np.mean([Levenshtein.normalized_similarity(norm[(m, s)], norm[(o, s)])
                                               for o in others])))
    model_agree = {m: float(np.mean(v)) for m, v in agree_sum.items() if len(v) >= 20}
    cutoff = float(np.median(list(model_agree.values())))
    top_models = {m for m, a in model_agree.items() if a >= cutoff}
    print(f"mbr_top pool: {len(top_models)} of {len(model_agree)} models (agreement >= {cutoff:.3f})")

    rng = np.random.default_rng(0)
    pick = {"mbr_all": {}, "mbr_top": {}, "oracle": {}, "random": {}}
    for s in lines:
        pool = per_line[s]
        pick["mbr_all"][s] = medoid(pool, norm, s)[0]
        top_pool = [m for m in pool if m in top_models]
        pick["mbr_top"][s] = medoid(top_pool, norm, s)[0] if len(top_pool) >= 3 else pick["mbr_all"][s]
        pick["oracle"][s] = min(pool, key=lambda m: counts[(m, s)]["raw"][0] / max(counts[(m, s)]["raw"][1], 1))
        pick["random"][s] = pool[int(rng.integers(len(pool)))]

    run_dir = REPO / "results" / "consensus" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{args.dataset}"
    run_dir.mkdir(parents=True)
    with (run_dir / "picks.jsonl").open("w", encoding="utf-8") as f:
        for s in lines:
            f.write(json.dumps({"sample_id": s, "n_pool": len(per_line[s]),
                                **{v: pick[v][s] for v in pick}}) + "\n")

    eval_models = sorted(m for m in {m for s in lines for m in per_line[s]}
                         if sum((m, s) in rows for s in lines) >= args.min_lines)
    systems = ["mbr_all", "mbr_top", "oracle", "random"]

    def cer(items: list[tuple[int, int]]) -> float:
        e, n = sum(x[0] for x in items), sum(x[1] for x in items)
        return e / n if n else 0.0

    out_rows = []
    print(f"\n{'model':38}{'n':>5}{'profile':>17}{'model':>8}{'mbr_all':>9}{'mbr_top':>9}{'oracle':>8}{'random':>8}"
          f"{'mbr_all-model  [95% CI]':>32}")
    for prof in PROFILES:
        for m in eval_models:
            ls = [s for s in lines if (m, s) in rows]
            own = [counts[(m, s)][prof] for s in ls]
            sysc = {v: [counts[(pick[v][s], s)][prof] for s in ls] for v in systems}
            diffs = np.array([sysc["mbr_all"][i][0] - own[i][0] for i in range(len(ls))], dtype=float)
            lens = np.array([own[i][1] for i in range(len(ls))], dtype=float)
            boots = []
            for _ in range(args.boot):
                ix = rng.integers(0, len(ls), len(ls))
                boots.append(diffs[ix].sum() / max(lens[ix].sum(), 1))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            d = diffs.sum() / lens.sum()
            row = {"profile": prof, "model": m, "n": len(ls), "cer_model": cer(own),
                   **{f"cer_{v}": cer(sysc[v]) for v in systems},
                   "mbr_all_minus_model": d, "ci_lo": float(lo), "ci_hi": float(hi)}
            out_rows.append(row)
            print(f"{m:38}{len(ls):>5}{prof:>17}{row['cer_model']:8.3f}{row['cer_mbr_all']:9.3f}"
                  f"{row['cer_mbr_top']:9.3f}{row['cer_oracle']:8.3f}{row['cer_random']:8.3f}"
                  f"{d:+12.3f} [{lo:+.3f},{hi:+.3f}]")
    with (run_dir / "per_model.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
