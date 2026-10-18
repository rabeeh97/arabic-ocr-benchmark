"""Per-sample paired diff between two (model, dataset) score sets.

Generalizes the ad-hoc analysis used to diagnose the Baseer-Nakba regression
(docs/experiments/findings-and-notes.md's "Per-sample diagnosis of the Baseer
regression") into a reusable tool, so the next regression (e.g. PaliGemma's
vision-unfrozen kitab_hindawi result) doesn't need the same snippet rewritten
from scratch.

    uv run python scripts/pair_diff.py \
        --model-a exp3090-paligemma2-3b-ft-bf16 \
        --model-b exp3090-paligemma2-3b-visft-ft-bf16 \
        --dataset kitab_hindawi

Reads every results/**/scores.jsonl under --results-dir, keeps the LATEST
row per (model, dataset, sample_id) if a sample was scored more than once
(by run-dir name, lexicographic == chronological for the ...Z timestamp
format this repo uses), inner-joins model-a's and model-b's rows by
sample_id, and reports how many samples got worse / better / stayed flat
under --metric (default cer_light), plus the biggest individual swings in
each direction with their sample_ids so you can go look at the actual
predictions.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def _load(results_dir: str, model: str, dataset: str, metric: str) -> dict[str, float]:
    """sample_id -> metric value, latest-scored-run wins on a repeat."""
    best: dict[str, tuple[str, float]] = {}  # sample_id -> (run_dir, value)
    for f in sorted(glob.glob(f"{results_dir}/**/scores.jsonl", recursive=True)):
        run_dir = Path(f).parent.name
        try:
            lines = open(f, encoding="utf-8").read().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model") != model or row.get("dataset") != dataset:
                continue
            if row.get("error"):
                continue
            v = row.get(metric)
            if v is None:
                continue
            sid = row["sample_id"]
            prev = best.get(sid)
            if prev is None or run_dir >= prev[0]:
                best[sid] = (run_dir, float(v))
    return {sid: v for sid, (_, v) in best.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True, help="baseline model")
    ap.add_argument("--model-b", required=True, help="comparison model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", default="cer_light",
                     help="scores.jsonl field to compare (default: cer_light)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--flat-eps", type=float, default=1e-9,
                     help="|delta| below this counts as flat (default: 1e-9)")
    ap.add_argument("--top", type=int, default=15,
                     help="how many of the biggest regressions/improvements to print")
    args = ap.parse_args(argv)

    a = _load(args.results_dir, args.model_a, args.dataset, args.metric)
    b = _load(args.results_dir, args.model_b, args.dataset, args.metric)
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"no shared scored sample_ids between {args.model_a!r} and "
              f"{args.model_b!r} on {args.dataset!r} -- nothing to compare.")
        return 1

    deltas = [(sid, b[sid] - a[sid], a[sid], b[sid]) for sid in shared]
    worse = [d for d in deltas if d[1] > args.flat_eps]     # b's metric grew = worse (for an error metric)
    better = [d for d in deltas if d[1] < -args.flat_eps]
    flat = [d for d in deltas if abs(d[1]) <= args.flat_eps]

    print(f"{args.model_a}  vs  {args.model_b}  on {args.dataset}  (metric={args.metric})")
    print(f"n paired = {len(shared)}   worse = {len(worse)}   better = {len(better)}   flat = {len(flat)}")
    mean_a = statistics.mean(a[sid] for sid in shared)
    mean_b = statistics.mean(b[sid] for sid in shared)
    print(f"mean {args.metric}: {args.model_a}={mean_a:.4f}  {args.model_b}={mean_b:.4f}"
          f"  (delta {mean_b - mean_a:+.4f})")

    worse.sort(key=lambda d: -d[1])
    better.sort(key=lambda d: d[1])
    print(f"\n--- {min(args.top, len(worse))} biggest regressions (B worse than A) ---")
    for sid, delta, va, vb in worse[: args.top]:
        print(f"  {sid}  {va:.4f} -> {vb:.4f}  ({delta:+.4f})")
    print(f"\n--- {min(args.top, len(better))} biggest improvements (B better than A) ---")
    for sid, delta, va, vb in better[: args.top]:
        print(f"  {sid}  {va:.4f} -> {vb:.4f}  ({delta:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
