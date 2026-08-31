"""Run hosted-API models (Claude / Gemini / GPT) locally on a SMALL sample.

Same engine as notebook 02 (`run_sweep`, partial phase only) — just no git push.
Keeps cost down: `--limit` defaults to 20 and is hard-capped at 50.

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/run_hosted.py                       # claude x 20 muharaf lines
    uv run python scripts/run_hosted.py --dataset synthetic   # zero cost, no download
    uv run python scripts/run_hosted.py --models claude-haiku --limit 10
    uv run python scripts/run_hosted.py --push                # also commit + push results

Needs the model's API key in the environment (env_key from configs/models.yaml).
For `muharaf_lines`, set HF_TOKEN too if the anonymous download throttles.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
HARD_CAP = 50
DEFAULT_MODELS = ["claude-haiku", "claude-sonnet", "claude-opus"]


def _models_with_keys(names: list[str]) -> list[str]:
    cfg = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())
    keep = []
    for n in names:
        entry = cfg.get(n)
        if not entry:
            print(f"  skip {n}: not in configs/models.yaml")
            continue
        env_key = entry.get("env_key")
        if env_key and not os.environ.get(env_key):
            print(f"  skip {n}: {env_key} not set")
            continue
        keep.append(n)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated model names from configs/models.yaml")
    ap.add_argument("--dataset", default="muharaf_lines")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--push", action="store_true", help="git add/commit/push results after")
    args = ap.parse_args()

    limit = min(args.limit, HARD_CAP)
    if limit != args.limit:
        print(f"--limit clamped {args.limit} -> {HARD_CAP} (cost guard)")

    os.chdir(REPO)
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models = _models_with_keys(requested)
    if not models:
        print("\nno runnable models — set the API key(s) and retry, e.g.:\n"
              "  export ANTHROPIC_API_KEY=sk-ant-...")
        return 1
    print(f"running: {models}  x  {args.dataset}  ({limit} lines, partial only)\n")

    from arabic_ocr_bench.samples import export_run_samples
    from arabic_ocr_bench.sweep import run_sweep

    _ds_cache: dict = {}
    _hf_split = {"muharaf_lines": ("aamijar/muharaf-public", "test")}

    def get_image_fn(dataset: str):
        if dataset == "synthetic":
            from arabic_ocr_bench.datasets.synthetic import Synthetic

            imgs = [s.image for s in Synthetic(limit=limit)]
            return lambda i: imgs[int(i)] if int(i) < len(imgs) else None
        if dataset not in _hf_split:
            return lambda _i: None
        if dataset not in _ds_cache:
            import itertools

            from datasets import load_dataset

            hf_id, split = _hf_split[dataset]
            _ds_cache[dataset] = list(
                itertools.islice(load_dataset(hf_id, split=split, streaming=True), limit)
            )
        rows = _ds_cache[dataset]
        return lambda i: rows[int(i)]["image"] if int(i) < len(rows) else None

    def on_done(entry, run_dir):
        try:
            export_run_samples(run_dir, get_image=get_image_fn(entry["dataset"]), k=3)
        except Exception as e:  # noqa: BLE001
            print(f"  (sample export skipped: {e})")
        subprocess.run([sys.executable, "scripts/make_leaderboard.py", "results",
                        "-o", "results/leaderboard.html"], check=False)

    run_sweep(
        models=models,
        dataset=args.dataset,
        partial_limit=limit,
        run_full=False,
        concurrency=4,
        manifest_path="results/sweep_manifest.json",
        results_dir="results",
        on_progress=print,
        on_done=on_done,
    )

    latest = sorted((REPO / "results").glob("2*/summary.md"))
    if latest:
        print("\n" + latest[-1].read_text())
    print("leaderboard -> results/leaderboard.html")

    if args.push:
        subprocess.run(["git", "add", "-A", "results"], check=False)
        subprocess.run(["git", "commit", "-m",
                        f"hosted: {', '.join(models)} x {args.dataset} n={limit}"], check=False)
        subprocess.run(["git", "push"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
