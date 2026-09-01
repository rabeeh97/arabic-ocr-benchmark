"""Run hosted-API models (Gemini / Mistral / OpenRouter / Claude / GPT) on a
SMALL sample. Same engine as notebook 02 (`run_sweep`).

Resume-safe: every finished (model, dataset, phase) is recorded in
`results/sweep_manifest.json` AND committed to git in the same step, and every
finished line is cached in `results/_cache/` (also committed). Re-running skips
whole done phases and, within a half-finished phase, never re-calls a line that
already has an answer — so a free-tier quota is never spent twice on the same
work. Keys come from `.env`; `--limit` defaults to 20, hard-capped at 50.

    uv run python scripts/run_hosted.py                        # every keyed model x 20 kitab_khatt lines, commit+push
    uv run python scripts/run_hosted.py --dataset synthetic    # zero cost, no download
    uv run python scripts/run_hosted.py --models claude-haiku --limit 10
    uv run python scripts/run_hosted.py --no-push              # commit locally only (offline)
    uv run python scripts/run_hosted.py --no-commit            # scratch run, persist nothing

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
        if entry.get("api_base_env") and not os.environ.get(entry["api_base_env"]):
            print(f"  skip {n}: {entry['api_base_env']} not set")
            continue
        keep.append(n)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="",
                    help="comma-separated names from configs/models.yaml "
                         "(default: every hosted model whose API key is set)")
    ap.add_argument("--dataset", default="kitab_khatt",
                    help="one name, or comma-separated for several "
                         "(kitab_khatt,kitab_hindawi,...)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--full", action="store_true",
                    help="also run the full-split phase after the partial one")
    ap.add_argument("--no-commit", action="store_true",
                    help="don't git-commit results (default: commit after every phase)")
    ap.add_argument("--no-push", action="store_true",
                    help="commit locally but don't git push")
    args = ap.parse_args()

    do_commit = not args.no_commit
    do_push = do_commit and not args.no_push

    limit = min(args.limit, HARD_CAP)
    if limit != args.limit:
        print(f"--limit clamped {args.limit} -> {HARD_CAP} (cost guard)")

    os.chdir(REPO)
    from arabic_ocr_bench.models.registry import available_hosted_models

    if args.models.strip():
        models = _models_with_keys([m.strip() for m in args.models.split(",") if m.strip()])
    else:
        models = available_hosted_models()
    if not models:
        print("\nno runnable models — set the API key(s) and retry, e.g.:\n"
              "  export ANTHROPIC_API_KEY=sk-ant-...")
        return 1
    phases = "partial + full" if args.full else "partial only"
    print(f"running: {models}  x  {args.dataset}  ({limit} lines, {phases})")
    print(f"persist: {'commit + push' if do_push else 'commit (local)' if do_commit else 'none'}\n")

    from arabic_ocr_bench.samples import export_run_samples
    from arabic_ocr_bench.sweep import run_sweep

    _ds_cache: dict = {}
    _hf_split = {
        "muharaf_lines": ("aamijar/muharaf-public", "test"),
        "kitab_khatt": ("ahmedheakl/arocrbench_khatt", "train"),
        "kitab_muharaf": ("ahmedheakl/arocrbench_muharaf", "train"),
        "kitab_historicalbooks": ("ahmedheakl/arocrbench_historicalbooks", "train"),
        "kitab_hindawi": ("ahmedheakl/arocrbench_hindawi", "train"),
    }

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
        """Persist THIS phase before run_sweep marks it done. Mirrors the Colab
        notebook's checkpoint(): a phase recorded in the manifest is always
        already in git, so nothing (and no paid/free call) is ever repeated."""
        try:
            export_run_samples(run_dir, get_image=get_image_fn(entry["dataset"]), k=3)
        except Exception as e:  # noqa: BLE001
            print(f"  (sample export skipped: {e})")
        subprocess.run([sys.executable, "scripts/make_leaderboard.py", "results",
                        "-o", "results/leaderboard.html"], check=False)
        if not do_commit:
            return
        # results/_cache is committed too (see .gitignore) — that's the per-line
        # resume so an interrupted phase doesn't re-call the lines it finished.
        subprocess.run(["git", "add", "-A", "results"], check=False)
        msg = (f"hosted: {entry['model']} {entry['dataset']} {entry['phase']} "
               f"n={entry['n']} CER-light {entry['cer_micro_light'] * 100:.1f}%")
        subprocess.run(["git", "commit", "-m", msg], check=False)
        if not do_push:
            return
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
        p = subprocess.run(["git", "push", "origin", "HEAD:main"],
                           capture_output=True, text=True, check=False)
        if p.returncode != 0:
            # raising rolls the manifest entry back -> phase re-runs next time,
            # but the per-line cache means that costs ~nothing
            raise RuntimeError("git push failed: " + (p.stderr.strip()[:300] or "unknown"))
        print("  push: ok")

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    for ds in datasets:
        print(f"\n########## DATASET: {ds} ##########")
        try:
            run_sweep(
                models=models,
                dataset=ds,
                partial_limit=limit,
                run_full=args.full,
                concurrency=4,
                manifest_path="results/sweep_manifest.json",
                results_dir="results",
                on_progress=print,
                on_done=on_done,
            )
        except Exception as e:  # noqa: BLE001 - one bad dataset shouldn't sink the rest
            import traceback

            traceback.print_exc()
            print(f"!!! DATASET {ds} FAILED ({type(e).__name__}: {e}) — continuing !!!")

    latest = sorted((REPO / "results").glob("2*/summary.md"))
    if latest:
        print("\n" + latest[-1].read_text())
    print("leaderboard -> results/leaderboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
