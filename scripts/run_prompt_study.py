"""Prompt-sensitivity study driver (see docs/experiments/prompt-study.md).

Runs a matrix of (dataset, prompt_version) cells through the SAME small set
of samples per dataset, all on one cheap model (gemini-flash-lite by
default), so results are directly comparable and answer: what does the
choice of OCR prompt actually change?

Two phases, same matrix definition:
  --phase pilot  10-20 samples/dataset (default N per dataset below) — fast
                 read on every cell before spending quota on a full sweep.
  --phase full   the rest of each dataset (up to its real size), same cells.

Resumable by construction, no extra state file needed:
  - `indices=[0..N)` (pilot) / the dataset's full remaining range (full) is
    passed explicitly per (model, dataset) call, so the SAME sample_ids are
    requested every time this script runs.
  - the per-line result cache is keyed on (sample_id, model, prompt_version)
    — a sample already scored under a given prompt is never re-billed, so the
    `line-v2` baseline cells reuse whatever earlier benchmark runs already
    cached for gemini-flash-lite instead of re-fetching it.
  - if every Gemini key is daily-capped mid-cell, `runner.run` returns
    per-line rate-limit errors rather than hanging; this script logs the cell
    as incomplete and moves on to the next one rather than aborting the whole
    matrix. Just re-run the same command later (e.g. after the daily reset)
    to fill in whatever is still missing.

    uv run python scripts/run_prompt_study.py --phase pilot
    uv run python scripts/run_prompt_study.py --phase full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# HF's newer "xet" transfer backend gets stuck retrying the last ~1% of a
# download indefinitely on this network ("connection struggling" in its own
# transfer log — seen on nakba_lines and rasam1). Falling back to the classic
# HTTP downloader before any HF dataset is touched. Must be set before the
# first huggingface_hub / datasets import that could trigger a download.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import arabic_ocr_bench  # noqa: F401  -- loads .env

from arabic_ocr_bench.datasets.registry import build_dataset, load_dataset_configs
from arabic_ocr_bench.report import score_run
from arabic_ocr_bench.runner import run, scored_indices

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("results")
PROGRESS = ROOT / "prompt_study_progress.json"

MODEL = "gemini-flash-lite"
MODEL_PAGE = "gemini-flash-lite-640tok"   # page-v2 only: needs headroom past 256 tok

# All three of these have NEVER been successfully downloaded on this machine
# (unlike the other 5 datasets here, already cached locally from earlier
# sweeps — building them needs no network at all). muharaf_lines' ~36MB
# parquet drops mid-download via the classic HTTP path (IncompleteRead from
# the HF CDN — same reason run.free-accumulate.yaml excludes it).
# nakba_lines / rasam1 stalled identically via BOTH the xet backend
# ("connection struggling", stuck near 100%) and, after HF_HUB_DISABLE_XET,
# the classic downloader too (stuck dead at exactly 20MB) — a genuine network
# reliability problem on this machine for first-time large downloads, not an
# xet-specific bug. Cache-only until retried from a more reliable network.
CACHE_ONLY = {"muharaf_lines", "nakba_lines", "rasam1"}

# Datasets picked to look as different from each other as possible: script
# (Ruq'ah / Maghrebi / Naskh), condition (clean / degraded), and modality
# (handwritten / printed). kitab_historicalbooks only has 10 rows total.
PILOT_N = {
    "muharaf_lines": 15,          # HW, Ruq'ah, degraded (main training corpus)
    "nakba_lines": 15,            # HW, degraded 20th-c. (gated)
    "rasam1": 15,                 # HW, Maghrebi script
    "kitab_khatt": 15,            # HW, modern/clean
    "kitab_historyar": 15,        # HW, historical/degraded manuscripts
    "kitab_onlinekhatt": 15,      # HW, stylus-captured (cleaner strokes)
    "kitab_hindawi": 15,          # Printed, clean
    "kitab_historicalbooks": 10,  # Printed, degraded — only 10 rows exist
}

# prompt_version -> which datasets get it. `line-v2` (baseline) and the three
# always-on axes (no-modality, arabic, persona, no-diac) run on every dataset
# so every cell is comparable to the same baseline. `line-v2-claims-printed`
# also runs everywhere (it's the "correct" claim on the 2 printed sets AND the
# "wrong" claim on the 6 handwritten ones — one prompt, both control
# directions). `page-v2` only makes sense on the two page/paragraph-scale sets.
ALL_DATASETS = list(PILOT_N)
PAGE_DATASETS = ["kitab_hindawi", "kitab_historicalbooks"]

MATRIX = [
    # (prompt_version, model, datasets)
    ("line-v2", MODEL, ALL_DATASETS),                 # baseline (reuses old cache)
    ("line-v2-no-modality", MODEL, ALL_DATASETS),
    ("line-v2-claims-printed", MODEL, ALL_DATASETS),
    ("line-v2-arabic", MODEL, ALL_DATASETS),
    ("line-v2-persona", MODEL, ALL_DATASETS),
    ("line-v2-no-diac", MODEL, ALL_DATASETS),
    ("page-v2", MODEL_PAGE, PAGE_DATASETS),
]


def _dataset_size(name: str, dcfg: dict) -> int:
    """True row count — only called for --phase full. Reads the repo's cached
    results/_dataset_sizes.json first (already populated for most datasets by
    prior runs) so this never re-derives a size by fully building a dataset
    that's known to be slow/unreliable to enumerate on this network (e.g.
    muharaf_lines — see run.free-accumulate.yaml's own note on this)."""
    cache_path = ROOT / "_dataset_sizes.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = {}
    if name in cached:
        return int(cached[name])
    n = len(build_dataset(name, dcfg[name], limit=None, offset=0))
    cached[name] = n
    cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    return n


def _cell_indices(phase: str, dataset: str, n_pilot: int, dcfg: dict) -> list[int]:
    if phase == "pilot":
        # No need for the dataset's true size — n_pilot is far smaller than
        # any dataset here, and avoiding the size lookup sidesteps datasets
        # that are slow/unreliable to fully enumerate (e.g. muharaf_lines).
        return list(range(n_pilot))
    real_size = _dataset_size(dataset, dcfg)
    return list(range(min(n_pilot, real_size), real_size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "full"], default="pilot")
    args = ap.parse_args()

    dcfg = load_dataset_configs()
    print(f"phase: {args.phase}", flush=True)

    status: dict[str, dict] = {}
    for prompt_version, model, datasets in MATRIX:
        for ds in datasets:
            cell = f"{prompt_version} · {model} · {ds}"
            indices = _cell_indices(args.phase, ds, PILOT_N[ds], dcfg)
            if not indices:
                print(f"[skip] {cell}: nothing to score in this phase", flush=True)
                status[cell] = {"requested": 0, "done": True}
                continue
            already = set(scored_indices(ROOT, model, ds, prompt_version=prompt_version))
            todo = [i for i in indices if i not in already]
            if not todo:
                print(f"[done] {cell}: all {len(indices)} already cached", flush=True)
                status[cell] = {"requested": len(indices), "done": True}
                continue
            if ds in CACHE_ONLY:
                print(f"[cache-only] {cell}: {len(already)}/{len(indices)} cached, "
                      f"skipping {len(todo)} new (network-unreliable dataset)", flush=True)
                status[cell] = {"requested": len(indices), "done": False,
                                 "cached": len(already), "skipped_new": len(todo)}
                continue
            print(f"[run]  {cell}: {len(todo)}/{len(indices)} new", flush=True)
            run_cfg = {
                "models": [model],
                "datasets": [ds],
                "prompt_version": prompt_version,
                "indices": todo,
                "normalization_profiles": ["raw", "light", "diac_insensitive"],
                "concurrency": 3,
                "max_retries": 3,
                "cache": True,
                "results_dir": str(ROOT),
            }
            try:
                run_dir = run(run_cfg, results_root=ROOT)
                profiles = tuple(run_cfg.get("normalization_profiles") or ())
                score_run(run_dir, profiles=profiles) if profiles else score_run(run_dir)
            except Exception as e:  # noqa: BLE001 - log and move to the next cell
                print(f"[error] {cell}: {type(e).__name__}: {e}", flush=True)
                status[cell] = {"requested": len(indices), "done": False, "error": str(e)}
                continue
            still_missing = [i for i in todo
                              if i not in set(scored_indices(ROOT, model, ds,
                                                              prompt_version=prompt_version))]
            status[cell] = {
                "requested": len(indices),
                "done": not still_missing,
                "missing": len(still_missing),
            }
            if still_missing:
                print(f"[partial] {cell}: {len(still_missing)} still unscored "
                      f"(rate-limited?) — re-run this command later to fill in", flush=True)

        PROGRESS.write_text(json.dumps(status, indent=2), encoding="utf-8")

    remaining = sum(1 for v in status.values() if not v["done"])
    print(f"\n{'all cells complete' if remaining == 0 else f'{remaining} cell(s) incomplete — re-run to continue'}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
