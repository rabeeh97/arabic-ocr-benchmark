"""Drive configs/run.gemini-full.yaml to completion.

Loops `runner.run` (accumulate mode) until every (gemini model, dataset) pair has
a non-errored result for every sample. The per-line cache makes it fully
resumable: a killed process loses nothing, and a re-run skips everything already
done. When a pass makes no progress (a daily quota is exhausted) it sleeps ~1h
and tries again.

    nohup uv run python scripts/run_gemini_full.py > gemini_full.log 2>&1 &

Progress is written to results/gemini_full_progress.json after every pass.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

import arabic_ocr_bench  # noqa: F401  -- loads .env

from arabic_ocr_bench.datasets.registry import build_dataset, load_dataset_configs
from arabic_ocr_bench.report import score_run
from arabic_ocr_bench.runner import run, scored_indices

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "run.gemini-full.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
ROOT = Path(CFG.get("results_dir", "results"))
PROGRESS = ROOT / "gemini_full_progress.json"
CAPPED_SLEEP_S = 3600      # a zero-progress pass => daily quota; wait it out
ERROR_SLEEP_S = 120


def _dataset_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    cached = ROOT / "_dataset_sizes.json"
    if cached.exists():
        try:
            sizes.update({k: int(v) for k, v in json.loads(cached.read_text()).items()})
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    dcfg = load_dataset_configs()
    for d in CFG["datasets"]:
        if d not in sizes:
            sizes[d] = len(build_dataset(d, dcfg[d], limit=None, offset=0))
    return sizes


def _status(sizes: dict[str, int]) -> dict[str, dict[str, list[int]]]:
    return {
        m: {d: [len(scored_indices(ROOT, m, d)), sizes.get(d, 0)] for d in CFG["datasets"]}
        for m in CFG["models"]
    }


def _remaining(st: dict) -> int:
    return sum(max(0, total - done) for m in st.values() for done, total in m.values())


def _done(st: dict) -> bool:
    return all(done >= total > 0 for m in st.values() for done, total in m.values())


def main() -> int:
    sizes = _dataset_sizes()
    print(f"targets: {sizes}", flush=True)
    while True:
        st = _status(sizes)
        rem = _remaining(st)
        PROGRESS.write_text(json.dumps(
            {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "remaining": rem, "done": _done(st), "status": st},
            indent=2,
        ))
        print(f"[{time.strftime('%H:%M:%S')}] remaining rows: {rem}", flush=True)
        if _done(st):
            print("=== GEMINI FULL SWEEP COMPLETE ===", flush=True)
            return 0

        before = sum(done for m in st.values() for done, _ in m.values())
        try:
            rd = run(dict(CFG))
            score_run(rd, progress=False)
        except Exception as e:  # noqa: BLE001 - keep looping through anything transient
            print(f"  pass errored ({type(e).__name__}: {e}); retry in {ERROR_SLEEP_S}s", flush=True)
            time.sleep(ERROR_SLEEP_S)
            continue

        after = sum(done for m in _status(sizes).values() for done, _ in m.values())
        gained = after - before
        print(f"  pass -> {rd.name}: +{gained} rows scored", flush=True)
        if gained == 0:
            print(f"  no progress (daily quota?) — sleeping {CAPPED_SLEEP_S // 60} min", flush=True)
            time.sleep(CAPPED_SLEEP_S)


if __name__ == "__main__":
    sys.exit(main())
