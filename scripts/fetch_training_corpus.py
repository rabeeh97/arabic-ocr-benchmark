"""Pre-fetch every real training-data source once, shared across a whole suite run.

    python scripts/fetch_training_corpus.py --dir C:\\Users\\User\\suite-v1

NOT run automatically — the box operator runs this once, after the shared
venv exists (needs the `datasets`/`huggingface_hub` libraries, so run it
*inside* the suite venv, e.g. via `run_3090_experiment.py`'s bootstrap phase
having already created one, or any environment with this repo's `[local]`
extra installed) and before `run_experiment_suite.py` starts the sequential
model queue.

Without this, each of the ~4 sequential model runs in the suite would
independently re-download the same HF corpora (Muharaf, RASAM-1, RASAM-2)
via its own `phase_fetch`. This script warms **the same** shared cache the
suite's runs use — `<--dir>/_shared/hf_cache`, which is exactly what
`run_experiment_suite.py` passes as `--shared <--dir>/_shared` to both
`run_3090_experiment.py` and `run_3090_benchmark.py`, and what those pin as
`HF_HOME`. Keep these paths in sync: if the shared layout changes, this
prefetch silently warms a cache nobody reads.

Also runs `fetch_arams.py` (AraMS-28k-HTR, the one non-HF source) into the
SAME repo-relative `data/arams_28k_htr` that `configs/datasets.yaml`'s
`arams_htr` entry and every `arams_train` training source already hardcode as
`params.root` — deliberately not a suite-specific path, so eval and every
model's training config keep agreeing on where the corpus lives without a
per-suite override mechanism. Override with `--arams-dest` only if you
genuinely want a different location (and update every config that
references it to match).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fetch_arams  # noqa: E402  (local script import, path set above)


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from <dir>/experiment.env into os.environ — same
    convention as run_3090_experiment.py's `_load_env_file`, duplicated here
    (not imported) because that module is a `__main__`-oriented launcher, not
    a library. Muharaf is gated; without this, HF_TOKEN is never set and the
    gated download fails with an auth error instead of using the token that's
    sitting right next to this script's --dir."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Every real (non-synthetic) training-data source in the round-1 mix — kept
# as a flat list here rather than re-parsing training YAML configs, since
# pre-fetching is deliberately a superset: it's fine (just a few extra small
# `val` splits) to always fetch everything any suite model config might use.
_HF_SOURCES: list[tuple[str, str]] = [
    ("aamijar/muharaf-public", "train"),
    ("calfa-ai/RASAM-1", "train"),
    ("calfa-ai/RASAM-1", "val"),
    ("calfa-ai/RASAM-2", "train"),
    ("calfa-ai/RASAM-2", "val"),
]


def _fetch_hf(hf_id: str, split: str) -> None:
    from datasets import load_dataset

    print(f"fetch {hf_id}:{split} ...")
    ds = load_dataset(hf_id, split=split)
    print(f"  {len(ds)} rows cached")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="the suite's shared directory (for the HF cache)")
    ap.add_argument(
        "--arams-dest",
        default="data/arams_28k_htr",
        help="repo-relative path AraMS is fetched into — must match every "
        "config's params.root (default matches configs/datasets.yaml's arams_htr entry)",
    )
    ap.add_argument("--skip-arams", action="store_true", help="HF sources only")
    ap.add_argument("--skip-hf", action="store_true", help="AraMS only")
    args = ap.parse_args()

    suite_dir = Path(args.dir)
    # Must match run_experiment_suite.py's _shared_dir() + the `hf_cache` key in
    # run_3090_experiment.py's _dirs(), or this warms a cache nothing reads.
    hf_cache = suite_dir / "_shared" / "hf_cache"
    arams_root = Path(args.arams_dest)
    hf_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache)
    # experiment.env lives next to --dir (same convention as
    # run_3090_experiment.py) — load it BEFORE any gated download.
    _load_env_file(suite_dir / "experiment.env")

    if not args.skip_hf:
        for hf_id, split in _HF_SOURCES:
            _fetch_hf(hf_id, split)

    if not args.skip_arams:
        arams_root.mkdir(parents=True, exist_ok=True)
        old_argv = sys.argv
        sys.argv = ["fetch_arams.py", "--dest", str(arams_root)]
        try:
            rc = fetch_arams.main()
        finally:
            sys.argv = old_argv
        if rc:
            print(f"fetch_arams.py exited {rc} — inspect {arams_root} before starting the suite")
            return rc

    print(f"\nShared cache ready under {suite_dir / '_shared'}")
    print(f"  HF_HOME:   {hf_cache}")
    print(f"  AraMS root: {arams_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
