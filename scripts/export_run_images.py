"""Dump every line image a run touched as `images.json` in the run dir:
{sample_id -> data-URI PNG}. The leaderboard's "Review — best & worst by model"
browser reads this so *every* row it lists can show (and enlarge) its image, not
just the handful `samples.json` curates.

    uv run python scripts/export_run_images.py results/<run_id> [--max-w 1100]

Loads each dataset the run used via its adapter with an explicit index list, so
it works offline against the HF cache (set HF_HUB_OFFLINE=1 if needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arabic_ocr_bench  # noqa: F401  -- loads .env

from arabic_ocr_bench.datasets.registry import build_dataset, load_dataset_configs
from arabic_ocr_bench.samples import image_data_uri


def _row_index(sample_id: str) -> int:
    try:
        return int(sample_id.rsplit("/", 1)[-1])
    except ValueError:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--max-w", type=int, default=560,
                    help="longest image edge, px (the leaderboard re-shrinks further)")
    args = ap.parse_args()

    preds_path = args.run_dir / "predictions.jsonl"
    if not preds_path.exists():
        sys.exit(f"no predictions.jsonl in {args.run_dir}")

    # dataset -> {row_index -> sample_id}
    wanted: dict[str, dict[int, str]] = {}
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ds, sid = r.get("dataset", ""), r.get("sample_id", "")
        if not ds or not sid:
            continue
        wanted.setdefault(ds, {})[_row_index(sid)] = sid

    dcfg = load_dataset_configs()
    out: dict[str, str] = {}
    for ds, rows in wanted.items():
        idxs = sorted(i for i in rows if i >= 0)
        adapter = build_dataset(ds, dcfg[ds], indices=idxs)
        got = 0
        for sample in adapter:
            sid = wanted[ds].get(_row_index(sample.id), sample.id)
            try:
                out[sid] = image_data_uri(sample.image, max_w=args.max_w)
                got += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {sid}: image failed: {e}")
        print(f"{ds}: {got}/{len(idxs)} images")

    dest = args.run_dir / "images.json"
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"-> {dest}  ({len(out)} images, {dest.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
