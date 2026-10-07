"""Image layer, first pass (task R6.0): does the image explain the line
difficulty that the text cannot?

Takes the newest results/characterize/<stamp>/samples.json (per-sample GT
features + consensus difficulty for every complete block), loads each block's
raw line images from the local HF copy, measures ``images.image_features``,
and fits the same rank regression three ways: text only, image only, text +
image. Reports R^2 for each, the image coefficients, and simple rank
correlations of every image feature with difficulty.

    PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python scripts/image_layer.py [--datasets a,b]

Writes results/characterize/<stamp>/image_layer.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from arabic_ocr_bench.characterize import samples as S
from arabic_ocr_bench.characterize.images import image_features, load_images

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sample_characteristics as SC  # noqa: E402

TEXT = SC.REGRESSORS
IMAGE = ("height", "aspect", "ink_density", "core_height", "stroke_width", "stroke_rel",
         "ink_px_per_char", "contrast", "blur_lapvar", "skew_deg", "ink_left_edge")


def regression(rows: list[dict], keys: tuple[str, ...]) -> dict:
    old = SC.REGRESSORS
    SC.REGRESSORS = keys
    try:
        return SC.rank_regression(rows)
    finally:
        SC.REGRESSORS = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()
    want = set(filter(None, args.datasets.split(",")))
    src = sorted((ROOT / "results/characterize").glob("*/samples.json"))[-1]
    doc = json.loads(src.read_text(encoding="utf-8"))
    cfgs = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text(encoding="utf-8"))
    out_blocks = []
    for b in doc["blocks"]:
        ds = b["dataset"]
        if want and ds not in want:
            continue
        ps = b["per_sample"]
        try:
            ims = load_images(ds, [p["sample_id"] for p in ps], cfgs[ds])
        except Exception as e:
            print(f"{ds:20s} skipped: {e!r}"[:160], file=sys.stderr)
            continue
        rows = []
        for p in ps:
            img, _ = ims[p["sample_id"]]
            rows.append({**p, **image_features(img, p["chars"]), "y": p["consensus_cer"]})
        res = {
            "dataset": ds, "n": len(rows),
            "r2_text": regression(rows, TEXT)["r2"],
            "r2_image": regression(rows, IMAGE)["r2"],
        }
        both = regression(rows, TEXT + IMAGE)
        res["r2_both"] = both["r2"]
        res["image_coef"] = {k: v for k, v in both["coef"].items() if k in IMAGE}
        res["image_rho"] = {k: S.spearman([r[k] for r in rows], [r["y"] for r in rows]) for k in IMAGE}
        res["image_median"] = {k: float(np.median([r[k] for r in rows])) for k in IMAGE + ("width",)}
        out_blocks.append(res)
        sig = {k: round(v["beta"], 2) for k, v in res["image_coef"].items() if abs(v["t"]) > 2}
        print(f"{ds:20s} n={len(rows):5d} R2 text {res['r2_text']:.2f} image {res['r2_image']:.2f} "
              f"both {res['r2_both']:.2f}  sig image: {sig}", file=sys.stderr)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results/characterize" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "image_layer.json").write_text(json.dumps({"generated": stamp, "samples_source": str(src.relative_to(ROOT)),
                                                     "blocks": out_blocks}, indent=1), encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
