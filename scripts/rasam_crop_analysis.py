"""CER of a RASAM prediction file by original crop height and by ink fraction (is the model failing on tall / multi-line crops?).

    PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/rasam_crop_analysis.py results/rasam/crnn/A1_w05_s0_best__r1__test.json RASAM-1 test
"""
from __future__ import annotations

import glob
import io
import json
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"


def main() -> None:
    pred, dataset, split = sys.argv[1:4]
    d = json.load(open(pred))
    imgs = []
    for f in sorted(glob.glob(f"{ROOT}{dataset}/snapshots/*/data/{split}-*.parquet")):
        for r in pq.read_table(f, columns=["image"]).to_pylist():
            imgs.append(Image.open(io.BytesIO(r["image"]["bytes"])))
    assert len(imgs) == len(d["lines"]), (len(imgs), len(d["lines"]))
    rows = []
    for l, im in zip(d["lines"], imgs):
        c = char_edit_counts(normalize(l["ref"], "raw").strip(), normalize(l["hyp"], "raw").strip())
        g = np.asarray(im.convert("L"))
        valid = g >= 12  # outside the black rotation mask
        h = im.size[1]
        chars = max(1, len(l["ref"]))
        # average pixel width available per reference character at the model's 64 px height
        rows.append({"h": h, "w": im.size[0], "err": c["sub"] + c["del"] + c["ins"], "ref": c["ref_len"],
                     "px_per_char_at64": im.size[0] * 64 / h / chars, "mask": 1 - valid.mean()})
    print(f"{pred.split('/')[-1]}  n={len(rows)}")
    for name, lo, hi in [("h<80", 0, 80), ("80-150", 80, 150), ("150-300", 150, 300), ("300+", 300, 10_000)]:
        sel = [r for r in rows if lo <= r["h"] < hi]
        if sel:
            print(f"  height {name:8s} n={len(sel):4d}  CER {sum(r['err'] for r in sel) / max(1, sum(r['ref'] for r in sel)):.3f}")
    qs = np.quantile([r["px_per_char_at64"] for r in rows], [0, .25, .5, .75, 1])
    for k in range(4):
        sel = [r for r in rows if qs[k] <= r["px_per_char_at64"] <= qs[k + 1]]
        print(f"  px per character at 64 px height, quartile {k + 1} ({qs[k]:.1f}-{qs[k + 1]:.1f}) n={len(sel)}  CER "
              f"{sum(r['err'] for r in sel) / max(1, sum(r['ref'] for r in sel)):.3f}")
    print("  median original height:", int(np.median([r["h"] for r in rows])), " median downscale factor to 64:", round(float(np.median([r["h"] / 64 for r in rows])), 2))


if __name__ == "__main__":
    main()
