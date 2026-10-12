"""How much of each 64 px line image does the writing fill, and how does CER depend on it?

    .geomvenv/bin/python scripts/geom_ink_extent.py results/geometry/transfer_eval.pkl results/geometry/base_evaldata.json

Ink extent = height of the band holding the central 96 % of ink pixels (Otsu threshold), as a share of the image height.
"""
import json
import pickle
import sys

import numpy as np
from skimage.filters import threshold_otsu

rows = [r for r in pickle.load(open(sys.argv[1], "rb"))["rows"] if "__" not in r["dataset"]]
per = json.load(open(sys.argv[2], encoding="utf-8"))["per_line"]
out = {}
for tag in ("kitab_khatt", "nakba_lines", "muharaf_lines"):
    ext, cer = [], []
    for r in rows:
        if r["dataset"] != tag:
            continue
        g = r["img"].astype(float)
        try:
            ink = g < threshold_otsu(g)
        except ValueError:
            continue
        prof = ink.sum(1)
        if prof.sum() == 0:
            continue
        c = np.cumsum(prof) / prof.sum()
        top, bot = np.searchsorted(c, 0.02), np.searchsorted(c, 0.98)
        ext.append((bot - top + 1) / g.shape[0])
        p = per.get(tag, {}).get(r["id"])
        cer.append(p[0] / p[1] if p else np.nan)
    ext, cer = np.array(ext), np.array(cer)
    bands = [(0, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.01)]
    res = {"lines": len(ext), "median_extent": float(np.median(ext)), "bands": []}
    for lo, hi in bands:
        s = (ext >= lo) & (ext < hi) & np.isfinite(cer)
        res["bands"].append({"extent": f"{lo:.2f}-{min(hi, 1):.2f}", "share_of_lines": float(((ext >= lo) & (ext < hi)).mean()), "mean_line_cer": float(np.mean(cer[s])) if s.sum() else None, "n": int(s.sum())})
    out[tag] = res
    print(tag, json.dumps(res))
json.dump(out, open("results/geometry/ink_extent.json", "w"), indent=1)
