"""X-G7 CPU proxy: how much dot contrast survives resizing a line to height 32 / 48 / 64, plain and dot-preserving.

    .geomvenv/bin/python scripts/geom_dots.py results/geometry/run1

Uses the dot centroids that geom_measure.py saved (native pixels). For every dot, contrast = (local paper level -
darkest pixel in the 3x3 patch at the mapped position) / (local paper level - darkest pixel of the dot in the native
image). 1 = fully kept, 0 = gone. Also reports the dot diameter after resizing and how often it falls under 2 px.
Variants: "plain" = Lanczos (what the CRNN pipeline does); "minpool" = grey minimum over a window of the shrink
factor, then Lanczos (ink-preserving: dots cannot dilute, strokes thicken by about one source window).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import load_lines  # noqa: E402

run = Path(sys.argv[1])
dots = json.loads((run / "dots.json").read_text())
TAGS = ["muharaf", "nakba", "khatt", "rasam1"]
HEIGHTS = [32, 48, 64]
N_LINES = 120


def resized(g: np.ndarray, h: int, variant: str) -> tuple[np.ndarray, float]:
    f = g.shape[0] / h
    src = g
    if variant == "minpool" and f > 1:
        k = int(np.ceil(f))
        src = ndi.minimum_filter(g, size=(k, k))
    w = max(16, round(g.shape[1] / f))
    return np.asarray(Image.fromarray(src).resize((w, h), Image.LANCZOS), dtype=np.float32), f


def patch_min(a, x, y, r):
    x0, y0 = int(round(x)), int(round(y))
    p = a[max(0, y0 - r) : y0 + r + 1, max(0, x0 - r) : x0 + r + 1]
    return float(p.min()) if p.size else np.nan


res = {}
for tag in TAGS:
    acc = {(h, v): {"contrast": [], "diam": [], "stroke_ratio": []} for h in HEIGHTS for v in ("plain", "minpool")}
    for i, (g, _) in enumerate(load_lines(tag, N_LINES)):  # same seed, so line i matches geom_measure's index i
        if f"{tag}/{i}" not in dots:
            continue
        d = dots[f"{tag}/{i}"]
        if not d:
            continue
        gf = g.astype(np.float32)
        paper = ndi.median_filter(gf, size=(15, 15)) if g.size < 4e5 else gf
        for h in HEIGHTS:
            for v in ("plain", "minpool"):
                rs, f = resized(g, h, v)
                paper_r = ndi.median_filter(rs, size=(9, 9))
                ink_med = float(np.percentile(rs, 5))
                for cx, cy, diam in d[:40]:
                    r0 = max(1, int(diam / 2))
                    nat_min = patch_min(gf, cx, cy, r0)
                    pap = float(paper[int(min(cy, g.shape[0] - 1)), int(min(cx, g.shape[1] - 1))])
                    if pap - nat_min < 25:  # dot not clearly darker than paper in the source: not usable
                        continue
                    x, y = cx / f, cy / f
                    if not (0 <= int(round(y)) < rs.shape[0] and 0 <= int(round(x)) < rs.shape[1]):
                        continue
                    pr = float(paper_r[int(round(y)), int(round(x))])
                    c = (pr - patch_min(rs, x, y, 1)) / max(pap - nat_min, 1)
                    acc[(h, v)]["contrast"].append(float(np.clip(c, 0, 1.2)))
                    acc[(h, v)]["diam"].append(diam / f)
    res[tag] = {}
    for (h, v), a in acc.items():
        c, dm = np.array(a["contrast"]), np.array(a["diam"])
        res[tag][f"{h}_{v}"] = {"dots": int(c.size), "mean_contrast": float(c.mean()), "share_contrast_below_half": float((c < 0.5).mean()),
                                "median_diam_px": float(np.median(dm)), "share_diam_below_2px": float((dm < 2).mean())}
    print(tag, {k: (v["dots"], round(v["mean_contrast"], 2), round(v["share_diam_below_2px"], 2)) for k, v in res[tag].items()}, flush=True)

(run / "dots_resize.json").write_text(json.dumps(res, indent=1))
