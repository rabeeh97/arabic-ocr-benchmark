"""Ink-cropped copy of the aligner training pickle (X-S1 follow-up), for training on cropped lines. Runs on the 3090.

    python xai_inkcrop_traindata.py aligner_data.pkl aligner_data_inkcrop.pkl

The stored rows are already 64 px high, so the crop works on those: keep the rows between the 1st and 99th percentile of ink
(Otsu threshold) plus a quarter of that band above and below, then resize back to 64 px high. Muharaf lines are mostly tight,
so few change; the point is that training and test lines then go through the same step. Rows keep their order.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from xai_align import otsu  # noqa: E402

H = 64


def inkcrop64(g: np.ndarray) -> np.ndarray:
    ink = g < otsu(g)
    prof = ink.sum(1).astype(float)
    if prof.sum() == 0:
        return g
    c = np.cumsum(prof) / prof.sum()
    top, bot = int(np.searchsorted(c, 0.01)), int(np.searchsorted(c, 0.99))
    m = int(0.25 * (bot - top + 1)) + 2
    t, b = max(0, top - m), min(H, bot + m + 1)
    if b - t >= H - 4:  # already tight
        return g
    crop = g[t:b]
    w = max(16, min(1536, round(g.shape[1] * H / crop.shape[0])))
    return np.asarray(Image.fromarray(crop).resize((w, H), Image.LANCZOS), dtype=np.uint8)


blob = pickle.load(open(sys.argv[1], "rb"))
changed = 0
for r in blob["rows"]:
    new = inkcrop64(r["img"])
    if new.shape[1] // 4 >= len(r["label"]) + 4 and new is not r["img"]:
        r["img"] = new
        changed += 1
pickle.dump(blob, open(sys.argv[2], "wb"), protocol=4)
print("rows", len(blob["rows"]), "cropped", changed, flush=True)
