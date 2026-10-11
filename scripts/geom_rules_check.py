"""Montage of NAKBA lines before and after ruled-line removal (from the export pickle), for a visual check."""
import pickle
import sys

import numpy as np
from PIL import Image

rows = pickle.load(open(sys.argv[1], "rb"))["rows"]
a = {r["id"]: r["img"] for r in rows if r["dataset"] == "nakba_lines"}
b = {r["id"]: r["img"] for r in rows if r["dataset"] == "nakba_lines__norules"}
chg = [k for k in a if a[k].shape == b[k].shape and (a[k] != b[k]).any()][:6]
tiles = []
for k in chg:
    w = min(900, a[k].shape[1])
    tiles += [a[k][:, -w:], b[k][:, -w:], np.full((6, w), 128, np.uint8)]
W = max(t.shape[1] for t in tiles)
canvas = np.full((sum(t.shape[0] for t in tiles), W), 255, np.uint8)
y = 0
for t in tiles:
    canvas[y : y + t.shape[0], : t.shape[1]] = t
    y += t.shape[0]
Image.fromarray(canvas).save(sys.argv[2])
print(len(chg), "changed lines shown")
