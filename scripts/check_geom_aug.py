"""Montage of real lines and four random draws of geom_aug each, to check the augmentation by eye.

    .geomvenv/bin/python scripts/check_geom_aug.py results/geometry/tiny OUT.png
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from geom_tiny import img, load  # noqa: E402
from xai_train_aligner import geom_aug, normalise  # noqa: E402

torch.manual_seed(1)
mm, meta = load(Path(sys.argv[1]), "eval")
pick = [m for m in meta if m["dataset"] == "muharaf_lines" and m["w"] >= 500][:2] + [m for m in meta if m["dataset"] == "kitab_khatt" and m["w"] >= 500][:2]
rows = []
for m in pick:
    g = img(mm, m)[:, -500:]
    x = torch.from_numpy(np.stack([normalise(g)] * 4)).float()[:, None]
    y, _ = geom_aug(x, torch.full((4,), 500))
    tiles = [np.pad(g, ((0, 0), (12, 12)), constant_values=255)] + [((t[0].clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8) for t in y]
    rows += tiles + [np.full((4, tiles[0].shape[1]), 128, np.uint8)]
Image.fromarray(np.concatenate(rows)).save(sys.argv[2])
print("saved")
