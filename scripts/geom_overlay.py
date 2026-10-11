"""Overlay check for geom_measure: dots (red), body skeleton (blue), fitted baseline (green), slant ticks (orange)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import load_lines  # noqa: E402
from geom_measure import measure  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for tag in sys.argv[2:]:
    tiles = []
    for i, (g, lab) in enumerate(load_lines(tag, 6)):
        r = measure((tag, i, g, lab, True))
        if not r.get("ok"):
            continue
        d = r["_dbg"]
        rgb = np.stack([d["g1"]] * 3, -1).astype(np.uint8)
        rgb[d["skel"] & d["body"]] = (30, 90, 255)
        rgb[d["dots"] & ~d["skel"]] = (255, 40, 40)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        w = im.width
        dr.line([(0, d["bA"]), (w, d["aA"] * w + d["bA"])], fill=(0, 170, 0), width=1)
        tiles.append(im)
    W = max(t.width for t in tiles)
    sheet = Image.new("RGB", (W, sum(t.height + 4 for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height + 4
    sheet.save(out / f"{tag}.png")
