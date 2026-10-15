"""Contact sheet of CTC-span letter crops (base CRNN alignment) to judge whether spans cut letters cleanly enough
for a shape library, shape tokens and stitching. Uses the small inkcrop_eval.pkl (as-stored images = panel images)."""
import json
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = "/Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike"
J = json.load(open(f"{W}/results/panel/letters/aligner_run/letters.json", encoding="utf-8"))
L = [dict(zip(J["fields"], r)) for r in J["rows"]]
rows = pickle.load(open(f"{W}/results/geometry/inkcrop_eval.pkl", "rb"))["rows"]
img = {r["id"]: r["img"] for r in rows if "__" not in r["dataset"]}  # legacy ids: kitab_khatt/NNNNNN
want = [("ع", "ini"), ("ه", "fin"), ("س", "med"), ("ك", "med"), ("ح", "ini"), ("ي", "fin")]
pick = defaultdict(list)
rng = np.random.default_rng(1)
for r in rng.permutation(len(L)):
    x = L[int(r)]
    if x["tag"] not in ("kitab_khatt", "nakba_lines"):
        continue
    ds, _, idx = x["id"].split("/")
    lid = f"{ds}/{idx}"
    if lid not in img:
        continue
    k = (x["ch"], x["form"])
    if k in want and len(pick[k]) < 12:
        pick[k].append((lid, x))
CW, CH, PAD = 56, 64, 12
sheet = Image.new("L", (12 * (CW + 4) + 70, len(want) * (CH + 18) + 10), 255)
d = ImageDraw.Draw(sheet)
for row, k in enumerate(want):
    y = 5 + row * (CH + 18)
    d.text((4, y + 24), f"{k[0]} {k[1]}", fill=0)
    for col, (lid, x) in enumerate(pick[k]):
        g = img[lid]
        x0, x1 = max(0, x["x0"] - PAD), min(g.shape[1], x["x1"] + PAD)
        crop = Image.fromarray(g[:, x0:x1]).resize((CW, CH))
        sheet.paste(crop, (70 + col * (CW + 4), y))
        d.rectangle([70 + col * (CW + 4) + int(PAD * CW / max(1, x1 - x0)), y, 70 + col * (CW + 4) + CW - int(PAD * CW / max(1, x1 - x0)), y + CH - 1], outline=160)
        d.text((70 + col * (CW + 4), y + CH + 2), ("ok" if x["ok"] else "err") + f" {x['x1'] - x['x0']}px", fill=0)
sheet.save(W + "/results/analysis/S/span_sheet.png")
w = [x["x1"] - x["x0"] for x in L if x["tag"] == "kitab_khatt"]
print("span width px at 64 px height: median", np.median(w), "p10", np.percentile(w, 10), "p90", np.percentile(w, 90), "| share of spans = one CTC column (4 px):", np.mean(np.array(w) <= 4))
