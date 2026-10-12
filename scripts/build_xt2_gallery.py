"""Gallery for the transfer test (X-T2) on the 3090 CRNN: which lines slant and stroke changes helped or hurt.

    .geomvenv/bin/python scripts/build_xt2_gallery.py results/geometry/transfer_eval.pkl results/geometry/transfer_cer.json OUT.html [--frag FRAG.html]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from change_gallery import gallery, page  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pkl")
ap.add_argument("cer")
ap.add_argument("out")
ap.add_argument("--frag")
a = ap.parse_args()
rows = pickle.load(open(a.pkl, "rb"))["rows"]
imgs = {(r["dataset"], r["id"]): r["img"] for r in rows}
gts = {r["id"]: r["label"] for r in rows}
PL = json.loads(open(a.cer, encoding="utf-8").read())["per_line"]

COMPS = [
    ("kitab_khatt", "kitab_khatt__as_muharaf_only_slant", "KHATT, sheared to Muharaf’s slant", ("KHATT original", "sheared like Muharaf"),
     "The slant step alone carries the whole KHATT gain (−1.1 points). Lines that improve should be ones whose upright strokes the model misread; lines that get worse show what shearing costs."),
    ("muharaf_lines", "muharaf_lines__as_khatt_only_slant", "Muharaf, sheared upright like KHATT", ("Muharaf original", "sheared upright"),
     "The reverse test: removing Muharaf’s rightward lean raises its CER from 6.2 % to 8.2 %."),
    ("nakba_lines", "nakba_lines__as_muharaf_only_stroke", "NAKBA, strokes thinned to Muharaf’s width", ("NAKBA original", "thinner strokes"),
     "Thinning NAKBA’s strokes to Muharaf’s measured width gains 1.1 points."),
]
frags = []
for base, var, title, labels, note in COMPS:
    items = []
    for i, (ea, n, ha) in PL[base].items():
        if i not in PL[var]:
            continue
        eb, _, hb = PL[var][i]
        items.append({"id": i, "gt": gts[i], "img_a": imgs[(base, i)], "img_b": imgs[(var, i)], "hyp_a": ha, "hyp_b": hb, "err_a": ea, "err_b": eb, "n": n})
    frags.append(f"<section>{gallery(title, note, items, labels)}</section>")
body = ('<p>3090 CRNN (the aligner, 7.6 % held-out CER on Muharaf). Each card: the line before and after the change, the ground truth, '
        'and both readings with wrong characters highlighted (▾ marks missing ones). Lines are picked by the change in their error count, '
        'relative to their length.</p>' + "".join(frags))
Path(a.out).write_text(page("Slant and stroke transfer: line by line", body), encoding="utf-8")
if a.frag:
    Path(a.frag).write_text("".join(frags), encoding="utf-8")
print("wrote", a.out)
