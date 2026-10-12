"""Gallery for one before/after image comparison read by one model (per-line JSON from geom_transfer_eval.py).

    .geomvenv/bin/python scripts/build_pair_gallery.py --pkl P.pkl --cer C.json --base TAG --var TAG__x --labels A B --title T --note N OUT.html [--frag F]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from change_gallery import gallery, page  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--pkl", required=True)
ap.add_argument("--cer", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--var", required=True)
ap.add_argument("--labels", nargs=2, required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--note", default="")
ap.add_argument("--frag")
a = ap.parse_args()
rows = pickle.load(open(a.pkl, "rb"))["rows"]
img = {(r["dataset"], r["id"]): r["img"] for r in rows}
gt = {r["id"]: r["label"] for r in rows}
PL = json.loads(open(a.cer, encoding="utf-8").read())["per_line"]
items = []
for i, (ea, n, ha) in PL[a.base].items():
    if i in PL[a.var]:
        eb, _, hb = PL[a.var][i]
        items.append({"id": i, "gt": gt[i], "img_a": img[(a.base, i)], "img_b": img[(a.var, i)], "hyp_a": ha, "hyp_b": hb, "err_a": ea, "err_b": eb, "n": n})
frag = f"<section>{gallery(a.title, a.note, items, a.labels)}</section>"
Path(a.out).write_text(page(a.title, frag), encoding="utf-8")
if a.frag:
    Path(a.frag).write_text(frag, encoding="utf-8")
print("wrote", a.out)
