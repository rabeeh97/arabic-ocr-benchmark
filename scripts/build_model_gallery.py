"""Gallery comparing two models' readings of the same lines (per-line JSON from geom_transfer_eval.py or xai_e8_spike.py).

    .geomvenv/bin/python scripts/build_model_gallery.py --pkl eval_data.pkl --a base.json --b aug.json --labels "CRNN" "CRNN + slant aug" \
        --tags kitab_khatt nakba_lines --title "..." --note "..." OUT.html [--frag FRAG.html]

For E8 output (per_line.json with gt / crnn / e8 fields) pass the same file as --a and --b with --e8.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from change_gallery import gallery, page  # noqa: E402
from xai_train_aligner import edit_distance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--pkl", required=True)
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--labels", nargs=2, required=True)
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--names", nargs="+")
ap.add_argument("--title", required=True)
ap.add_argument("--note", default="")
ap.add_argument("--e8", action="store_true")
ap.add_argument("--frag")
ap.add_argument("--ids", help="FILE.json:KEY, keep only the line ids listed under KEY (e.g. shape/halves.json:half_b)")
ap.add_argument("--b-suffix", default="", help="read B's lines from tag + suffix (e.g. __tta in xai_geom_tta.py output)")
a = ap.parse_args()
keep = None
if a.ids:
    f, k = a.ids.rsplit(":", 1)
    keep = set(json.loads(open(f, encoding="utf-8").read())[k])
rows = pickle.load(open(a.pkl, "rb"))["rows"]
img = {(r["dataset"], r["id"]): r["img"] for r in rows}
gt = {(r["dataset"], r["id"]): r["label"] for r in rows}
names = dict(zip(a.tags, a.names or a.tags))
frags = []
if a.e8:
    P = json.loads(open(a.a, encoding="utf-8").read())
    for t in a.tags:
        items = []
        for i, v in P[t].items():
            if (t, i) not in img or (keep is not None and i not in keep):
                continue
            ea, eb = edit_distance(list(v["crnn"]), list(v["gt"])), edit_distance(list(v["e8"]), list(v["gt"]))
            items.append({"id": i, "gt": v["gt"], "img_a": img[(t, i)], "img_b": None, "hyp_a": v["crnn"], "hyp_b": v["e8"], "err_a": ea, "err_b": eb, "n": max(1, len(v["gt"]))})
        frags.append(f"<section>{gallery(names[t], a.note, items, a.labels)}</section>")
else:
    A = json.loads(open(a.a, encoding="utf-8").read())["per_line"]
    B = json.loads(open(a.b, encoding="utf-8").read())["per_line"]
    for t in a.tags:
        items = []
        Bt = B[t + a.b_suffix]
        for i, (ea, n, ha) in A[t].items():
            if i not in Bt or (t, i) not in img:  # only lines whose image is in the local pickle
                continue
            if keep is not None and i not in keep:
                continue
            eb, _, hb = Bt[i]
            items.append({"id": i, "gt": gt[(t, i)], "img_a": img[(t, i)], "img_b": None, "hyp_a": ha, "hyp_b": hb, "err_a": ea, "err_b": eb, "n": n})
        frags.append(f"<section>{gallery(names[t], a.note, items, a.labels)}</section>")
Path(a.out).write_text(page(a.title, "".join(frags)), encoding="utf-8")
if a.frag:
    Path(a.frag).write_text("".join(frags), encoding="utf-8")
print("wrote", a.out)
