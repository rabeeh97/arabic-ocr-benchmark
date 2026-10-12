"""Gallery: NAKBA lines scored as written against scored with hamza folded (is the hamza in the image or only in the label?).

    .geomvenv/bin/python scripts/build_hamza_gallery.py results/geometry/base_evaldata.json results/geometry/inkcrop_eval.pkl OUT.html --frag F.html
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from change_gallery import gallery, page  # noqa: E402
from xai_train_aligner import edit_distance  # noqa: E402

FOLD = str.maketrans({c: "ا" for c in "أإآٱ"})
ap = argparse.ArgumentParser()
ap.add_argument("cer")
ap.add_argument("pkl")
ap.add_argument("out")
ap.add_argument("--frag")
a = ap.parse_args()
per = json.load(open(a.cer, encoding="utf-8"))["per_line"]
rows = {(r["dataset"], r["id"]): r for r in pickle.load(open(a.pkl, "rb"))["rows"] if "__" not in r["dataset"]}
items = []
for i, (err, n, hyp) in per["nakba_lines"].items():
    r = rows.get(("nakba_lines", i))
    if not r or len(r["label"]) < 30:
        continue
    lab = r["label"]
    items.append({"id": i, "gt": lab, "img_a": r["img"], "img_b": None, "hyp_a": hyp, "hyp_b": hyp.translate(FOLD),
                  "err_a": edit_distance(list(hyp), list(lab)), "err_b": edit_distance(list(hyp.translate(FOLD)), list(lab.translate(FOLD))), "n": len(lab)})
frag = "<section>" + gallery("NAKBA: the CRNN’s reading scored as written, and with hamza folded",
                             "Same reading, two scorings. Look at the alifs in the images of the top cards: if no hamza is drawn, the label added it and the error is a convention, not a misreading. Highlights in the second reading are against the folded truth.",
                             items, ("as written", "hamza folded"), k=(6, 0, 4)) + "</section>"
Path(a.out).write_text(page("Hamza convention in NAKBA", frag), encoding="utf-8")
if a.frag:
    Path(a.frag).write_text(frag, encoding="utf-8")
print("wrote", a.out)
