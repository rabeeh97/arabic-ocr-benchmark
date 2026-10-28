"""AraMS-28k Maghrebi books as an unseen-manuscript test for the RASAM CRNN (2026-10-28).

    ~/.venvs/kraken-rasam/bin/python scripts/arams_prepare_crnn_pack.py --base PACK.pkl --out OUT.pkl [--height 96]

AraMS-28k (arXiv 2608.26921, CC BY-NC-SA 4.0) has 4 Maghrebi books: book_03 (its test book, 3,661 lines) and book_19 / 20 / 24 (its
train books, 666 / 2,163 / 2,400 lines). None of them is in RASAM or TariMa, so all four are unseen manuscripts for our models.
Every line becomes a test row tagged am03 / am19 / am20 / am24, preprocessed exactly like the RASAM lines (rasam_prepare_crnn_data.prep).

The base pack (a rasam_prepare_crnn_data.py pickle) supplies the LM's training text and the validation lines that tune the LM weight;
its train rows keep only their labels (images dropped) and its test rows are dropped, so the pack stays small enough for the Mac.
Images are read from the extracted zip under --root (only the four books need extracting).
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rasam_prepare_crnn_data as P  # noqa: E402

BOOKS = {"book_03": "am03", "book_19": "am19", "book_20": "am20", "book_24": "am24"}  # the Maghrebi books (paper Table 6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("/Volumes/PortableSSD/data/arams_28k_htr"))
    ap.add_argument("--height", type=int, default=96)
    a = ap.parse_args()
    P.H, P.MAX_W = a.height, 24 * a.height
    base = pickle.load(open(a.base, "rb"))
    assert base["H"] == a.height, (base["H"], a.height)
    empty = np.zeros((a.height, 16), np.uint8)
    rows = []
    for r in base["rows"]:
        if r["role"] == "test":
            continue
        if r["role"] == "train":
            r = {**r, "img": empty}
        rows.append(r)
    del base
    n = {}
    for rec in csv.DictReader(open(a.root / "AraMs-28k-HTR/metadata.csv", encoding="utf-8")):
        tag = BOOKS.get(rec["book_id"])
        if not tag:
            continue
        img = Image.open(a.root / rec["img_path"][2:])
        rows.append({"id": f"arams/{rec['book_id']}/{rec['page_id']}/{int(rec['line_idx']):03d}", "dataset": tag, "role": "test",
                     "label": P.clean(rec["gt_text"], "raw"), "img": P.prep(img), "manuscript": rec["book_id"], "page": rec["page_id"],
                     "region": rec["line_type"], "orig_split": rec["split"]})
        n[tag] = n.get(tag, 0) + 1
    print("AraMS lines:", n, flush=True)
    pickle.dump({"H": a.height, "split": "arams-maghrebi-unseen", "rows": rows}, open(a.out, "wb"), protocol=4)


if __name__ == "__main__":
    main()
