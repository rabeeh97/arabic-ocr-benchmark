"""B2: rewrite the RASAM-2 rows of a rasam_prepare_crnn_data.py pickle with a MANUSCRIPT-disjoint split.

    python scripts/rasam_make_msplit.py rasam_raw.pkl rasam_msplit.pkl

Rule (fixed in advance, not tuned): sort the 15 RASAM-2 manuscript names; the manuscripts at sorted positions 2, 7 and 12 are held out
(every line of them, from every published split, becomes a test line). The other 12 keep their published train / validation split.
Dataset tags in the output:
  r2m  train and validation lines of the 12 training manuscripts (train role "train" / "heldout"), plus ALL lines of the 3 held-out
       manuscripts with role "test"  (unseen-manuscript test)
  r2s  the published TEST lines of the 12 training manuscripts, role "test"  (seen-manuscript test, same recipe as a random split)
Line ids are kept, so a model trained on the published random split (A2) can be compared on exactly the same lines.
"""
from __future__ import annotations

import pickle
import sys

HELD_POSITIONS = (2, 7, 12)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    blob = pickle.load(open(src, "rb"))
    r2 = [r for r in blob["rows"] if r["dataset"] == "r2"]
    names = sorted({r["manuscript"] for r in r2})
    held = {names[i] for i in HELD_POSITIONS}
    print("RASAM-2 manuscripts:", len(names), " held out:", sorted(held))
    out = []
    for r in r2:
        n = dict(r)
        if r["manuscript"] in held:
            n["dataset"], n["role"] = "r2m", "test"
        elif r["role"] == "test":
            n["dataset"], n["role"] = "r2s", "test"
        else:
            n["dataset"] = "r2m"
        out.append(n)
    counts = {}
    for r in out:
        counts[(r["dataset"], r["role"])] = counts.get((r["dataset"], r["role"]), 0) + 1
    print(counts)
    pickle.dump({"H": blob["H"], "split": "rasam2-manuscript-disjoint", "held_out": sorted(held), "rows": out}, open(dst, "wb"), protocol=4)


if __name__ == "__main__":
    main()
