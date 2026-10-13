"""Derive the standard-split training pickle from an existing aligner_data.pkl, without the HF cache (for the 3090).

    python xai_derive_standard_split.py aligner_data.pkl aligner_data_standard.pkl

Same rows and preprocessing as `xai_prepare_aligner_data.py --standard-split`: Muharaf train-shard rows -> train,
Muharaf validation rows -> heldout; every test row and every RASAM row is dropped. Ids are rewritten to the
"muharaf_lines/train0000N/..." form that the standard-split builder uses, so the two pickles can be compared.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    blob = pickle.load(open(src, "rb"))
    rows = []
    for r in blob["rows"]:
        if r["dataset"] != "muharaf_lines":
            continue
        split = r["id"].split("/")[1]
        if split == "train":
            rows.append({**r, "role": "train"})
        elif split == "validation":
            rows.append({**r, "role": "heldout"})
    pickle.dump({"H": blob["H"], "split": "standard", "rows": rows}, open(dst, "wb"), protocol=4)
    print(Counter(r["role"] for r in rows), "->", dst)


if __name__ == "__main__":
    main()
