"""Pack original and transformed lines for the 3090, so CER can tell whether the transformed geometry carries a transfer gap.

    .geomvenv/bin/python scripts/geom_transfer_export.py results/geometry/transfer OUT.pkl [--n 300]

Output has the format of xai_prepare_eval_data.py (rows with id, dataset, role, label, img at 64 px high) and holds, for
every source line, the untouched version and each transformed version under its own dataset tag:

    kitab_khatt            kitab_khatt__as_muharaf
    nakba_lines            nakba_lines__as_muharaf            nakba_lines__as_khatt            nakba_lines__norules (ruled lines painted out)
    muharaf_lines          muharaf_lines__as_khatt            muharaf_lines__as_nakba

Then on the 3090:  python xai_eval_aligner.py --data OUT.pkl --run aligner_run --ckpt model_final.pt --out transfer_cer.json
Paired comparison: the same line ids appear under both tags, so per-line differences are available too.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import load_lines  # noqa: E402
from geom_measure import binarize, strip_rules, stroke_width  # noqa: E402
from geom_transfer import apply, only, to64  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")
TAG = {"muharaf": "muharaf_lines", "khatt": "kitab_khatt", "nakba": "nakba_lines"}
ABLATE = {("khatt", "muharaf"), ("muharaf", "khatt"), ("nakba", "muharaf")}


def clean(t: str) -> str:
    return " ".join(STRIP.sub("", unicodedata.normalize("NFC", t)).split())


def remove_rules(img):
    """Paint paper over ruled lines and table borders (long thin horizontal ink). Returns the image and whether it changed."""
    ink = binarize(img)
    sw = stroke_width(ink)
    if not np.isfinite(sw):
        return img, False
    clean, share = strip_rules(ink, sw)
    if share == 0:
        return img, False
    rm = ndi.binary_dilation(ink & ~clean, iterations=2)
    out = img.copy()
    out[rm] = int(np.median(img[~ink]))
    return out, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    saved = pickle.load(open(a.run / "transfer_params.pkl", "rb"))
    params = {tuple(k.split(":")): v["P"] for k, v in saved.items()}
    rows = []
    changed_n = 0
    for src in ("khatt", "nakba", "muharaf"):
        targets = [t for (s, t) in params if s == src]
        kept = 0
        for i, g, lab in load_lines(src, a.n * 2, with_index=True):
            label = clean(lab)
            img = to64(g)
            if not 2 <= len(label) <= 120 or img.shape[1] // 4 < len(label) + 4:
                continue
            rows.append({"id": f"{TAG[src]}/{i:06d}", "dataset": TAG[src], "role": "eval", "label": label, "img": img})
            for t in targets:
                out = apply(img, params[(src, t)], seed=i)
                if out.shape[1] // 4 < len(label) + 4:
                    continue
                rows.append({"id": f"{TAG[src]}/{i:06d}", "dataset": f"{TAG[src]}__as_{t}", "role": "eval", "label": label, "img": out})
                if (src, t) in ABLATE:  # one group of knobs at a time
                    for part in ("slant", "stroke", "paper"):
                        o2 = apply(img, only(params[(src, t)], part), seed=i)
                        if o2.shape[1] // 4 >= len(label) + 4:
                            rows.append({"id": f"{TAG[src]}/{i:06d}", "dataset": f"{TAG[src]}__as_{t}_only_{part}", "role": "eval", "label": label, "img": o2})
            if src == "nakba":
                nr, changed = remove_rules(img)
                changed_n += int(changed)
                rows.append({"id": f"{TAG[src]}/{i:06d}", "dataset": f"{TAG[src]}__norules", "role": "eval", "label": label, "img": nr})
            kept += 1
            if kept >= a.n:
                break
        print(src, kept, "lines, transformed to", targets)
        if src == "nakba":
            print("  lines with ruled lines removed:", changed_n)
    pickle.dump({"H": 64, "rows": rows}, open(a.out, "wb"), protocol=4)
    from collections import Counter
    print(Counter(r["dataset"] for r in rows), f"{a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
