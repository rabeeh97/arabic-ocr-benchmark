"""Original and ink-cropped versions of the evaluation lines, for the size-normalization test.

    .geomvenv/bin/python scripts/geom_inkcrop_export.py OUT.pkl [--n 600]

Ink crop: on the native image, keep the rows between the 1st and 99th percentile of ink (Otsu threshold), plus a margin of 25 %
of that band above and below (dots and ascenders), then resize to 64 px high as usual. Same line ids as eval_data.pkl, so the
rows pair with the earlier per-line results. Tags: kitab_khatt, nakba_lines, nakba_blind and each with __inkcrop.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import load_lines  # noqa: E402
from geom_transfer import to64  # noqa: E402

STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")
TAG = {"khatt": "kitab_khatt", "nakba": "nakba_lines", "nakba_blind": "nakba_blind"}


def clean(t):
    return " ".join(STRIP.sub("", unicodedata.normalize("NFC", t or "")).split())


def inkcrop(g: np.ndarray) -> np.ndarray:
    try:
        t = threshold_otsu(g)
    except ValueError:
        return g
    # On a two-level (binarised) image, e.g. KHATT v1.0 from Teklia ({0, 255} only), Otsu returns the lower level itself, so
    # `g < t` finds no ink and the crop silently returned the whole image (bug found 2026-10-18). Only that degenerate case
    # switches to `<=`; grey-level images keep exactly the earlier behaviour.
    ink = g <= t if t <= g.min() else g < t
    prof = ink.sum(1).astype(float)
    if prof.sum() == 0:
        return g
    c = np.cumsum(prof) / prof.sum()
    top, bot = int(np.searchsorted(c, 0.01)), int(np.searchsorted(c, 0.99))
    m = int(0.25 * (bot - top + 1)) + 2
    return g[max(0, top - m) : min(g.shape[0], bot + m + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    rows, changed = [], {}
    for src, tag in TAG.items():
        kept = 0
        for i, g, lab in load_lines(src, 100000, with_index=True):
            label = clean(lab)
            img = to64(g)
            if not 2 <= len(label) <= 120 or img.shape[1] // 4 < len(label) + 4:
                continue
            crop = to64(inkcrop(g))
            if crop.shape[1] // 4 < len(label) + 4:
                crop = img
            rid = f"{tag}/{i:06d}"
            rows.append({"id": rid, "dataset": tag, "role": "eval", "label": label, "img": img})
            rows.append({"id": rid, "dataset": f"{tag}__inkcrop", "role": "eval", "label": label, "img": crop})
            changed[tag] = changed.get(tag, 0) + int(crop.shape != img.shape or (crop != img).any())
            kept += 1
            if kept >= a.n:
                break
        print(tag, kept, "lines;", changed.get(tag, 0), "changed by the crop")
    pickle.dump({"H": 64, "rows": rows}, open(a.out, "wb"), protocol=4)


if __name__ == "__main__":
    main()
