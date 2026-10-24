"""RS4: pack RASAM-1/2 line images with the PUBLISHED train / validation / test splits for scripts/xai_train_aligner.py.

    .venv/bin/python scripts/rasam_prepare_crnn_data.py OUT.pkl [--sets RASAM-1 RASAM-2] [--labels raw|stripped]

Roles: train -> "train", validation -> "heldout" (the trainer's model selection), test -> "test" (ignored by the trainer; scored
by scripts/rasam_crnn_eval.py). Dataset tags are "r1" / "r2" so the trainer's 4x repeat of datasets named "rasam*" does not apply.
Every row keeps manuscript and page, so leave-manuscript-out and data-diversity arms (A6) can filter on them.

Image: grey, black rotation-mask pixels (< 12) filled with the median paper tone (the published Kraken models failed on the raw
mask, 74 % CER vs 14 % binarised), height 64 like the aligner. Labels: NFC, bidi / zero-width marks removed, whitespace collapsed.
"raw" keeps diacritics and tatweel; "stripped" also removes harakat and tatweel (the aligner's old policy).
"""
from __future__ import annotations

import argparse
import glob
import io
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

H = 64
MAX_W = 1536
import os

ROOT = os.environ.get("RASAM_HF_ROOT", "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--")  # Colab: /content/hfcache/datasets--calfa-ai--
BIDI = re.compile("[​-‏‪-‮⁦-⁩﻿؜]")
HARAKAT = re.compile("[ً-ْٰـ]")
TAGS = {"RASAM-1": "r1", "RASAM-2": "r2"}
ROLES = {"train": "train", "validation": "heldout", "test": "test"}


def clean(t: str, labels: str) -> str:
    t = BIDI.sub("", unicodedata.normalize("NFC", t or ""))
    if labels == "stripped":
        t = HARAKAT.sub("", t)
    return " ".join(t.split())


def prep(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L")).astype(np.float32)
    mask = g < 12
    if mask.any() and (~mask).any():
        g = np.where(mask, float(np.median(g[~mask])), g)
    w, h = g.shape[1], g.shape[0]
    nw = max(16, min(MAX_W, round(w * H / h)))
    return np.asarray(Image.fromarray(g.astype(np.uint8)).resize((nw, H), Image.LANCZOS), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sets", nargs="+", default=["RASAM-1", "RASAM-2"], choices=list(TAGS))
    ap.add_argument("--labels", default="raw", choices=["raw", "stripped"])
    a = ap.parse_args()
    rows, dropped = [], {}
    for name in a.sets:
        for split, role in ROLES.items():
            n = 0
            for f in sorted(glob.glob(f"{ROOT}{name}/snapshots/*/data/{split}-*.parquet")):
                for rec in pq.read_table(f).to_pylist():
                    label = clean(rec["transcription"], a.labels)
                    arr = prep(Image.open(io.BytesIO(rec["image"]["bytes"])))
                    # CTC needs a frame per character; test lines are never dropped (scored as they are)
                    if role != "test" and (not label or arr.shape[1] // 4 < len(label) + 2):
                        dropped[(name, split)] = dropped.get((name, split), 0) + 1
                        continue
                    rows.append({"id": f"{name}/{split}/{n:06d}", "dataset": TAGS[name], "role": role, "label": label, "img": arr,
                                 "manuscript": rec["manuscript_name"], "page": rec["image_name"], "region": rec["region_type"]})
                    n += 1
            print(name, split, n, flush=True)
    pickle.dump({"H": H, "split": f"rasam-published-{a.labels}", "rows": rows}, open(a.out, "wb"), protocol=4)
    print("dropped (no CTC room / empty):", {f"{k[0]}/{k[1]}": v for k, v in dropped.items()}, "->", a.out, f"{a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
