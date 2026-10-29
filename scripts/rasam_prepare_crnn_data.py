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
import json
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
TAGS = {"RASAM-1": "r1", "RASAM-2": "r2", "tarima": "tm"}  # TariMa (calfa-ai/tarima) has the same columns and Calfa's transcription guidelines
ROLES = {"train": "train", "validation": "heldout", "test": "test"}


def _standardise(t: str) -> str:
    """Lucas et al. 2022 standardised RASAM's character set (54 -> 38 classes). Our equivalent is the project's diac_insensitive profile:
    diacritics and tatweel removed, alef / hamza / yaa / taa-marbuta variants and digits folded. On Colab the module is uploaded as /content/arabic_normalize.py."""
    try:
        from arabic_ocr_bench.normalize import normalize
    except ImportError:
        from arabic_normalize import normalize
    return normalize(t, "diac_insensitive")


def clean(t: str, labels: str) -> str:
    t = BIDI.sub("", unicodedata.normalize("NFC", t or ""))
    if labels == "stripped":
        t = HARAKAT.sub("", t)
    elif labels == "standardised":
        t = _standardise(t)
    return " ".join(t.split())


CHANNEL = "grey"  # --channel darkest: per-pixel minimum of R, G, B, so red and blue rubrics stay as dark as black ink


def prep(img: Image.Image) -> np.ndarray:
    if CHANNEL == "darkest":
        g = np.asarray(img.convert("RGB")).min(axis=2).astype(np.float32)
    else:
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
    ap.add_argument("--labels", default="raw", choices=["raw", "stripped", "standardised"])
    ap.add_argument("--height", type=int, default=64, help="line height in px (multiple of 32); the width cap scales with it")
    ap.add_argument("--channel", default="grey", choices=["grey", "darkest"], help="grey (default) or the darkest of R, G, B per pixel")
    ap.add_argument("--repair", type=Path, help="pickle from rasam_recut_broken.py: replacement images for degenerate crops, keyed by (set, split, index)")
    ap.add_argument("--drop-broken", action="store_true", help="drop TRAIN/validation crops under 8 px tall that have no repair (test lines are always kept)")
    ap.add_argument("--all-train-sets", nargs="*", default=[], choices=list(TAGS),
                    help="sets whose validation AND test lines also become training lines (e.g. RASAM-1 when only RASAM-2 test is scored)")
    ap.add_argument("--holdout", nargs="*", default=[], help="manuscript names held out entirely (leave-manuscript-out): ALL their lines "
                    "(train, validation and test) become test lines with dataset tag <tag>ho (e.g. r2ho), so no page of them is seen in training")
    ap.add_argument("--drop", type=Path, help="JSON list of line keys '<set>/<split>/<parquet index>' to leave out (data cleaning; any split)")
    ap.add_argument("--fold", type=Path, help="JSON {'<manuscript>|<page>': fold}: cross-fitting. With --fold-index i, train/validation lines on "
                    "fold-i pages keep their roles, those on other pages become test lines tagged <tag>cf; the official test lines are left out")
    ap.add_argument("--fold-index", type=int, default=0)
    a = ap.parse_args()
    drop = set(json.loads(a.drop.read_text())) if a.drop else set()
    fold = json.loads(a.fold.read_text()) if a.fold else None
    global H, MAX_W, CHANNEL
    H, MAX_W = a.height, 24 * a.height
    CHANNEL = a.channel
    rows, dropped = [], {}
    repairs = pickle.load(open(a.repair, "rb")) if a.repair else {}
    n_repaired = 0
    for name in a.sets:
        for split, role in ROLES.items():
            if name in a.all_train_sets:
                role = "train"
            n = 0
            k = -1  # index in the split's parquet order (the key used by the repair file)
            for f in sorted(glob.glob(f"{ROOT}{name}/snapshots/*/data/{split}-*.parquet")):
                for rec in pq.read_table(f).to_pylist():
                    k += 1
                    key = f"{name}/{split}/{k:06d}"
                    pk = f"{rec['manuscript_name']}|{rec['image_name']}"
                    if key in drop:
                        dropped[(name, split, "drop list")] = dropped.get((name, split, "drop list"), 0) + 1
                        continue
                    if fold is not None and (split == "test" or pk not in fold):
                        continue
                    label = clean(rec["transcription"], a.labels)
                    raw = Image.open(io.BytesIO(rec["image"]["bytes"]))
                    if (name, split, k) in repairs:
                        raw = Image.open(io.BytesIO(repairs[(name, split, k)]["bytes"]))
                        n_repaired += 1
                    elif a.drop_broken and split != "test" and raw.size[1] < 8:
                        dropped[(name, split)] = dropped.get((name, split), 0) + 1
                        continue
                    arr = prep(raw)
                    held = rec["manuscript_name"] in a.holdout
                    r_role, r_tag = ("test", TAGS[name] + "ho") if held else (role, TAGS[name])
                    if fold is not None and fold[pk] != a.fold_index:
                        r_role, r_tag = "test", TAGS[name] + "cf"
                    # CTC needs a frame per character; test lines are never dropped (scored as they are)
                    if r_role != "test" and (not label or arr.shape[1] // 4 < len(label) + 2):
                        dropped[(name, split)] = dropped.get((name, split), 0) + 1
                        continue
                    rows.append({"id": f"{name}/{split}/{n:06d}", "dataset": r_tag, "role": r_role, "label": label, "img": arr,
                                 "manuscript": rec["manuscript_name"], "page": rec["image_name"], "region": rec["region_type"], "orig_split": split, "key": key})
                    n += 1
            print(name, split, n, flush=True)
    split_name = f"rasam-published-{a.labels}" + (f"-alltrain-{'-'.join(a.all_train_sets)}" if a.all_train_sets else "") + ("-repaired" if repairs else "") \
        + (f"-holdout-{len(a.holdout)}ms" if a.holdout else "")
    print("repaired crops used:", n_repaired, flush=True)
    pickle.dump({"H": H, "split": split_name, "rows": rows}, open(a.out, "wb"), protocol=4)
    print("dropped:", {"/".join(map(str, k)): v for k, v in dropped.items()}, "->", a.out, f"{a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
