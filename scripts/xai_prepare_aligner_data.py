"""Pack line images + labels into one pickle for the CTC aligner (X-B1).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/xai_prepare_aligner_data.py OUT.pkl PANEL_MANIFEST [PANEL_MANIFEST ...]
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/xai_prepare_aligner_data.py OUT.pkl --standard-split

Note (2026-10-13): the default mode trains on every Muharaf and RASAM-1 *test* line except its held-out quota. It is kept
for the existing runs; use --standard-split for anything compared with published numbers.

Reads the cached Arrow files (offline), turns every line into a height-64 grey uint8 array, strips
harakat / tatweel / bidi marks from the label, and tags each line:
  train    everything except the held-out lines (the panel lines are deliberately included: we only
           need their character positions, and forced alignment of labelled lines is standard)
  heldout  300 Muharaf-test + 100 RASAM-1-test lines that are NOT panel lines, for an honest CER
  panel    a train line that belongs to the attribution panels (id kept so spans can be looked up)
"""
from __future__ import annotations

import glob
import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from datasets import Dataset
from PIL import Image

H = 64
MAX_W = 1536
HF = Path.home() / ".cache/huggingface/datasets"
STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")
SOURCES = [  # (folder, arrow glob, text column, dataset tag)
    ("aamijar___muharaf-public", "muharaf-public-*.arrow", "text", "muharaf_lines"),
    ("calfa-ai___rasam-1", "rasam-1-*.arrow", "transcription", "rasam1"),
    ("calfa-ai___rasam-2", "rasam-2-*.arrow", "transcription", "rasam2"),
]


def clean(t: str) -> str:
    return " ".join(STRIP.sub("", unicodedata.normalize("NFC", t)).split())


def prep(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    w, h = g.size
    nw = max(16, min(MAX_W, round(w * H / h)))
    return np.asarray(g.resize((nw, H), Image.LANCZOS), dtype=np.uint8)


def standard_split(out: Path) -> None:
    """--standard-split: Muharaf train shards -> train, Muharaf validation -> heldout (model selection). No test lines and no
    RASAM (only its test split is published), so every test set stays unseen; score them with xai_prepare_panel.py."""
    rows = []
    for path in sorted(glob.glob(str(HF / "aamijar___muharaf-public" / "*" / "*" / "*" / "muharaf-public-*.arrow"))):
        name = Path(path).name
        if "test" in name:
            continue
        role = "heldout" if "valid" in name else "train"
        shard = "validation" if role == "heldout" else "train" + name.split("-train-")[1][:5]
        ds = Dataset.from_file(path)
        for i in range(len(ds)):
            row = ds[i]
            label = clean(row["text"] or "")
            if not 2 <= len(label) <= 120:
                continue
            arr = prep(row["image"])
            if arr.shape[1] // 4 < len(label) + 4:  # CTC needs room for every character
                continue
            rows.append({"id": f"muharaf_lines/{shard}/{i:06d}", "dataset": "muharaf_lines", "role": role, "label": label, "img": arr})
        print(name, len(ds), "->", len(rows), flush=True)
    pickle.dump({"H": H, "split": "standard", "rows": rows}, open(out, "wb"), protocol=4)
    roles = {k: sum(1 for r in rows if r["role"] == k) for k in ("train", "heldout")}
    print(roles, "->", out, f"{out.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    if "--standard-split" in sys.argv:
        return standard_split(Path(sys.argv[1]))
    out = Path(sys.argv[1])
    panel = set()
    for m in sys.argv[2:]:
        for it in json.loads(Path(m).read_text(encoding="utf-8")):
            panel.add(it["id"])
    rows, held = [], {"muharaf_lines": 0, "rasam1": 0}
    quota = {"muharaf_lines": 300, "rasam1": 100}
    for folder, pat, col, tag in SOURCES:
        for path in sorted(glob.glob(str(HF / folder / "*" / "*" / "*" / pat))):
            split = "test" if "test" in Path(path).name else ("validation" if "valid" in Path(path).name else "train")
            ds = Dataset.from_file(path)
            for i in range(len(ds)):  # train shards repeat row indices; only test ids matter (panel lookup)
                row = ds[i]
                label = clean(row[col] or "")
                if not 2 <= len(label) <= 120:
                    continue
                lid = f"{tag}/{split}/{i:06d}"
                arr = prep(row["image"])
                if arr.shape[1] // 4 < len(label) + 4:  # CTC needs room for every character
                    continue
                if lid in panel:
                    role = "panel"
                elif split == "test" and tag in quota and held[tag] < quota[tag] and 8 <= len(label) <= 60:
                    held[tag] += 1
                    role = "heldout"
                else:
                    role = "train"
                rows.append({"id": lid, "dataset": tag, "role": role, "label": label, "img": arr})
            print(Path(path).name, len(ds), "->", len(rows), flush=True)
    missing = panel - {r["id"] for r in rows if r["role"] == "panel"}
    print("panel ids missing:", sorted(missing))
    pickle.dump({"H": H, "rows": rows}, open(out, "wb"), protocol=4)
    roles = {k: sum(1 for r in rows if r["role"] == k) for k in ("train", "heldout", "panel")}
    print(roles, "->", out, f"{out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
