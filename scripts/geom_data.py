"""Shared loader for the geometry measurements (X-G1 to X-G6): reads cached Arrow files, no network.

    from geom_data import SOURCES, load_lines
    for img, label in load_lines("khatt", n=300): ...

Datasets are identified by a short tag. Images come back as 2-D uint8 greyscale at their native
resolution (geometry measures must not depend on our 64 px resize).
"""
from __future__ import annotations

import glob
import io
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

HF = Path.home() / ".cache/huggingface/datasets"
# tag -> (glob under HF, text column, kind, style)
SOURCES = {
    "muharaf_train": ("aamijar___muharaf-public/default/*/*/muharaf-public-train-00000-of-00003.arrow", "text", "manuscript lines (training pool)", "mixed hands"),
    "muharaf": ("aamijar___muharaf-public/default/*/*/muharaf-public-test.arrow", "text", "manuscript lines", "mixed hands"),
    "rasam1": ("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow", "transcription", "Maghrebi manuscript lines", "Maghrebi"),
    "rasam2": ("calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow", "transcription", "Maghrebi manuscript lines", "Maghrebi"),
    "nakba": ("U4RASD___omar-al-saleh-manuscripts-segments/default-2645abedcb4f16f8/*/*/*-test.arrow", None, "manuscript lines", "single hand"),
    "nakba_blind": ("U4RASD___omar-al-saleh-manuscripts-segments/default-0150fd3dc57fdfd3/*/*/*-blind_test.arrow", None, "manuscript lines", "single hand"),
    "khatt": ("ahmedheakl___arocrbench_khatt/default/*/*/*.arrow", "text", "modern handwriting lines", "modern pen"),
}


def _table(path: str) -> pa.Table:
    mm = pa.memory_map(path)
    try:
        return pa.ipc.open_stream(mm).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_file(mm).read_all()


def load_lines(tag: str, n: int = 300, seed: int = 0):
    """Yield up to n (grey uint8 array, label) pairs, sampled with a fixed seed."""
    pat, col, _, _ = SOURCES[tag]
    path = sorted(glob.glob(str(HF / pat)))[0]
    t = _table(path)
    if col is None:
        col = next(c for c in ("text", "transcription", "answer", "label") if c in t.column_names)
    idx = list(range(t.num_rows))
    random.Random(seed).shuffle(idx)
    for i in idx[:n]:
        row = t.slice(i, 1).to_pylist()[0]
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
        yield np.asarray(img, dtype=np.uint8), (row[col] or "")
