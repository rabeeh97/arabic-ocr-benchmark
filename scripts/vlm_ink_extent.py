"""Do loose KHATT crops hurt the VLMs too? Per-line CER of every benchmarked model against the ink extent of the native image.

    .geomvenv/bin/python scripts/vlm_ink_extent.py /Volumes/PortableSSD/repos/arabic_ocr/suite-v1-artifacts OUT.json [--paddle-order stored|unreverse]

Ink extent as in geom_ink_extent.py (central 98 % of ink rows over image height). CER: edit distance over the cleaned labels
(harakat, tatweel and bidi marks removed on both sides).

Rows are grouped by each prediction's `model` field, not by the artifact folder: a suite folder holds the base model and its
fine-tune in one predictions file, and the old folder grouping pooled the two (checklist 1.9). For each model the latest
run (by run-directory name) that contains it is used; a repeated sample id within that run keeps its last row.

--paddle-order: the stored arabic-paddleocr-v5 runs (2026-09-18) were made with reverse_output: true, which scrambled
readings that were already in logical order (CORRECTIONS C5). `stored` scores them as written (correct for a run made
after the patch, which turns the flag off); `unreverse` undoes the reversal post hoc and reports the model as
`arabic-paddleocr-v5-unreversed`, the same derived run as store_build.py.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")
PADDLE = "arabic-paddleocr"
LOOSE = 0.35


def clean(t):
    return " ".join(STRIP.sub("", unicodedata.normalize("NFC", t or "")).split())


def extent(g):
    from skimage.filters import threshold_otsu

    ink = g < threshold_otsu(g)
    prof = ink.sum(1).astype(float)
    c = np.cumsum(prof) / max(prof.sum(), 1)
    return (np.searchsorted(c, 0.98) - np.searchsorted(c, 0.02) + 1) / g.shape[0]


def latest_rows_by_model(files):
    """{model field: {sample_id: row}} using, per model, the latest run file (sorted by run-directory name) containing it."""
    out = {}
    for f in sorted(files, key=lambda p: Path(p).parent.name):
        per = {}
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                per.setdefault(r["model"], {})[r["sample_id"]] = r
        out.update(per)  # a later run replaces the model's earlier one wholesale
    return out


def hypothesis(row, paddle_order):
    hyp = row.get("cleaned_text") or row.get("raw_text") or ""
    if paddle_order == "unreverse" and row["model"].startswith(PADDLE):
        hyp = hyp[::-1]
    return hyp


def score(rows, ext, paddle_order, edit_distance):
    loose, tight = [0, 0], [0, 0]
    for r in rows:
        i = int(r["sample_id"].rsplit("/", 1)[1])
        gt, hyp = clean(r["ground_truth"]), clean(hypothesis(r, paddle_order))
        b = loose if ext.get(i, 1) < LOOSE else tight
        b[0] += edit_distance(list(hyp), list(gt))
        b[1] += max(1, len(gt))
    return {"lines": len(rows), "cer_loose": loose[0] / max(loose[1], 1), "cer_tight": tight[0] / max(tight[1], 1),
            "cer_all": (loose[0] + tight[0]) / max(loose[1] + tight[1], 1),
            "loose_share_of_errors": loose[0] / max(loose[0] + tight[0], 1)}


def native_extents():
    import pyarrow as pa
    from PIL import Image

    from geom_data import HF

    path = sorted((HF / "ahmedheakl___arocrbench_khatt/default").glob("*/*/*.arrow"))[0]
    try:
        t = pa.ipc.open_stream(pa.memory_map(str(path))).read_all()
    except pa.ArrowInvalid:
        t = pa.ipc.open_file(pa.memory_map(str(path))).read_all()
    return {i: extent(np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("L")))
            for i, im in enumerate(t.column("image").to_pylist())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--paddle-order", choices=("stored", "unreverse"), default="stored")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).parent))
    from xai_train_aligner import edit_distance

    ext = native_extents()
    out = {"lines": len(ext), "share_extent_below_35": float(np.mean([v < LOOSE for v in ext.values()])),
           "paddle_order": a.paddle_order, "grouping": "model field; latest run per model", "models": {}}
    by_model = latest_rows_by_model(a.root.glob("*/bench/full/kitab_khatt/*/predictions.jsonl"))
    for model in sorted(by_model):
        name = f"{model}-unreversed" if a.paddle_order == "unreverse" and model.startswith(PADDLE) else model
        m = out["models"][name] = score(by_model[model].values(), ext, a.paddle_order, edit_distance)
        print(f"{name:32s} n {m['lines']:3d}  all {100 * m['cer_all']:5.1f} %  loose {100 * m['cer_loose']:5.1f} %  "
              f"tight {100 * m['cer_tight']:5.1f} %  loose share of errors {100 * m['loose_share_of_errors']:4.1f} %")
    print("share of lines with ink in under 35 % of the height:", round(out["share_extent_below_35"], 3))
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
