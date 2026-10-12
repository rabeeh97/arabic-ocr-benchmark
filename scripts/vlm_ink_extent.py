"""Do loose KHATT crops hurt the VLMs too? Per-line CER of every benchmarked model against the ink extent of the native image.

    .geomvenv/bin/python scripts/vlm_ink_extent.py /Volumes/PortableSSD/repos/arabic_ocr/suite-v1-artifacts OUT.json

Ink extent as in geom_ink_extent.py (central 98 % of ink rows over image height). CER: edit distance over the cleaned labels
(harakat, tatweel and bidi marks removed on both sides), one prediction file per model (the latest full run on kitab_khatt).
"""
import glob
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import HF  # noqa: E402
from xai_train_aligner import edit_distance  # noqa: E402

STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")
clean = lambda t: " ".join(STRIP.sub("", unicodedata.normalize("NFC", t or "")).split())  # noqa: E731


def extent(g):
    ink = g < threshold_otsu(g)
    prof = ink.sum(1).astype(float)
    c = np.cumsum(prof) / max(prof.sum(), 1)
    return (np.searchsorted(c, 0.98) - np.searchsorted(c, 0.02) + 1) / g.shape[0]


path = sorted(glob.glob(str(HF / "ahmedheakl___arocrbench_khatt/default/*/*/*.arrow")))[0]
mm = pa.memory_map(path)
try:
    t = pa.ipc.open_stream(mm).read_all()
except pa.ArrowInvalid:
    t = pa.ipc.open_file(pa.memory_map(path)).read_all()
imgs = t.column("image").to_pylist()
ext = {i: extent(np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("L"))) for i, im in enumerate(imgs)}
root = Path(sys.argv[1])
out = {"lines": len(ext), "share_extent_below_35": float(np.mean([v < 0.35 for v in ext.values()])), "models": {}}
for f in sorted(root.glob("*/bench/full/kitab_khatt/*/predictions.jsonl")):
    model = f.parts[-6]
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    loose, tight = [0, 0], [0, 0]
    for r in rows:
        i = int(r["sample_id"].rsplit("/", 1)[1])
        gt, hyp = clean(r["ground_truth"]), clean(r.get("cleaned_text") or r.get("raw_text") or "")
        e, n = edit_distance(list(hyp), list(gt)), max(1, len(gt))
        b = loose if ext.get(i, 1) < 0.35 else tight
        b[0] += e
        b[1] += n
    out["models"][model] = {"lines": len(rows), "cer_loose": loose[0] / max(loose[1], 1), "cer_tight": tight[0] / max(tight[1], 1),
                            "cer_all": (loose[0] + tight[0]) / max(loose[1] + tight[1], 1), "loose_share_of_errors": loose[0] / max(loose[0] + tight[0], 1)}
    print(f"{model:28s} all {100 * out['models'][model]['cer_all']:5.1f} %  loose crops {100 * out['models'][model]['cer_loose']:5.1f} %  tight {100 * out['models'][model]['cer_tight']:5.1f} %  loose share of errors {100 * out['models'][model]['loose_share_of_errors']:4.1f} %")
print("share of lines with ink in under 35 % of the height:", round(out["share_extent_below_35"], 3))
json.dump(out, open(sys.argv[2], "w"), indent=1)
