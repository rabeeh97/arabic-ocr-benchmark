"""Compute the real examples behind the timeline's term pop-ups -> docs/research/timeline_popups.json.

- the normalisation ladder on one real line (strings and edit counts per rung),
- a character alignment of that line (substitutions, deletions, insertions),
- one model's corpus CER at every rung.
Needs the project environment (pandas, jiwer). Usage:
  PYTHONPATH=src /Volumes/PortableSSD/repos/arabic_ocr/.venv/bin/python scripts/extract_popup_examples.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pandas as pd

from PIL import Image

from arabic_ocr_bench.characterize import errors as E
from arabic_ocr_bench.characterize.images import image_features
from arabic_ocr_bench.characterize.samples import gt_features

SAMPLE = "muharaf_lines/test/000724"
MODEL = "suite-v1-qwen3vl4b-ft-bf16"
STAIR = ("suite-v1-qwen3vl8b-ft-bf16", "nakba_lines")

lines = pd.read_parquet("results/store/lines.parquet")
lines = lines[lines["transform"].astype(str) == "none"]
samples = pd.read_parquet("results/store/samples.parquet", columns=["sample_key", "gt_raw"])
row = lines[(lines.model == MODEL) & (lines.sample_key == SAMPLE)].iloc[0]
gt = samples[samples.sample_key == SAMPLE].gt_raw.iloc[0]
hyp = row.hyp
gs, hs = E.ladder_strings(gt), E.ladder_strings(hyp)
ed = E.ladder_edits(gt, hyp)
rungs = [{"name": n, "gt": g, "hyp": h, "edits": e, "ref_len": r} for n, g, h, (e, r) in zip(E.RUNGS, gs, hs, ed)]


def align(a: str, b: str):
    """Levenshtein alignment: list of (op, a_char, b_char); op in eq / sub / del (a only) / ins (b only)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("eq" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1]))
            j -= 1
    return ops[::-1]


canon = next(r for r in rungs if r["name"] == "canonical")
ops = align(canon["gt"], canon["hyp"])
counts = {k: sum(1 for o in ops if o[0] == k) for k in ("eq", "sub", "del", "ins")}
sub = lines[(lines.model == STAIR[0]) & (lines.dataset == STAIR[1])]
stair = [{"name": r, "cer": float(sub["e_" + r].sum() / max(1, sub["n_" + r].sum()))} for r in E.RUNGS]
out = {"example": {"sample": SAMPLE, "model": MODEL, "gt": gt, "hyp": hyp, "rungs": rungs, "ops": ops, "counts": counts},
       "stair": {"model": STAIR[0], "dataset": STAIR[1], "n_lines": int(len(sub)), "rungs": stair}}

# --- text and image features on real examples
assets = json.loads(Path("docs/research/brief_assets.json").read_text(encoding="utf8"))
crop = assets["crop"][0]
text_lines = [("A KHATT line", crop["truth"]), ("A Muharaf line with digits and a dash", gt)]
text_keys = ["chars", "words", "word_len", "dotted_share", "ambiguous_share", "paws_per_word", "hamza_share", "digit_share", "punct_share"]
text_ex = []
for lab, t in text_lines:
    f = gt_features(t)
    text_ex.append({"label": lab, "text": t, "vals": {k: round(float(f[k]), 3) for k in text_keys}})
img_keys = ["width", "height", "aspect", "ink_density", "ink_height_frac", "core_height", "stroke_width", "contrast", "blur_lapvar", "skew_deg", "ink_px_per_char"]
n_chars = len(crop["truth"].replace(" ", ""))
img_ex = []
for lab, key in [("The line as stored", "stored"), ("The same line cropped to its ink", "cropped")]:
    im = Image.open(io.BytesIO(base64.b64decode(crop[key].split(",", 1)[1]))).convert("L")
    f = image_features(im, n_chars)
    img_ex.append({"label": lab, "src": crop[key], "vals": {k: round(float(f[k]), 3) for k in img_keys if k in f}})
out["features"] = {"text": text_ex, "image": img_ex}
Path("docs/research/timeline_popups.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf8")
print("example edits per rung:", [r["edits"] for r in rungs], "alignment:", counts)
print("text feats:", [t["vals"] for t in text_ex]); print("image feats:", [i["vals"] for i in img_ex])
print("stair:", [(s["name"], round(s["cer"] * 100, 1)) for s in stair])
