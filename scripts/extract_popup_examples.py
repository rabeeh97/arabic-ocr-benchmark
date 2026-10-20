"""Compute the real examples behind the timeline's term pop-ups -> docs/research/timeline_popups.json.

- the normalisation ladder on one real line (strings and edit counts per rung),
- a character alignment of that line (substitutions, deletions, insertions),
- one model's corpus CER at every rung.
Needs the project environment (pandas, jiwer). Usage:
  PYTHONPATH=src /Volumes/PortableSSD/repos/arabic_ocr/.venv/bin/python scripts/extract_popup_examples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from arabic_ocr_bench.characterize import errors as E

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
Path("docs/research/timeline_popups.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf8")
print("example edits per rung:", [r["edits"] for r in rungs], "alignment:", counts)
print("stair:", [(s["name"], round(s["cer"] * 100, 1)) for s in stair])
