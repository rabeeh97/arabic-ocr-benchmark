"""Shared loaders for the dots / skeleton / spacing studies (checklist 2.1, 2.3, 2.4, 2.5, 2.13, 2.18).

    from dots_common import load_store, load_spans, load_panel, gt_index_map, align, line_geom, cluster_boot

Data (all read-only, CPU):
  results/panel/panel.pkl                         64 px panel images (as stored), label = CRNN label of the line
  results/panel/letters/aligner_run/letters.json  CTC forced-alignment spans per GT letter (base aligner, 4 px frames)
  results/store/{samples,lines}.parquet           GT and every system's reading
Spans index the panel label; `gt_index_map` maps them to the letters view of the store GT, which is where the
per-letter outcomes of every reading live. Families follow analyze_dots.fam (lines a run trained on excluded; AIN,
PaddleOCR and TrOCR left out, CORRECTIONS C5/C6).
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein
from scipy import ndimage as ndi

W = Path(__file__).resolve().parent.parent
OUT = W / "results/characterize/dots"
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
sys.path.insert(0, str(W / "scripts"))
from arabic_ocr_bench.characterize.samples import _letters_view  # noqa: E402,F401
from arabic_ocr_bench.characterize.text import DOTTED, analyze_letters, rasm_of  # noqa: E402,F401
from analyze_dots import SKIP, fam  # noqa: E402

DATASETS = ("kitab_khatt", "nakba_lines", "muharaf_lines", "rasam1", "rasam2")
FAMILIES = ("CRNN", "CRNN → LLM (E8)", "VLM fine-tuned", "VLM base", "API", "OCR engine")
PACKS = ("panel-2026-10-13", "bench", "eval_data+heldout")


def load_store(datasets=DATASETS):
    """(samples by key, {family: [line rows]}). As-stored transform only (fam() maps inkcrop to its own family; dropped)."""
    S = {s["sample_key"]: s for s in pq.read_table(W / "results/store/samples.parquet").to_pylist()}
    cols = ["run_id", "family", "transform", "sample_key", "dataset", "pack", "hyp", "in_train"]
    by = defaultdict(list)
    for r in pq.read_table(W / "results/store/lines.parquet", columns=cols).to_pylist():
        if r["pack"] not in PACKS or r.get("in_train") == "yes" or r["dataset"] not in datasets or any(x in r["run_id"] for x in SKIP):
            continue
        f = fam(r)
        if f in FAMILIES and r["hyp"] is not None:
            by[f].append(r)
    return S, dict(by)


def load_spans(path: Path = W / "results/panel/letters/aligner_run/letters.json"):
    """{sample_key: [(label_index, ch, x0, x1)]} for as-stored images, de-duplicated, in logical order."""
    J = json.loads(path.read_text(encoding="utf-8"))
    out = defaultdict(dict)
    for r in J["rows"]:
        k, tag, i, ch, _, x0, x1 = r[:7]
        if tag.endswith("__inkcrop"):
            continue
        out[k].setdefault(i, (i, ch, int(x0), int(x1)))
    return {k: [v[i] for i in sorted(v)] for k, v in out.items()}


def load_panel():
    """{sample_key: (img uint8 64xW, label)} for the as-stored panel images."""
    d = pickle.load(open(W / "results/panel/panel.pkl", "rb"))
    return {r["id"]: (r["img"], r["label"]) for r in d["rows"] if "__" not in r["dataset"]}


def gt_index_map(label: str, g: str) -> dict[int, int]:
    """Panel-label index -> letters-view GT index, from the equal blocks of a Levenshtein alignment."""
    m = {}
    for op in Levenshtein.opcodes(label, g):
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                m[op.src_start + k] = op.dest_start + k
    return m


def align(g: str, p: str):
    """Per GT index: the hyp character read (None = deleted), and insertions keyed by the GT index they precede."""
    read: list[str | None] = [None] * len(g)
    ins: dict[int, str] = defaultdict(str)
    for op in Levenshtein.opcodes(g, p):
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                read[op.src_start + k] = g[op.src_start + k]
        elif op.tag == "replace":
            n_s, n_d = op.src_end - op.src_start, op.dest_end - op.dest_start
            for k in range(n_s):
                read[op.src_start + k] = p[op.dest_start + k] if k < n_d else None
            if n_d > n_s:
                ins[op.src_end] += p[op.dest_start + n_s : op.dest_end]
        elif op.tag == "insert":
            ins[op.src_start] += p[op.dest_start : op.dest_end]
    return read, ins


def otsu(g: np.ndarray) -> float:
    h = np.bincount(g.ravel(), minlength=256).astype(float)
    p = h / h.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    s = np.where(np.isfinite(s), s, -1.0)
    return float(np.argmax(s)) if s.max() > 0 else float(g.min())  # constant image: no ink


def line_geom(img: np.ndarray) -> dict:
    """Binary ink, components, stroke width, main band (x-height) of a 64 px line. Ink is dark."""
    img = img.copy()
    pad_r = (img == 0).mean(1) > 0.9  # letterbox bars (RASAM) are exact zeros
    pad_c = (img == 0).mean(0) > 0.9
    img[pad_r, :] = 255
    img[:, pad_c] = 255
    ink = img < otsu(img)
    ink[(ink.mean(1) > 0.6), :] = False  # ruling lines
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    sl = ndi.find_objects(lab)
    area = ndi.sum_labels(ink, lab, np.arange(1, n + 1)) if n else np.zeros(0)
    dt = ndi.distance_transform_edt(ink)
    # stroke width: 2 x the median of local maxima of the distance transform (ridge pixels)
    ridge = ink & (dt >= ndi.maximum_filter(dt, size=3))
    sw = float(2 * np.median(dt[ridge])) if ridge.any() else 2.0
    sw = max(sw, 1.0)
    big = np.zeros(n + 1, bool)
    if n:
        big[1:] = area > max(6.0, 4 * sw * sw)
    body = big[lab]
    prof = body.sum(1).astype(float)
    if prof.max() > 0:
        prof = ndi.gaussian_filter1d(prof, 1.0)
        pk = int(prof.argmax())
        hi = prof >= 0.5 * prof.max()
        top = bot = pk  # contiguous rows around the peak: neighbouring lines' ink (RASAM, Muharaf) stays out
        while top > 0 and hi[top - 1]:
            top -= 1
        while bot < len(hi) - 1 and hi[bot + 1]:
            bot += 1
    else:
        top, bot = 20, 44
    xh = max(3.0, bot - top + 1.0)
    return {"ink": ink, "lab": lab, "sl": sl, "area": area, "sw": sw, "body": body, "band": (top, bot), "xh": xh,
            "base": float(bot)}


def cluster_boot(num: np.ndarray, den: np.ndarray, reps: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap of sum(num)/sum(den) with lines as clusters. num, den: per-line arrays (... x lines)."""
    rng = np.random.default_rng(seed)
    n = num.shape[-1]
    idx = rng.integers(0, n, (reps, n))
    return num[..., idx].sum(-1) / np.maximum(1e-12, den[..., idx].sum(-1))


def ci(a: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def readings(S, rows):
    """Yield (sample_key, run_id, g, read, ins) for each reading: letters view, per-GT-index alignment."""
    for r in rows:
        s = S.get(r["sample_key"])
        if s is None:
            continue
        g = _letters_view(s["gt_raw"])
        read, ins = align(g, _letters_view(r["hyp"]))
        yield r["sample_key"], r["run_id"], g, read, ins


def load_features():
    z = np.load(OUT / "features.npz")
    T = {"L": {}, "B": {}, "N": {}}
    for k in z.files:
        T[k[0]][k[2:]] = z[k]
    return T


def wmedian(vals: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(vals, kind="stable")
    c = np.cumsum(w[o])
    if c[-1] <= 0:
        return float("nan")
    return float(vals[o][np.searchsorted(c, c[-1] / 2)])


def boot_p_ratio(boots: np.ndarray) -> float:
    """Two-sided p for log ratio = 0 from the bootstrap SE of the log ratio (as eval_stats.boot_p)."""
    from eval_stats import boot_p
    lb = np.log(boots[np.isfinite(boots) & (boots > 0)])
    if lb.size < 10:
        return float("nan")
    return boot_p(float(np.median(lb)), lb)
