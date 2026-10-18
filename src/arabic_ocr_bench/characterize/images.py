"""Image-layer features of a line image (characterisation L6, task R6.0).

:func:`load_images` fetches the *raw* line images (no benchmark resize) for
given sample ids from the dataset's Hugging Face copy, using the row index in
the sample id (``<dataset>/<split>/<row>``) and checking the GT text matches.

:func:`image_features` measures one image after Otsu binarisation:

- geometry: width, height, aspect;
- ink: ink density, horizontal and vertical ink extent, ink touching the left
  or right edge (text cut by the crop; for Arabic the line *ends* at the left);
- scale: core (x-height-like) band height from the horizontal projection,
  median stroke width (run lengths), ink width per GT character;
- quality: contrast (ink vs background grey level), blur (variance of the
  Laplacian), skew (angle maximising projection-profile sharpness).

Pure numpy + PIL. All features are cheap enough for every block line.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# loading


def _hf_source(cfg: dict) -> tuple[str, str]:
    params = cfg.get("params") or {}
    hf_id = cfg.get("hf_id") or params.get("subset")
    if hf_id and "/" not in hf_id:
        hf_id = f"ahmedheakl/{hf_id}"
    return hf_id, cfg.get("split", "test")


def load_images(dataset: str, sample_ids: Iterable[str], cfg: dict) -> dict[str, tuple[Image.Image, str]]:
    """{sample_id: (raw RGB image, GT text)} from the local HF cache."""
    from datasets import load_dataset

    hf_id, _ = _hf_source(cfg)
    by_split: dict[str, list[tuple[str, int]]] = {}
    for sid in sample_ids:
        _, split, row = sid.rsplit("/", 2)
        by_split.setdefault(split, []).append((sid, int(row)))
    out = {}
    for split, items in by_split.items():
        try:
            # fetch only this split's parquet shards, not the whole repository
            ds = load_dataset(hf_id, split=split, data_files={split: f"data/{split}-*.parquet"})
        except Exception:
            ds = load_dataset(hf_id, split=split)
        sub = ds.select([r for _, r in items])
        for (sid, _), row in zip(items, sub):
            gt = next((row[k] for k in ("text", "answer", "transcription", "label", "gt", "ground_truth")
                       if isinstance(row.get(k), str)), "")
            out[sid] = (row["image"].convert("RGB"), gt)
    return out


# ---------------------------------------------------------------------------
# features


def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    sum_all = (np.arange(256) * hist).sum()
    w0 = s0 = 0.0
    best, thr = -1.0, 127
    for t in range(256):
        w0 += hist[t]
        if w0 == 0 or w0 == total:
            continue
        s0 += t * hist[t]
        m0, m1 = s0 / w0, (sum_all - s0) / (total - w0)
        between = w0 * (total - w0) * (m0 - m1) ** 2
        if between > best:
            best, thr = between, t
    return thr


def _runs(mask_1d: np.ndarray) -> list[int]:
    d = np.diff(np.concatenate([[0], mask_1d.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(ends - starts)


def _laplacian_var(g: np.ndarray) -> float:
    g = g.astype(float)
    lap = (g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1])
    return float(lap.var()) if lap.size else 0.0


def _skew(ink: np.ndarray, max_deg: int = 5) -> float:
    """Angle (deg) whose rotated horizontal projection is sharpest: the sum of
    squared row counts, which empty rows added by the rotation do not change."""
    h, w = ink.shape
    scale = 96 / max(h, 1)
    small = Image.fromarray((ink * 255).astype(np.uint8)).resize(
        (max(int(w * scale), 8), max(int(h * scale), 8)))
    best, angle = -1.0, 0.0
    for a in range(-max_deg, max_deg + 1):
        prof = (np.asarray(small.rotate(a, expand=True, fillcolor=0)) > 127).sum(axis=1).astype(float)
        v = float((prof ** 2).sum())
        if v > best:
            best, angle = v, float(a)
    return angle


def image_features(img: Image.Image, n_chars: int | None = None) -> dict[str, float]:
    gray = np.asarray(img.convert("L"))
    h, w = gray.shape
    thr = otsu(gray)
    ink = gray <= thr  # Otsu's class 0 is the values up to and including the threshold
    # light text on a dark ground: flip so ink is the minority
    if ink.mean() > 0.5:
        ink = ~ink
    density = float(ink.mean())
    cols = ink.any(axis=0)
    rows = ink.sum(axis=1)
    xs = np.where(cols)[0]
    ys = np.where(rows > 0)[0]
    ink_w = int(xs[-1] - xs[0] + 1) if xs.size else 0
    ink_h = int(ys[-1] - ys[0] + 1) if ys.size else 0
    core = int((rows >= 0.5 * rows.max()).sum()) if rows.size and rows.max() else 0
    runs = [r for c in range(0, w, max(w // 200, 1)) for r in _runs(ink[:, c])]
    stroke = float(np.median(runs)) if runs else 0.0
    ink_px, bg_px = gray[ink], gray[~ink]
    contrast = float(bg_px.mean() - ink_px.mean()) if ink_px.size and bg_px.size else 0.0
    edge = max(int(0.02 * w), 1)
    out = {
        "width": float(w), "height": float(h), "aspect": w / max(h, 1),
        "ink_density": density,
        "ink_width_frac": ink_w / max(w, 1), "ink_height_frac": ink_h / max(h, 1),
        "ink_left_edge": float(ink[:, :edge].mean()), "ink_right_edge": float(ink[:, -edge:].mean()),
        "core_height": float(core), "core_frac": core / max(h, 1),
        "stroke_width": stroke, "stroke_rel": stroke / max(core, 1),
        "contrast": contrast, "blur_lapvar": _laplacian_var(gray),
        "skew_deg": _skew(ink),
    }
    if n_chars:
        out["ink_px_per_char"] = ink_w / n_chars
    return out


__all__ = ["image_features", "load_images", "otsu"]
