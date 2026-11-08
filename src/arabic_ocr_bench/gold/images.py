"""Image signatures for de-duplication, and the image tags attached to each gold line."""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image

THUMB = (48, 12)
_RES = getattr(Image, "Resampling", Image)


def decode(raw: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def _gray(im: Image.Image) -> Image.Image:
    return im.convert("L")


def signature(raw: bytes, bits: int = 16) -> dict:
    """Exact pixel hash + difference hash + a small thumbnail (for confirming near-duplicates)."""
    g = _gray(decode(raw))
    sha = hashlib.sha256(f"{g.size}".encode() + g.tobytes()).hexdigest()
    a = np.asarray(g.resize((bits + 1, bits), _RES.BILINEAR), dtype=np.int16)
    dh = 0
    for bit in (a[:, 1:] > a[:, :-1]).ravel():
        dh = (dh << 1) | int(bit)
    thumb = np.asarray(g.resize(THUMB, _RES.BILINEAR), dtype=np.uint8).tobytes()
    return {"sha": sha, "dhash": dh, "thumb": thumb, "w": g.size[0], "h": g.size[1]}


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def thumb_corr(t1: bytes, t2: bytes) -> float:
    a = np.frombuffer(t1, dtype=np.uint8).astype(np.float64)
    b = np.frombuffer(t2, dtype=np.uint8).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def near_duplicate(s1: dict, s2: dict, max_hamming: int, min_corr: float, aspect_tol: float) -> bool:
    if hamming(s1["dhash"], s2["dhash"]) > max_hamming:
        return False
    r1, r2 = s1["w"] / max(s1["h"], 1), s2["w"] / max(s2["h"], 1)
    if abs(r1 - r2) / max(r1, r2) > aspect_tol:
        return False
    return thumb_corr(s1["thumb"], s2["thumb"]) >= min_corr


def _otsu(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    w0 = 0.0
    sum0 = 0.0
    best, thr = -1.0, 127.0
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        m0, m1 = sum0 / w0, (sum_all - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best:
            best, thr = var, t
    return thr


def image_tags(raw: bytes) -> dict:
    im = decode(raw)
    w, h = im.size
    g = np.asarray(_gray(im))
    thr = _otsu(g)
    dark = g <= thr
    ink = float(dark.mean())
    # ink is the minority class; if the "dark" side is the majority (dark background) flip it
    if ink > 0.5:
        ink = 1.0 - ink
    p5, p95 = np.percentile(g, [5, 95])
    sat = 0.0
    if im.mode not in ("L", "1", "LA"):
        hsv = np.asarray(im.convert("RGB").convert("HSV"), dtype=np.float64)
        sat = float(hsv[..., 1].mean() / 255.0)
    mean = float(g.mean())
    background = "coloured" if sat > 0.25 else ("dark" if mean < 100 else "light")
    return {
        "img_w": w, "img_h": h, "aspect": round(w / h, 2),
        "height_tier": "low_le64" if h <= 64 else ("mid_65_128" if h <= 128 else "high_gt128"),
        "ink_density": round(ink, 4),
        "contrast": round(float(p95 - p5) / 255.0, 3),
        "background": background,
    }
