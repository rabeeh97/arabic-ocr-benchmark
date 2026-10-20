"""Run the real geometry detectors (scripts/geom_measure.py) on a few real lines and draw each detector's own output
-> docs/research/timeline_detectors.json (small WebP overlays plus the measured values).

Needs the geometry environment and the local dataset copies:
  PYTHONPATH=scripts HF_HUB_OFFLINE=1 .geomvenv/bin/python scripts/extract_detector_examples.py
"""
from __future__ import annotations

import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, "scripts")
import geom_measure as G  # noqa: E402
from geom_data import load_lines  # noqa: E402

OUT = Path("docs/research/timeline_detectors.json")
TAGS = [("khatt", "KHATT"), ("muharaf", "Muharaf"), ("rasam1", "RASAM-1 (Maghrebi)")]
TARGET = G.TARGET_SW


def webp(im: Image.Image, scale: int = 2, q: int = 82) -> str:
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=q, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def normalise(g: np.ndarray):
    """Same rescale as measure(): stroke ~ TARGET px wide. Returns (g1, ink, rule_removed, rule_share, sw)."""
    ink0 = G.binarize(g)
    sw0 = G.stroke_width(ink0, skeletonize(ink0))
    s = float(np.clip(TARGET / sw0, 0.2, 5.0))
    im = Image.fromarray(g).resize((max(16, round(g.shape[1] * s)), max(8, round(g.shape[0] * s))), Image.LANCZOS if s < 1 else Image.BICUBIC)
    g1 = np.asarray(im)
    ink_raw = G.binarize(g1)
    ink, share = G.strip_rules(ink_raw, TARGET)
    return g1, ink_raw, ink, share, G.stroke_width(ink)


def rgb_of(g1):
    base = (255 - (255 - g1.astype(np.float32)) * 0.55).astype(np.uint8)  # faded grey so overlays stand out
    return np.stack([base] * 3, -1)


def pick(tag: str, n: int = 200, want_rules: bool = False):
    """Return (index, grey, label, result) for a tight, clean, representative line (or the line with the most ruled-line ink)."""
    best = None
    fb = None
    for i, (g, lab) in enumerate(load_lines(tag, n)):
        if g.shape[1] < 200 or g.shape[0] < 25:
            continue
        r = G.measure((tag, i, g, lab, True))
        if not r.get("ok"):
            continue
        if want_rules:
            if best is None or r.get("rule_share", 0) > best[3].get("rule_share", 0):
                best = (i, g, lab, r)
            continue
        body = r["_dbg"]["body"]
        rows = np.flatnonzero(body.any(1))
        span = (rows[-1] - rows[0] + 1) / body.shape[0]
        w = body.shape[1]
        hb = max(int(0.12 * body.shape[0]), 2)
        edge = float(body[:hb].sum() + body[-hb:].sum()) / max(float(body.sum()), 1.0)
        finite = all(np.isfinite(r[k]) for k in ("slant_skel_deg", "sw_vertical_stroke", "sw_horizontal_stroke"))
        if finite and r.get("rule_share", 0) < 0.1 and 150 <= w <= 1000 and r["n_dots"] >= 3 and (fb is None or span - 3 * edge > fb[4]):
            fb = (i, g, lab, r, span - 3 * edge)
        if (250 <= w <= 800 and 0.45 <= span <= 0.95 and edge < 0.04 and r.get("rule_share", 0) < 0.03 and r["n_dots"] >= 4
                and np.isfinite(r["slant_skel_deg"]) and np.isfinite(r["sw_vertical_stroke"]) and np.isfinite(r["sw_horizontal_stroke"])):
            return i, g, lab, r
    return best or (fb[:4] if fb else None)


def bbox(mask, pad=4):
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    h, w = mask.shape
    return max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, h), max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, w)


def ticks(body, sk, color=(255, 120, 0)):
    vx, vy, coh = G.orientation(sk & body)
    ang_h = np.abs(np.degrees(np.arctan2(vy, np.abs(vx))))
    vert = sk & body & (coh > 0.5) & (ang_h > 45)
    return vx, vy, vert


out = {"lines": [], "constants": {"target_stroke_px": TARGET}}
for tag, name in TAGS:
    i, g, lab, r = pick(tag)
    d = r["_dbg"]
    r0, r1, c0, c1 = bbox(d["body"])
    g1, body, sk = d["g1"][r0:r1, c0:c1], d["body"][r0:r1, c0:c1], d["skel"][r0:r1, c0:c1]
    dots_m = d["dots"][r0:r1, c0:c1]
    vx, vy, vert = ticks(body, sk)
    H, W = g1.shape
    sc = 3 if W < 350 else 2
    # slant: orange segments along the local stroke direction on near-vertical skeleton pixels
    im = Image.fromarray(rgb_of(g1))
    dr = ImageDraw.Draw(im)
    ys, xs = np.nonzero(vert)
    for k in range(0, len(xs), 5):
        x, y = xs[k], ys[k]
        dx, dy = vx[y, x] * 5, vy[y, x] * 5
        dr.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(255, 110, 0), width=1)
    slant_img = webp(im, sc)
    # slant sweep (the projection estimator's score at every trial shear)
    by, bx = np.nonzero(body)
    yc = by.mean()
    sweep = []
    for deg in np.arange(-30, 30.01, 1.0):
        t = math.tan(math.radians(deg))
        xn = bx + np.round((by - yc) * t).astype(int)
        sweep.append(float((np.bincount(xn - xn.min()).astype(np.float64) ** 2).sum()))
    sweep = [round(v / max(sweep), 4) for v in sweep]
    # stroke: skeleton coloured by stroke direction (vertical blue, horizontal orange, other grey)
    vx2, vy2, coh = G.orientation(sk & body)
    ang_h = np.abs(np.degrees(np.arctan2(vy2, np.abs(vx2))))
    arr = rgb_of(g1)
    arr[sk & body & (coh <= 0.5)] = (130, 130, 130)
    arr[sk & body & (coh > 0.5) & (ang_h >= 20) & (ang_h <= 70)] = (130, 130, 130)
    arr[sk & body & (coh > 0.5) & (ang_h > 70)] = (30, 90, 255)
    arr[sk & body & (coh > 0.5) & (ang_h < 20)] = (255, 130, 0)
    stroke_img = webp(Image.fromarray(arr), sc)
    # dots
    arr = rgb_of(g1)
    arr[dots_m] = (230, 30, 30)
    arr[sk & body] = (30, 90, 255)
    dots_img = webp(Image.fromarray(arr), sc)
    # baseline
    im = Image.fromarray(rgb_of(g1))
    dr = ImageDraw.Draw(im)
    dr.line([(0, d["aA"] * c0 + d["bA"] - r0), (W, d["aA"] * (W + c0) + d["bA"] - r0)], fill=(0, 170, 0), width=1)
    baseline_img = webp(im, sc)
    # tone: grey levels of paper and ink at native resolution
    ink_native = G.binarize(g)
    paper = g[~ink_native]
    inkv = g[ink_native]
    hist, _ = np.histogram(g, bins=32, range=(0, 256))
    out["lines"].append({
        "tag": tag, "name": name, "label": lab, "index": i, "size": [int(g.shape[1]), int(g.shape[0])],
        "orig": webp(Image.fromarray(g1), sc), "slant": slant_img, "stroke": stroke_img, "dots": dots_img, "baseline": baseline_img,
        "sweep": sweep,
        "values": {k: (None if (isinstance(r.get(k), float) and not np.isfinite(r.get(k))) else (round(float(r[k]), 2) if r.get(k) is not None else None))
                   for k in ["slant_skel_deg", "slant_proj_deg", "sw", "sw_vertical_stroke", "sw_horizontal_stroke", "n_dots", "exp_marks", "exp_dots",
                             "slope_proj_deg", "slope_rot_deg", "slope_skel_deg", "dot_diam", "asc_over_sw", "xheight_over_sw"]},
        "tone": {"hist": hist.tolist(), "paper": round(float(np.median(paper)), 1), "ink": round(float(np.median(inkv)), 1)},
    })
    print(name, i, W, H, {k: out["lines"][-1]["values"][k] for k in ["slant_skel_deg", "sw_vertical_stroke", "sw_horizontal_stroke", "n_dots", "exp_marks"]})

# ruled lines: the line (across tags) whose ink has the largest share of removed ruled-line components
cands = []
for tag in ("nakba", "rasam1", "rasam2", "muharaf"):
    pk = pick(tag, 60, want_rules=True)
    if pk:
        cands.append((pk[3].get("rule_share", 0), tag, pk))
cands.sort(key=lambda c: -c[0])
share, tag, (i, g, lab, r) = cands[0]
g1, ink_raw, ink, sh, sw = normalise(g)
arr = rgb_of(g1)
arr[ink_raw & ~ink] = (220, 0, 200)
arr[ink] = (60, 60, 60)
out["rules"] = {"tag": tag, "share": round(float(share), 3), "image": webp(Image.fromarray(arr), 2), "orig": webp(Image.fromarray(g1), 2)}
print("rules", tag, round(float(share), 3))

# loci: probe pixels with rays to the first ink run in N, S, E, W (counts of ink runs, capped at 2)
pk = out["lines"][0]
g0 = load_lines("khatt", pk["index"] + 1)
for j, (g, lab) in enumerate(g0):
    if j == pk["index"]:
        r = G.measure(("khatt", j, g, lab, True))
        break
d = r["_dbg"]
r0, r1, c0, c1 = bbox(d["body"], 6)
body, g1 = d["body"][r0:r1, c0:c1], d["g1"][r0:r1, c0:c1]
H, W = body.shape
rng = np.random.default_rng(3)
rows = np.flatnonzero(body.any(1))
cols = np.flatnonzero(body.any(0))


def runs(a):  # number of ink runs along a 1-D mask
    a = np.asarray(a, bool)
    return int((a & ~np.r_[False, a[:-1]]).sum())


def hits(line):  # indices where an ink run starts along a 1-D mask
    a = np.asarray(line, bool)
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])


SC = 4
im = Image.fromarray(rgb_of(g1)).convert("RGB").resize((W * SC, H * SC), Image.NEAREST)
dr = ImageDraw.Draw(im)
probes = []
tries = 0
while len(probes) < 3 and tries < 6000:
    tries += 1
    y = int(rng.integers(rows[0] + 3, rows[-1] - 3))
    x = int(rng.integers(cols[0] + 6, cols[-1] - 6))
    if body[y, x] or any(abs(x - px) < max(W // 9, 14) for px, py, _ in probes):
        continue
    n_, s_, w_, e_ = runs(body[:y, x]), runs(body[y + 1:, x]), runs(body[y, :x]), runs(body[y, x + 1:])
    if n_ + s_ + w_ + e_ >= 3 and max(n_, s_, w_, e_) <= 2 and (n_, s_, e_, w_) not in [pp[2] for pp in probes]:
        probes.append((x, y, (n_, s_, e_, w_)))
P = lambda v: v * SC + SC // 2
for (x, y, (n_, s_, e_, w_)) in probes:
    for (dx, dy, col) in [(0, -1, (30, 90, 255)), (0, 1, (30, 90, 255)), (1, 0, (255, 130, 0)), (-1, 0, (255, 130, 0))]:
        pts = [(xx, yy) for k in range(1, max(H, W)) for xx, yy in [(x + dx * k, y + dy * k)] if 0 <= xx < W and 0 <= yy < H]
        line = np.array([body[yy, xx] for xx, yy in pts], bool)
        ends = pts[-1] if pts else (x, y)
        dr.line([(P(x), P(y)), (P(ends[0]), P(ends[1]))], fill=col, width=2)
        for k in hits(line):
            xx, yy = pts[int(k)]
            dr.ellipse([P(xx) - 5, P(yy) - 5, P(xx) + 5, P(yy) + 5], outline=(0, 0, 0), fill=(255, 255, 0))
    dr.ellipse([P(x) - 5, P(y) - 5, P(x) + 5, P(y) + 5], fill=(220, 0, 0))
code_rows = [{"n": p[2][0], "s": p[2][1], "e": p[2][2], "w": p[2][3]} for p in probes]
L = r["_loci"]
out["loci"] = {"image": webp(im, 1, 85), "probes": code_rows, "zones": [[int(v) for v in row] for row in L], "name": "KHATT"}
print("loci probes", code_rows)
OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf8")
print("wrote", OUT, round(OUT.stat().st_size / 1024), "KB")
