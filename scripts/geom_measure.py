"""Per-line geometry of Arabic text lines: stroke width, dots, baseline, slant, hinge, characteristic loci.

    .geomvenv/bin/python scripts/geom_measure.py OUT_DIR [--n 300] [--tags muharaf_train muharaf ...]

Implements X-G1 (loci), X-G2 (baseline, slope, slant), X-G4 (hinge, curvature) and X-G5 (stroke width, dots,
ascender height) on the cached Arrow files, CPU only. Every line is first rescaled so its median stroke width is
4 px, so histograms from datasets scanned at different resolutions are comparable ("nuqta-style units": the
stroke is the unit, not the pixel). Native-resolution quantities are kept next to the normalized ones.

Sign convention: slant > 0 means strokes lean right (top displaced to the right, like italic Latin); slope > 0
means the baseline descends towards the right of the image (image y points down).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.measure import find_contours
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import SOURCES, load_lines  # noqa: E402

TARGET_SW = 4.0
NB_HINGE = 12
LEG = 5.0  # hinge leg length in px at stroke width 4 (about 1.25 stroke widths)
DOTS = {**{c: 1 for c in "بجخذزضظغفنأإؤئ"}, **{c: 2 for c in "تقية"}, **{c: 3 for c in "ثش"}}
DOTS["ة"] = 2


def expected_dots(label: str) -> tuple[int, int]:
    """(letters that carry at least one mark, total dots) from the transcription."""
    marks = sum(1 for c in label if c in DOTS)
    return marks, sum(DOTS.get(c, 0) for c in label)


def binarize(g: np.ndarray) -> np.ndarray:
    """Ink mask. Sauvola handles stained manuscripts; the Otsu cap removes speckle in empty paper."""
    h = g.shape[0]
    win = int(np.clip(h // 3 | 1, 15, 51))
    gf = g.astype(np.float32)
    ink = gf < threshold_sauvola(gf, window_size=win, k=0.2)
    try:
        ink &= gf < threshold_otsu(gf) + 10
    except ValueError:
        pass
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if n:
        area = ndi.sum_labels(ink, lab, np.arange(1, n + 1))
        ink = np.isin(lab, np.flatnonzero(area >= 3) + 1)
    return ink


def strip_rules(ink: np.ndarray, sw: float):
    """Remove ruled lines and table borders: components much wider than a word and no taller than a stroke or two.
    Returns the cleaned mask and the share of ink pixels that were removed."""
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if not n:
        return ink, 0.0
    sl = ndi.find_objects(lab)
    bad = [i + 1 for i, o in enumerate(sl) if (o[1].stop - o[1].start) > 45 * sw and (o[0].stop - o[0].start) <= 3 * sw]
    if not bad:
        return ink, 0.0
    rm = np.isin(lab, bad)
    return ink & ~rm, float(rm.sum() / max(ink.sum(), 1))


def stroke_width(ink: np.ndarray, skel: np.ndarray | None = None) -> float:
    dt = ndi.distance_transform_edt(ink)
    sk = skeletonize(ink) if skel is None else skel
    return trimmed_width(dt[sk])


def trimmed_width(v: np.ndarray) -> float:
    """Stroke width from local half-widths on the skeleton: 10-90 % trimmed mean (continuous, unlike the median
    of integer distance values), times two, plus one pixel because the skeleton pixel itself is ink."""
    if v.size < 20:
        return float("nan")
    lo, hi = np.percentile(v, [10, 90])
    return float(2 * v[(v >= lo) & (v <= hi)].mean())


def orientation(skel: np.ndarray):
    """Direction of the stroke at every skeleton pixel (structure tensor of the blurred skeleton) and its coherence."""
    s = ndi.gaussian_filter(skel.astype(np.float32), 1.5)
    gy, gx = np.gradient(s)
    jxx, jyy, jxy = (ndi.gaussian_filter(a, 2.5) for a in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # gradient direction
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-9)
    # stroke direction is perpendicular to the gradient: vector (vx, vy) with vy >= 0
    vx, vy = -np.sin(theta), np.cos(theta)
    flip = vy < 0
    vx, vy = np.where(flip, -vx, vx), np.where(flip, -vy, vy)
    return vx, vy, coh


def robust_line(xs, ys, w=None, iters=3):
    """Weighted least squares with MAD-based trimming. Returns slope and intercept (y = a*x + b)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    w = np.ones_like(xs) if w is None else np.asarray(w, float)
    if xs.size < 3:
        return float("nan"), float(np.median(ys)) if ys.size else float("nan")
    keep = np.ones(xs.size, bool)
    a = b = 0.0
    for _ in range(iters):
        if keep.sum() < 3:
            break
        a, b = np.polyfit(xs[keep], ys[keep], 1, w=np.sqrt(w[keep]))
        r = ys - (a * xs + b)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        keep = np.abs(r) < 2.5 * 1.4826 * mad + 1.0
    return float(a), float(b)


def baseline_projection(body: np.ndarray, sw: float):
    """Estimator A: densest row of the body ink in sliding windows, then a robust line through those rows."""
    h, w = body.shape
    win, step = int(16 * sw), int(8 * sw)
    xs, ys, ws = [], [], []
    for x0 in range(0, max(1, w - win + 1), step):
        seg = body[:, x0 : x0 + win]
        p = ndi.gaussian_filter1d(seg.sum(1).astype(np.float32), 1.0)
        if p.max() < 2 * sw:
            continue
        xs.append(x0 + win / 2), ys.append(float(p.argmax())), ws.append(float(seg.sum()))
    return robust_line(xs, ys, ws)


def baseline_rotation(body: np.ndarray):
    """Estimator B: global shear of the rows that maximises the sharpness (sum of squares) of the row profile."""
    h, w = body.shape
    cols = np.flatnonzero(body.any(0))
    if cols.size < 10:
        return float("nan")
    xs = np.arange(w)
    best, ang = -1.0, 0.0
    for deg in np.arange(-10, 10.01, 0.5):
        t = math.tan(math.radians(deg))
        shift = np.round((xs - w / 2) * t).astype(int)
        prof = np.zeros(h + 2 * (abs(int(w / 2 * abs(t))) + 2), np.float32)
        off = (prof.size - h) // 2
        ys, xi = np.nonzero(body)
        np.add.at(prof, ys + off - shift[xi], 1)
        s = float((prof**2).sum())
        if s > best:
            best, ang = s, float(deg)
    return ang  # degrees; positive means the baseline descends to the right


def slant_projection(body: np.ndarray):
    """Vinciarelli-style estimator: shear that makes the vertical projection most peaked."""
    ys, xs = np.nonzero(body)
    if ys.size < 50:
        return float("nan")
    yc = ys.mean()
    best, ang = -1.0, 0.0
    for deg in np.arange(-30, 30.01, 1.0):
        t = math.tan(math.radians(deg))
        xn = xs + np.round((ys - yc) * t).astype(int)
        col = np.bincount(xn - xn.min())
        s = float((col.astype(np.float64) ** 2).sum())
        if s > best:
            best, ang = s, float(deg)
    return ang


def hinge_features(ink: np.ndarray):
    """Hinge (pairs of contour direction at ±LEG), single direction and mean turning angle on the contours."""
    pad = np.pad(ink, 1).astype(np.float32)
    hist = np.zeros((NB_HINGE, NB_HINGE))
    f0 = np.zeros(18)
    turn_sum, turn_n = 0.0, 0
    for c in find_contours(pad, 0.5):
        if len(c) < 3 * LEG:
            continue
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
        if d[-1] < 3 * LEG:
            continue
        s = np.arange(0, d[-1], 1.0)
        p = np.c_[np.interp(s, d, c[:, 1]), np.interp(s, d, c[:, 0])]  # (x, y)
        n, k = len(p), int(round(LEG))
        if n <= 2 * k + 1:
            continue
        i = np.arange(n)
        a, b = p[(i + k) % n] - p, p[(i - k) % n] - p
        pa, pb = np.arctan2(a[:, 1], a[:, 0]) % (2 * np.pi), np.arctan2(b[:, 1], b[:, 0]) % (2 * np.pi)
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        bl = np.minimum((lo / (2 * np.pi) * NB_HINGE).astype(int), NB_HINGE - 1)
        bh = np.minimum((hi / (2 * np.pi) * NB_HINGE).astype(int), NB_HINGE - 1)
        np.add.at(hist, (bl, bh), 1)
        f0 += np.bincount(np.minimum(((pa % np.pi) / np.pi * 18).astype(int), 17), minlength=18)
        ang = np.abs(np.pi - np.abs(((pa - pb) + np.pi) % (2 * np.pi) - np.pi))  # 0 = straight, pi = hairpin
        turn_sum += float(ang.sum())
        turn_n += n
    iu = np.triu_indices(NB_HINGE)
    h = hist[iu]
    return h / max(h.sum(), 1), f0 / max(f0.sum(), 1), turn_sum / max(turn_n, 1)


def loci(body: np.ndarray, yb: np.ndarray):
    """Glucksman loci on background pixels: runs of ink met along N, S, E, W (capped at 2), split into three
    zones relative to the baseline (above, near, below). Returns a (3, 81) count array."""
    h, w = body.shape
    rows = np.flatnonzero(body.any(1))
    cols = np.flatnonzero(body.any(0))
    out = np.zeros((3, 81))
    if rows.size < 5 or cols.size < 5:
        return out
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    b = body[r0:r1, c0:c1]

    def runs_before(a):  # number of ink runs strictly before each pixel along axis 1
        start = a & ~np.pad(a, ((0, 0), (1, 0)))[:, :-1]
        return np.cumsum(start, 1) - start  # a background pixel is not inside a run, so this equals runs before it

    w_ = runs_before(b)
    start_all = b & ~np.pad(b, ((0, 0), (1, 0)))[:, :-1]
    tot_row = start_all.sum(1, keepdims=True)
    e_ = tot_row - np.cumsum(start_all, 1)
    bt = b.T
    st_t = bt & ~np.pad(bt, ((0, 0), (1, 0)))[:, :-1]
    n_t = np.cumsum(st_t, 1) - st_t
    s_t = st_t.sum(1, keepdims=True) - np.cumsum(st_t, 1)
    n_, s_ = n_t.T, s_t.T
    code = np.minimum(n_, 2) * 27 + np.minimum(s_, 2) * 9 + np.minimum(e_, 2) * 3 + np.minimum(w_, 2)
    ys, xs = np.mgrid[r0:r1, c0:c1]
    zone = np.where(ys < yb[xs] - 2 * TARGET_SW, 0, np.where(ys > yb[xs] + 2 * TARGET_SW, 2, 1))
    bg = ~b
    sel = bg & ((np.arange(b.shape[0])[:, None] % 2 == 0) & (np.arange(b.shape[1])[None, :] % 2 == 0))
    for z in range(3):
        m = sel & (zone == z)
        out[z] = np.bincount(code[m], minlength=81)
    return out


def measure(job):
    tag, idx, g, label = job[:4]
    debug = len(job) > 4
    res = {"tag": tag, "idx": idx, "h0": int(g.shape[0]), "w0": int(g.shape[1])}
    try:
        ink0 = binarize(g)
        sk0 = skeletonize(ink0)
        sw0 = stroke_width(ink0, sk0)
        if not np.isfinite(sw0) or sw0 < 0.8:
            return {**res, "ok": False, "why": "no stroke"}
        res["sw_native"] = sw0
        # rescale so the stroke is TARGET_SW wide, then measure everything on the normalized image
        s = float(np.clip(TARGET_SW / sw0, 0.2, 5.0))
        im = Image.fromarray(g).resize((max(16, round(g.shape[1] * s)), max(8, round(g.shape[0] * s))), Image.LANCZOS if s < 1 else Image.BICUBIC)
        g1 = np.asarray(im)
        ink, res["rule_share"] = strip_rules(binarize(g1), TARGET_SW)
        sk = skeletonize(ink)
        sw = stroke_width(ink, sk)
        if not np.isfinite(sw):
            return {**res, "ok": False, "why": "no stroke after scale"}
        lab, n = ndi.label(ink, structure=np.ones((3, 3)))
        sl = ndi.find_objects(lab)
        area = ndi.sum_labels(ink, lab, np.arange(1, n + 1))
        hh = np.array([x[0].stop - x[0].start for x in sl])
        ww = np.array([x[1].stop - x[1].start for x in sl])
        is_dot = (np.maximum(hh, ww) <= 3.2 * sw) & (area >= 0.6 * sw * sw)
        dots = np.flatnonzero(is_dot) + 1
        body = ink & ~np.isin(lab, dots)
        res["n_dots"] = int(dots.size)
        cen = ndi.center_of_mass(ink, lab, dots) if dots.size else []
        res["_dots"] = [(float(cx / s), float(cy / s), float(2 * math.sqrt(a / math.pi) / s)) for (cy, cx), a in zip(cen, area[is_dot])]
        res["dot_diam"] = float(np.sqrt(4 * np.median(area[is_dot]) / np.pi)) if dots.size else float("nan")
        res["dot_diam_over_sw"] = res["dot_diam"] / sw
        marks, ndots = expected_dots(label)
        res["exp_marks"], res["exp_dots"] = marks, ndots
        res["line_len"] = len(label)
        if body.sum() < 100:
            return {**res, "ok": False, "why": "no body"}
        # stroke thickness by stroke direction (pen contrast)
        vx, vy, coh = orientation(sk & body)
        dt = ndi.distance_transform_edt(body)
        ang_h = np.abs(np.degrees(np.arctan2(vy, np.abs(vx))))  # 0 = horizontal, 90 = vertical stroke
        sk_b = sk & body & (coh > 0.5)
        horiz = sk_b & (ang_h < 20)
        vert = sk_b & (ang_h > 70)
        res["sw"] = trimmed_width(dt[sk & body])
        res["sw_horizontal_stroke"] = trimmed_width(dt[horiz])
        res["sw_vertical_stroke"] = trimmed_width(dt[vert])
        res["sw_cv"] = float(np.std(dt[sk & body]) / (np.mean(dt[sk & body]) + 1e-9))
        # baseline, three estimators
        aA, bA = baseline_projection(body, sw)
        res["slope_proj_deg"] = math.degrees(math.atan(aA)) if np.isfinite(aA) else float("nan")
        res["slope_rot_deg"] = baseline_rotation(body)
        hz = sk & body & (coh > 0.5) & (ang_h < 20)
        ys_h, xs_h = np.nonzero(hz)
        if xs_h.size > 40:
            # lowest dense row cluster in each window, using the lower half of horizontal skeleton pixels
            win = int(16 * sw)
            X, Y, W = [], [], []
            for x0 in range(0, max(1, body.shape[1] - win + 1), int(8 * sw)):
                m = (xs_h >= x0) & (xs_h < x0 + win)
                if m.sum() < 8:
                    continue
                hist = np.bincount(ys_h[m], minlength=body.shape[0]).astype(float)
                hist = ndi.gaussian_filter1d(hist, 1.0)
                X.append(x0 + win / 2), Y.append(float(hist.argmax())), W.append(float(m.sum()))
            aC, bC = robust_line(X, Y, W)
            res["slope_skel_deg"] = math.degrees(math.atan(aC)) if np.isfinite(aC) else float("nan")
        else:
            res["slope_skel_deg"] = float("nan")
        # slant, two estimators
        res["slant_proj_deg"] = slant_projection(body)
        vert2 = sk & body & (coh > 0.5) & (ang_h > 45)
        if vert2.sum() > 30:
            al = np.degrees(np.arctan2(-vx[vert2], vy[vert2]))  # positive = leans right
            res["slant_skel_deg"] = float(np.median(al))
            res["slant_skel_iqr"] = float(np.subtract(*np.percentile(al, [75, 25])))
        else:
            res["slant_skel_deg"] = res["slant_skel_iqr"] = float("nan")
        # heights relative to the fitted baseline (stroke widths and dot diameters as units)
        xs_all = np.arange(body.shape[1])
        yb = aA * xs_all + bA if np.isfinite(aA) else np.full(body.shape[1], float(bA))
        colhas = body.any(0)
        top = np.where(colhas, body.argmax(0), np.nan)
        bot = np.where(colhas, body.shape[0] - 1 - body[::-1].argmax(0), np.nan)
        above = yb - top
        below = bot - yb
        ok = colhas
        res["asc_over_sw"] = float(np.nanpercentile(above[ok], 95) / sw)
        res["xheight_over_sw"] = float(np.nanmedian(above[ok]) / sw)
        res["desc_over_sw"] = float(np.nanpercentile(below[ok], 95) / sw)
        res["asc_over_dot"] = res["asc_over_sw"] / res["dot_diam_over_sw"] if dots.size else float("nan")
        res["ink_density"] = float(ink.mean())
        res["ink_height_over_sw"] = float((np.flatnonzero(body.any(1))[-1] - np.flatnonzero(body.any(1))[0]) / sw)
        # hinge, direction, curvature
        hinge, f0, turn = hinge_features(ink)
        res["turn_mean_rad"] = turn
        # loci
        L = loci(body, yb)
        res["ok"] = True
        if debug:
            res["_dbg"] = dict(g1=g1, body=body, dots=np.isin(lab, dots), skel=sk, aA=aA, bA=bA, slant=res["slant_skel_deg"], slantp=res["slant_proj_deg"])
        res["_hinge"], res["_f0"], res["_loci"] = hinge, f0, L
        return res
    except Exception as e:  # keep the sweep going; the failure is reported in the summary
        return {**res, "ok": False, "why": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tags", nargs="*", default=["muharaf_train", "muharaf", "rasam1", "rasam2", "nakba", "nakba_blind", "khatt"])
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    allres = []
    for tag in a.tags:
        jobs = [(tag, i, g, lab) for i, (g, lab) in enumerate(load_lines(tag, a.n))]
        with Pool(a.procs) as p:
            rs = p.map(measure, jobs, chunksize=4)
        ok = [r for r in rs if r.get("ok")]
        print(f"{tag}: {len(ok)}/{len(rs)} lines measured", {r['why'] for r in rs if not r.get('ok')} or "", flush=True)
        allres += rs
    scal = [{k: v for k, v in r.items() if not k.startswith("_")} for r in allres]
    (a.out / "dots.json").write_text(json.dumps({f"{r['tag']}/{r['idx']}": r["_dots"] for r in allres if r.get("ok")}))
    (a.out / "per_line.json").write_text(json.dumps(scal, ensure_ascii=False))
    np.savez_compressed(
        a.out / "hists.npz",
        tag=np.array([r["tag"] for r in allres if r.get("ok")]),
        idx=np.array([r["idx"] for r in allres if r.get("ok")]),
        hinge=np.array([r["_hinge"] for r in allres if r.get("ok")]),
        f0=np.array([r["_f0"] for r in allres if r.get("ok")]),
        loci=np.array([r["_loci"] for r in allres if r.get("ok")]),
    )


if __name__ == "__main__":
    main()
