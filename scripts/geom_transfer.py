"""Turn lines of one dataset into the look of another, using the geometry we measured (CPU).

    .geomvenv/bin/python scripts/geom_transfer.py OUT_DIR [--n 150] [--pairs muharaf:khatt khatt:muharaf ...]

Works in the CRNN's input space: every line is first resized to 64 px high. A transform has these knobs, each tied
to one measured difference between datasets:

    tone     map ink and paper grey levels to the target's           (ink / paper level)
    slant    shear rows                                              (slant_skel_deg)
    stroke   grey-level erosion / dilation, separately for vertical
             strokes (tx) and horizontal strokes (ty)                (stroke width, pen contrast)
    blur     Gaussian blur or unsharp mask                           (edge sharpness)
    noise    paper noise                                             (paper noise level)

For each (source, target) pair the knobs are calibrated on 40 source lines so that the measured medians of the
transformed lines meet the target's medians, then applied to held-out source lines, re-measured, and scored against
real target lines. Nothing here changes the text, so the label of a transformed line is the label of its source.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import load_lines  # noqa: E402
from geom_measure import measure  # noqa: E402

H = 64
MAX_W = 1536
FEATS = ["slant_skel_deg", "sw_abs", "sw_v_abs", "sw_h_abs", "pen_contrast", "dot_diam_over_sw", "ink_level", "paper_level", "paper_noise", "sharpness", "turn_mean_rad", "ink_density"]


def to64(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    nw = max(16, min(MAX_W, round(w * H / h)))
    return np.asarray(Image.fromarray(g).resize((nw, H), Image.LANCZOS), dtype=np.uint8)


# ------------------------------------------------------------------ image statistics (cheap, no binarisation)
def tone_stats(g: np.ndarray) -> dict:
    f = g.astype(np.float32)
    thr = np.percentile(f, 50)
    paper = f[f >= np.percentile(f, 55)]
    ink = f[f <= np.percentile(f, 3)]
    pl = float(np.median(paper))
    il = float(np.mean(ink))
    noise = float(np.std(paper[paper > pl - 25])) if paper.size else 0.0
    n = (np.clip(f, il, pl) - il) / max(pl - il, 1.0)  # ink 0, paper 1
    gy, gx = np.gradient(ndi.gaussian_filter(n, 0.5))
    sharp = float(np.percentile(np.hypot(gx, gy), 99))
    return {"ink_level": il, "paper_level": pl, "paper_noise": noise, "sharpness": sharp}


def line_feats(g64: np.ndarray, label: str = "") -> dict | None:
    r = measure(("x", 0, g64, label))
    if not r.get("ok"):
        return None
    s = r["norm_scale"]
    out = {k: r.get(k, np.nan) for k in ["slant_skel_deg", "dot_diam_over_sw", "turn_mean_rad", "ink_density"]}
    out["sw_abs"] = r["sw_native"]
    out["sw_v_abs"] = r["sw_vertical_stroke"] / s if np.isfinite(r["sw_vertical_stroke"]) else np.nan
    out["sw_h_abs"] = r["sw_horizontal_stroke"] / s if np.isfinite(r["sw_horizontal_stroke"]) else np.nan
    out["pen_contrast"] = out["sw_v_abs"] / out["sw_h_abs"] if out["sw_h_abs"] else np.nan
    out.update(tone_stats(g64))
    out["_hist"] = np.concatenate([r["_hinge"], r["_f0"], (r["_loci"] / np.maximum(r["_loci"].sum(1, keepdims=True), 1)).ravel()])
    return out


def _feat_job(a):
    return line_feats(*a)


# ------------------------------------------------------------------ transform steps
def op_tone(g, ink_to, paper_to):
    st = tone_stats(g)
    f = g.astype(np.float32)
    n = (f - st["ink_level"]) / max(st["paper_level"] - st["ink_level"], 1.0)
    return np.clip(ink_to + n * (paper_to - ink_to), 0, 255)


def op_shear(f, deg):
    if abs(deg) < 0.05:
        return f
    t = np.tan(np.radians(deg))
    h, w = f.shape
    pad = int(np.ceil(abs(t) * h / 2)) + 2
    fill = float(np.median(f[:, : min(w, 10)]))
    out = np.full((h, w + 2 * pad), fill, np.float32)
    ys = np.arange(h)
    for y in ys:  # sub-pixel row shift by linear interpolation
        dx = -(y - h / 2) * t
        xs = np.arange(w + 2 * pad) - pad - dx
        out[y] = np.interp(xs, np.arange(w), f[y], left=fill, right=fill)
    return out


def _blend_filter(f, rx, ry, fn):
    """Rectangular min/max filter with fractional radii, by bilinear blend of the four integer neighbours."""
    def at(ix, iy):
        return fn(f, size=(2 * iy + 1, 2 * ix + 1)) if (ix or iy) else f
    fx, fy = int(np.floor(rx)), int(np.floor(ry))
    ax, ay = rx - fx, ry - fy
    r = (1 - ax) * (1 - ay) * at(fx, fy)
    if ax: r = r + ax * (1 - ay) * at(fx + 1, fy)
    if ay: r = r + (1 - ax) * ay * at(fx, fy + 1)
    if ax and ay: r = r + ax * ay * at(fx + 1, fy + 1)
    return r


def op_stroke(f, tx, ty):
    """tx / ty = change in width (px) of vertical / horizontal strokes. Positive thickens (dark = ink, so a min filter)."""
    out = f
    for t_x, t_y in [(tx, ty)]:
        pos_x, pos_y = max(t_x, 0) / 2, max(t_y, 0) / 2
        neg_x, neg_y = max(-t_x, 0) / 2, max(-t_y, 0) / 2
        if pos_x or pos_y:
            out = _blend_filter(out, pos_x, pos_y, ndi.minimum_filter)
        if neg_x or neg_y:
            out = _blend_filter(out, neg_x, neg_y, ndi.maximum_filter)
    return out


def op_blur(f, sigma, unsharp=0.0):
    out = ndi.gaussian_filter(f, sigma) if sigma > 0.05 else f
    if unsharp > 0.01:
        out = out + unsharp * (out - ndi.gaussian_filter(out, 1.0))
    return out


def op_paper(f, noise_to, rng):
    """Make the paper noise equal to noise_to (grey levels): flatten it if the source is noisier, add smooth noise if cleaner."""
    st = tone_stats(np.clip(f, 0, 255).astype(np.uint8))
    n_s, pl = st["paper_noise"], st["paper_level"]
    if n_s > noise_to + 0.2:
        m = f > pl - 3 * max(n_s, 1.0)
        f = np.where(m, pl + (f - pl) * (noise_to / max(n_s, 1e-3)), f)
    elif noise_to > n_s + 0.2:
        add = np.sqrt(noise_to**2 - n_s**2)
        f = f + ndi.gaussian_filter(rng.normal(0, add, f.shape), 0.6) * 1.6
    return f


def pipeline(g64, P, seed=0):
    rng = np.random.default_rng(seed)
    f = g64.astype(np.float32)
    steps = []
    f = op_shear(f, P["shear"]); steps.append(f)
    f = op_stroke(f, P["tx"], P["ty"]); steps.append(f)
    f = op_blur(f, P["blur"], P["unsharp"]); steps.append(f)
    f = op_tone(np.clip(f, 0, 255).astype(np.uint8), P["ink_level"], P["paper_level"]); steps.append(f)
    f = op_paper(f, P["noise"], rng); steps.append(f)
    return [np.clip(x, 0, 255).astype(np.uint8) for x in steps]


def apply(g64, P, seed=0):
    return pipeline(g64, P, seed)[-1]


def apply_steps(g64, P):
    """The image after each cumulative step (shear, stroke, blur, tone, paper noise), for the step-by-step figure."""
    return pipeline(g64, P, 0)


# ------------------------------------------------------------------ calibration
def med(rows, k):
    v = np.array([r[k] for r in rows if r and np.isfinite(r.get(k, np.nan))], float)
    return float(np.median(v)) if v.size else float("nan")


def measure_set(imgs, labels, pool):
    return [r for r in pool.map(_feat_job, list(zip(imgs, labels)), chunksize=4) if r is not None]


def calibrate(src_imgs, src_labels, src_feats, tgt_feats, pool):
    """Choose the knobs so the measured medians of transformed source lines meet the target's medians."""
    T = {k: med(tgt_feats, k) for k in FEATS}
    S = {k: med(src_feats, k) for k in FEATS}
    P = dict(ink_level=T["ink_level"], paper_level=T["paper_level"], shear=0.0, tx=0.0, ty=0.0, blur=0.0, unsharp=0.0, noise=T["paper_noise"])
    log = []

    def run(P):
        ims = [apply(g, P) for g in src_imgs]
        return ims, measure_set(ims, src_labels, pool)

    # 1. slant: secant search on the shear; the skeleton estimator responds with a gain below one
    want = T["slant_skel_deg"]
    pts = []
    sh = want - S["slant_skel_deg"]
    for _ in range(7):
        P["shear"] = float(np.clip(sh, -35, 35))
        _, F = run(P)
        got = med(F, "slant_skel_deg")
        pts.append((P["shear"], got))
        log.append(("shear", P["shear"], got, want))
        if abs(got - want) < 0.5:
            break
        if len(pts) >= 2 and abs(pts[-1][0] - pts[-2][0]) > 1e-6:
            gain = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
            gain = gain if 0.2 < gain < 2 else 0.8
        else:
            gain = 0.8
        sh = P["shear"] + (want - got) / gain
    # 2. stroke widths, vertical and horizontal separately, damped fixed-point iteration
    for _ in range(5):
        ims, F = run(P)
        ev = T["sw_v_abs"] - med(F, "sw_v_abs")
        eh = T["sw_h_abs"] - med(F, "sw_h_abs")
        log.append(("stroke", (P["tx"], P["ty"]), (med(F, "sw_v_abs"), med(F, "sw_h_abs")), (T["sw_v_abs"], T["sw_h_abs"])))
        if abs(ev) < 0.12 and abs(eh) < 0.12:
            break
        P["tx"] = float(np.clip(P["tx"] + 0.9 * ev, -2.5, 4))
        P["ty"] = float(np.clip(P["ty"] + 0.9 * eh, -2.5, 4))
    # 3. sharpness: blur if too sharp, unsharp mask if too soft
    for _ in range(4):
        ims, F = run(P)
        e = T["sharpness"] - med(F, "sharpness")
        log.append(("sharp", (P["blur"], P["unsharp"]), med(F, "sharpness"), T["sharpness"]))
        if abs(e) < 0.02:
            break
        if e < 0:
            P["blur"] = float(np.clip(P["blur"] + min(0.6, -e * 3), 0, 2.5))
            P["unsharp"] = max(0.0, P["unsharp"] - 0.3)
        else:
            P["unsharp"] = float(np.clip(P["unsharp"] + e * 3, 0, 3))
            P["blur"] = max(0.0, P["blur"] - 0.2)
    # 4. one more slant pass, because stroke and blur changes move the estimator a little
    _, F = run(P)
    got = med(F, "slant_skel_deg")
    log.append(("shear-final", P["shear"], got, want))
    if abs(got - want) > 0.7:
        P["shear"] = float(np.clip(P["shear"] + 0.9 * (want - got), -35, 35))
    return P, log


# ------------------------------------------------------------------ scoring
def zscores(a, b, sd):
    return {k: (med(a, k) - med(b, k)) / sd[k] for k in FEATS if k in sd}


def centroid_classify(real: dict, test_feats, names):
    """Nearest centroid on (hinge, direction, loci) histograms, trained on real 64 px lines; returns the label shares."""
    X = {n: np.sqrt(np.array([r["_hist"] for r in real[n]])) for n in names}
    cm = {n: X[n].mean(0) for n in names}
    out = {n: 0 for n in names}
    for r in test_feats:
        v = np.sqrt(r["_hist"])
        out[min(names, key=lambda n: ((cm[n] - v) ** 2).sum())] += 1
    tot = max(1, sum(out.values()))
    return {n: out[n] / tot for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--pairs", nargs="*", default=["muharaf:khatt", "khatt:muharaf", "nakba:khatt", "muharaf:nakba", "nakba:muharaf"])
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tags = sorted({t for p in a.pairs for t in p.split(":")})
    data, feats = {}, {}
    with Pool(a.procs) as pool:
        for t in tags:
            lines = list(load_lines(t, a.n + 100, with_index=True))
            data[t] = [(i, to64(g), lab) for i, g, lab in lines]
            feats[t] = measure_set([d[1] for d in data[t]], [d[2] for d in data[t]], pool)
            print(t, "measured", len(feats[t]), {k: round(med(feats[t], k), 2) for k in FEATS[:6]}, flush=True)
        sd = {k: float(np.nanstd([r[k] for t in tags for r in feats[t] if np.isfinite(r.get(k, np.nan))])) for k in FEATS}
        result = {"pairs": {}, "sd": sd, "feats_median": {t: {k: med(feats[t], k) for k in FEATS} for t in tags}}
        saved = {}
        for pair in a.pairs:
            s, t = pair.split(":")
            cal = data[s][:100]
            test = data[s][100 : 100 + a.n]
            P, log = calibrate([c[1] for c in cal], [c[2] for c in cal], feats[s][:100], feats[t], pool)
            new_imgs = [apply(d[1], P, seed=d[0]) for d in test]
            new_feats = measure_set(new_imgs, [d[2] for d in test], pool)
            src_feats = measure_set([d[1] for d in test], [d[2] for d in test], pool)
            before = zscores(src_feats, feats[t], sd)
            after = zscores(new_feats, feats[t], sd)
            names = [s, t]
            real = {n: feats[n] for n in names}
            cls_before = centroid_classify(real, src_feats, names)
            cls_after = centroid_classify(real, new_feats, names)
            result["pairs"][pair] = {
                "params": P, "log": [[a_, str(b_), str(c_), str(d_)] for a_, b_, c_, d_ in log],
                "z_before": before, "z_after": after,
                "mean_abs_z_before": float(np.nanmean([abs(v) for v in before.values()])),
                "mean_abs_z_after": float(np.nanmean([abs(v) for v in after.values()])),
                "classifier_share_target_before": cls_before[t], "classifier_share_target_after": cls_after[t],
                "median_before": {k: med(src_feats, k) for k in FEATS}, "median_after": {k: med(new_feats, k) for k in FEATS},
                "median_target": {k: med(feats[t], k) for k in FEATS},
                "p10_p90": {name: {k: [float(x) for x in np.nanpercentile([r[k] for r in fs if np.isfinite(r.get(k, np.nan))], [10, 90])] for k in ["slant_skel_deg", "sw_abs", "pen_contrast", "sharpness", "paper_noise"]}
                            for name, fs in (("source", src_feats), ("transformed", new_feats), ("target", feats[t]))},
            }
            saved[pair] = {"P": P, "test": [(d[0], d[2]) for d in test], "tgt": [(d[0], d[1]) for d in data[t][100:] if d[1].shape[1] >= 400][:6], "src": [d[1] for d in test if d[1].shape[1] >= 400][:8], "tgt_name": t, "src_name": s}
            print(pair, "mean |z| before/after", round(result["pairs"][pair]["mean_abs_z_before"], 2), round(result["pairs"][pair]["mean_abs_z_after"], 2),
                  "classified as target before/after", round(cls_before[t], 2), round(cls_after[t], 2), flush=True)
    (a.out / "transfer.json").write_text(json.dumps(result, indent=1, default=float))
    import pickle
    pickle.dump(saved, open(a.out / "transfer_params.pkl", "wb"))


if __name__ == "__main__":
    main()
