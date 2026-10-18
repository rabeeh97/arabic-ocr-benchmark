"""Image features per GT letter and per letter boundary, from CTC spans on the 64 px panel images (checklist 2.1, 2.4, 2.5,
2.18). Pre-registered rules: results/characterize/dots/prereg.json.

    .geomvenv/bin/python scripts/dots_extract.py --native   (KHATT and RASAM at native resolution; Muharaf and NAKBA are
                                                             stored at 60 and ~37 px, so 64 px is already their native)

Writes results/characterize/dots/features.npz holding three tables (letters, boundaries, lines) as column arrays:
  letters    sample_key, gi (letters-view GT index), ch, form, paw, x0, x1, cy_body (body-ink centroid in the span columns),
             n_exp, side, style, n_small, small_aspect, small_area
  boundaries sample_key, gi_a, gi_b (GT indices of the letters either side; for a word gap gi_b = gi_a + 2), kind (paw, word,
             joined), gap_px (longest ink-free column run between the two span centres)
  lines      sample_key, dataset, xh, sw, band_top, band_bot, base_sub (Abu-Ain-style sub-word baseline), n_letters_spanned
"""
from __future__ import annotations

import glob
import sys
import time

import numpy as np
from scipy import ndimage as ndi

from dots_common import OUT, W, _letters_view, analyze_letters, gt_index_map, line_geom, load_panel, load_spans  # noqa: F401

DOTS = {"ب": (1, "below"), "ت": (2, "above"), "ث": (3, "above"), "ج": (1, "below"), "خ": (1, "above"), "ذ": (1, "above"),
        "ز": (1, "above"), "ش": (3, "above"), "ض": (1, "above"), "ظ": (1, "above"), "غ": (1, "above"), "ف": (1, "above"),
        "ق": (2, "above"), "ن": (1, "above"), "ي": (2, "below"), "ة": (2, "above"), "ئ": (2, "below")}
UNDOTTED = set("احدرسصطعكلمهوى")


def longest_false_run(a: np.ndarray) -> int:
    if a.size == 0:
        return 0
    best = cur = 0
    for v in a:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def subword_baseline(G) -> float:
    """Abu-Ain-style: per main-body component, the densest row of its lower half; area-weighted median over components."""
    lab, sl, area, sw = G["lab"], G["sl"], G["area"], G["sw"]
    ys, ws = [], []
    for k, s in enumerate(sl):
        if s is None or area[k] <= max(6.0, 4 * sw * sw) or (s[1].stop - s[1].start) < 2 * sw:
            continue
        m = lab[s] == k + 1
        rows = m.sum(1).astype(float)
        cy = (np.arange(m.shape[0]) * rows).sum() / rows.sum()
        lo = int(np.floor(cy))
        r = rows.copy()
        r[:lo] = 0
        ys.append(s[0].start + float(r.argmax()))
        ws.append(float(area[k]))
    if not ys:
        return float("nan")
    o = np.argsort(ys)
    c = np.cumsum(np.array(ws)[o])
    return float(np.array(ys)[o][np.searchsorted(c, c[-1] / 2)])


def native_images(S) -> dict:
    """Native-resolution greyscale for KHATT and RASAM (the panel row index is the arrow row index; labels checked)."""
    import io

    from PIL import Image
    sys.path.insert(0, str(W / "scripts"))
    import geom_data
    out = {}
    for tag, ds in (("khatt", "kitab_khatt"), ("rasam1", "rasam1"), ("rasam2", "rasam2")):
        pat, col, _, _ = geom_data.SOURCES[tag]
        t = geom_data._table(sorted(glob.glob(str(geom_data.HF / pat)))[0])
        for k, s in S.items():
            if s["dataset"] != ds or s["index"] >= t.num_rows:
                continue
            row = t.slice(s["index"], 1).to_pylist()[0]
            if (row[col] or "").strip() != (s["gt_raw"] or "").strip():
                continue
            out[k] = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("L"), dtype=np.uint8)
    print("native images", len(out), flush=True)
    return out


def main() -> None:
    import pyarrow.parquet as pq
    t0 = time.time()
    S = {s["sample_key"]: s for s in pq.read_table(W / "results/store/samples.parquet").to_pylist()}
    spans = load_spans()
    native = native_images(S) if "--native" in sys.argv else {}
    panel = load_panel()
    print("loaded", len(spans), len(panel), f"{time.time() - t0:.0f}s", flush=True)
    L = {k: [] for k in ("sample_key", "gi", "ch", "form", "paw", "x0", "x1", "cy_body", "n_exp", "side", "style", "n_small", "small_aspect", "small_area", "dot_y")}
    B = {k: [] for k in ("sample_key", "gi_a", "gi_b", "kind", "gap_px")}
    N = {k: [] for k in ("sample_key", "dataset", "xh", "sw", "band_top", "band_bot", "base_sub", "n_letters_spanned", "scale")}
    skipped = 0
    for n_done, (key, sp) in enumerate(spans.items()):
        if key not in panel or key not in S:
            skipped += 1
            continue
        img, label = panel[key]
        sx = 1.0
        if key in native:  # KHATT / RASAM: native scans are 110-300 px tall; spans scale with the width
            sx = native[key].shape[1] / img.shape[1]
            img = native[key]
        g = _letters_view(S[key]["gt_raw"])
        LI = {li.index: li for li in analyze_letters(g)}
        m = gt_index_map(label, g)
        G = line_geom(img)
        ink, lab, sl, area, sw, xh = G["ink"], G["lab"], G["sl"], G["area"], G["sw"], G["xh"]
        top, bot = G["band"]
        # small components (dot candidates)
        sm = []
        for k, s in enumerate(sl):
            if s is None:
                continue
            h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
            if 0.4 * sw * sw <= area[k] <= 7 * sw * sw and max(h, w) <= 0.8 * xh and h <= 0.6 * xh:
                cy, cx = ndi.center_of_mass(lab[s] == k + 1)
                sm.append((s[0].start + cy, s[1].start + cx, w / max(1, h), float(area[k])))
        sm = np.array(sm) if sm else np.zeros((0, 4))
        body = G["body"]
        pos = {}
        for li_idx, ch, x0, x1 in sp:
            x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
            gi = m.get(li_idx)
            if gi is None or gi >= len(g) or g[gi] != ch or gi not in LI:
                continue
            pos[gi] = (x0, x1)
            cols = body[:, x0:x1]
            yy = np.nonzero(cols)[0]
            cyb = float(yy.mean()) if yy.size else float("nan")
            n_exp, side = DOTS.get(ch, (0, "above" if ch in UNDOTTED else ""))
            style, n_small, asp, ar = "", -1, float("nan"), float("nan")
            dot_y = float("nan")  # largest small component in the span +-1 frame, either side (baseline check, 2.18)
            if len(sm):
                inw = sm[(sm[:, 1] >= x0 - 4 * sx) & (sm[:, 1] <= x1 + 4 * sx)]
                if len(inw):
                    dot_y = float(inw[int(np.argmax(inw[:, 3])), 0])
            if side:
                def on_side(a, b):
                    if not len(sm):
                        return sm
                    xs, ys = sm[:, 1], sm[:, 0]
                    sel = (xs >= a) & (xs <= b)
                    sel &= (ys < top + 0.25 * xh) if side == "above" else (ys > bot - 0.25 * xh)
                    return sm[sel]
                c = on_side(x0 - 4 * sx, x1 + 4 * sx)
                n_small = len(c)
                if n_small:
                    j = int(np.argmax(c[:, 3]))
                    asp, ar = float(c[j, 2]), float(c[j, 3])
                if n_exp:
                    med = float(np.median(sm[:, 3])) if len(sm) else 1.0
                    if n_small == 0:
                        wdt = max(8 * sx, x1 - x0)
                        style = "displaced" if len(on_side(x0 - 4 * sx - wdt, x1 + 4 * sx + wdt)) else "touching"
                    elif n_exp == 1:
                        style = "separate"
                    elif n_exp == 2:
                        style = "separate" if n_small >= 2 else "dash" if asp > 2 else "blob" if ar > 1.6 * med else "separate"
                    else:
                        style = ("separate" if n_small >= 3 else "partial" if n_small == 2 else
                                 "dash" if asp > 2 else "caret")
            li = LI[gi]
            for k2, v in (("sample_key", key), ("gi", gi), ("ch", ch), ("form", li.position), ("paw", li.paw), ("x0", x0), ("x1", x1),
                          ("cy_body", cyb), ("n_exp", n_exp), ("side", side), ("style", style), ("n_small", n_small),
                          ("small_aspect", asp), ("small_area", ar), ("dot_y", dot_y)):
                L[k2].append(v)
        r0, r1 = max(0, int(top - 1.0 * xh)), min(ink.shape[0], int(bot + 0.5 * xh) + 1)  # own line only
        colink = ink[r0:r1].any(0)
        for gi, (x0, x1) in pos.items():
            for gb, kind in ((gi + 1, None), (gi + 2, "word")):
                if gb not in pos:
                    continue
                if kind == "word":
                    if g[gi + 1] != " ":
                        continue
                else:
                    if g[gb] == " ":
                        continue
                    kind = "paw" if LI[gi].paw != LI[gb].paw else "joined"
                    if LI[gi].word != LI[gb].word:
                        continue
                ca = (x0 + x1) / 2
                cb = (pos[gb][0] + pos[gb][1]) / 2
                gap = longest_false_run(colink[int(np.ceil(cb)) : int(ca)]) if cb < ca else 0
                for k2, v in (("sample_key", key), ("gi_a", gi), ("gi_b", gb), ("kind", kind), ("gap_px", gap)):
                    B[k2].append(v)
        for k2, v in (("sample_key", key), ("dataset", S[key]["dataset"]), ("xh", xh), ("sw", sw), ("band_top", top), ("band_bot", bot),
                      ("base_sub", subword_baseline(G)), ("n_letters_spanned", len(pos)), ("scale", sx)):
            N[k2].append(v)
        if n_done % 500 == 0:
            print(n_done, f"{time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "features.npz", **{f"L_{k}": np.array(v) for k, v in L.items()}, **{f"B_{k}": np.array(v) for k, v in B.items()},
                        **{f"N_{k}": np.array(v) for k, v in N.items()})
    print("lines", len(N["sample_key"]), "letters", len(L["gi"]), "boundaries", len(B["gi_a"]), "skipped", skipped, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
