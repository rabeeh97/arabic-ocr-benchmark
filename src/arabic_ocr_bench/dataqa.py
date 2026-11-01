"""Automatic data-QA checks for line-level HTR/OCR datasets.

Each check finds one kind of defect where the reference (ground truth, GT)
does not describe its line image, or the image is not a usable line. The
checks need no human labels and no dataset-specific code. Any dataset that
yields ``(id, PIL image, GT)`` can be scanned (see ``scripts/dataqa_scan.py``).

They come in four tiers by what they need:

  text      the GT string only
  image     the image only
  joint     image and GT, compared against the rest of the dataset (no model)
  reader    an automatic reading of the line by any recogniser (``hyp``)
  dataset   all lines together (duplicates)

Each check returns a :class:`Flag` with a severity score and a short,
human-readable reason. ``CHECKS`` documents every rule and its thresholds.
The thresholds were set on RASAM-1/2 and TariMa, where we have lines labelled
by eye (``results/rasam/push/data_issues_curated.json``). They are meant to
stay conservative: a flag says "look at this line", not "this line is wrong".

The rules generalise ``scripts/rasam_clean_lists.py``.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image

from .characterize.images import otsu
from .metrics import char_edit_counts

# --------------------------------------------------------------------------- #
# registry of definitions


@dataclass(frozen=True)
class Check:
    name: str
    tier: str  # text | image | joint | reader | dataset
    side: str  # which side is likely wrong: "gt", "image" or "either"
    rule: str  # the exact rule, in words
    fix: str  # what to do with a flagged line


CHECKS: dict[str, Check] = {c.name: c for c in [
    Check("empty_gt", "text", "gt",
          "GT has no letters and no digits after removing whitespace, punctuation and format characters.",
          "Drop, or re-transcribe."),
    Check("repeated_gt", "text", "gt",
          "Spaces removed, some span of >= 15 characters appears again within 12 characters of its end "
          "(the GT, or part of it, was pasted twice). Also fires when the GT is a whole-line repeat X+X.",
          "Confirm with a reader (reader tier: doubled_gt); then cut the copy."),
    Check("char_run", "text", "gt",
          "The same non-space, non-tatweel character repeated >= 6 times in a row.",
          "Look: usually a keyboard or HTR artefact."),
    Check("off_script", "text", "gt",
          "Fewer than 50 % of the GT's letters are Arabic-script letters (GT >= 5 letters).",
          "Look: wrong language, transliteration or encoding error."),
    Check("degenerate", "image", "image",
          "Image height or width < 8 px.",
          "Re-cut from the page if a polygon exists, else drop."),
    Check("vertical", "image", "image",
          "Image height >= 1.8 x width: a vertical or marginal strip, not a horizontal line "
          "(skipped when the GT has <= 2 letters, which can be narrow legitimately).",
          "Drop, or rotate if the text runs vertically."),
    Check("blank", "image", "image",
          "No ink: the image is one flat colour (grey std < 2), or < 0.3 % of its non-padding pixels are darker "
          "than 0.8 x the local paper brightness (max filter over 1/8-scale neighbourhoods, so vignetting, "
          "ruling and coloured ink do not matter) while the GT has >= 3 letters. Deliberately conservative: "
          "faint but present text passes and is left to the reader tier.",
          "Drop, or re-cut."),
    Check("multi_line", "image", "image",
          "After deskewing (+-8 deg), the horizontal ink projection (smoothed over H/15 rows, padding "
          "excluded) has >= 2 prominent peaks: each >= 45 % of the highest with >= 40 % of the main line's ink mass "
          "(summed over one core height), apart by >= max(15 % of the height, 0.8 x the main line's core height), "
          "with a valley between them <= 50 % of the lower peak, and the extra peak >= 12 % of the height "
          "from the top and bottom edge (a whole extra line, not a clipped one: see neighbour_clip). Ascender tips and descender loops clipped "
          "from neighbour lines are too low to count.",
          "Re-cut to the line polygon; if the GT holds only one line, the extra line is noise."),
    Check("neighbour_clip", "image", "image",
          "Like multi_line, but the extra peak lies within 12 % of the top or bottom edge: a neighbour "
          "line cut in half by the crop. Informational: common in polygon-bounding-box crops.",
          "Usually keep; re-cut to the polygon if the reader is confused by it."),
    Check("length_mismatch", "joint", "either",
          "Robust z-score (median / MAD over the dataset) of log(GT letters) - log(core width / core height), "
          "where the core is the main line's band on the deskewed ink (neighbour fragments excluded), "
          "is beyond +-3.5: far too many letters for the ink (GT longer than the image: cut crop, doubled "
          "GT, wrong line) or far too few (truncated GT, extra ink).",
          "Look; the sign says which way."),
    Check("gt_truncated", "reader", "gt",
          "The reader reads all of the GT and much more: sub + del <= 15 % of GT length and "
          "insertions >= 15 % of GT length + 5.",
          "Complete the GT, or drop."),
    Check("image_truncated", "reader", "image",
          "The GT holds all the reader reads and much more: sub + ins <= 15 % of GT length and "
          "deletions >= 30 % of GT length + 5 (the crop misses part of the line, or the GT holds extra text).",
          "Re-cut from the page, or drop."),
    Check("doubled_gt", "reader", "gt",
          "Spaces removed, the longest span of >= 5 characters that repeats within 12 characters of its end "
          "is a paste: deleting its second copy from the GT cuts the edit distance to the reading by >= 60 % "
          "of the span length. Natural repeats (\u201cالاثنين في الاثنين\u201d) are read twice and gain nothing.",
          "Cut the copy from the GT."),
    Check("shifted", "reader", "gt",
          "CER(GT, own reading) >= 30 %, GT >= 15 characters, and the reading of another line on the same "
          "page matches the GT with CER < min(30 %, half the own CER): the GT belongs to a neighbour.",
          "Re-assign GTs along the page (or drop)."),
    Check("unreadable", "reader", "either",
          "CER(GT, reading) >= 60 % with GT >= 10 characters and no more specific flag.",
          "Look: garbled GT, wrong crop, or a hard line."),
    Check("duplicate_image", "dataset", "either",
          "Two or more lines have the same image (hash of the 64 x 16 greyscale thumbnail); "
          "detail says whether their GTs agree and whether they cross splits (leakage).",
          "Keep one; if the GTs differ, one is wrong."),
    Check("duplicate_gt", "dataset", "either",
          "Two or more lines share an identical GT of >= 25 characters.",
          "Look: repeated content, or the same GT pasted on different images."),
]}

# --------------------------------------------------------------------------- #
# flags


@dataclass
class Flag:
    check: str
    score: float
    detail: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# text helpers

_FORMAT = {"Cf", "Mn", "Me"}


def letters(t: str) -> str:
    """Letters of ``t`` (Unicode category L*), dropping marks and format characters."""
    return "".join(c for c in t if unicodedata.category(c).startswith("L"))


def is_arabic_letter(c: str) -> bool:
    o = ord(c)
    return (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF
            or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF)


def longest_repeat(t: str, n: int = 5, gap: int = 12) -> tuple[int, int, int]:
    """(i, j, L): the longest span t[i:i+L] (L >= n) whose copy starts at j, at
    most ``gap`` characters after the span's end. (0, 0, 0) when none."""
    best = (0, 0, 0)
    for i in range(len(t) - n + 1):
        seed = t[i:i + n]
        j = t.find(seed, i + n)
        while j >= 0:
            L = n
            while j + L < len(t) and i + L < j and t[i + L] == t[j + L]:
                L += 1
            if j - (i + L) <= gap and L > best[2]:
                best = (i, j, L)
            j = t.find(seed, j + 1)
    return best


def repeated_span(t: str, n: int = 15, gap: int = 12) -> str:
    """First span of ``n`` characters that occurs again starting within ``gap``
    characters of its own end; '' when none."""
    for i in range(len(t) - n + 1):
        s = t[i:i + n]
        j = t.find(s, i + n)
        if s.strip() and 0 <= j <= i + n + gap:
            return s
    return ""


def whole_repeat(t: str) -> str:
    """X when ``t`` (spaces removed) is X repeated >= 2 times, with len(X) >= 4."""
    s = re.sub(r"\s+", "", t)
    for k in range(4, len(s) // 2 + 1):
        if len(s) % k == 0 and s[:k] * (len(s) // k) == s:
            return s[:k]
    return ""


def check_text(gt: str) -> list[Flag]:
    out: list[Flag] = []
    lt = letters(gt)
    if not lt:
        if any(unicodedata.category(c) == "Nd" for c in gt):
            return []  # digits only (page or verse numbers): content, not empty
        return [Flag("empty_gt", 1.0, "GT has no letters or digits")]
    s = re.sub(r"\s+", "", gt)
    rep = repeated_span(s) or whole_repeat(gt)
    if rep:
        out.append(Flag("repeated_gt", min(1.0, len(rep) / max(len(s) / 2, 1)),
                        f"span of {len(rep)} chars repeats", {"span": rep}))
    m = re.search(r"([^\sـ])\1{5,}", gt)
    if m:
        out.append(Flag("char_run", len(m.group(0)) / 10, f"'{m.group(1)}' x{len(m.group(0))}"))
    if len(lt) >= 5:
        ar = sum(is_arabic_letter(c) for c in lt) / len(lt)
        if ar < 0.5:
            out.append(Flag("off_script", 1 - ar, f"{ar:.0%} of letters are Arabic"))
    return out


# --------------------------------------------------------------------------- #
# image helpers


def gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))


def fill_mask(g: np.ndarray, dark: int = 40) -> np.ndarray:
    """Padding added around rotated or polygon crops: dark (<= ``dark``) regions
    connected to the image border that cover >= 1 % of the image. Ink strokes
    touching the border are thin; padding is a solid strip or corner wedge, so a
    region also needs >= 60 % of its bounding box filled."""
    from scipy import ndimage
    m = g <= dark
    if m.mean() < 0.01:
        return np.zeros_like(m)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(m)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    out = np.zeros_like(m)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if i not in edge or sl is None:
            continue
        reg = lab[sl] == i
        if reg.sum() >= 0.003 * m.size and reg.mean() >= 0.3:
            out[sl] |= reg
    return out


def ink_mask(g: np.ndarray) -> np.ndarray:
    """Otsu ink over non-padding pixels, flipped when light-on-dark so ink is the
    minority class. Padding is never ink."""
    pad = fill_mask(g)
    vals = g[~pad] if (~pad).any() else g
    ink = g <= otsu(vals)
    if ink[~pad].mean() > 0.5:
        ink = ~ink
    ink &= ~pad
    return ink


def deskew(ink: np.ndarray, max_deg: int = 8) -> np.ndarray:
    """Rotate the ink mask by the angle (+-max_deg, 1 deg steps) that makes its
    horizontal projection sharpest (sum of squared row counts)."""
    im = Image.fromarray((ink * 255).astype(np.uint8))
    best, out = -1.0, ink
    for a in range(-max_deg, max_deg + 1):
        r = np.asarray(im.rotate(a, expand=True, fillcolor=0)) > 127
        v = float((r.sum(axis=1).astype(float) ** 2).sum())
        if v > best:
            best, out = v, r
    return out


def ink_fraction(g: np.ndarray, rel: float = 0.8, k: int = 8) -> float:
    """Share of non-padding pixels darker than ``rel`` x the local paper
    brightness (a 5 x 5 max filter on the 1/``k``-scale image, upsampled)."""
    h, w = g.shape
    pad = fill_mask(g)
    if (~pad).sum() < 16:
        return 0.0
    small = Image.fromarray(g).resize((max(w // k, 1), max(h // k, 1)), Image.BILINEAR)
    from PIL import ImageFilter
    bg = np.asarray(small.filter(ImageFilter.MaxFilter(5)).resize((w, h), Image.BILINEAR)).astype(float)
    n = g / np.maximum(bg, 1.0)
    return float((n[~pad] < rel).mean())


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def text_bands(ink: np.ndarray, low: float = 0.12) -> list[dict]:
    """Horizontal ink bands: maximal row runs whose smoothed projection is above
    ``low`` x peak. Each band reports top, bottom, height and ink mass."""
    h = ink.shape[0]
    prof = _smooth(ink.sum(axis=1).astype(float), max(h // 40, 1))
    if not prof.size or prof.max() <= 0:
        return []
    on = prof >= low * prof.max()
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [{"top": int(a), "bottom": int(b), "height": int(b - a), "mass": float(prof[a:b].sum())}
            for a, b in zip(starts, ends)]


PEAK_FRAC = 0.45
VALLEY_FRAC = 0.5
MASS_FRAC = 0.4


def line_peaks(ink: np.ndarray, peak_frac: float = PEAK_FRAC, valley_frac: float = VALLEY_FRAC,
               min_sep: float = 0.15, skewed: bool = True) -> list[dict]:
    """Prominent peaks of the deskewed horizontal ink projection (smoothed over
    H/15 rows). A peak is a text line's core when it is >= ``peak_frac`` x the
    highest peak and, against every line already accepted, at least ``min_sep`` x H
    away with a valley between them <= ``valley_frac`` x the lower of the two.
    Neighbour fragments clipped at the edges (ascender tips, descender loops) are
    lower than that, and loops bridging two lines do not fill the valley."""
    if skewed:
        ink = deskew(ink)
    H = ink.shape[0]
    prof = _smooth(ink.sum(axis=1).astype(float), max(H // 15, 1))
    if not prof.size or prof.max() <= 0:
        return []
    top = prof.max()
    y0 = int(prof.argmax())
    a0 = b0 = y0  # the main line's core: rows around the top peak >= 35 % of it
    while a0 > 0 and prof[a0 - 1] >= 0.35 * top:
        a0 -= 1
    while b0 < H - 1 and prof[b0 + 1] >= 0.35 * top:
        b0 += 1
    core = b0 - a0 + 1
    half = max(core // 2, 1)

    def mass(y):
        return float(prof[max(y - half, 0):y + half + 1].sum())

    m0 = mass(y0)
    sep = max(min_sep * H, 0.8 * core)
    cand = [y for y in range(H) if prof[y] >= peak_frac * top and mass(y) >= MASS_FRAC * m0
            and prof[y] >= prof[max(y - 1, 0)] and prof[y] >= prof[min(y + 1, H - 1)]]
    cand.sort(key=lambda y: -prof[y])
    acc: list[int] = []
    for y in cand:
        ok = True
        for a in acc:
            lo, hi = sorted((a, y))
            if hi - lo < sep or prof[lo:hi + 1].min() > valley_frac * min(prof[a], prof[y]):
                ok = False
                break
        if ok:
            acc.append(y)
    return [{"y": int(y), "rel": round(float(prof[y] / top), 2), "H": H} for y in sorted(acc)]


def core_extent(ink: np.ndarray, frac: float = 0.35) -> tuple[int, int]:
    """(width, height) of the main line's core on a deskewed ink mask: the rows
    around the highest projection peak while the smoothed projection stays
    >= ``frac`` x that peak, and the 1st-99th percentile of ink columns in them.
    Ink clipped from neighbour lines lies outside the core and is ignored."""
    H = ink.shape[0]
    prof = _smooth(ink.sum(axis=1).astype(float), max(H // 15, 1))
    if not prof.size or prof.max() <= 0:
        return 0, 0
    y0 = int(prof.argmax())
    a = b = y0
    while a > 0 and prof[a - 1] >= frac * prof[y0]:
        a -= 1
    while b < H - 1 and prof[b + 1] >= frac * prof[y0]:
        b += 1
    cols = np.where(ink[a:b + 1].any(axis=0))[0]
    if cols.size < 2:
        return 0, b - a + 1
    lo, hi = np.percentile(cols, [1, 99])
    return int(hi - lo + 1), b - a + 1


def count_lines(ink: np.ndarray, edge: float = 0.12, skewed: bool = True) -> tuple[int, int, list[dict]]:
    """(whole lines, lines clipped at the top/bottom edge, peaks). A peak within
    ``edge`` x H of the top or bottom is a neighbour cut by the crop, not a whole line."""
    p = line_peaks(ink, skewed=skewed)
    for b in p:
        b["edge"] = b["y"] < edge * b["H"] or b["y"] > (1 - edge) * b["H"]
    whole = [b for b in p if not b["edge"]]
    return max(len(whole), 1 if p else 0), len(p) - len(whole), p


def _cap(img: Image.Image, max_h: int = 240, max_w: int = 1600) -> Image.Image:
    w, h = img.size
    s = min(1.0, max_h / max(h, 1), max_w / max(w, 1))
    return img if s >= 1 else img.resize((max(int(w * s), 1), max(int(h * s), 1)))


def image_stats(img: Image.Image) -> dict:
    """Cheap per-image numbers the image and joint checks use."""
    w, h = img.size
    g = gray(_cap(img))
    st = {"w": w, "h": h, "std": float(g.std()) if g.size else 0.0, "ink_frac": ink_fraction(g)}
    if min(g.shape) >= 4:
        ink = deskew(ink_mask(g))
        st["ink_w"], st["ink_h"] = core_extent(ink)
        n, clip, bands = count_lines(ink, skewed=False)
        st["n_lines"], st["n_clipped"], st["bands"] = n, clip, bands
    else:
        st.update(ink_w=w, ink_h=h, n_lines=1, n_clipped=0, bands=[])
    return st


def check_image(st: dict, gt: str = "") -> list[Flag]:
    w, h = st["w"], st["h"]
    if min(w, h) < 8:
        return [Flag("degenerate", 1.0, f"{w}x{h} px")]
    out: list[Flag] = []
    if h >= 1.8 * w and len(letters(gt)) > 2:
        out.append(Flag("vertical", min(1.0, h / w / 4), f"{w}x{h} px (h/w {h / w:.1f})"))
    if st["std"] < 2 or (st["ink_frac"] < 0.003 and len(letters(gt)) >= 3):
        why = "flat image" if st["std"] < 2 else f"ink {st['ink_frac']:.2%} of pixels"
        out.append(Flag("blank", 1.0, why))
    elif st.get("n_lines", 1) >= 2 and w >= 2 * h:
        out.append(Flag("multi_line", min(1.0, (st["n_lines"] - 1) / 2),
                        f"{st['n_lines']} whole text lines (peaks at rows {[b['y'] for b in st['bands']]} of "
                        f"{st['bands'][0]['H']})", {"bands": st["bands"]}))
    elif st.get("n_clipped", 0) >= 1:
        out.append(Flag("neighbour_clip", 0.3, f"{st['n_clipped']} neighbour line(s) cut at the edge",
                        {"bands": st["bands"]}))
    return out


# --------------------------------------------------------------------------- #
# joint (dataset-relative, no model)


def length_ratio(st: dict, gt: str) -> float | None:
    n = len(letters(gt))
    iw, ih = st.get("ink_w", 0), st.get("ink_h", 0)
    if n < 3 or iw < 8 or ih < 4 or min(st["w"], st["h"]) < 8:
        return None
    return math.log(n) - math.log(iw / ih)


def check_length(ratios: dict[str, float], z_thr: float = 3.5) -> dict[str, Flag]:
    """Robust z of the letters-per-aspect ratio; needs >= 30 lines."""
    if len(ratios) < 30:
        return {}
    v = np.array(list(ratios.values()))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826 or 1e-6
    out = {}
    for k, r in ratios.items():
        z = (r - med) / mad
        if abs(z) >= z_thr:
            way = "GT too long for the image" if z > 0 else "GT too short for the image"
            out[k] = Flag("length_mismatch", min(1.0, abs(z) / 10), f"z={z:+.1f}: {way}", {"z": z})
    return out


# --------------------------------------------------------------------------- #
# reader-assisted


def cer(ref: str, hyp: str) -> tuple[float, dict]:
    e = char_edit_counts(ref, hyp)
    n = max(e["ref_len"], 1)
    return 100.0 * (e["sub"] + e["del"] + e["ins"]) / n, e


def check_reader(gt: str, hyp: str) -> list[Flag]:
    out: list[Flag] = []
    c, e = cer(gt, hyp)
    n = max(e["ref_len"], 1)
    if e["sub"] + e["del"] <= 0.15 * n and e["ins"] >= 0.15 * n + 5:
        out.append(Flag("gt_truncated", min(1.0, e["ins"] / n), f"reader reads {e['ins']} more chars"))
    if e["sub"] + e["ins"] <= 0.15 * n and e["del"] >= 0.30 * n + 5:
        out.append(Flag("image_truncated", min(1.0, e["del"] / n), f"GT has {e['del']} chars the reader misses"))
    rs, hs = re.sub(r"\s+", "", gt), re.sub(r"\s+", "", hyp)
    i, j, L = longest_repeat(rs)
    if L:
        e0 = sum(cer(rs, hs)[1][k] for k in ("sub", "del", "ins"))
        e1 = sum(cer(rs[:j] + rs[j + L:], hs)[1][k] for k in ("sub", "del", "ins"))
        if e0 - e1 >= 0.6 * L:
            out.append(Flag("doubled_gt", min(1.0, (e0 - e1) / L),
                            f"repeat of {L} chars; dropping the copy removes {e0 - e1} of {e0} edits",
                            {"span": rs[i:i + L]}))
    if not out and c >= 60 and len(gt) >= 10:
        out.append(Flag("unreadable", min(1.0, c / 100), f"CER {c:.0f} %"))
    return out


def check_shifted(items: dict[str, tuple[str, str]], groups: dict[str, list[str]]) -> dict[str, Flag]:
    """items: id -> (gt, hyp); groups: page key -> ids on that page."""
    out = {}
    for ids in groups.values():
        if len(ids) < 2:
            continue
        for i in ids:
            gt, hyp = items[i]
            if len(gt) < 15:
                continue
            c, _ = cer(gt, hyp)
            if c < 30:
                continue
            best, bj = 1e9, None
            for j in ids:
                if j == i:
                    continue
                cj, _ = cer(gt, items[j][1])
                if cj < best:
                    best, bj = cj, j
            if bj is not None and best < min(30.0, 0.5 * c):
                out[i] = Flag("shifted", 1 - best / c, f"own CER {c:.0f} %, matches {bj} at {best:.0f} %",
                              {"match": bj})
    return out


# --------------------------------------------------------------------------- #
# dataset-level


def image_hash(img: Image.Image) -> str:
    g = np.asarray(img.convert("L").resize((64, 16)), dtype=np.uint8)
    q = (g // 32).tobytes()  # quantise, so re-encodes still collide
    return hashlib.md5(q + bytes(str(img.size), "ascii")).hexdigest()


def split_of(sid: str) -> str:
    p = sid.split("/")
    return p[-2] if len(p) >= 3 else ""


def check_duplicates(hashes: dict[str, str], gts: dict[str, str]) -> dict[str, list[Flag]]:
    out: dict[str, list[Flag]] = defaultdict(list)
    by_h: dict[str, list[str]] = defaultdict(list)
    for k, h in hashes.items():
        by_h[h].append(k)
    for ids in by_h.values():
        if len(ids) < 2:
            continue
        same = len({gts[i] for i in ids}) == 1
        cross = len({split_of(i) for i in ids}) > 1
        d = f"{len(ids)} copies; GT {'same' if same else 'DIFFERS'}{'; crosses splits' if cross else ''}"
        for i in ids:
            out[i].append(Flag("duplicate_image", 1.0 if not same else 0.5, d,
                               {"others": [j for j in ids if j != i][:5]}))
    by_t: dict[str, list[str]] = defaultdict(list)
    for k, t in gts.items():
        t2 = " ".join(t.split())
        if len(t2) >= 25:
            by_t[t2].append(k)
    for ids in by_t.values():
        if len(ids) < 2:
            continue
        cross = len({split_of(i) for i in ids}) > 1
        for i in ids:
            out[i].append(Flag("duplicate_gt", 0.5, f"{len(ids)} lines share this GT"
                               f"{'; crosses splits' if cross else ''}",
                               {"others": [j for j in ids if j != i][:5]}))
    return out


# --------------------------------------------------------------------------- #
# driver


def scan(samples: Iterable, reads: dict[str, dict] | None = None, page_of=None,
         keep_stats: bool = False) -> dict:
    """Run every check over ``samples`` (objects with ``id``, ``image``, ``ground_truth``).

    reads    optional {id: {"hyp": str, ...}} from any recogniser; enables the reader tier.
    page_of  optional callable(id, read_entry) -> page key; enables ``shifted``.
    Returns {"flags": {id: [flag dicts]}, "counts": {...}, "n": int, "stats": {...}?}.
    """
    flags: dict[str, list[Flag]] = defaultdict(list)
    ratios, hashes, gts, stats = {}, {}, {}, {}
    items, groups = {}, defaultdict(list)
    n = 0
    for s in samples:
        n += 1
        gt = s.ground_truth or ""
        gts[s.id] = gt
        flags[s.id] += check_text(gt)
        st = image_stats(s.image)
        flags[s.id] += check_image(st, gt)
        hashes[s.id] = image_hash(s.image)
        r = length_ratio(st, gt)
        if r is not None:
            ratios[s.id] = r
        if keep_stats:
            stats[s.id] = {k: v for k, v in st.items() if k != "bands"}
        if reads and s.id in reads:
            hyp = reads[s.id].get("hyp", "")
            items[s.id] = (gt, hyp)
            flags[s.id] += check_reader(gt, hyp)
            if page_of is not None:
                pk = page_of(s.id, reads[s.id])
                if pk:
                    groups[pk].append(s.id)
    for k, f in check_length(ratios).items():
        flags[k].append(f)
    if groups:
        for k, f in check_shifted(items, groups).items():
            flags[k].append(f)
            flags[k] = [x for x in flags[k] if x.check != "unreadable"]
    for k, fs in check_duplicates(hashes, gts).items():
        flags[k] += fs
    flags = {k: v for k, v in flags.items() if v}
    counts: dict[str, int] = defaultdict(int)
    for fs in flags.values():
        for f in {x.check for x in fs}:
            counts[f] += 1
    res = {"n": n, "n_read": len(items), "counts": dict(sorted(counts.items())),
           "any": len(flags), "flags": {k: [f.to_dict() for f in v] for k, v in flags.items()}}
    if keep_stats:
        res["stats"] = stats
    return res


__all__ = ["CHECKS", "Check", "Flag", "check_text", "check_image", "check_reader", "check_shifted",
           "check_length", "check_duplicates", "image_stats", "scan", "repeated_span", "whole_repeat",
           "count_lines", "text_bands"]
