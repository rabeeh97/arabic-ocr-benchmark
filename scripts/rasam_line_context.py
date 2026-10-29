"""Show a RASAM line in its page: the PAGE-XML polygon of the line (matched by text, like `rasam_recut_broken.py`) drawn on a band of the page.

    from rasam_line_context import context
    img, info = context("RASAM-1", "BULAC_MS_ARA_609", "00203", reference_text)

Returns (PIL image or None, info dict with the matched XML line text, match score and polygon box). Page images and the GitHub PAGE-XML
(calfa-co/rasam-dataset) live in /Volumes/PortableSSD/data/rasam-pages; TariMa has no page files here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw
from rapidfuzz import fuzz

from rasam_recut_broken import PAGES, norm, xml_lines

SUB = {"RASAM-1": "rasam1", "RASAM-2": "rasam2"}


@lru_cache(maxsize=64)
def _page(sub: str, stem: str):
    xp = PAGES / "rasam-dataset-main/page" / sub / f"{stem}.xml"
    ip = PAGES / "images" / f"{stem}.jpg"
    if not (xp.exists() and ip.exists()):
        return None
    lines, xml_w = xml_lines(xp)
    return lines, xml_w, Image.open(ip).convert("RGB")


def context(dataset: str, ms: str, page: str, text: str, max_w: int = 1000):
    sub = SUB.get(dataset)
    got = _page(sub, f"{ms}_{page}") if sub else None
    if got is None:
        return None, {"error": "no page files"}
    lines, xml_w, im = got
    t = norm(text)
    best = max(lines, key=lambda L: fuzz.ratio(L["text"], t))
    score = fuzz.ratio(best["text"], t)
    if score < 60:
        return None, {"error": f"no XML line matches the reference (best {score:.0f})"}
    s = im.size[0] / xml_w
    poly = [(x * s, y * s) for x, y in best["poly"]]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    lh = max(30.0, max(ys) - min(ys))
    box = (max(0, int(min(xs) - 0.15 * lh)), max(0, int(min(ys) - 1.6 * lh)),
           min(im.size[0], int(max(xs) + 0.15 * lh)), min(im.size[1], int(max(ys) + 1.6 * lh)))
    band = im.crop(box)
    d = ImageDraw.Draw(band)
    for L in lines:  # every other line faint, this line red
        q = [(x * s - box[0], y * s - box[1]) for x, y in L["poly"]]
        if len(q) > 2:
            d.polygon(q, outline=(230, 0, 0) if L is best else (80, 140, 255), width=max(2, int(band.size[0] / 500)) if L is best else 1)
    if band.size[0] > max_w:
        band = band.resize((max_w, max(1, round(band.size[1] * max_w / band.size[0]))), Image.LANCZOS)
    return band, {"xml_text": best["text"], "match": round(score), "xml_line": best["id"],
                  "poly_box_px": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))], "page_size": list(im.size)}
