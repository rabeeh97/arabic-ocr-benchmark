"""Does each published RASAM line crop show the region its PAGE-XML polygon outlines?

    PYTHONPATH=src .venv/bin/python scripts/rasam_crop_check.py

For every RASAM-1/2 line in the Hugging Face parquet: find the TextLine on the same page with the same text (GitHub PAGE-XML), cut its
polygon's bounding box from the page image (as `rasam_recut_broken.py` does), and compare it with the published crop (both grey, 48 x 384,
Pearson correlation over the pixels inside the published crop's polygon mask, best over small shifts and rotations of +-3 deg). A crop
that shows another place of the page scores near 0; faint lines (MS 1943) can also score low, so a low score marks a candidate. Writes results/rasam/push/crop_check.json (per-line score, per-manuscript share below the threshold).
"""
from __future__ import annotations

import glob
import io
import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from rapidfuzz import fuzz

from rasam_line_context import SUB
from rasam_recut_broken import PAGES, norm, xml_lines

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
OUT = "results/rasam/push/crop_check.json"
THR = 0.2  # calibrated on test: the 12 MS 1936 crops cut from the wrong place score < 0.2 (models 60-76 % CER); correct crops 0.59+; other manuscripts: 3 of 1,050 test lines < 0.16, all read correctly


def prep(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L").resize((384, 48), Image.BILINEAR), dtype=np.float32)


def mcorr(hf: np.ndarray, pg: np.ndarray) -> float:
    """Pearson correlation over the pixels inside the published crop's polygon mask (non-black), best over small shifts."""
    m = hf > 3
    best = -1.0
    for dx in (-16, -8, 0, 8, 16):
        for dy in (-4, 0, 4):
            x, y = hf[m], np.roll(np.roll(pg, dx, 1), dy, 0)[m]
            x, y = x - x.mean(), y - y.mean()
            d = float(np.sqrt((x * x).sum() * (y * y).sum()))
            if d > 0:
                best = max(best, float((x * y).sum()) / d)
    return best


def match(hf_img: Image.Image, page: Image.Image, poly, s: float) -> float:
    xs = [x * s for x, _ in poly]
    ys = [y * s for _, y in poly]
    box = (max(0, int(min(xs)) - 2), max(0, int(min(ys)) - 2), min(page.size[0], int(max(xs)) + 2), min(page.size[1], int(max(ys)) + 2))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return -1.0
    crop, hf = page.crop(box), prep(hf_img)
    return max(mcorr(hf, prep(crop.rotate(a, resample=Image.BILINEAR, fillcolor=255) if a else crop)) for a in (-3, -1.5, 0, 1.5, 3))


def load_page(sub: str, stem: str):
    """PAGE-XML lines and the page image, shrunk to <= 1600 px wide (enough for a 48 px comparison)."""
    xp = PAGES / "rasam-dataset-main/page" / sub / f"{stem}.xml"
    ip = PAGES / "images" / f"{stem}.jpg"
    if not (xp.exists() and ip.exists()):
        return None
    lines, xml_w = xml_lines(xp)
    im = Image.open(ip)
    im.draft("L", (1600, 1600 * im.size[1] // im.size[0]))  # fast JPEG downscale on decode
    im = im.convert("L")
    if im.size[0] > 1600:
        im = im.resize((1600, round(im.size[1] * 1600 / im.size[0])), Image.BILINEAR)
    return lines, xml_w, im


def main() -> None:
    recs = []
    for name in ("RASAM-1", "RASAM-2"):
        for split in ("train", "validation", "test"):
            k = -1
            for f in sorted(glob.glob(f"{ROOT}{name}/snapshots/*/data/{split}-*.parquet")):
                for rec in pq.read_table(f, columns=["image", "transcription", "manuscript_name", "image_name"]).to_pylist():
                    k += 1
                    recs.append((f"{name}/{split}/{k:06d}", name, rec))
    by_page = defaultdict(list)
    for x in recs:
        by_page[(x[1], f"{x[2]['manuscript_name']}_{x[2]['image_name']}")].append(x)
    print(len(recs), "lines on", len(by_page), "pages", flush=True)
    rows = []
    for n_done, ((name, stem), group) in enumerate(sorted(by_page.items())):
        got = load_page(SUB[name], stem)
        for lid, _, rec in group:
            hf = Image.open(io.BytesIO(rec["image"]["bytes"]))
            r = {"id": lid, "ms": rec["manuscript_name"], "page": rec["image_name"], "hf_size": list(hf.size)}
            if got is None:
                r["status"] = "no page files"
            elif hf.size[1] < 8:
                r["status"] = "degenerate"
            else:
                lines, xml_w, page = got
                t = norm(rec["transcription"])
                cand = [L for L in lines if L["text"] == t] or sorted(lines, key=lambda L: -fuzz.ratio(L["text"], t))[:1]
                if not cand or fuzz.ratio(cand[0]["text"], t) < 90:
                    r["status"] = "no XML line"
                else:
                    s = page.size[0] / xml_w
                    best = max(match(hf, page, L["poly"], s) for L in cand[:3])  # identical texts on one page: best polygon
                    r["status"], r["corr"] = "ok", round(best, 3)
            rows.append(r)
        if n_done % 50 == 0:
            print(n_done, "pages", flush=True)
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = r["ms"]
        per[key]["lines"] += 1
        if r["status"] == "ok" and r["corr"] < THR:
            per[key]["mismatch"] += 1
        elif r["status"] != "ok":
            per[key][r["status"]] += 1
    tab = {m: dict(v) for m, v in sorted(per.items(), key=lambda x: -x[1].get("mismatch", 0))}
    json.dump({"threshold": THR, "per_manuscript": tab, "lines": rows}, open(OUT, "w"), indent=1)
    for m, v in tab.items():
        print(m, v)
    print("->", OUT)


if __name__ == "__main__":
    main()
