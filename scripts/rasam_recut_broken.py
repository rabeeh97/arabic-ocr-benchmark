"""Re-cut the degenerate RASAM-2 line crops (MS 1936: 80 crops 1 to 7 px tall in the Hugging Face copy) from the original page images and PAGE-XML.

    python scripts/rasam_recut_broken.py [--out results/rasam/repairs/rasam2_recut.pkl]

For every HF line whose crop is under 8 px tall: find the PAGE-XML TextLine of the same page with the same text (exact after normalisation, else the
closest on that page with similarity >= 90), crop the polygon's bounding box (+6 px) from the full page image (coordinates scaled from the XML image size
to the downloaded image), and black out pixels outside the polygon, like the HF crops. Writes a pickle of repairs keyed by (dataset, split, index in the
split's parquet order) for `rasam_prepare_crnn_data.py --repair`, and an HTML contact sheet (before / after / text) for a visual check.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import pickle
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from rapidfuzz import fuzz

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--RASAM-2"
PAGES = Path("/Volumes/PortableSSD/data/rasam-pages")
BIDI = re.compile("[​-‏‪-‮⁦-⁩﻿؜]")


def norm(t: str) -> str:
    return " ".join(BIDI.sub("", unicodedata.normalize("NFC", t or "")).split())


def xml_lines(xml_path: Path) -> tuple[list[dict], int]:
    r = ET.parse(xml_path).getroot()
    ns = {"p": r.tag.split("}")[0].strip("{")}
    page = r.find("p:Page", ns)
    out = []
    for L in page.findall(".//p:TextLine", ns):
        u = L.find(".//p:Unicode", ns)
        pts = [tuple(map(float, x.split(","))) for x in L.find("p:Coords", ns).get("points").split()]
        out.append({"id": L.get("id"), "text": norm(u.text if u is not None else ""), "poly": pts})
    return out, int(page.get("imageWidth"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rasam/repairs/rasam2_recut.pkl")
    ap.add_argument("--crop-check", type=Path, help="also re-cut every RASAM-1/2 crop that rasam_crop_check.py scores below --thr "
                    "(published crop shows another place of the page), not only the degenerate ones")
    ap.add_argument("--thr", type=float, default=0.2)
    a = ap.parse_args()
    flagged = set()
    if a.crop_check:
        flagged = {r["id"] for r in json.loads(a.crop_check.read_text())["lines"] if r["status"] == "ok" and r["corr"] < a.thr}
    repairs, cards, unmatched = {}, [], []
    page_cache: dict = {}
    for name, sub in (("RASAM-1", "rasam1"), ("RASAM-2", "rasam2")) if a.crop_check else (("RASAM-2", "rasam2"),):
      for split in ("train", "validation", "test"):
        idx = 0
        for f in sorted(glob.glob(f"{ROOT.rsplit('--', 1)[0]}--{name}/snapshots/*/data/{split}-*.parquet")):
            for r in pq.read_table(f, columns=["image", "transcription", "manuscript_name", "image_name"]).to_pylist():
                i = idx
                idx += 1
                old = Image.open(io.BytesIO(r["image"]["bytes"]))
                if old.size[1] >= 8 and f"{name}/{split}/{i:06d}" not in flagged:
                    continue
                stem = f"{r['manuscript_name']}_{r['image_name']}"
                xp = PAGES / "rasam-dataset-main/page" / sub / f"{stem}.xml"
                ip = PAGES / "images" / f"{stem}.jpg"
                if not (xp.exists() and ip.exists()):
                    unmatched.append((name, split, i, stem, "page files missing"))
                    continue
                if stem not in page_cache:
                    if len(page_cache) >= 8:  # pages can be 20 Mpx: keep a few
                        page_cache.pop(next(iter(page_cache)))
                    page_cache[stem] = (xml_lines(xp), Image.open(ip).convert("RGB"))
                (lines, xml_w), page = page_cache[stem]
                t = norm(r["transcription"])
                cand = [L for L in lines if L["text"] == t] or sorted(lines, key=lambda L: -fuzz.ratio(L["text"], t))[:1]
                if not cand or fuzz.ratio(cand[0]["text"], t) < 90:
                    unmatched.append((name, split, i, stem, "no matching XML line"))
                    continue
                s = page.size[0] / xml_w
                poly = [(x * s, y * s) for x, y in cand[0]["poly"]]
                xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                box = (max(0, int(min(xs)) - 6), max(0, int(min(ys)) - 6), min(page.size[0], int(max(xs)) + 6), min(page.size[1], int(max(ys)) + 6))
                crop = page.crop(box)
                m = Image.new("L", crop.size, 0)
                ImageDraw.Draw(m).polygon([(x - box[0], y - box[1]) for x, y in poly], fill=255)
                black = Image.new("RGB", crop.size, (0, 0, 0))
                new = Image.composite(crop, black, m)
                buf = io.BytesIO()
                new.save(buf, "PNG")
                repairs[(name, split, i)] ={"bytes": buf.getvalue(), "xml_line": cand[0]["id"], "page": stem, "match": fuzz.ratio(cand[0]["text"], t)}
                th = new.copy()
                th.thumbnail((900, 120))
                tb = io.BytesIO()
                th.save(tb, "JPEG", quality=70)
                cards.append(f"<div class=c><div class=m>{name} {split} #{i} · {html.escape(stem)} · XML {cand[0]['id']} · old crop {old.size[0]}x{old.size[1]} px → new {new.size[0]}x{new.size[1]} px</div>"
                             f"<img src='data:image/jpeg;base64,{base64.b64encode(tb.getvalue()).decode()}'><div dir=rtl class=t>{html.escape(r['transcription'])}</div></div>")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(repairs, open(a.out, "wb"))
    page_html = ("<!doctype html><meta charset=utf-8><title>RASAM-2 re-cut lines</title><style>body{font:14px system-ui;margin:16px;max-width:1000px}"
                 ".c{border:1px solid #8884;border-radius:8px;padding:8px;margin:8px 0}img{max-width:100%;display:block}.m{font-size:12px;opacity:.75}.t{font-size:17px}</style>"
                 f"<h1>RASAM-2: {len(repairs)} degenerate crops re-cut from the original pages</h1><p>unmatched: {len(unmatched)} {html.escape(str(unmatched[:10]))}</p>{''.join(cards)}")
    Path(a.out).with_suffix(".html").write_text(page_html)
    if not a.crop_check:
        (Path.home() / "colab_staging/dashboard/rasam2_recut.html").write_text(page_html)
    print(f"re-cut {len(repairs)} lines; unmatched {len(unmatched)}: {unmatched[:5]}")
    print("by split:", {sp: sum(1 for k in repairs if k[1] == sp) for sp in ("train", "validation", "test")})


if __name__ == "__main__":
    main()
