"""Sample gallery of the hand-checked data defects for the Maghrebi HTR Record (the {{GALLERY}} block).

    PYTHONPATH=src:scripts python scripts/record/data_gallery.py   ->  results/rasam/clean/record_gallery.html

Top: an overview strip, one tile per defect type (an example crop and a one-line statement of the error), linking to its category.
Then one card per example in results/rasam/push/data_issues_curated.json. Each card starts with the error in words. Below it:
  crop defects (empty, wrong place, neighbouring lines, vertical): the image is wrong and the reference is right. The card shows the
      published crop (red frame), our re-cut from the page (green frame) where rasam_recut_broken.py made one, and the best system's
      reading of each, compared with the reference.
  reference defects (shifted, repeated, truncated, garbled): the image is right and the reference is wrong. The card shows the crop, the
      reference with the parts the image does not show in red, and the text the image does show (struck through where the reference
      leaves it out). "What the image shows" is our best system's reading (2-model ensemble + char LM + AraGPT2), checked by eye.
Also on every card: the line in its page (PAGE-XML polygon of this reference in red, other lines in blue; RASAM only), and what the
automatic cleaning did with the line. Images are small WebP so the page stays light.
"""
from __future__ import annotations

import base64
import difflib
import html
import io
import json
import pickle
from pathlib import Path

from PIL import Image

from push_failures import PACK, readings
from rasam_line_context import context

CURATED = Path("results/rasam/push/data_issues_curated.json")
REPAIRS = Path("results/rasam/repairs/rasam_recut_v2.pkl")
REPORT = Path("results/rasam/clean/clean_report.json")
RECUT_READ = Path("results/rasam/clean/rerun/ours_lm.json")  # clean_rerun_local.py --stage apply
OUT = Path("results/rasam/clean/record_gallery.html")
DS = {"r1": "RASAM-1", "r2": "RASAM-2", "tm": "tarima"}
RULE = {"garbled": "garbled reference", "garbled_page": "garbled page (uncorrected HTR)", "shifted": "reference of another line",
        "truncated": "reference misses text", "doubled": "reference repeats itself", "vertical": "vertical crop", "blank": "blank crop",
        "external": "invalid in the manual check (PR #7)"}
# defect type -> (short label, which side is wrong, one-line statement for the overview)
KIND = {
    "degenerate": ("Empty crop", "crop", "The published crop is 1 px tall: the model gets a black strip, not the line."),
    "wrong-place": ("Crop from the wrong place", "crop", "The crop was cut from another part of the page, not from the line's outline."),
    "neighbours": ("Neighbouring lines in the crop", "crop", "The crop holds most of the lines above or below as well."),
    "vertical": ("Vertical line", "crop", "A vertical margin note, unreadable for a horizontal-line recogniser."),
    "shifted": ("Reference of another line", "ref", "The reference is the text of a neighbouring line."),
    "doubled": ("Reference repeats text", "ref", "Part of the reference is typed twice."),
    "truncated": ("Reference misses text", "ref", "The image holds more words than the reference."),
    "garbled": ("Garbled reference", "ref", "The reference is uncorrected HTR output; the image reads as fluent Arabic."),
}


def webp(im: Image.Image, h: int | None = None, w: int | None = None, q: int = 60, color: bool = False) -> str:
    im = im.convert("RGB" if color else "L")
    if h:
        im = im.resize((max(1, min(900, round(im.width * h / max(1, im.height)))), h), Image.LANCZOS)
    if w and im.width > w:
        im = im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "WEBP", quality=q)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


def cer(ref: str, hyp: str) -> float:
    m = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    e = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in m.get_opcodes() if op != "equal")
    return 100 * e / max(1, len(ref))


def mark(base: str, other: str, extra_cls: str, miss_cls: str) -> str:
    """`other` written out, with the parts not in `base` in extra_cls and the parts of `base` it lacks (struck) in miss_cls."""
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, base, other, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape(other[j1:j2]))
        if op in ("replace", "insert"):
            out.append(f"<span class='{extra_cls}'>{html.escape(other[j1:j2])}</span>")
        if op in ("replace", "delete"):
            out.append(f"<span class='{miss_cls}'>{html.escape(base[i1:i2])}</span>")
    return "".join(out)


def main() -> None:
    cur = json.loads(CURATED.read_text())
    rows = {r["id"]: r for r in pickle.load(open(PACK, "rb"))["rows"] if r["role"] == "test"}
    reps = pickle.load(open(REPAIRS, "rb"))
    reasons = json.loads(REPORT.read_text())["reasons"]
    recut_read = json.loads(RECUT_READ.read_text())["ENS"] if RECUT_READ.exists() else {}
    ens = {}
    for t in DS:
        ens.update(readings("ENS", t)[0])
    overview, out, n = [], [], 0
    for c in cur["categories"]:
        lab, side, one = KIND[c["key"]]
        cards = []
        for ex in c["examples"]:
            i = ex["id"]
            r = rows.get(i)
            if r is None:
                continue
            name, split, k = i.split("/")
            rep = reps.get((name, split, int(k)))
            why = reasons.get(i)
            if rep is not None and why:
                fate = ("p-kept", "re-cut, then dropped: " + ", ".join(RULE.get(x, x) for x in why))
            elif rep is not None:
                fate = ("p-kept", "fixed: re-cut from the page")
            elif why:
                fate = ("p-kept", "dropped: " + ", ".join(RULE.get(x, x) for x in why))
            else:
                fate = ("p-neg", "not caught by the automatic rules")
            gt, hyp = r["label"], ens.get(i, "")
            crop = Image.fromarray(r["img"])
            if side == "crop":
                verdict = "<span class='vx'>image wrong</span><span class='vok'>reference right</span>"
                imgs = (f"<figure class='crop bad'><figcaption>✗ published crop, what the models see</figcaption>"
                        f"<img alt='published crop' loading='lazy' src='{webp(crop, h=44)}'></figure>")
                texts = (f"<div class='rd wide'><b>Reference</b><div class='ar txt'>{html.escape(gt)}</div></div>"
                         f"<div class='rd wide'><b>Read from published crop</b><div class='ar txt'>{mark(gt, hyp, 'bad', 'miss')}"
                         f"<span class='lab cer' dir='ltr'>CER {cer(gt, hyp):.0f} %</span></div></div>")
                if rep is not None:
                    rc = Image.open(io.BytesIO(rep["bytes"]))
                    imgs += (f"<figure class='crop ok'><figcaption>✓ re-cut from the page along the line's outline</figcaption>"
                             f"<img alt='re-cut crop' loading='lazy' src='{webp(rc, h=44)}'></figure>")
                    if i in recut_read:
                        texts += (f"<div class='rd wide'><b>Read from re-cut</b><div class='ar txt'>{mark(gt, recut_read[i], 'bad', 'miss')}"
                                  f"<span class='lab cer' dir='ltr'>CER {cer(gt, recut_read[i]):.0f} %</span></div></div>")
                key = "red = read but not in the reference, struck = in the reference but not read"
            else:
                verdict = "<span class='vok'>image right</span><span class='vx'>reference wrong</span>"
                imgs = (f"<figure class='crop'><figcaption>the crop</figcaption>"
                        f"<img alt='crop' loading='lazy' src='{webp(crop, h=44)}'></figure>")
                texts = (f"<div class='rd wide'><b>Reference</b><div class='ar txt'>{mark(hyp, gt, 'bad', 'none')}</div></div>"
                         f"<div class='rd wide'><b>Image shows</b><div class='ar txt'>{mark(gt, hyp, 'add', 'none')}</div></div>")
                key = "red = in the reference but not in the image, green = in the image but not in the reference"
            ctx, _ = context(DS[r["dataset"]], r["manuscript"], r["page"], gt, max_w=560)
            if ctx is not None:
                imgs += (f"<figure class='crop'><figcaption>the line in its page: <span class='kr'>red</span> = outline of the line this "
                         f"reference belongs to, <span class='kb'>blue</span> = other lines</figcaption>"
                         f"<img alt='page context' loading='lazy' src='{webp(ctx, w=560, q=55, color=True)}'></figure>")
            if not overview or overview[-1][0] != c["key"]:
                overview.append((c["key"], lab, one, webp(crop, h=36), side))
            cards.append(
                f"<div class='card'><p class='err'><span class='etag'>Error · {html.escape(lab)}</span> {html.escape(ex['note'])}</p>"
                f"<div class='cmeta'>{verdict}<span class='mono'>{html.escape(i)}</span> · {html.escape(r['manuscript'] or '')} p. "
                f"{html.escape(str(r['page']))} <span class='pill {fate[0]}'>{html.escape(fate[1])}</span></div>"
                f"{imgs}{texts}<div class='lab'>{key}</div></div>")
            n += 1
        if cards:
            out.append(f"<details class='cat' id='defect-{c['key']}' open><summary>{html.escape(c['title'])} <span class='mono'>({len(cards)})</span></summary>"
                       f"<p>{html.escape(c['what'])}</p><p><b>What to do:</b> {html.escape(c['fix'])}</p>{''.join(cards)}</details>")
    strip = "".join(f"<a class='ov' href='#defect-{k}'><span class='etag'>{html.escape(lab)}</span>"
                    f"<img alt='' src='{im}'><span class='ovt'>{html.escape(one)}</span>"
                    f"<span class='lab'>{'the image is wrong' if side == 'crop' else 'the reference is wrong'}</span></a>"
                    for k, lab, one, im, side in overview)
    OUT.write_text(f"<div class='ovs'>{strip}</div>" + "".join(out))
    print(n, "cards,", len(out), "categories,", OUT.stat().st_size // 1024, "KB ->", OUT)


if __name__ == "__main__":
    main()
