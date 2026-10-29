"""Compact sample gallery of the hand-checked data defects for the Maghrebi HTR Record (the {{GALLERY}} block).

    PYTHONPATH=src:scripts python scripts/record/data_gallery.py   ->  results/rasam/clean/record_gallery.html

One card per example in results/rasam/push/data_issues_curated.json: the published crop the models see, the line in its page (PAGE-XML
polygon in red; RASAM only), our re-cut where rasam_recut_broken.py made one, the reference, the best system's reading (2-model ensemble
+ char LM + AraGPT2), and what the automatic cleaning did with the line (re-cut / dropped and why / not caught). Images are small WebP so
the page stays light.
"""
from __future__ import annotations

import base64
import html
import io
import json
import pickle
from pathlib import Path

from PIL import Image

from push_failures import PACK, diff_html, readings
from rasam_line_context import context

CURATED = Path("results/rasam/push/data_issues_curated.json")
REPAIRS = Path("results/rasam/repairs/rasam_recut_v2.pkl")
REPORT = Path("results/rasam/clean/clean_report.json")
OUT = Path("results/rasam/clean/record_gallery.html")
DS = {"r1": "RASAM-1", "r2": "RASAM-2", "tm": "tarima"}
RULE = {"garbled": "garbled reference", "garbled_page": "garbled page (uncorrected HTR)", "shifted": "reference of another line",
        "truncated": "reference misses text", "doubled": "reference repeats itself", "vertical": "vertical crop", "blank": "blank crop",
        "external": "invalid in the manual check (PR #7)"}


def webp(im: Image.Image, h: int | None = None, w: int | None = None, q: int = 60) -> str:
    im = im.convert("L")
    if h:
        im = im.resize((max(1, min(900, round(im.width * h / max(1, im.height)))), h), Image.LANCZOS)
    if w and im.width > w:
        im = im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "WEBP", quality=q)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


def main() -> None:
    cur = json.loads(CURATED.read_text())
    rows = {r["id"]: r for r in pickle.load(open(PACK, "rb"))["rows"] if r["role"] == "test"}
    reps = pickle.load(open(REPAIRS, "rb"))
    reasons = json.loads(REPORT.read_text())["reasons"]
    ens = {}
    for t in DS:
        ens.update(readings("ENS", t)[0])
    out, n = [], 0
    for c in cur["categories"]:
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
            imgs = f"<div class='lab'>what the models see (published crop)</div><img alt='' loading='lazy' src='{webp(Image.fromarray(r['img']), h=44)}'>"
            if rep is not None:
                imgs += (f"<div class='lab'>our re-cut from the page</div>"
                         f"<img alt='' loading='lazy' src='{webp(Image.open(io.BytesIO(rep['bytes'])), h=44)}'>")
            ctx, _ = context(DS[r["dataset"]], r["manuscript"], r["page"], r["label"], max_w=560)
            if ctx is not None:
                imgs += (f"<details><summary>the line in its page</summary><img alt='' loading='lazy' "
                         f"src='{webp(ctx, w=560, q=50)}'></details>")
            cards.append(
                f"<div class='card'><div class='cmeta'><span class='mono'>{html.escape(i)}</span> · {html.escape(r['manuscript'] or '')} p. "
                f"{html.escape(str(r['page']))} <span class='pill {fate[0]}'>{html.escape(fate[1])}</span></div>"
                f"<p class='cnote'>{html.escape(ex['note'])}</p>{imgs}"
                f"<div class='rd'><b>GT</b><div class='ar txt'>{html.escape(r['label'])}</div></div>"
                f"<div class='rd'><b>Best</b><div class='ar txt'>{diff_html(r['label'], ens.get(i, ''))}</div></div></div>")
            n += 1
        if cards:
            out.append(f"<details class='cat'><summary>{html.escape(c['title'])} <span class='mono'>({len(cards)})</span></summary>"
                       f"<p>{html.escape(c['what'])}</p><p><b>What to do:</b> {html.escape(c['fix'])}</p>{''.join(cards)}</details>")
    OUT.write_text("".join(out))
    print(n, "cards,", len(out), "categories,", OUT.stat().st_size // 1024, "KB ->", OUT)


if __name__ == "__main__":
    main()
