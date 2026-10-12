"""Galleries of the lines a change helped most, hurt most, and a few at random (images, ground truth, both readings).

Library used by the gallery builders (build_xt2_gallery.py and others). A comparison is a list of items:
    {"id", "gt", "img_a", "img_b" (None when the image is the same), "hyp_a", "hyp_b", "err_a", "err_b", "n"}
`gallery(title, note, items, labels)` returns an HTML fragment; `page(title, body)` wraps fragments in a standalone page.
Wrong characters in a reading are highlighted against the ground truth (difflib alignment), missed ones are marked with ▾.
"""
from __future__ import annotations

import base64
import difflib
import html
import io
import random

import numpy as np
from PIL import Image

SHOW_W = 640


def b64(img: np.ndarray) -> str:
    if img.shape[1] > SHOW_W:  # keep the right-hand end: Arabic starts there
        img = img[:, -SHOW_W:]
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def marked(gt: str, hyp: str) -> str:
    """The reading, with characters that differ from the ground truth highlighted and deletions marked."""
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gt, hyp, autojunk=False).get_opcodes():
        seg = html.escape(hyp[j1:j2])
        if op == "equal":
            out.append(seg)
        elif op in ("replace", "insert"):
            out.append(f'<mark class="bad">{seg}</mark>')
        if op in ("delete", "replace") and i2 - i1 > (j2 - j1):
            out.append('<mark class="miss">▾</mark>')
    return "".join(out)


def select(items, k_best=5, k_worst=4, k_rand=4, seed=0):
    d = sorted(items, key=lambda it: (it["err_b"] - it["err_a"]) / it["n"])
    best = [it for it in d if it["err_b"] < it["err_a"]][:k_best]
    worst = [it for it in reversed(d) if it["err_b"] > it["err_a"]][:k_worst]
    rest = [it for it in items if it not in best and it not in worst]
    rnd = random.Random(seed).sample(rest, min(k_rand, len(rest)))
    return best, worst, rnd


def card(it, labels):
    la, lb = labels
    imgs = f'<img alt="line" src="{b64(it["img_a"])}">'
    if it.get("img_b") is not None:
        imgs = f'<div class="pair"><figure><img alt="{la}" src="{b64(it["img_a"])}"><figcaption>{la}</figcaption></figure><figure><img alt="{lb}" src="{b64(it["img_b"])}"><figcaption>{lb}</figcaption></figure></div>'
    dlt = it["err_b"] - it["err_a"]
    tag = "better" if dlt < 0 else ("worse" if dlt > 0 else "same")
    return (f'<div class="gcard {tag}">{imgs}<div class="gtxt">'
            f'<div class="r"><span class="k">truth</span><bdi dir="rtl" lang="ar">{html.escape(it["gt"])}</bdi></div>'
            f'<div class="r"><span class="k">{la}</span><span class="e">{it["err_a"]} err</span><bdi dir="rtl" lang="ar">{marked(it["gt"], it["hyp_a"])}</bdi></div>'
            f'<div class="r"><span class="k">{lb}</span><span class="e">{it["err_b"]} err</span><bdi dir="rtl" lang="ar">{marked(it["gt"], it["hyp_b"])}</bdi></div>'
            f'<div class="r id"><span class="k">line</span>{html.escape(it["id"])} · {it["n"]} characters · change {dlt:+d} errors</div></div></div>')


def gallery(title, note, items, labels, k=(5, 4, 4)):
    best, worst, rnd = select(items, *k)
    n_b = sum(it["err_b"] < it["err_a"] for it in items)
    n_w = sum(it["err_b"] > it["err_a"] for it in items)
    ea, eb, nn = (sum(it[x] for it in items) for x in ("err_a", "err_b", "n"))
    head = (f'<h3>{title}</h3><p>{note}</p><p class="gsum">{len(items)} lines · CER {100 * ea / nn:.1f} % → {100 * eb / nn:.1f} % · '
            f'<span class="better">{n_b} better</span> · <span class="worse">{n_w} worse</span> · {len(items) - n_b - n_w} unchanged</p>')
    sec = lambda name, its: (f'<h4>{name}</h4><div class="gl">' + "".join(card(it, labels) for it in its) + "</div>") if its else ""  # noqa: E731
    return head + sec("Helped most", best) + sec("Hurt most", worst) + sec("Random lines", rnd)


CSS = """
.gl{display:flex;flex-direction:column;gap:10px}
.gcard{border:1px solid var(--line);border-left:4px solid var(--line);border-radius:6px;padding:8px 10px;display:flex;flex-direction:column;gap:6px;background:var(--surface)}
.gcard.better{border-left-color:var(--good)}.gcard.worse{border-left-color:var(--bad)}
.gcard img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:3px;background:#fff}
.gcard .pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.gcard figure{margin:0;min-width:0}.gcard figcaption{font-size:11.5px;color:var(--muted)}
.gtxt{display:flex;flex-direction:column;gap:2px;font-size:14px}
.gtxt .r{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.gtxt .k{font:500 11px var(--f-mono,monospace);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);min-width:110px}
.gtxt .e{font-size:11.5px;color:var(--muted);min-width:48px}
.gtxt bdi{font-family:"Noto Naskh Arabic","Geeza Pro",serif;font-size:17px;line-height:1.7}
.gtxt .id{font-size:11.5px;color:var(--muted)}
mark.bad{background:var(--bad-soft);color:inherit;border-radius:2px}mark.miss{background:none;color:var(--bad);font-size:11px}
.gsum .better{color:var(--good)}.gsum .worse{color:var(--bad)}
@media (max-width:760px){.gcard .pair{grid-template-columns:1fr}}
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@500&display=swap">
<style>:root{{--bg:#f4f6f8;--surface:#fff;--ink:#18202b;--muted:#5a6676;--line:#dde2e8;--good:#2c7a4b;--bad:#b23a26;--bad-soft:#f8d9d2;--f-mono:ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11161d;--surface:#18202a;--ink:#e4e9ef;--muted:#9aa6b4;--line:#2a3542;--good:#7cc79a;--bad:#f0907a;--bad-soft:#4a2620;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}}.wrap{{max-width:1080px;margin:0 auto;padding-inline:20px;padding-block:28px 60px}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin-top:20px}}p{{max-width:80ch}}{css}</style></head>
<body><div class="wrap"><h1>{title}</h1>{body}</div></body></html>"""


def page(title, body):
    return PAGE.format(title=html.escape(title), body=body, css=CSS)
