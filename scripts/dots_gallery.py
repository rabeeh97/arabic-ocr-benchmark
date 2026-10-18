"""One HTML gallery for the dots / skeleton / spacing studies (2.1, 2.3, 2.4, 2.5): worst, best and random cases with the
full-width 64 px line image (letters of interest boxed in red from the CTC spans), the GT and the readings.

    .geomvenv/bin/python scripts/dots_gallery.py     # results/characterize/dots/gallery.html
"""
from __future__ import annotations

import base64
import html
import io
import json
import random

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from dots_common import OUT, W, _letters_view, gt_index_map, load_panel, load_spans

K = 8
SHOW_RUNS = ("crnn/base", "vlm/suite-v1-qwen3vl8b-ft-bf16", "vlm/nakba-misraj-baseer", "vlm/qwen3-vl-8b-bf16", "vlm/gemini-flash")


def img_b64(img, boxes):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    for x0, x1 in boxes:
        d.rectangle([x0, 0, x1, img.shape[0] - 1], outline=(220, 30, 30), width=2)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def main() -> None:
    gp = json.loads((OUT / "gallery_picks.json").read_text(encoding="utf-8"))
    tp = json.loads((OUT / "text_picks.json").read_text(encoding="utf-8"))
    rnd = random.Random(0)
    secs = []
    g = gp["gaps"]
    secs.append(("2.1 Split PAW gaps (a reading inserted a space inside a word, at a PAW boundary)", "gap_xh", [
        ("Widest split gaps (look like word gaps: ambiguous spacing)", sorted(g, key=lambda x: -x["gap_xh"])[:K]),
        ("Narrowest split gaps (true over-splits)", sorted(g, key=lambda x: x["gap_xh"])[:K]),
        ("Random split gaps", rnd.sample(g, min(K, len(g))))]))
    s = [x for x in gp["styles"] if x["n_exp"] == 3]
    s2 = [x for x in gp["styles"] if x["n_exp"] < 3]
    secs.append(("2.4 Dot styles (non-separate dots; 3-dot letters first)", "style", [
        ("3-dot letters, dot error", [x for x in s if x["dot_error"]][:K]),
        ("3-dot letters, read right", [x for x in s if not x["dot_error"]][:K]),
        ("1-2-dot letters, dot error", [x for x in s2 if x["dot_error"]][:K]),
        ("Random", rnd.sample(gp["styles"], min(K, len(gp["styles"]))))]))
    st = gp["stack"]
    secs.append(("2.5 Vertically stacked joined pairs (A's ink >= 1 x-height above B's)", "dy_xh", [
        ("Stacked, deleted or transposed", [x for x in st if x["event"]][:K]),
        ("Stacked, read right", [x for x in st if not x["event"]][:K]),
        ("Random stacked", rnd.sample(st, min(K, len(st))))]))
    t = tp["teeth"]
    secs.append(("2.3 Tooth runs with >= 4 teeth", "teeth", [
        ("Tooth letter inserted or deleted", [x for x in t if x["event"]][:K]),
        ("Read right", [x for x in t if not x["event"]][:K]),
        ("Random", rnd.sample(t, min(K, len(t))))]))
    keys = {x["key"] for _, _, grp in secs for _, xs in grp for x in xs}
    S = {r["sample_key"]: r for r in pq.read_table(W / "results/store/samples.parquet").to_pylist() if r["sample_key"] in keys}
    hyp = {}
    for r in pq.read_table(W / "results/store/lines.parquet", columns=["run_id", "sample_key", "transform", "hyp", "pack"]).to_pylist():
        if r["sample_key"] in keys and r["transform"] == "none" and r["hyp"] is not None:
            hyp.setdefault((r["sample_key"], r["run_id"]), r["hyp"])
    panel = load_panel()
    spans = load_spans()
    parts = []
    for title, field, grp in secs:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        for sub, xs in grp:
            parts.append(f"<h3>{html.escape(sub)} ({len(xs)})</h3>")
            for x in xs:
                k = x["key"]
                if k not in panel or k not in S:
                    continue
                img, label = panel[k]
                gtv = _letters_view(S[k]["gt_raw"])
                m = gt_index_map(label, gtv)
                want = set(range(x.get("gi_a", x.get("gi", x.get("a"))), x.get("gi_b", x.get("gi", x.get("b"))) + 1))
                boxes = [(x0, x1) for li, ch, x0, x1 in spans.get(k, []) if m.get(li) in want]
                rows = [f"<tr><th>GT</th><td>{html.escape(gtv)}</td></tr>",
                        f"<tr><th>{html.escape(x['run'])} ({html.escape(x['family'])})</th><td>{html.escape(_letters_view(hyp.get((k, x['run']), '')))}</td></tr>"]
                for r in SHOW_RUNS:
                    if r != x["run"] and (k, r) in hyp:
                        rows.append(f"<tr><th>{r}</th><td>{html.escape(_letters_view(hyp[(k, r)]))}</td></tr>")
                v = x.get(field)
                meta = f"{k} · {field} = {v:.2f}" if isinstance(v, float) else f"{k} · {field} = {v}"
                parts.append(f"<div class=card><div class=meta>{html.escape(meta)}</div><img src='data:image/png;base64,{img_b64(img, boxes)}'>"
                             f"<table>{''.join(rows)}</table></div>")
    doc = ("<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Dots and Spacing Gallery</title>"
           "<style>:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd}@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161616;--fg:#eee;--mut:#aaa;--line:#333}}"
           "body{background:var(--bg);color:var(--fg);font:14px system-ui;margin:0 16px}.card{border-top:1px solid var(--line);padding:8px 0}"
           "img{max-width:100%;height:auto;background:#fff;image-rendering:pixelated}table{width:100%;border-collapse:collapse}"
           "th{text-align:left;color:var(--mut);font-weight:400;width:16em;font-size:12px;vertical-align:top}td{direction:rtl;text-align:right;font-size:16px}"
           ".meta{color:var(--mut);font-size:12px}</style></head><body><h1>Dots, skeleton and spacing: per-sample gallery</h1>"
           "<p>Checklist 2.1, 2.3, 2.4, 2.5. Red boxes: CTC spans of the letters in question (base aligner, 64 px panel image). Numbers: "
           "results/characterize/dots/image_studies.json and text_studies.json.</p>" + "".join(parts) + "</body></html>")
    (OUT / "gallery.html").write_text(doc, encoding="utf-8")
    print("gallery", OUT / "gallery.html", len(doc) // 1024, "KB")


if __name__ == "__main__":
    main()
