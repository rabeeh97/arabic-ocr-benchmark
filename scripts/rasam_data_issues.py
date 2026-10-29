"""The "Data issues" tab of results/rasam/push/both_fail.html: hand-checked examples of broken crops and wrong references in the
RASAM-1/2 and TariMa test sets, with the screen counts over all splits.

Called by `push_failures.py`. Inputs: results/rasam/push/data_issues_curated.json (the hand-checked examples, by category),
results/rasam/push/data_audit.json (`rasam_data_audit.py`), results/rasam/push/crop_check.json (`rasam_crop_check.py`).
Each card shows the crop the models see, the line in its page (PAGE-XML polygon in red; RASAM only), the reference and the readings.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from rasam_line_context import context

CURATED = Path("results/rasam/push/data_issues_curated.json")
AUDIT = Path("results/rasam/push/data_audit.json")
CROPS = Path("results/rasam/push/crop_check.json")
DS = {"r1": "RASAM-1", "r2": "RASAM-2", "tm": "tarima"}


def _b64(im) -> str:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()


def section(rows: dict, readings: dict, diff_html, png_b64, cands=("E60", "PT", "ENS")) -> str:
    cur = json.loads(CURATED.read_text())
    audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else None
    crops = json.loads(CROPS.read_text()) if CROPS.exists() else None
    out = [f"<p class=warn><b>These are examples, not a complete list.</b> {html.escape(cur['note'])}</p>"]
    if cur.get("evidence"):
        out.append("<h3>Already reported?</h3><ul>" + "".join(f"<li>{x}</li>" for x in cur["evidence"]) + "</ul>")
    # screen counts over all splits
    if audit or crops:
        out.append("<h3>How common is it? Automatic screens over all splits (candidates, not verdicts)</h3>")
        if crops:
            rows_ms = "".join(f"<tr><td>{html.escape(m)}</td><td>{v.get('lines', 0)}</td><td>{v.get('mismatch', 0)}</td><td>{v.get('degenerate', 0)}</td>"
                              f"<td>{v.get('no XML line', 0)}</td></tr>" for m, v in crops["per_manuscript"].items()
                              if v.get("mismatch") or v.get("degenerate") or v.get("no XML line"))
            out.append(f"<p>Published crop vs. the page region its PAGE-XML polygon outlines (correlation &lt; {crops['threshold']} = shows "
                       f"another place of the page):</p><table><tr><th>manuscript</th><th>lines</th><th>crop shows another place</th>"
                       f"<th>degenerate (&lt; 8 px)</th><th>no XML line with this text</th></tr>{rows_ms}</table>")
        if audit:
            keys = sorted({k for v in audit["counts"].values() for k in v})
            head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
            body = "".join(f"<tr><td>{s}</td>" + "".join(f"<td>{v.get(k, '')}</td>" for k in keys) + "</tr>" for s, v in audit["counts"].items())
            out.append(f"<p>Line screens (the reference screens need our readings, so they only run on test):</p><table><tr><th>set/split</th>{head}</tr>{body}</table>")
            pg = "".join(f"<tr><td>{r['set']}</td><td>{html.escape(r['ms'])}</td><td>{html.escape(str(r['page']))}</td><td>{r['lines']}</td>"
                         f"<td>{r['median_bpc']}</td></tr>" for r in audit.get("per_page_top40", [])[:15])
            if pg:
                out.append(f"<details><summary>Pages whose references are least plausible as Arabic (OpenITI char LM, bits/char; corpus median "
                           f"≈ {audit['median_bpc_test_reference']}, MS 1936 is partly Berber)</summary><table><tr><th>set</th><th>manuscript</th>"
                           f"<th>page</th><th>lines</th><th>median bits/char</th></tr>{pg}</table></details>")
    # nav
    out.append("<p>" + " · ".join(f'<a href="#c-{c["key"]}">{html.escape(c["title"])} ({len(c["examples"])})</a>' for c in cur["categories"]) + "</p>")
    for c in cur["categories"]:
        out.append(f'<h2 id="c-{c["key"]}">{html.escape(c["title"])}</h2><p>{html.escape(c["what"])}<br><b>What to do:</b> {html.escape(c["fix"])}</p>')
        for ex in c["examples"]:
            i = ex["id"]
            r = rows.get(i)
            if r is None:
                continue
            ds = DS[r["dataset"]]
            ctx, info = context(ds, r["manuscript"], r["page"], r["label"])
            rd = readings.get(i, {})
            reads = "".join(f'<div class="rd"><b>{k}</b><span class="cer"></span><div class="txt">{diff_html(r["label"], rd[k])}</div></div>'
                            for k in cands if k in rd)
            ctx_html = (f'<div class=lab>in the page (red = this line\'s polygon, blue = other lines)</div><img src="data:image/jpeg;base64,{_b64(ctx)}">'
                        if ctx is not None else f'<div class=lab>no page view: {html.escape(info.get("error", ""))}</div>')
            out.append(f'<div class="card"><div class="meta">{html.escape(i)} · <span class=tag>{html.escape(r["manuscript"] or "")}</span> '
                       f'<span class=tag>page {html.escape(str(r["page"]))}</span> <span class=tag>{html.escape(r["region"] or "")}</span></div>'
                       f'<p class=note>{html.escape(ex["note"])}</p><div class=lab>what the models see (published crop, 96 px)</div>'
                       f'<img src="data:image/png;base64,{png_b64(r["img"])}">{ctx_html}'
                       f'<div class="rd"><b>GT</b><span class="cer"></span><div class="txt ref">{html.escape(r["label"])}</div></div>{reads}</div>')
    return "".join(out)
