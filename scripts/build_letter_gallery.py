"""Gallery for the letter-shape library (X-G10): novelty bands against errors, allograph clusters, novel letters.

    .geomvenv/bin/python scripts/build_letter_gallery.py results/geometry/letters OUT.html [--frag FRAG.html]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from change_gallery import page  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dir", type=Path)
ap.add_argument("out")
ap.add_argument("--frag")
a = ap.parse_args()
S = json.loads((a.dir / "summary.json").read_text(encoding="utf-8"))
EX = json.loads((a.dir / "examples.json").read_text(encoding="utf-8"))
EXI = np.load(a.dir / "examples.npz")
AL = np.load(a.dir / "allographs.npz")
NAME = {"muharaf_lines": "Muharaf test", "nakba_lines": "NAKBA test", "nakba_blind": "NAKBA blind", "kitab_khatt": "KHATT"}
COL = {"muharaf_lines": "#2a78d6", "nakba_lines": "#1baf7a", "nakba_blind": "#7bc9a9", "kitab_khatt": "#eb6834"}
FORM = {"iso": "isolated", "ini": "initial", "med": "medial", "fin": "final"}


def png(c):
    b = io.BytesIO()
    Image.fromarray(c).resize((c.shape[1] * 2, c.shape[0] * 2), Image.NEAREST).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def band_chart():
    W, H, x0, x1, y0, y1 = 720, 300, 60, 600, 20, 250
    labels = ["most typical 20 %", "20–40 %", "40–60 %", "60–80 %", "80–95 %", "beyond 95 %"]
    ymax = 0.6
    X = lambda i: x0 + i * (x1 - x0) / 5  # noqa: E731
    Y = lambda v: y1 - v / ymax * (y1 - y0)  # noqa: E731
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Letter error rate by novelty band">']
    for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/><text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" class="t">{int(v * 100)} %</text>')
    for i, lab in enumerate(labels):
        o.append(f'<text x="{X(i):.1f}" y="{y1 + 18}" text-anchor="middle" class="t">{lab}</text>')
    o.append(f'<text x="{(x0 + x1) / 2}" y="{H - 6}" text-anchor="middle" class="t">novelty band (cut points from Muharaf test letters)</text>')
    for t, d in S["datasets"].items():
        pts = [(X(i), Y(v)) for i, v in enumerate(d["error_rate_by_band"]) if v is not None]
        o.append(f'<path d="{"".join(("M" if k == 0 else "L") + f"{x:.1f} {y:.1f}" for k, (x, y) in enumerate(pts))}" fill="none" stroke="{COL[t]}" stroke-width="2.5"/>')
        o += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COL[t]}" stroke="var(--surface)" stroke-width="1.5"/>' for x, y in pts]
        dy = {"nakba_lines": -6, "nakba_blind": 12}.get(t, 4)
        o.append(f'<text x="{pts[-1][0] + 10:.1f}" y="{pts[-1][1] + dy:.1f}" class="l" fill="{COL[t]}">{NAME[t]}</text>')
    o.append("</svg>")
    return "".join(o)


def dataset_table():
    rows = []
    for t, d in S["datasets"].items():
        rows.append(f'<tr><td>{NAME[t]}</td><td class="n">{d["letters"]:,}</td><td class="n">{d["median_novelty"]:.2f}</td><td class="n">{100 * d["share_above_p95_of_muharaf"]:.0f} %</td><td class="n">{100 * d["letter_error_rate"]:.1f} %</td>'
                    + "".join(f'<td class="n">{"–" if v is None else f"{100 * v:.0f} %"}</td>' for v in d["error_rate_by_band"]) + "</tr>")
    return ('<div class="qwrap"><table class="dt"><thead><tr><th>Set</th><th>Letters</th><th>Median novelty</th><th>Beyond Muharaf’s 95th pct</th><th>Letters misread</th>'
            '<th>Error, most typical 20 %</th><th>20–40</th><th>40–60</th><th>60–80</th><th>80–95</th><th>beyond 95 %</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def letter_table():
    rows = []
    items = [(ch, r) for ch, r in S["per_letter"].items() if all(k in r for k in ("muharaf_lines", "nakba_lines", "kitab_khatt"))]
    items.sort(key=lambda x: -x[1]["kitab_khatt"]["median_novelty"])
    for ch, r in items:
        rows.append(f'<tr><td><bdi class="arb">{ch}</bdi></td><td class="n">{r["bank"]:,}</td>'
                    + "".join(f'<td class="n">{r[t]["median_novelty"]:.2f}</td>' for t in ("muharaf_lines", "nakba_lines", "kitab_khatt"))
                    + "".join(f'<td class="n">{100 * r[t]["error_rate"]:.0f} %</td>' for t in ("muharaf_lines", "nakba_lines", "kitab_khatt")) + "</tr>")
    return ('<div class="qwrap"><table class="dt"><thead><tr><th>Letter</th><th>Training examples</th><th>Novelty Muharaf</th><th>NAKBA</th><th>KHATT</th><th>Misread Muharaf</th><th>NAKBA</th><th>KHATT</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def allographs():
    out = []
    for key in AL.files:
        ch, form = key.split("|")
        sizes = S["allograph_sizes"][key]
        tot = sum(sizes)
        cells = "".join(f'<figure><img alt="cluster" src="{png(c)}"><figcaption>{100 * s / tot:.0f} %</figcaption></figure>' for c, s in zip(AL[key], sizes))
        out.append(f'<div class="al"><div class="alh"><bdi class="arb">{ch}</bdi> {FORM[form]} · {tot} examples</div><div class="alr">{cells}</div></div>')
    return '<div class="algrid">' + "".join(out) + "</div>"


def novel(tag, kind, k):
    sel = [(i, e) for i, e in enumerate(EX) if e["tag"] == tag and e["kind"] == kind][:k]
    cells = []
    for i, e in sel:
        mark = "ok" if e["ok"] else "bad"
        cells.append(f'<figure class="nv {mark}"><div class="two"><img alt="letter" src="{png(EXI["crops"][i])}"><img alt="nearest training letter" src="{png(EXI["nn"][i])}"></div>'
                     f'<figcaption><bdi class="arb">{html.escape(e["ch"])}</bdi> {FORM[e["form"]]} · novelty {e["novelty"]:.1f} · {"read right" if e["ok"] else "misread"}</figcaption></figure>')
    return '<div class="nvgrid">' + "".join(cells) + "</div>"


CSS2 = """
.arb{font-family:"Noto Naskh Arabic","Geeza Pro",serif;font-size:20px}
.gbox2 svg{width:100%;max-width:760px;height:auto}.gbox2 .t{fill:var(--muted);font-size:11.5px}.gbox2 .l{font-size:12.5px;font-weight:600}
.algrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}.al{border:1px solid var(--line);border-radius:6px;padding:8px}
.alh{font-size:13px;color:var(--muted)}.alr{display:flex;gap:6px;flex-wrap:wrap}.alr figure{margin:0;text-align:center}.alr img{width:48px;height:48px;border:1px solid var(--line);border-radius:3px}
.alr figcaption{font-size:11px;color:var(--muted)}
.nvgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}.nv{margin:0;border:1px solid var(--line);border-left:4px solid var(--line);border-radius:5px;padding:6px}
.nv.bad{border-left-color:var(--bad)}.nv.ok{border-left-color:var(--good)}.nv .two{display:flex;gap:4px}.nv img{width:72px;height:72px;border:1px solid var(--line);border-radius:3px}
.nv figcaption{font-size:11.5px;color:var(--muted)}
table.dt{border-collapse:collapse;font-size:13px;width:100%}table.dt th,table.dt td{padding:4px 7px;border-bottom:1px solid var(--line);text-align:left}table.dt td.n{text-align:right;font-variant-numeric:tabular-nums}
.qwrap{overflow-x:auto}
"""

frag = f"""
<section><h3>Letter-shape novelty against errors (X-G10)</h3>
<p>Every letter of {S['bank_lines']:,} Muharaf training lines ({S['bank_letters']:,} letters) forms the training bank. Each test letter is located by CTC forced alignment, described by the CRNN’s CNN features inside its span, and scored by its distance to the nearest training letter of the same character and positional form, relative to the typical distance inside the bank (1 = as typical as a training letter). The line marks whether the model’s greedy reading got that letter right.</p>
<div class="gbox2">{band_chart()}</div>
{dataset_table()}
<p>In every set the error rate climbs steeply with novelty, and KHATT has twice NAKBA’s share of letters beyond the most unusual 5 % of Muharaf’s own test letters. Within the same band KHATT letters are still misread more often, so novelty is a large part of the gap but not all of it. Caveat: novelty is measured with the same model’s features, so letters it is unsure of may also look unusual to it; a second model trained separately is the planned check.</p>
<h4>Per letter, sorted by KHATT novelty</h4>{letter_table()}
<h4>Allographs in the training bank</h4><p>k-means (k = 6) on the bank’s features for common letters and forms; each tile is the letter closest to a cluster centre, with the cluster’s share. These are the shapes the model has seen most.</p>{allographs()}
<h4>KHATT’s most novel letters, next to their nearest training letter</h4><p>Left: the KHATT letter crop; right: the closest Muharaf training letter of the same character and form. Red edge: misread by the CRNN.</p>{novel("kitab_khatt", "most_novel", 24)}
<h4>Random KHATT letters, for calibration</h4>{novel("kitab_khatt", "random", 16)}
<h4>NAKBA’s most novel letters</h4>{novel("nakba_lines", "most_novel", 16)}
</section>"""
Path(a.out).write_text(page("Letter-shape library and novelty", "<style>" + CSS2 + "</style>" + frag), encoding="utf-8")
if a.frag:
    Path(a.frag).write_text(frag, encoding="utf-8")
print("wrote", a.out)
