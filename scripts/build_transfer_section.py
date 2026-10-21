"""Insert the "Geometry transfer" article into docs/research/paper-ledger.html (after the geometry measurements).

    .geomvenv/bin/python scripts/build_transfer_section.py results/geometry/transfer [--cer results/geometry/replica/transfer_cer.json] [--verdict FILE]

Reads transfer.json and transfer_params.pkl (geom_transfer.py), an optional CER file (geom_transfer_eval.py) and an optional
verdict paragraph (HTML). Idempotent: it removes an earlier copy of the article first.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from geom_transfer import apply, apply_steps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--cer", type=Path)
ap.add_argument("--verdict", type=Path)
ap.add_argument("--extra", type=Path, help="HTML paragraphs placed after the CER table")
a = ap.parse_args()
R = json.loads((a.run / "transfer.json").read_text())
SAVED = pickle.load(open(a.run / "transfer_params.pkl", "rb"))
LEDGER = Path("docs/research/paper-ledger.html")
NICE = {"muharaf": "Muharaf", "khatt": "KHATT", "nakba": "NAKBA"}
FNAME = {"slant_skel_deg": "Slant (°, + leans right)", "sw_v_abs": "Vertical-stroke width (px)", "sw_h_abs": "Horizontal-stroke width (px)", "pen_contrast": "Pen contrast",
         "ink_level": "Ink grey level", "paper_level": "Paper grey level", "paper_noise": "Paper noise", "sharpness": "Edge sharpness", "turn_mean_rad": "Contour curviness (rad)", "dot_diam_over_sw": "Dot ÷ stroke"}
SHOW_W = 560


def f(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def b64(img):
    if img.shape[1] > SHOW_W:
        img = img[:, -SHOW_W:]
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def im(img, cap, cls=""):
    return f'<figure class="xln {cls}"><img alt="{cap}" src="{b64(img)}"><figcaption>{cap}</figcaption></figure>'


def pair_block(pair, nrows, steps=False, spread=False):
    s, t = pair.split(":")
    S, v = SAVED[pair], R["pairs"][pair]
    P = S["P"]
    rows = "".join(f'<div class="xrow">{im(g, f"{NICE[s]} original")}{im(apply(g, P), f"made to look like {NICE[t]}", "new")}</div>' for g in S["src"][:nrows])
    real = "".join(im(g, f"real {NICE[t]}") for _, g in S["tgt"][:2])
    out = (f'<h4>{NICE[s]} → {NICE[t]}</h4><p>Mean gap to the {NICE[t]} medians over twelve measures: {f(v["mean_abs_z_before"])} SD before, <b>{f(v["mean_abs_z_after"])}</b> after. '
           f'Knobs fitted on 100 {NICE[s]} lines: shear {f(P["shear"], 1)}°, vertical strokes {f(P["tx"], 1)} px, horizontal strokes {f(P["ty"], 1)} px, unsharp {f(P["unsharp"], 1)}, ink grey {P["ink_level"]:.0f}, paper grey {P["paper_level"]:.0f}, paper noise {P["noise"]:.1f}.</p>'
           f'<div class="xpairs">{rows}</div><div class="xstrip">{real}</div>')
    if steps:
        st = apply_steps(S["src"][0], P)
        names = ["+ slant", "+ stroke width", "+ blur / sharpen", "+ ink and paper tone", "+ paper noise"]
        out += '<p class="xcap">One line, step by step:</p><div class="xstrip">' + im(S["src"][0], "original") + "".join(im(x, n) for x, n in zip(st, names)) + "</div>"
    return out


def summary_table():
    rows = []
    for pair, v in R["pairs"].items():
        s, t = pair.split(":")
        rows.append(f'<tr><td>{NICE[s]} → {NICE[t]}</td><td class="n">{f(v["mean_abs_z_before"])}</td><td class="n"><b>{f(v["mean_abs_z_after"])}</b></td>'
                    f'<td class="n">{v["classifier_share_target_before"] * 100:.0f} % → {v["classifier_share_target_after"] * 100:.0f} %</td></tr>')
    return ('<div class="qwrap"><table class="dt"><thead><tr><th>Pair</th><th>Mean gap to target before (SD)</th><th>after (SD)</th><th>Called target by a contour classifier</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def measure_table(pair):
    v = R["pairs"][pair]
    rows = []
    for k, name in FNAME.items():
        d = 1 if abs(v["median_target"][k]) > 5 else 2
        rows.append(f'<tr><td>{name}</td><td class="n">{f(v["median_before"][k], d)}</td><td class="n">{f(v["median_after"][k], d)}</td><td class="n">{f(v["median_target"][k], d)}</td></tr>')
    s, t = pair.split(":")
    return (f'<div class="qwrap"><table class="dt"><thead><tr><th>{NICE[s]} → {NICE[t]}, median of lines</th><th>Source</th><th>Transformed</th><th>Target</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


LABEL = {
    "kitab_khatt__as_muharaf": "KHATT made to look like Muharaf (all steps)",
    "kitab_khatt__as_muharaf_only_slant": "… slant only",
    "kitab_khatt__as_muharaf_only_stroke": "… stroke width and blur only",
    "kitab_khatt__as_muharaf_only_paper": "… ink and paper tone and noise only",
    "nakba_lines__as_muharaf": "NAKBA made to look like Muharaf (all steps)",
    "nakba_lines__as_muharaf_only_slant": "… slant only",
    "nakba_lines__as_muharaf_only_stroke": "… stroke width and blur only",
    "nakba_lines__as_muharaf_only_paper": "… ink and paper tone and noise only",
    "nakba_lines__norules": "NAKBA with ruled lines painted out",
    "nakba_lines__as_khatt": "NAKBA made to look like KHATT",
    "muharaf_lines__as_khatt": "Muharaf made to look like KHATT (all steps)",
    "muharaf_lines__as_khatt_only_slant": "… slant only",
    "muharaf_lines__as_khatt_only_stroke": "… stroke width and blur only",
    "muharaf_lines__as_khatt_only_paper": "… ink and paper tone and noise only",
    "muharaf_lines__as_nakba": "Muharaf made to look like NAKBA",
}


def cer_table(C):
    rows = []
    for tag, name in LABEL.items():
        p = C["paired"].get(tag)
        if not p:
            continue
        lo, hi = p["ci95"]
        sig = "" if lo <= 0 <= hi else " class=\"sig\""
        rows.append(f'<tr{sig}><td>{name}</td><td class="n">{p["lines"]}</td><td class="n">{p["cer_original"] * 100:.1f} %</td><td class="n">{p["cer_transformed"] * 100:.1f} %</td>'
                    f'<td class="n"><b>{p["cer_transformed"] * 100 - p["cer_original"] * 100:+.1f}</b> [{lo * 100:+.1f}, {hi * 100:+.1f}]</td><td class="n">{p["share_lines_better"] * 100:.0f} / {p["share_lines_worse"] * 100:.0f} %</td></tr>'.replace("-", "−"))
    base = C["cer"]
    return ('<div class="qwrap"><table class="dt"><thead><tr><th>Lines read by the model</th><th>Lines</th><th>CER original</th><th>CER transformed</th><th>Change in points [95 % interval]</th><th>Lines better / worse</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


CSS = """
.xpairs{display:flex;flex-direction:column;gap:8px}.xrow{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.xstrip{display:flex;flex-direction:column;gap:6px;margin-top:6px}.xln{margin:0;min-width:0}
.xln img{display:block;width:100%;max-width:560px;height:auto;border:1px solid var(--line);border-radius:3px}.xln.new img{border-color:var(--s1)}
.xln figcaption{font-size:12px;color:var(--muted);margin-top:2px}.xrow .xln img{max-width:none}
.xcap{color:var(--muted);font-size:13px;margin:6px 0 0}
table.dt tr.sig td{background:var(--accent-soft)}
@media (max-width:760px){.xrow{grid-template-columns:1fr}}
"""

cer_html = ""
C = None
if a.cer and a.cer.exists():
    C = json.loads(a.cer.read_text())
    cer_html = f"<h4>Does it move the results? (X-T2)</h4>{cer_table(C)}"
verdict = a.verdict.read_text() if a.verdict and a.verdict.exists() else "<b>The CER test has not run yet.</b> It needs a model, and the 3090 was unreachable when this was written."
extra = a.extra.read_text() if a.extra and a.extra.exists() else ""

ARTICLE = f"""
  <!-- 6g. geometry transfer -->
  <article class="dpart" id="d-xfer">
    <h3>Geometry transfer: making one dataset look like another (X-T1, X-T2)</h3>
    <div class="verdict"><p>{verdict}</p></div>
    <p>The measurements above say which differences between datasets are measurable. This section turns them into a transformation and asks what it changes. Each line is resized to 64 px high (the CRNN’s input) and passed through five steps whose strengths are fitted per pair of datasets so that the <i>measured</i> medians of the output match the target: <b>slant</b> (row shear), <b>stroke width</b> (grey-level erosion or dilation, separately for vertical and horizontal strokes, which also sets the pen contrast), <b>blur or sharpening</b>, <b>ink and paper tone</b>, and <b>paper noise</b>. The text is untouched, so a transformed line keeps its label. Code: <span class="mono">scripts/geom_transfer.py</span>; full page with every pair and measure: <span class="mono">results/geometry/transfer/index.html</span>.</p>
    {summary_table()}
    <p>The scalar measures move to the target as intended, from about 0.75 SD to about 0.25 SD. A nearest-centroid classifier on contour histograms moves little. These steps change how a line is scanned and inked, not how its letters are shaped, and the curviness and dot-to-stroke ratio stay where they were. At 64 px the stroke widths of Muharaf and KHATT are already close, so most of the work is tone, slant and noise.</p>
    {pair_block("muharaf:khatt", 4, steps=False)}
    {pair_block("khatt:muharaf", 4, steps=True)}
    {pair_block("nakba:muharaf", 3)}
    <details><summary>Measured medians before and after, KHATT → Muharaf</summary>{measure_table("khatt:muharaf")}</details>
    {cer_html}
    {extra}
  </article>
"""
html = LEDGER.read_text(encoding="utf-8")
html = re.sub(r"\n  <!-- 6g\. geometry transfer -->.*?</article>\n", "\n", html, flags=re.S)
assert "\n  <!-- 6b. attribution spike -->" in html
html = html.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
if ".xpairs{" not in html:
    html = html.replace("</style>\n\n<div class=\"layout\">", CSS + "</style>\n\n<div class=\"layout\">", 1)
if 'href="#d-xfer"' not in html:
    html = html.replace('<li><a href="#d-geom">Geometry measurements</a></li>', '<li><a href="#d-geom">Geometry measurements</a></li>\n        <li><a href="#d-xfer">Geometry transfer</a></li>', 1)
    html = html.replace('<a href="#d-geom">Geometry measurements</a><a href="#d-xai">', '<a href="#d-geom">Geometry measurements</a><a href="#d-xfer">Geometry transfer</a><a href="#d-xai">', 1)

done_cer = C is not None
x1 = ' ["X-T1","Geometry transfer: make lines of one dataset look like another from the measured geometry","CPU","Done 2026-10-11: mean gap to target over twelve measures 0.7 to 0.8 SD before, 0.23 to 0.28 SD after, five pairs; contour classifier barely moves"],\n'
x2 = (' ["X-T2","CRNN CER on original and transformed lines, with single-step ablations (scripts/geom_transfer_eval.py, runbook in docs/research/xai-tasks-and-runbook.md)","GPU (3090)","'
      + ("Done: see Geometry transfer" if done_cer else "Not run. A half-width replica could not be trained on the 8 GB laptop (swap thrashing, no finished epoch in 19 minutes); the 3090 was unreachable") + '"],\n')
if '["X-T1"' not in html:
    m = re.search(r'^ \["X-G12",.*?\],\n', html, flags=re.M)
    assert m
    html = html.replace(m.group(0), m.group(0) + x1 + x2, 1)
QROW = ' ["Geometry transfer CER test", "Does making KHATT look like Muharaf lower the CRNN\'s CER, and which step carries it?", "glucksman1967loci, vinciarelli2001norm", "CER on original and transformed lines with paired intervals; single-step ablations (slant, stroke and blur, tone and noise).", "gpu", "X-T2"],\n'
if "Geometry transfer CER test" not in html and not done_cer:
    html = html.replace(' ["Per-module gradient probe by letter family"', QROW + ' ["Per-module gradient probe by letter family"', 1)
POOL14 = ('        <tr><td>14</td><td><b>Geometry transfer as augmentation and as a diagnostic.</b> Fit the five-step transformation from measured medians and apply it to training lines (to look like the target) or to target lines (to look like the training data).</td>'
          '<td>Domain shift without target labels; separates what slant, tone and stroke weight explain from what letter shape explains.</td><td>X-T1 done on CPU; X-T2 on the 3090 (CER on original and transformed lines)</td><td>Leaves letter shapes alone by construction, so it can only explain the part of the gap that is scan and pen.</td></tr>\n')
if "Geometry transfer as augmentation" not in html:
    html = html.replace("      </tbody>\n    </table></div>\n    <h4>E8: CTC-encoder-to-LLM spike (proposed)</h4>", POOL14 + "      </tbody>\n    </table></div>\n    <h4>E8: CTC-encoder-to-LLM spike (proposed)</h4>", 1)
LEDGER.write_text(html, encoding="utf-8")
print("ledger updated", len(html))
