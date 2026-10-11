"""Build a self-contained HTML page of the dataset-to-dataset transformations.

    .geomvenv/bin/python scripts/build_transfer_html.py results/geometry/transfer
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from geom_transfer import apply, apply_steps  # noqa: E402

run = Path(sys.argv[1])
R = json.loads((run / "transfer.json").read_text())
SAVED = pickle.load(open(run / "transfer_params.pkl", "rb"))
NICE = {"muharaf": "Muharaf", "khatt": "KHATT", "nakba": "NAKBA"}
FNAME = {"slant_skel_deg": "Slant (°, + leans right)", "sw_abs": "Stroke width (px at 64 px height)", "sw_v_abs": "Width of vertical strokes (px)", "sw_h_abs": "Width of horizontal strokes (px)",
         "pen_contrast": "Pen contrast (vertical ÷ horizontal width)", "dot_diam_over_sw": "Dot diameter ÷ stroke width", "ink_level": "Ink grey level (0 black)",
         "paper_level": "Paper grey level (255 white)", "paper_noise": "Paper noise (grey levels)", "sharpness": "Edge sharpness", "turn_mean_rad": "Contour curviness (rad)",
         "ink_density": "Ink density"}
CHANGEABLE = {"slant_skel_deg", "sw_abs", "sw_v_abs", "sw_h_abs", "pen_contrast", "ink_level", "paper_level", "paper_noise", "sharpness"}
SHOW_W = 640  # displayed crop width in px (right-most part of the line, since the script runs right to left)


def b64(img: np.ndarray, crop: bool = True) -> str:
    if crop and img.shape[1] > SHOW_W:
        img = img[:, -SHOW_W:]
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def im(img, cap="", cls=""):
    return f'<figure class="ln {cls}"><img alt="{cap}" src="{b64(img)}"><figcaption>{cap}</figcaption></figure>'


def f(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def range_svg(pair) -> str:
    keys = ["slant_skel_deg", "sw_abs", "pen_contrast", "sharpness", "paper_noise"]
    rows = []
    W, rh = 720, 22
    out = [f'<svg viewBox="0 0 {W} {len(keys) * (3 * rh + 14) + 6}" role="img" aria-label="Spread per feature before and after">']
    colors = {"source": "var(--s2)", "transformed": "var(--s1)", "target": "var(--s3)"}
    y = 4
    pr = R["pairs"][pair]["p10_p90"]
    med = {"source": R["pairs"][pair]["median_before"], "transformed": R["pairs"][pair]["median_after"], "target": R["pairs"][pair]["median_target"]}
    for k in keys:
        lo = min(pr[n][k][0] for n in pr)
        hi = max(pr[n][k][1] for n in pr)
        pad = (hi - lo) * 0.08 or 1
        lo, hi = lo - pad, hi + pad
        X = lambda v: 210 + (v - lo) / (hi - lo) * (W - 210 - 70)  # noqa: E731
        out.append(f'<text x="0" y="{y + 14}" class="lbl">{FNAME[k]}</text>')
        for i, n in enumerate(("source", "transformed", "target")):
            yy = y + i * rh + rh / 2
            a, b = pr[n][k]
            out.append(f'<line x1="{X(a):.1f}" x2="{X(b):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{colors[n]}" stroke-width="3" stroke-linecap="round"/>')
            out.append(f'<circle cx="{X(med[n][k]):.1f}" cy="{yy:.1f}" r="4.5" fill="{colors[n]}" stroke="var(--surface)" stroke-width="1.5"/>')
            out.append(f'<text x="{X(b) + 8:.1f}" y="{yy + 4:.1f}">{f(med[n][k], 1 if abs(med[n][k]) > 5 else 2)}</text>')
        y += 3 * rh + 14
    out.append("</svg>")
    return "".join(out)


def table(pair) -> str:
    r = R["pairs"][pair]
    rows = []
    for k, name in FNAME.items():
        zb, za = r["z_before"][k], r["z_after"][k]
        tag = "" if k in CHANGEABLE else '<span class="pill">not changed by these steps</span>'
        rows.append(f'<tr><td>{name} {tag}</td><td class="n">{f(r["median_before"][k], 1 if abs(r["median_before"][k]) > 5 else 2)}</td><td class="n">{f(r["median_after"][k], 1 if abs(r["median_after"][k]) > 5 else 2)}</td>'
                    f'<td class="n">{f(r["median_target"][k], 1 if abs(r["median_target"][k]) > 5 else 2)}</td><td class="n">{f(zb)}</td><td class="n"><b>{f(za)}</b></td></tr>')
    return ('<div class="qwrap"><table class="dt"><thead><tr><th>Measure (median of lines)</th><th>Source</th><th>Transformed</th><th>Target</th><th>Gap before (SD)</th><th>Gap after (SD)</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


sections = []
summary_rows = []
for pair, v in R["pairs"].items():
    s, t = pair.split(":")
    S = SAVED[pair]
    P = S["P"]
    srcs = S["src"][:6]
    tg = [x[1] for x in S["tgt"]][:3]
    grid = []
    for i, g in enumerate(srcs):
        grid.append(f'<div class="row">{im(g, f"{NICE[s]} original")}{im(apply(g, P), f"{NICE[s]} made to look like {NICE[t]}", "new")}</div>')
    steps = apply_steps(srcs[0], P)
    step_names = ["+ slant", "+ stroke width", "+ blur / sharpen", "+ ink and paper tone", "+ paper noise"]
    stepfig = "".join(im(x, n) for x, n in zip(steps, step_names))
    real = "".join(im(g, f"Real {NICE[t]}") for g in tg)
    sections.append(f"""
<section id="p-{s}-{t}">
  <h2>{NICE[s]} → {NICE[t]}</h2>
  <p class="lead">Mean gap to the {NICE[t]} medians over twelve measures falls from <b>{f(v["mean_abs_z_before"])}</b> to <b>{f(v["mean_abs_z_after"])}</b> SD. A nearest-centroid classifier on contour histograms calls {v["classifier_share_target_before"] * 100:.0f} % of the lines {NICE[t]} before and {v["classifier_share_target_after"] * 100:.0f} % after.
  Knobs fitted on 100 {NICE[s]} lines: shear {f(P["shear"], 1)}°, vertical strokes {f(P["tx"], 1)} px, horizontal strokes {f(P["ty"], 1)} px, blur σ {f(P["blur"], 1)}, unsharp {f(P["unsharp"], 1)}, ink grey {P["ink_level"]:.0f}, paper grey {P["paper_level"]:.0f}, paper noise {P["noise"]:.1f}.</p>
  <h3>Side by side (held-out lines, labels unchanged)</h3>
  <div class="pairs">{"".join(grid)}</div>
  <h3>Real {NICE[t]} lines for comparison</h3>
  <div class="strip">{real}</div>
  <h3>One line, step by step</h3>
  <div class="strip">{im(srcs[0], "original")}{stepfig}</div>
  <h3>Spread of each measure: source, transformed, target</h3>
  <div class="legend"><span><i style="background:var(--s2)"></i>source {NICE[s]}</span><span><i style="background:var(--s1)"></i>transformed</span><span><i style="background:var(--s3)"></i>real {NICE[t]}</span></div>
  <div class="gbox">{range_svg(pair)}</div>
  <p class="cap">Line = 10th to 90th percentile of lines, dot = median.</p>
  <details open><summary>All twelve measures</summary>{table(pair)}</details>
</section>""")
    summary_rows.append(f'<tr><td><a href="#p-{s}-{t}">{NICE[s]} → {NICE[t]}</a></td><td class="n">{f(v["mean_abs_z_before"])}</td><td class="n"><b>{f(v["mean_abs_z_after"])}</b></td>'
                        f'<td class="n">{v["classifier_share_target_before"] * 100:.0f} % → {v["classifier_share_target_after"] * 100:.0f} %</td></tr>')

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset geometry transfer</title>
<style>
/* Layout: one section per source-to-target pair; paired line images first, then step-by-step, then the measured spread. */
:root{{--bg:#f4f6f8;--surface:#fff;--ink:#18202b;--muted:#5a6676;--line:#dde2e8;--accent:#1f5f8b;--accent-soft:#e3eef6;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--grid:#e6e9ee;--axis:#8a95a3;--f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#11161d;--surface:#18202a;--ink:#e4e9ef;--muted:#9aa6b4;--line:#2a3542;--accent:#7db7e0;--accent-soft:#1c3346;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--grid:#26303b;--axis:#7d8896;color-scheme:dark}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--f-body)}}
.wrap{{max-width:1080px;margin:0 auto;padding-inline:20px;padding-block:32px 64px}}
h1{{font-size:28px;margin:0 0 6px}}h2{{font-size:21px;margin:0 0 6px}}h3{{font-size:14px;margin:18px 0 8px}}
p{{max-width:78ch}}.lead{{color:var(--ink)}}.cap{{color:var(--muted);font-size:13px;margin:4px 0 0}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px 20px;margin-top:22px}}
a{{color:var(--accent)}}
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);padding:8px 12px;border-radius:0 6px 6px 0;max-width:none}}
.pairs{{display:flex;flex-direction:column;gap:10px}}.row{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.strip{{display:flex;flex-direction:column;gap:8px}}
.ln{{margin:0;min-width:0}}.ln img{{display:block;width:100%;max-width:{SHOW_W * 1.0}px;height:auto;border:1px solid var(--line);border-radius:3px;image-rendering:auto}}
.ln.new img{{border-color:var(--s1)}}.ln figcaption{{font-size:12px;color:var(--muted);margin-top:2px}}
.row .ln img{{width:100%;max-width:none}}
@media (max-width:760px){{.row{{grid-template-columns:1fr}}}}
.gbox{{overflow-x:auto}}.gbox svg{{display:block;width:100%;min-width:560px;height:auto}}.gbox text{{fill:var(--muted);font-size:11.5px}}.gbox .lbl{{fill:var(--ink);font-size:12px;font-weight:600}}
.legend{{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12.5px;color:var(--muted)}}.legend i{{display:inline-block;width:14px;height:4px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.qwrap{{overflow-x:auto}}table.dt{{width:100%;border-collapse:collapse;font-size:13px}}table.dt th,table.dt td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
table.dt th{{font:500 11px var(--f-mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}table.dt td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.pill{{font-size:11px;padding:1px 7px;border-radius:99px;border:1px solid var(--line);color:var(--muted);white-space:nowrap}}
summary{{cursor:pointer;color:var(--accent);font-size:13px;margin-top:6px}}
</style></head><body><div class="wrap">
<h1>Dataset geometry transfer</h1>
<p class="cap">Generated by <code>scripts/geom_transfer.py</code> and <code>scripts/build_transfer_html.py</code>. Local page, not yet in the ledger.</p>
<p class="note">Each line of one dataset is resized to 64 px high (the CRNN's input) and passed through five steps whose strengths were fitted so that the measured medians of the output match the target dataset: <b>slant</b> (row shear), <b>stroke width</b> (grey-level erosion or dilation, separately for vertical and horizontal strokes), <b>blur or sharpening</b>, <b>ink and paper tone</b>, and <b>paper noise</b>. The text is untouched, so every transformed line keeps its label. Images show the right-most 640 px of each line, since the script runs right to left.</p>
<section><h2>Result in one table</h2>
<div class="qwrap"><table class="dt"><thead><tr><th>Pair</th><th>Mean gap to target before (SD)</th><th>after (SD)</th><th>Called target by a contour classifier</th></tr></thead><tbody>{"".join(summary_rows)}</tbody></table></div>
<p>The measured scalars move to the target as intended. The contour classifier moves little, because these steps change how a line is scanned and inked, not the shape of its letters. Curviness and dot-to-stroke ratio stay where they were (see each table). What the transformation therefore tests is how much of a transfer gap is carried by slant, tone and stroke weight, which a model run on the transformed lines can answer.</p></section>
{"".join(sections)}
</div></body></html>"""
(run / "index.html").write_text(HTML, encoding="utf-8")
print("wrote", run / "index.html", f"{len(HTML) / 1e6:.1f} MB")
