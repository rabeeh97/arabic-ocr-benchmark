"""Insert the "Laptop experiments" article (tiny CRNN, transfer test, augmentation, text distance, PAW coverage) into the ledger.

    .geomvenv/bin/python scripts/build_tiny_section.py results/geometry --verdict FILE [--notes FILE]

Reads tiny/analysis.json (geom_tiny_analyze.py), tiny/base.log and tiny/geom.log, text_shift.json and paw_coverage.json.
Idempotent.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--verdict", type=Path, required=True)
ap.add_argument("--notes", type=Path)
a = ap.parse_args()
A = json.loads((a.root / "tiny/analysis.json").read_text())
TS = json.loads((a.root / "text_shift.json").read_text())
PW = json.loads((a.root / "paw_coverage.json").read_text())
LEDGER = Path("docs/research/paper-ledger.html")
NAME = {"kitab_khatt": "KHATT", "nakba_lines": "NAKBA test", "muharaf_lines": "Muharaf test"}
LABEL = [
    ("kitab_khatt__as_muharaf", "KHATT made to look like Muharaf, all steps"),
    ("kitab_khatt__as_muharaf_only_slant", "… slant only"),
    ("kitab_khatt__as_muharaf_only_stroke", "… stroke width and blur only"),
    ("kitab_khatt__as_muharaf_only_paper", "… ink and paper tone and noise only"),
    ("muharaf_lines__as_khatt", "Muharaf made to look like KHATT, all steps"),
    ("muharaf_lines__as_khatt_only_slant", "… slant only"),
    ("muharaf_lines__as_khatt_only_stroke", "… stroke width and blur only"),
    ("muharaf_lines__as_khatt_only_paper", "… ink and paper tone and noise only"),
    ("nakba_lines__as_muharaf", "NAKBA made to look like Muharaf, all steps"),
    ("nakba_lines__as_muharaf_only_slant", "… slant only"),
    ("nakba_lines__as_muharaf_only_stroke", "… stroke width and blur only"),
    ("nakba_lines__as_muharaf_only_paper", "… ink and paper tone and noise only"),
    ("nakba_lines__as_khatt", "NAKBA made to look like KHATT"),
    ("nakba_lines__norules", "NAKBA with ruled lines painted out"),
    ("muharaf_lines__as_nakba", "Muharaf made to look like NAKBA"),
]


def pct(x, d=1):
    return f"{100 * x:.{d}f}".replace("-", "−")


def sgn(x):
    return f"{100 * x:+.1f}".replace("-", "−")


def paired_table(run):
    P = A["runs"][run]["paired"]
    rows = []
    for tag, name in LABEL:
        p = P.get(tag)
        if not p:
            continue
        lo, hi = p["ci95"]
        cls = ' class="sig"' if (lo > 0 or hi < 0) else ""
        rows.append(f'<tr{cls}><td>{name}</td><td class="n">{p["lines"]}</td><td class="n">{pct(p["cer_original"])} %</td><td class="n">{pct(p["cer_transformed"])} %</td>'
                    f'<td class="n"><b>{sgn(p["change"])}</b> [{sgn(lo)}, {sgn(hi)}]</td><td class="n">{p["share_lines_better"] * 100:.0f} / {p["share_lines_worse"] * 100:.0f} %</td></tr>')
    return ('<div class="qwrap"><table class="dt"><thead><tr><th>Lines read by the tiny CRNN</th><th>Lines</th><th>CER original</th><th>CER transformed</th><th>Change, points [95 %]</th><th>Lines better / worse</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div><p class=\"xcap\">Shaded rows: the 95 % interval excludes zero. Intervals are bootstrapped over lines; the same lines are read in both versions.</p>")


def aug_table():
    E = A.get("aug_effect")
    if not E:
        return "<p>The augmented model has not finished.</p>"
    rows = []
    for t in ("muharaf_lines", "nakba_lines", "kitab_khatt"):
        e = E[t]
        lo, hi = e["ci95"]
        cls = ' class="sig"' if (lo > 0 or hi < 0) else ""
        rows.append(f'<tr{cls}><td>{NAME[t]}</td><td class="n">{e["lines"]}</td><td class="n">{pct(e["cer_base"])} %</td><td class="n">{pct(e["cer_geom"])} %</td><td class="n"><b>{sgn(e["change"])}</b> [{sgn(lo)}, {sgn(hi)}]</td></tr>')
    return ('<div class="qwrap"><table class="dt"><thead><tr><th>Untouched lines</th><th>Lines</th><th>Baseline augmentation</th><th>+ geometry augmentation</th><th>Change, points [95 %]</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def corr_table():
    C = A["line_corr"]
    keys = [("slant_skel_deg", "Slant (+ leans right)"), ("sw_abs", "Stroke width"), ("dot_diam_over_sw", "Dot ÷ stroke"), ("ink_level", "Ink grey level"),
            ("paper_level", "Paper grey level"), ("paper_noise", "Paper noise"), ("sharpness", "Edge sharpness"), ("turn_mean_rad", "Curviness"), ("ink_density", "Ink density")]
    cols = ["muharaf_lines", "nakba_lines", "kitab_khatt", "pooled"]
    head = "<tr><th>Measure of the line</th>" + "".join(f"<th>{NAME.get(c, 'All three')}</th>" for c in cols) + "</tr>"
    rows = []
    for k, nm in keys:
        cells = []
        for c in cols:
            r, n = C[c][k]
            strong = abs(r) >= 0.25
            cells.append(f'<td class="n">{"<b>" if strong else ""}{f"{r:+.2f}".replace("-", "−")}{"</b>" if strong else ""}</td>')
        rows.append(f"<tr><td>{nm}</td>{''.join(cells)}</tr>")
    return f'<div class="qwrap"><table class="dt"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def text_table():
    S = TS["sets"]
    cols = [("muharaf", "Muharaf test"), ("nakba", "NAKBA test"), ("nakba_blind", "NAKBA blind"), ("khatt", "KHATT")]
    rows = [
        ("Characters outside the training charset", lambda t: f'{pct(S[t]["char_oov"])} %'),
        ("Word tokens never seen in training", lambda t: f'{pct(S[t]["word_oov"])} %'),
        ("Character 5-gram cross-entropy, bits per character", lambda t: f'{S[t]["xent_bits"]:.2f}'),
        ("PAW tokens never seen in training", lambda t: f'{pct(PW[t]["dotted"]["unseen_token_share"])} %'),
        ("… same, dotless (rasm) PAWs", lambda t: f'{pct(PW[t]["rasm"]["unseen_token_share"])} %'),
        ("Letters whose positional shape has fewer than 50 training examples", lambda t: f'{pct(S[t]["rare_shape_50"], 2)} %'),
        ("PAWs per word", lambda t: f'{PW[t]["dotted"]["paws_per_word"]:.2f}'),
        ("Type-token ratio, first 2,000 words", lambda t: f'{S[t]["ttr_2000"]:.2f}'),
    ]
    head = "<tr><th>Against the CRNN's training text</th>" + "".join(f"<th>{n}</th>" for _, n in cols) + "</tr>"
    body = "".join(f"<tr><td>{nm}</td>" + "".join(f'<td class="n">{fn(t)}</td>' for t, _ in cols) + "</tr>" for nm, fn in rows)
    return f'<div class="qwrap"><table class="dt"><thead>{head}</thead><tbody>{body}</tbody></table></div>'


def log_last(name):
    p = a.root / "tiny" / f"{name}.log"
    if not p.exists():
        return None
    ls = [l for l in p.read_text().splitlines() if "heldout CER" in l]
    return json.loads(ls[-1].split("heldout CER ")[1]) if ls else None


base_h, geom_h = log_last("base"), log_last("geom")
held = f"Held-out CER after 12 epochs: Muharaf {pct(base_h['muharaf_lines'])} %, RASAM-1 {pct(base_h['rasam1'])} % (baseline)" + (f"; {pct(geom_h['muharaf_lines'])} % and {pct(geom_h['rasam1'])} % with geometry augmentation." if geom_h else ".")
notes = a.notes.read_text() if a.notes and a.notes.exists() else ""
PW_train = PW["train"]

ARTICLE = f"""
  <!-- 6h. laptop experiments -->
  <article class="dpart" id="d-tiny">
    <h3>Laptop experiments: a tiny CRNN, the transfer test, augmentation, and the text side (2026-10-11)</h3>
    <div class="verdict"><p>{a.verdict.read_text()}</p></div>
    <h4>The model</h4>
    <p>Same architecture as the 3090 aligner at a quarter of the channels (0.61 M parameters against 9.5 M), 64 px input, trained on the laptop CPU for 12 epochs on 6,000 Muharaf train lines plus the cached RASAM lines (about 2 hours per model; images in a memory-mapped file so the process stays under 0.7 GB). {held} The 3090 model reaches 7.6 %, so this is a much weaker reader. Its pattern across datasets is the same, though: Muharaf test {pct(A["runs"]["base"]["cer"]["muharaf_lines"])} %, NAKBA {pct(A["runs"]["base"]["cer"]["nakba_lines"])} %, KHATT {pct(A["runs"]["base"]["cer"]["kitab_khatt"])} %. KHATT is the outlier for both models. Code: <span class="mono">scripts/geom_tiny.py</span>, analysis <span class="mono">scripts/geom_tiny_analyze.py</span>.</p>
    <h4>Does the geometry transfer move the results? (X-T2, on the tiny model)</h4>
    {paired_table("base")}
    <h4>Does training with geometry augmentation help? (pool idea 7 and 14)</h4>
    <p>Same model, same data and schedule, plus the geometry-transfer steps as augmentation with random strengths inside the range measured across the three datasets (shear −12° to +20°, mild stroke thinning and thickening, blur or sharpening, tone and noise): catalogue items 1, 3 (mild only) and 4, all marked safe or caution-with-limits there.</p>
    {aug_table()}
    <h4>Which line properties go with errors? (X-G2 “correlation with CER”)</h4>
    <p>Spearman correlation between each untouched line’s measured geometry and its CER under the baseline tiny model (negative means higher values go with fewer errors). Bold where |r| ≥ 0.25.</p>
    {corr_table()}
    <h4>The text side: how far is each set from the training text? (R1.6, R1.7, PAW coverage)</h4>
    {text_table()}
    <p>The training text holds {PW_train["dotted"]["distinct"]:,} distinct PAWs; the {PW_train["dotted"]["types_for_95pct"]:,} most frequent cover 95 % of PAW tokens ({PW_train["rasm"]["types_for_95pct"]:,} without dots, {PW_train["rasm"]["distinct"]:,} distinct). Code: <span class="mono">scripts/geom_text_shift.py</span>, <span class="mono">scripts/geom_paw_coverage.py</span>.</p>
    {notes}
  </article>
"""
html = LEDGER.read_text(encoding="utf-8")
html = re.sub(r"\n  <!-- 6h\. laptop experiments -->.*?</article>\n", "\n", html, flags=re.S)
html = html.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
if 'href="#d-tiny"' not in html:
    html = html.replace('<li><a href="#d-xfer">Geometry transfer</a></li>', '<li><a href="#d-xfer">Geometry transfer</a></li>\n        <li><a href="#d-tiny">Laptop experiments</a></li>', 1)
    html = html.replace('<a href="#d-xfer">Geometry transfer</a><a href="#d-xai">', '<a href="#d-xfer">Geometry transfer</a><a href="#d-tiny">Laptop experiments</a><a href="#d-xai">', 1)
LEDGER.write_text(html, encoding="utf-8")
print("ledger updated", len(html))
