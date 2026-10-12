"""Insert the 2026-10-12 results (ink crop, letter novelty, augmented CRNN, E8, galleries) into the ledger and update the
key-findings table and experiment rows.

    .geomvenv/bin/python scripts/build_day2_section.py results/geometry
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

R = Path(sys.argv[1])
G = R / "galleries"
LEDGER = Path("docs/research/paper-ledger.html")
J = lambda p: json.loads((R / p).read_text(encoding="utf-8"))  # noqa: E731
base_t, aug_t = J("transfer_cer.json"), J("aug_transfer.json")
base_e, aug_e = J("base_evaldata.json"), J("aug_evaldata.json")
ic_b, ic_a = J("inkcrop_base.json"), J("inkcrop_aug.json")
e8 = J("e8/results.json")
L1, L2 = J("letters/summary.json"), J("letters_indep/summary.json")
ext = J("ink_extent.json")
frag = lambda n: (G / f"{n}_frag.html").read_text(encoding="utf-8")  # noqa: E731
pc = lambda x: f"{100 * x:.1f}".replace("-", "−")  # noqa: E731
sg = lambda x: f"{100 * x:+.1f}".replace("-", "−")  # noqa: E731
NAME = {"kitab_khatt": "KHATT", "nakba_lines": "NAKBA test", "nakba_blind": "NAKBA blind", "muharaf_lines": "Muharaf test", "rasam1": "RASAM-1"}


def ci(p):
    return f'<b>{sg(p["change"])}</b> [{sg(p["ci95"][0])}, {sg(p["ci95"][1])}]'


def table(head, rows):
    return '<div class="qwrap"><table class="dt"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(rows_i) + "</tr>" for rows_i in rows) + "</tbody></table></div>"


# 1 ink crop
k = ext["kitab_khatt"]["bands"]
ink_rows = []
for tag in ("kitab_khatt", "nakba_lines", "nakba_blind"):
    pb, pa = ic_b["paired"][f"{tag}__inkcrop"], ic_a["paired"][f"{tag}__inkcrop"]
    ink_rows.append([f"<td>{NAME[tag]}</td>", f'<td class="n">{pb["lines"]}</td>', f'<td class="n">{pc(pb["cer_original"])} → {pc(pb["cer_transformed"])} %</td>', f'<td class="n">{ci(pb)}</td>',
                     f'<td class="n">{pc(pa["cer_original"])} → {pc(pa["cer_transformed"])} %</td>', f'<td class="n">{ci(pa)}</td>'])
ink = (f"""<h4>1 · Cropping to the ink before resizing (new, X-S1)</h4>
<p>The E8 gallery showed KHATT lines whose writing fills only a thin strip of a tall image: resized to 64 px, the letters become tiny. Measured on the stored images, {pc(k[0]['share_of_lines'])} % of KHATT lines have writing in less than 35 % of the height, and those lines average {pc(k[0]['mean_line_cer'])} % CER under the CRNN, against about 23 % for the rest (NAKBA: {pc(ext['nakba_lines']['bands'][0]['share_of_lines'])} %). The fix is to keep the rows between the 1st and 99th percentile of ink, plus a quarter of that band above and below for dots and ascenders, before resizing (<span class="mono">scripts/geom_inkcrop_export.py</span>).</p>"""
       + table(["Set", "Lines", "Original CRNN", "Change [95 %]", "Augmented CRNN", "Change [95 %]"], ink_rows)
       + "<p>No retraining, and it only touches lines whose crop is loose, so NAKBA is unchanged. It is the largest single gain we have found for KHATT, three times the slant effect.</p>" + frag("inkcrop"))

# 2 letter novelty
nov_rows = []
for tag in ("muharaf_lines", "nakba_lines", "nakba_blind", "kitab_khatt"):
    a_, b_ = L1["datasets"][tag], L2["datasets"][tag]
    nov_rows.append([f"<td>{NAME[tag]}</td>", f'<td class="n">{a_["median_novelty"]:.2f}</td>', f'<td class="n">{pc(a_["share_above_p95_of_muharaf"])} %</td>',
                     f'<td class="n">{pc(a_["error_rate_by_band"][0])} → {pc(a_["error_rate_by_band"][-1])} %</td>',
                     f'<td class="n">{b_["median_novelty"]:.2f}</td>', f'<td class="n">{pc(b_["share_above_p95_of_muharaf"])} %</td>',
                     f'<td class="n">{pc(b_["error_rate_by_band"][0])} → {pc(b_["error_rate_by_band"][-1])} %</td>'])
nov = ("""<h4>2 · Letter-shape novelty predicts letter errors (X-G10)</h4>
<p>Independent check passed: novelty measured with the features of a second, separately trained CRNN (the augmented one) predicts the original CRNN’s letter errors just as well as its own features do, so the result is not only the model judging its own uncertainty.</p>"""
       + table(["Set", "Median novelty (own features)", "Beyond Muharaf’s 95th pct", "Error: most typical → most novel", "Median novelty (second model)", "Beyond 95th pct", "Error: typical → novel"], nov_rows)
       + frag("letters"))

# 3 augmentation
aug_rows = []
for tag in ("kitab_khatt", "nakba_lines", "nakba_blind"):
    aug_rows.append([f"<td>{NAME[tag]}</td>", f'<td class="n">{pc(base_e["cer"][tag])} %</td>', f'<td class="n">{pc(aug_e["cer"][tag])} %</td>'])
sens = []
for tag, nm in (("muharaf_lines__as_khatt_only_slant", "Muharaf sheared upright (slant only)"), ("kitab_khatt__as_muharaf_only_slant", "KHATT sheared toward Muharaf (slant only)"),
                ("nakba_lines__as_muharaf_only_stroke", "NAKBA strokes thinned (stroke only)")):
    sens.append([f"<td>{nm}</td>", f'<td class="n">{ci(base_t["paired"][tag])}</td>', f'<td class="n">{ci(aug_t["paired"][tag])}</td>'])
aug = ("""<h4>3 · Training with slant and stroke-width augmentation (X-T3)</h4>
<p>Same CRNN recipe (24 k lines, 30 epochs) plus, on the GPU, a random shear of −12° to +20° (p = 0.8) and fractional thickening or thinning of vertical and horizontal strokes (p = 0.5 each), checked by eye first (<span class="mono">geom_aug</span> in <span class="mono">xai_train_aligner.py</span>). Held-out Muharaf CER is unchanged (7.62 % against 7.6 %).</p>"""
       + table(["Set (all lines)", "Original CRNN", "Augmented CRNN"], aug_rows)
       + table(["Sensitivity test (CER change, points)", "Original CRNN", "Augmented CRNN"], sens)
       + "<p>The augmentation does what it should for robustness (an upright Muharaf costs +0.6 points instead of +2.0), but it does not improve the unseen sets; KHATT and NAKBA get slightly worse. Slant robustness alone is not where the transfer gap is.</p>" + frag("aug"))

# 4 E8
e8_rows = []
for tag in ("muharaf_lines", "nakba_lines", "nakba_blind", "kitab_khatt", "rasam1"):
    v = e8[tag]
    e8_rows.append([f"<td>{NAME[tag]}</td>", f'<td class="n">{v["lines"]}</td>', f'<td class="n">{pc(v["cer_crnn"])} %</td>', f'<td class="n">{pc(v["cer_e8"])} %</td>',
                    f'<td class="n">{pc(v["fluent_wrong_share_crnn"])} %</td>', f'<td class="n">{pc(v["fluent_wrong_share_e8"])} %</td>'])
e8h = ("""<h4>4 · E8: the CRNN encoder read by a small LLM</h4>
<p>Frozen CRNN, its BiLSTM output pooled by two along the line, a two-layer projector into Qwen3-0.6B; one epoch training the projector alone, then two epochs with LoRA (r = 16) on all of the LLM’s linear layers (<span class="mono">scripts/xai_e8_spike.py</span>, about 45 minutes on the 3090). “Fluent wrong” is the share of wrong output words that are nonetheless real words of the training text.</p>"""
       + table(["Set", "Lines", "CRNN (CTC)", "E8 (LLM decoder)", "Fluent-wrong, CRNN", "Fluent-wrong, E8"], e8_rows)
       + "<p><b>No-go at this scale.</b> The LLM decoder is worse on every set, by 4 to 14 points, and a larger share of its mistakes are plausible words: the language prior overrides what the encoder saw. On KHATT lines where the CRNN reads nothing (tiny writing), E8 writes a fluent sentence that is not there. A larger LLM, more training or feeding CTC posteriors instead of features might change this; as it stands, a CTC decoder on these features is better.</p>" + frag("e8"))

# 5 X-T2 galleries
xt2 = "<h4>5 · The slant and stroke transfer, line by line (X-T2)</h4><p>The lines behind yesterday’s transfer numbers: which lines the slant and stroke changes helped and hurt.</p>" + frag("xt2")

verdict = (f"<b>The biggest lever for KHATT is image size, not slant: cropping each line to its ink band before resizing lowers the CRNN’s KHATT CER from {pc(ic_b['paired']['kitab_khatt__inkcrop']['cer_original'])} % to {pc(ic_b['paired']['kitab_khatt__inkcrop']['cer_transformed'])} % with no retraining.</b> "
           "Letter-shape novelty is the strongest predictor of errors and survives an independent check; KHATT has twice NAKBA’s share of unusual letters. "
           "Training with slant and stroke augmentation makes the model slant-robust but does not help the unseen sets. The E8 encoder-to-LLM spike is a no-go at 0.6 B: worse CER and more fluent but wrong words.")

CSS = """
:root{--good:#2c7a4b;--bad:#b23a26;--bad-soft:#f8d9d2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--good:#7cc79a;--bad:#f0907a;--bad-soft:#4a2620}}
:root[data-theme="dark"]{--good:#7cc79a;--bad:#f0907a;--bad-soft:#4a2620}
.gl{display:flex;flex-direction:column;gap:10px}
.gcard{border:1px solid var(--line);border-left:4px solid var(--line);border-radius:6px;padding:8px 10px;display:flex;flex-direction:column;gap:6px;background:var(--surface)}
.gcard.better{border-left-color:var(--good)}.gcard.worse{border-left-color:var(--bad)}
.gcard img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:3px;background:#fff}
.gcard .pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.gcard figure{margin:0;min-width:0}.gcard figcaption{font-size:11.5px;color:var(--muted)}
.gtxt{display:flex;flex-direction:column;gap:2px;font-size:14px}.gtxt .r{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.gtxt .k{font:500 11px var(--f-mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);min-width:110px}
.gtxt .e{font-size:11.5px;color:var(--muted);min-width:48px}.gtxt bdi{font-family:var(--f-ar);font-size:17px;line-height:1.7}.gtxt .id{font-size:11.5px;color:var(--muted)}
mark.bad{background:var(--bad-soft);color:inherit;border-radius:2px}mark.miss{background:none;color:var(--bad);font-size:11px}
.gsum .better{color:var(--good)}.gsum .worse{color:var(--bad)}
.dpart section{border-top:1px solid var(--line);padding-top:10px;display:flex;flex-direction:column;gap:8px}
.arb{font-family:var(--f-ar);font-size:20px}
.gbox2 svg{width:100%;max-width:760px;height:auto}.gbox2 .t{fill:var(--muted);font-size:11.5px}.gbox2 .l{font-size:12.5px;font-weight:600}
.algrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}.al{border:1px solid var(--line);border-radius:6px;padding:8px}
.alh{font-size:13px;color:var(--muted)}.alr{display:flex;gap:6px;flex-wrap:wrap}.alr figure{margin:0;text-align:center}.alr img{width:48px;height:48px;border:1px solid var(--line);border-radius:3px;background:#fff}
.alr figcaption{font-size:11px;color:var(--muted)}
.nvgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}.nv{margin:0;border:1px solid var(--line);border-left:4px solid var(--line);border-radius:5px;padding:6px}
.nv.bad{border-left-color:var(--bad)}.nv.ok{border-left-color:var(--good)}.nv .two{display:flex;gap:4px}.nv img{width:72px;height:72px;border:1px solid var(--line);border-radius:3px;background:#fff}
.nv figcaption{font-size:11.5px;color:var(--muted)}
details.gal>summary{cursor:pointer;color:var(--accent);font-size:13.5px}
@media (max-width:760px){.gcard .pair{grid-template-columns:1fr}}
"""


def fold(title, body):
    return f'<details class="gal"><summary>{title}</summary>{body}</details>'


ARTICLE = f"""
  <!-- 6i. results 2026-10-12 -->
  <article class="dpart" id="d-day2">
    <h3>Ink cropping, letter novelty, augmentation and E8 (2026-10-12)</h3>
    <div class="verdict"><p>{verdict}</p></div>
    {ink}
    {nov}
    {aug}
    {e8h}
    {fold("5 · X-T2 line by line: open the slant and stroke galleries", xt2)}
  </article>
"""
html = LEDGER.read_text(encoding="utf-8")
html = re.sub(r"\n  <!-- 6i\. results 2026-10-12 -->.*?</article>\n(?=\n  <!-- 6b)", "\n", html, flags=re.S)
html = html.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
if ".gcard{" not in html:
    html = html.replace("</style>\n\n<div class=\"layout\">", CSS + "</style>\n\n<div class=\"layout\">", 1)
if 'href="#d-day2"' not in html:
    html = html.replace('<li><a href="#d-xfer">Geometry transfer</a></li>', '<li><a href="#d-xfer">Geometry transfer</a></li>\n        <li><a href="#d-day2">Ink crop, novelty, E8</a></li>', 1)
    html = html.replace('<a href="#d-xfer">Geometry transfer</a><a href="#d-xai">', '<a href="#d-xfer">Geometry transfer</a><a href="#d-day2">Ink crop, novelty, E8</a><a href="#d-xai">', 1)

# key findings: add the ink-extent row at the top, update novelty caveat, add E8 and augmentation rows
kb = ic_b["paired"]["kitab_khatt__inkcrop"]
row_ink = (f'      <tr><td><b>Writing size inside the line image</b></td><td>Height of the band holding the central ink rows (Otsu), as a share of the image; crop to that band plus a margin before resizing · <span class="mono">geom_ink_extent.py</span>, <span class="mono">geom_inkcrop_export.py</span></td>'
           f'<td>10 % of KHATT lines have writing in under 35 % of the height (1 % of NAKBA); those lines average 85 % CER</td><td><b>Largest gain found:</b> KHATT {pc(kb["cer_original"])} → {pc(kb["cer_transformed"])} % with no retraining; NAKBA unchanged</td><td>High and immediate: make ink cropping part of every line pipeline (CRNN and VLM); check all benchmark sets for loose crops</td></tr>\n')
if "Writing size inside the line image" not in html:
    html = html.replace('    <tbody>\n      <tr><td><b>Letter-shape novelty</b></td>', '    <tbody>\n' + row_ink + '      <tr><td><b>Letter-shape novelty</b></td>', 1)
html = html.replace("Caveat: the feature comes from the same model, so a model-independent check is running", "Confirmed with the features of a second, separately trained model")
row_more = ('      <tr><td>Slant and stroke augmentation in training</td><td>Random shear −12° to +20° and fractional stroke thickening or thinning on the GPU · <span class="mono">geom_aug</span> in <span class="mono">xai_train_aligner.py</span></td><td>The model becomes slant-robust (an upright Muharaf costs +0.6 points instead of +2.0)</td><td>None on the unseen sets (KHATT 26.4 → 26.9 %, NAKBA 13.8 → 14.4 %)</td><td>Low on its own; keep as a robustness default</td></tr>\n'
            '      <tr><td>LLM decoder on CRNN features (E8)</td><td>Frozen CRNN BiLSTM features, projector into Qwen3-0.6B, LoRA · <span class="mono">xai_e8_spike.py</span></td><td>More fluent but wrong words (41 % against 24 % of wrong words on KHATT)</td><td>Worse everywhere: KHATT 26.4 → 40.0 %, NAKBA 13.8 → 20.8 %</td><td>Low at this scale; would need a larger LLM or CTC posteriors as input</td></tr>\n')
if "LLM decoder on CRNN features" not in html:
    html = html.replace('      <tr><td>Measured clearly, little effect</td>', row_more + '      <tr><td>Measured clearly, little effect</td>', 1)
html = html.replace("Summary · updated 2026-10-12</div>", "Summary · updated 2026-10-12 (evening)</div>")


def add_exp(gid, txt):
    global html
    line = f' ["{gid}",{txt}],\n'
    if f'["{gid}",' not in html:
        m = re.search(r'^ \["X-T2",.*?\],\n', html, flags=re.M)
        html = html.replace(m.group(0), m.group(0) + line, 1)


add_exp("X-T3", '"CRNN retrained with slant and stroke-width augmentation","GPU (3090)","Done 2026-10-12: held-out 7.6 % unchanged; slant-robust; KHATT 26.9 %, NAKBA 14.4 % (no gain)"')
add_exp("X-S1", '"Crop each line to its ink band before resizing","CPU + GPU","Done 2026-10-12: KHATT 26.3 → 23.2 % (−3.1 [−5.1, −1.5]) with no retraining; NAKBA unchanged"')
m = re.search(r'^ \["X-G10",(".*?"),(".*?"),(".*?")\],?$', html, flags=re.M)
if m and "Done 2026-10-12" not in m.group(0):
    html = html.replace(m.group(0), f' ["X-G10",{m.group(1)},{m.group(2)},"Done 2026-10-12: letter error rises from 1–5 % (typical) to 30–53 % (most novel); KHATT 21 % novel letters against 10 % for NAKBA; independent check passed"],', 1)
m = re.search(r'^ \["E8",(".*?"),(".*?"),(".*?")\]', html, flags=re.M)
if m and "Done" not in m.group(3):
    html = html.replace(m.group(0), f' ["E8",{m.group(1)},{m.group(2)},"Done 2026-10-12: no-go at 0.6 B; worse on every set (KHATT 40.0 % against 26.4 %), more fluent but wrong words"]', 1)
LEDGER.write_text(html, encoding="utf-8")
print("ledger", len(html))
