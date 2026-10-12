"""Insert the second 2026-10-12 results block (ink crop trained, VLMs and loose crops, hamza convention, shape-targeted data).

    .geomvenv/bin/python scripts/build_day3_section.py results/geometry
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
pc = lambda x: f"{100 * x:.1f}".replace("-", "−")  # noqa: E731
sg = lambda x: f"{100 * x:+.1f}".replace("-", "−")  # noqa: E731


def table(head, rows):
    return '<div class="qwrap"><table class="dt"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(r) + "</tr>" for r in rows) + "</tbody></table></div>"


ib, it = J("inkcrop_base.json")["paired"], J("inkcrop_trained.json")["paired"]
crop = ("<h4>1 · Ink cropping: at test time is enough</h4>"
        + table(["Set", "Original CRNN, cropped at test time", "CRNN trained on cropped lines, cropped at test time"],
                [[f"<td>{n}</td>", f'<td class="n">{pc(ib[t]["cer_original"])} → <b>{pc(ib[t]["cer_transformed"])} %</b></td>', f'<td class="n">{pc(it[t]["cer_original"])} → <b>{pc(it[t]["cer_transformed"])} %</b></td>']
                 for t, n in (("kitab_khatt__inkcrop", "KHATT"), ("nakba_lines__inkcrop", "NAKBA test"), ("nakba_blind__inkcrop", "NAKBA blind"))])
        + "<p>Retraining on cropped lines (11 % of the training lines change) gives the same result as cropping only at test time. The crop is a preprocessing step for any fixed-height recogniser, not a training change (<span class=\"mono\">xai_inkcrop_traindata.py</span>).</p>")

V = J("vlm_ink_extent.json")
vrows = [[f"<td>{m}</td>", f'<td class="n">{pc(v["cer_all"])} %</td>', f'<td class="n">{pc(v["cer_loose"])} %</td>', f'<td class="n">{pc(v["cer_tight"])} %</td>'] for m, v in V["models"].items()]
vlm = (f"<h4>2 · Do loose crops hurt the VLMs? No</h4><p>The same KHATT benchmark lines ({V['lines']}, {pc(V['share_extent_below_35'])} % with writing in under 35 % of the height), scored per line from the stored predictions of every benchmarked model (<span class=\"mono\">vlm_ink_extent.py</span>).</p>"
       + table(["Model", "All lines", "Loose crops", "Tight crops"], vrows)
       + "<p>The VLMs read loose and tight crops equally well, because they receive the image at native resolution. Only PaddleOCR, which also resizes to a fixed height, suffers. The ink crop belongs in fixed-height pipelines (CRNN, PaddleOCR), not in front of the VLMs.</p>")

Hm = J("hamza.json")
hs = Hm["hamza_share_of_alifs"]
hrows = [[f"<td>{n}</td>", f'<td class="n">{pc(Hm["sets"][t]["cer"])} %</td>', f'<td class="n">{pc(Hm["sets"][t]["cer_hamza_folded"])} %</td>', f'<td class="n">{pc(hs[t])} %</td>']
         for t, n in (("nakba_lines", "NAKBA test"), ("nakba_blind", "NAKBA blind (118 lines with local images)"), ("kitab_khatt", "KHATT"))]
hamza = (f"<h4>3 · NAKBA’s alif-hamza errors are an annotation convention</h4><p>Muharaf’s training labels write the hamza on {pc(hs['muharaf_train'])} % of alifs; NAKBA’s labels on {pc(hs['nakba_lines'])} %. Of NAKBA’s ground-truth أ / إ / آ, the CRNN writes a bare ا at {Hm['sets']['nakba_lines']['gt_hamza_read_as'].get('ا', 0)} places and matches the label at {Hm['sets']['nakba_lines']['gt_hamza_read_as'].get('same', 0)}. The images below show why: the NAKBA writers wrote او, اية, الى, ايضا without a hamza and the annotators added it (<span class=\"mono\">geom_hamza.py</span>).</p>"
         + table(["Set", "CER as scored", "CER, hamza folded on both sides", "Alifs with hamza in the labels"], hrows)
         + "<p>About 2 points of NAKBA’s CER are spelling convention, not misreading. Either score NAKBA with hamza folded (the R2.1 orthographic rung), or treat hamza restoration as a separate post-OCR step; KHATT’s writers draw the hamza, and the model reads most of them.</p>"
         + (G / "hamza_frag.html").read_text(encoding="utf-8"))

shape = "<h4>4 · Shape-targeted training data (X-G11 lite)</h4><p>Pending: the targeted and random-control CRNNs are training on the 3090.</p>"
sp = R / "shape" / "analysis.json"
if sp.exists():
    A = json.loads(sp.read_text(encoding="utf-8"))
    S = json.loads((R / "shape" / "summary.json").read_text(encoding="utf-8"))
    b = A["half_b"]
    pr = b["paired"]
    le = b["letter_errors"]
    rate = lambda m, k: le[m][k][0] / max(1, le[m][k][1])  # noqa: E731
    rows = [[f"<td>{n}</td>", f'<td class="n">{pc(b["cer"][m])} %</td>', f'<td class="n">{pc(rate(m, "focus letters"))} %</td>', f'<td class="n">{pc(rate(m, "all letters"))} %</td>']
            for m, n in (("base", "Original CRNN"), ("random", "+ 2,904 random lines ×4 (control)"), ("targeted", "+ 2,904 shape-matched lines ×4"))]
    prow = [[f"<td>{k}</td>", f'<td class="n"><b>{sg(v["change"])}</b> [{sg(v["ci95"][0])}, {sg(v["ci95"][1])}]</td>'] for k, v in pr.items()]
    shape = (f"""<h4>4 · Shape-targeted training data (X-G11 lite)</h4>
<p>No KHATT labels are used for the selection. KHATT is split by line id into half A ({S['half_a_letters']:,} predicted letters) and half B. On half A, the CRNN’s own reading gives every predicted letter a span and a feature; {S['unusual_letters']} of them are unusual (novelty above Muharaf’s 95th percentile). Each unusual letter marks its 25 nearest training letters of the same character, and the {S['lines_selected']:,} most-marked training lines ({pc(S['lines_selected'] / S['train_lines'])} %) get three extra copies. The control gives three extra copies to the same number of random lines. Same recipe otherwise; scored on half B, which played no part in the selection (<span class="mono">xai_shape_select.py</span>, <span class="mono">shape_analyze.py</span>).</p>"""
             + table(["KHATT half B (" + str(b["lines"]) + " lines)", "CER", "Error on ح خ ش ط ج ظ ى غ ع ه ك م", "Error on all letters"], rows)
             + table(["Paired difference (points)", "Change [95 %]"], prow)
             + ((G / "shape_frag.html").read_text(encoding="utf-8") if (G / "shape_frag.html").exists() else ""))

verdict = ("<b>Three practical results.</b> Cropping to the ink is a preprocessing fix for fixed-height recognisers (no retraining needed) and does not concern the VLMs, which read loose crops fine. "
           "About 2 points of NAKBA’s CER are an annotation convention (hamza added by the annotators). " + ("" if not sp.exists() else "Shape-targeted training data is tested against a random control on held-out KHATT lines (section 4)."))
ARTICLE = f"""
  <!-- 6j. results 2026-10-12 (2) -->
  <article class="dpart" id="d-day3">
    <h3>Ink crop in training, VLMs, hamza convention and shape-targeted data (2026-10-12)</h3>
    <div class="verdict"><p>{verdict}</p></div>
    {crop}
    {vlm}
    {hamza}
    {shape}
  </article>
"""
html = LEDGER.read_text(encoding="utf-8")
html = re.sub(r"\n  <!-- 6j\. results 2026-10-12 \(2\) -->.*?</article>\n(?=\n  <!-- 6b)", "\n", html, flags=re.S)
html = html.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
if 'href="#d-day3"' not in html:
    html = html.replace('<li><a href="#d-day2">Ink crop, novelty, E8</a></li>', '<li><a href="#d-day2">Ink crop, novelty, E8</a></li>\n        <li><a href="#d-day3">Hamza, VLMs, targeted data</a></li>', 1)
    html = html.replace('<a href="#d-day2">Ink crop, novelty, E8</a><a href="#d-xai">', '<a href="#d-day2">Ink crop, novelty, E8</a><a href="#d-day3">Hamza, VLMs, targeted data</a><a href="#d-xai">', 1)
row = ('      <tr><td><b>Hamza spelling convention</b></td><td>Hamza share of alifs in each label set; CER with أ إ آ folded to ا · <span class="mono">geom_hamza.py</span></td>'
       '<td>Muharaf labels write the hamza on 4 % of alifs, NAKBA’s on 16 %; NAKBA writers do not draw it</td><td>About 2 points of NAKBA CER (13.8 → 11.6 % folded)</td><td>High for evaluation: fold hamza for NAKBA, or treat hamza restoration as a post-OCR step</td></tr>\n')
if "Hamza spelling convention" not in html:
    html = html.replace('      <tr><td><b>Letter-shape novelty</b></td>', row + '      <tr><td><b>Letter-shape novelty</b></td>', 1)
html = html.replace("Largest gain found:</b> KHATT", "Largest gain found (fixed-height models only; VLMs unaffected):</b> KHATT")
LEDGER.write_text(html, encoding="utf-8")
print("ledger", len(html))
