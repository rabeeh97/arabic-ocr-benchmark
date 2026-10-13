"""Insert the 2026-10-13 results block (overnight 3090 queue: replicate noise, shape-targeted data on held-out lines, combined recipe,
crop plus slant TTA, letter novelty under the new models, E8 at 1.7B) and update the key-findings and EXP rows.

    .geomvenv/bin/python scripts/build_day4_section.py results/geometry
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
G = R / "galleries"
LEDGER = Path("docs/research/paper-ledger.html")
J = lambda p: json.loads((R / p).read_text(encoding="utf-8"))  # noqa: E731
pc = lambda x: f"{100 * x:.1f}".replace("-", "−")  # noqa: E731
sg = lambda x: f"{100 * x:+.1f}".replace("-", "−")  # noqa: E731
frag = lambda n: (G / f"{n}_frag.html").read_text(encoding="utf-8") if (G / f"{n}_frag.html").exists() else ""  # noqa: E731


def table(head, rows):
    return '<div class="qwrap"><table class="dt"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(r) + "</tr>" for r in rows) + "</tbody></table></div>"


def cer(pl, ids):
    return sum(pl[i][0] for i in ids) / sum(pl[i][1] for i in ids)


def paired(a, b, ids, n_boot=4000):
    """CER(b) − CER(a) over the same lines, with a paired bootstrap 95 % interval."""
    ids = [i for i in ids if i in a and i in b]
    ea, eb, n = (np.array([d[i][k] for i in ids]) for d, k in ((a, 0), (b, 0), (a, 1)))
    rng = np.random.default_rng(0)
    s = rng.integers(0, len(ids), (n_boot, len(ids)))
    bs = (eb[s] - ea[s]).sum(1) / n[s].sum(1)
    return (eb - ea).sum() / n.sum(), np.percentile(bs, [2.5, 97.5])


def cell(d):
    ch, ci = d
    return f'<td class="n"><b>{sg(ch)}</b> [{sg(ci[0])}, {sg(ci[1])}]</td>'


H = J("shape/halves.json")
PL = lambda p, t="kitab_khatt": J(p)["per_line"][t]  # noqa: E731
base, rep2, targ, rand = PL("shape/base.json"), PL("night/rep2_eval.json"), PL("shape/targeted_eval.json"), PL("shape/random_eval.json")

# 1 · run-to-run noise and the shape result measured against it
noise_rows = []
for half, name in (("half_a", "Half A (used for the selection)"), ("half_b", "Half B (held out)")):
    ids = [i for i in H[half] if all(i in m for m in (base, rep2, targ, rand))]
    noise_rows.append([f"<td>{name}, {len(ids)} lines</td>", cell(paired(base, rep2, ids)), cell(paired(rand, targ, ids))])
nb = paired(base, rep2, [i for i in H["half_b"] if i in base and i in rep2])
sb = paired(rand, targ, [i for i in H["half_b"] if i in rand and i in targ])
noise = ("<h4>1 · How much does a retrain move KHATT by chance?</h4>"
         "<p>The original recipe trained a second time with a different seed (N1). The difference between two identical recipes is the noise yardstick for every retraining result on 91 or 92 KHATT lines.</p>"
         + table(["KHATT", "Replicate − original (same recipe)", "Shape-targeted − random control"], noise_rows)
         + f"<p>On half B a plain retrain moves CER by {sg(nb[0])} points, and the shape-targeted model beats the random control by {sg(-sb[0]).lstrip('+')} points with an interval that just reaches zero. "
         "The shape gain is in the expected direction on both halves and larger than the replicate’s drift, but on 91 held-out lines it is not established. Section 4 of the previous block has the full table and the gallery of half-B lines.</p>")

# 2 · stacking: crop, shape repeats, slant TTA
T = "kitab_khatt__inkcrop"
ib = J("inkcrop_base.json")["per_line"]
models = [("Original CRNN", ib), ("Replicate (N1)", J("night/rep2_inkcrop.json")["per_line"]),
          ("+ random lines ×4 (control)", J("night/randctl_inkcrop.json")["per_line"]), ("+ shape-matched lines ×4", J("night/shape_inkcrop.json")["per_line"]),
          ("Combo: trained on cropped lines + shape-matched lines ×4", J("night/combo_inkcrop.json")["per_line"])]
ids_all = list(ib[T])
srows = []
for name, m in models:
    srows.append([f"<td>{name}</td>", f'<td class="n">{pc(cer(m["kitab_khatt"], ids_all))} %</td>', f'<td class="n"><b>{pc(cer(m[T], ids_all))} %</b></td>',
                  cell(paired(ib[T], m[T], ids_all)) if m is not ib else '<td class="n">—</td>',
                  f'<td class="n">{pc(cer(m["nakba_lines__inkcrop"], list(m["nakba_lines__inkcrop"])))} %</td>'])
tta_rows, tta_note = [], ""
for f, name in (("night/tta_inkcrop_pl.json", "Original CRNN"), ("night/tta_inkcrop_combo.json", "Combo CRNN")):
    if not (R / f).exists():
        continue
    t = J(f)
    up, sel = t["per_line"][T], t["per_line"][T + "__tta"]
    tta_rows.append([f"<td>{name}</td>", f'<td class="n">{pc(t[T]["variant_cer"]["0"])} %</td>', f'<td class="n"><b>{pc(t[T]["confidence_selected_cer"])} %</b></td>',
                     cell(paired(up, sel, list(up))), f'<td class="n">{pc(t[T]["oracle_cer"])} %</td>',
                     f'<td class="n">{pc(t["nakba_lines__inkcrop"]["variant_cer"]["0"])} → {pc(t["nakba_lines__inkcrop"]["confidence_selected_cer"])} %</td>'])
if tta_rows:
    tta_note = (table(["Slant TTA on ink-cropped lines", "KHATT upright", "KHATT, confidence-selected shear (−6° to +18°)", "Change [95 %]", "Oracle shear", "NAKBA test upright → selected"], tta_rows)
                + "<p>Each line is read at five shears and the reading with the highest mean letter confidence is kept (<span class=\"mono\">xai_geom_tta.py</span>). On KHATT the gain adds to the crop; on NAKBA, which already leans like the training data, the selector picks a wrong shear often enough to cost 0.3 to 0.5 points, so TTA should be gated on measured slant rather than applied everywhere.</p>"
                + frag("tta"))
stack = ("<h4>2 · Do the fixes stack? Crop, targeted repeats and slant TTA</h4><p>All rows read the ink-cropped KHATT lines (183) and are compared with the original CRNN on the same cropped lines.</p>"
         + table(["Model", "KHATT as stored", "KHATT ink-cropped", "Change vs original, cropped [95 %]", "NAKBA test, cropped"], srows)
         + "<p>Training on cropped lines alone added nothing (previous block, section 1), and the shape repeats alone are within noise; together they give the lowest KHATT CER so far, and NAKBA does not get worse (slightly better if anything). "
           "Whether the combination is more than the sum of two small effects needs a second seed.</p>"
         + frag("combo") + tta_note)

# 3 · letter novelty under the new models
L = {n: J(p)["datasets"]["kitab_khatt"] for n, p in (("Original CRNN", "letters/summary.json"), ("+ random lines ×4 (control)", "night/letters_randctl/summary.json"), ("+ shape-matched lines ×4", "night/letters_shape/summary.json"))}
lrows = [[f"<td>{n}</td>", f'<td class="n">{pc(v["letter_error_rate"])} %</td>'] + [f'<td class="n">{pc(x)} %</td>' for x in v["error_rate_by_band"][3:]] for n, v in L.items()]
nov = ("<h4>3 · Letter novelty under the targeted model</h4><p>The X-G10 analysis repeated with each new model’s own features and forced alignment (band edges are recomputed per model and differ by under 0.02). All 183 KHATT lines, so half A, which steered the selection, is included.</p>"
       + table(["KHATT letters (9,019)", "Letter error", "Band 4", "Band 5", "Most novel band"], lrows)
       + "<p>The targeted model’s gain sits where it should: in the two most novel bands (about 2 points each below the control), while band 4 matches the control. KHATT’s letters are as novel to it as before (21–22 % beyond Muharaf’s 95th percentile): three extra copies of nearby training letters make the model more tolerant of the unusual shapes, without making them look familiar.</p>")

# 4 · E8 at 1.7B
E6, E17 = J("e8/results.json"), J("e8_1p7b/results.json")
erows = [[f"<td>{n}</td>", f'<td class="n">{pc(E6[t]["cer_crnn"])} %</td>', f'<td class="n">{pc(E6[t]["cer_e8"])} %</td>', f'<td class="n"><b>{pc(E17[t]["cer_e8"])} %</b></td>',
          f'<td class="n">{pc(E6[t]["fluent_wrong_share_crnn"])} / {pc(E6[t]["fluent_wrong_share_e8"])} / {pc(E17[t]["fluent_wrong_share_e8"])} %</td>']
         for t, n in (("kitab_khatt", "KHATT"), ("nakba_lines", "NAKBA test"), ("nakba_blind", "NAKBA blind"), ("muharaf_lines", "Muharaf"), ("rasam1", "RASAM"))]
e8 = ("<h4>4 · E8 with a three times larger LLM: still no-go</h4><p>Same spike as before (frozen CRNN BiLSTM features, projector, LoRA), Qwen3-1.7B in place of 0.6B (<span class=\"mono\">xai_e8_spike.py</span>).</p>"
      + table(["Set", "CRNN", "E8, Qwen3-0.6B", "E8, Qwen3-1.7B", "Fluent-but-wrong share of wrong words (CRNN / 0.6B / 1.7B)"], erows)
      + "<p>Scale does not help: CER is unchanged within a point and the share of wrong words that are real, fluent Arabic words rises further. The LLM rewrites what it cannot see into plausible text. Closing E8 in this form; an LLM would have to receive the CTC posteriors or be used only for rescoring constrained to the CRNN’s lattice.</p>"
      + frag("e8_1p7b"))

best = J("night/combo_inkcrop.json")["cer"][T]
ct = J("night/tta_inkcrop_combo.json")[T]["confidence_selected_cer"] if (R / "night/tta_inkcrop_combo.json").exists() else None
tta_txt = f"; with slant TTA on top, {pc(ct)} %" if ct is not None else ""
verdict = (f"<b>Fixes stack, scale does not.</b> Ink-cropped training plus shape-targeted repeats gives KHATT {pc(best)} % (from 26.3 % as stored and 23.2 % with the crop alone), with NAKBA unchanged{tta_txt} (TTA costs NAKBA about half a point, so it should be gated). "
           f"The shape-targeted repeats beat a random control by {sg(-sb[0]).lstrip('+')} points on held-out lines, about twice the drift between two identical trainings, but the interval reaches zero. E8 at 1.7B is as bad as at 0.6B.")
ARTICLE = f"""
  <!-- 6k. results 2026-10-13 -->
  <article class="dpart" id="d-day4">
    <h3>Overnight runs: noise yardstick, stacking fixes, targeted letters and E8 at 1.7B (2026-10-13)</h3>
    <div class="verdict"><p>{verdict}</p></div>
    {noise}
    {stack}
    {nov}
    {e8}
  </article>
"""
html = LEDGER.read_text(encoding="utf-8")
html = re.sub(r"\n  <!-- 6k\. results 2026-10-13 -->.*?</article>\n(?=\n  <!-- 6b)", "\n", html, flags=re.S)
html = html.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
if 'href="#d-day4"' not in html:
    html = html.replace('<li><a href="#d-day3">Hamza, VLMs, targeted data</a></li>', '<li><a href="#d-day3">Hamza, VLMs, targeted data</a></li>\n        <li><a href="#d-day4">Stacking, noise, E8 1.7B</a></li>', 1)
    html = html.replace('<a href="#d-day3">Hamza, VLMs, targeted data</a><a href="#d-xai">', '<a href="#d-day3">Hamza, VLMs, targeted data</a><a href="#d-day4">Stacking, noise, E8 1.7B</a><a href="#d-xai">', 1)

html = re.sub(r"<span>Snapshot 2026-10-\d\d</span>", "<span>Snapshot 2026-10-13</span>", html, count=1)

# key findings
html = re.sub(r'<div class="eyebrow">Summary · updated [^<]*</div>', '<div class="eyebrow">Summary · updated 2026-10-13</div>', html, count=1)
html = re.sub(r"(Largest gain found \(fixed-height models only; VLMs unaffected\):</b> KHATT 26\.3 → 23\.2 % with no retraining; NAKBA unchanged)(?:; [^<]*)?",
              rf"\1; {pc(best)} % with ink-cropped training plus shape-targeted repeats{tta_txt.replace(';', ',')}", html, count=1)
html = re.sub(r"(Confirmed with the features of a second, separately trained model)(?:\. [^<]*)?",
              rf"\1. Shape-targeted training repeats (no KHATT labels) lower the error in the two most novel bands by about 2 points; on held-out lines {sg(sb[0])} points CER against a random control [{sg(sb[1][0])}, {sg(sb[1][1])}]", html, count=1)
html = html.replace("<td>Low at this scale; would need a larger LLM or CTC posteriors as input</td>",
                    f"<td>Closed in this form: Qwen3-1.7B is no better (KHATT {pc(E17['kitab_khatt']['cer_e8'])} %, fluent-but-wrong {pc(E17['kitab_khatt']['fluent_wrong_share_e8'])} %); only CTC posteriors as input or lattice-constrained rescoring remain</td>")
if "Run-to-run noise" not in html:
    html = html.replace("      <tr><td>Measured clearly, little effect</td>",
                        f'      <tr><td>Run-to-run noise (yardstick)</td><td>The original recipe retrained with another seed · <span class="mono">xai_train_aligner.py</span></td><td>Two identical trainings differ by {sg(nb[0])} points on a 91-line KHATT half [{sg(nb[1][0])}, {sg(nb[1][1])}]</td><td>Any retraining effect under about 1 point on KHATT halves needs a second seed</td><td>Method: train two seeds for every retraining claim</td></tr>\n'
                        "      <tr><td>Measured clearly, little effect</td>", 1)

# EXP rows
exp_new = [
    ("X-G11L", "Shape-targeted training repeats: training lines whose letters are nearest to KHATT’s unusual letters (half A, no labels) get three extra copies; random control and replicate",
     "GPU (3090)", f"Done 2026-10-13: held-out half B {sg(sb[0])} points against random [{sg(sb[1][0])}, {sg(sb[1][1])}], replicate drift {sg(nb[0])}; error in the most novel bands about 2 points lower. Second seed pending"),
    ("X-S2", "Stack the fixes: ink-cropped training data + shape-targeted repeats, then slant TTA",
     "GPU (3090)", f"Done 2026-10-13: KHATT {pc(best)} % cropped (original CRNN cropped 23.2 %){tta_txt}, NAKBA unchanged without TTA; second seed pending"),
    ("E8b", "E8 with Qwen3-1.7B in place of 0.6B", "GPU (3090)",
     f"Done 2026-10-13: no-go, KHATT {pc(E17['kitab_khatt']['cer_e8'])} % (0.6B {pc(E6['kitab_khatt']['cer_e8'])} %), fluent-but-wrong share up"),
]
missing = []
for row in exp_new:
    if f'["{row[0]}",' in html:
        html = re.sub(rf'\["{row[0]}",[^\n]*\]', lambda _: json.dumps(list(row), ensure_ascii=False), html, count=1)
    else:
        missing.append(json.dumps(list(row), ensure_ascii=False))
if missing:
    e8_row = '","GPU","Done 2026-10-12: no-go at 0.6 B; worse on every set (KHATT 40.0 % against 26.4 %), more fluent but wrong words"]'
    html = html.replace(e8_row, e8_row + ",\n " + ",\n ".join(missing), 1)
LEDGER.write_text(html, encoding="utf-8")
print("ledger", len(html), "rows", all(f'["{c}",' in html for c, *_ in exp_new))
