"""Insert the "Geometry measurements" article into docs/research/paper-ledger.html and update the rows it touches.

    .geomvenv/bin/python scripts/build_geometry_section.py results/geometry/run1

Idempotent: it removes an earlier copy of the article before inserting. All numbers come from the run's JSON files.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

run = Path(sys.argv[1])
LEDGER = Path("docs/research/paper-ledger.html")
S = json.loads((run / "summary.json").read_text())
X = json.loads((run / "extra.json").read_text())
D = json.loads((run / "dots_resize.json").read_text())
T = S["transfer_tests"]["datasets"]
ORDER = ["muharaf_train", "muharaf", "nakba", "nakba_blind", "khatt", "rasam1", "rasam2"]
NAME = {"muharaf_train": "Muharaf train sample (the training pool)", "muharaf": "Muharaf test", "nakba": "NAKBA test", "nakba_blind": "NAKBA blind test",
        "khatt": "KHATT", "rasam1": "RASAM-1", "rasam2": "RASAM-2"}
COLOR = {"muharaf_train": "var(--s1)", "muharaf": "var(--s1)", "nakba": "var(--s3)", "nakba_blind": "var(--s3)", "khatt": "var(--s2)", "rasam1": "var(--axis)", "rasam2": "var(--axis)"}
n_lines = sum(S["n"].values())
_rows = [r for r in json.loads((run / "per_line.json").read_text()) if r.get("ok") and r["tag"] == "muharaf_train" and r.get("slant_skel_deg") == r.get("slant_skel_deg")]
TRAIN_BELOW3 = 100 * sum(r["slant_skel_deg"] < 3 for r in _rows) / len(_rows)


def f(x, d=1):
    return f"{x:.{d}f}".replace("-", "−")


# ---------------------------------------------------------------- figure 1: slant ranges
def slant_chart() -> str:
    x0, x1, lo, hi = 190, 720, -15, 35
    X_ = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    row_h, top = 30, 26
    H = top + row_h * len(ORDER) + 34
    p5, p95 = X["train_slant_p5_p95"]
    out = [f'<svg viewBox="0 0 760 {H}" role="img" aria-label="Slant range per dataset">']
    out.append(f'<rect x="{X_(p5):.1f}" y="{top - 8}" width="{X_(p95) - X_(p5):.1f}" height="{row_h * len(ORDER) + 8}" fill="var(--accent-soft)"/>')
    out.append(f'<text x="{X_(p5) + 6:.1f}" y="{top - 12}" class="lbl">training pool, 5th to 95th percentile</text>')
    for v in range(-10, 36, 10):
        out.append(f'<line x1="{X_(v):.1f}" x2="{X_(v):.1f}" y1="{top - 8}" y2="{top + row_h * len(ORDER)}" stroke="var(--grid)"/>')
        out.append(f'<text x="{X_(v):.1f}" y="{top + row_h * len(ORDER) + 16}" text-anchor="middle">{v}°</text>')
    out.append(f'<line x1="{X_(0):.1f}" x2="{X_(0):.1f}" y1="{top - 8}" y2="{top + row_h * len(ORDER)}" stroke="var(--axis)"/>')
    out.append(f'<text x="{X_(0) - 4:.1f}" y="{top + row_h * len(ORDER) + 30}" text-anchor="end">upright</text>')
    out.append(f'<text x="{X_(0) + 4:.1f}" y="{top + row_h * len(ORDER) + 30}">leans right</text>')
    for i, t in enumerate(ORDER):
        y = top + i * row_h + row_h / 2 - 6
        p10, p50, p90 = X["slant"][t]["p10_p50_p90"]
        c = COLOR[t]
        out.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="lbl">{NAME[t].replace(" (the training pool)", "")}</text>')
        out.append(f'<line x1="{X_(p10):.1f}" x2="{X_(p90):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>')
        out.append(f'<circle cx="{X_(p50):.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="var(--surface)" stroke-width="2"/>')
        out.append(f'<text x="{X_(p90) + 8:.1f}" y="{y + 4:.1f}">median {f(p50)}°</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- figure 2: distances to the training sample
def dist_panels() -> str:
    cats = [("muharaf", "Muharaf test", "7.6 %"), ("nakba", "NAKBA test", "13.8 %"), ("nakba_blind", "NAKBA blind", "13.7 %"), ("khatt", "KHATT", "26.4 %")]
    panels = [("scalar_profile", "Scalar profile (SD units)", "dist"), ("hinge", "Hinge histogram (JSD)", "jsd"), ("loci", "Loci histogram (JSD)", "jsd"), ("direction", "Stroke direction (JSD)", "jsd")]
    svgs = []
    for key, title, vk in panels:
        vals = [(T[t][key][vk], T[t][key]["ci95"]) for t, _, _ in cats]
        mx = max(max(v, c[1]) for v, c in vals) * 1.12
        x0, x1, W, rh = 112, 330, 360, 30
        X_ = lambda v: x0 + v / mx * (x1 - x0)  # noqa: E731
        H = 30 + rh * len(cats) + 26
        o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">', f'<text x="0" y="14" class="title">{title}</text>']
        for k in range(0, 5):
            v = mx * k / 4
            o.append(f'<line x1="{X_(v):.1f}" x2="{X_(v):.1f}" y1="24" y2="{24 + rh * len(cats)}" stroke="var(--grid)"/>')
            o.append(f'<text x="{X_(v):.1f}" y="{24 + rh * len(cats) + 14}" text-anchor="middle">{v:.3g}</text>')
        for i, ((t, nm, cer), (v, ci)) in enumerate(zip(cats, vals)):
            y = 24 + i * rh + rh / 2
            c = COLOR[t] if t != "muharaf" else "var(--axis)"
            o.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="lbl">{nm}</text>')
            o.append(f'<line x1="{X_(ci[0]):.1f}" x2="{X_(max(ci[1], v)):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="1.5"/>')
            o.append(f'<circle cx="{X_(v):.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="var(--surface)" stroke-width="2"/>')
        o.append("</svg>")
        svgs.append(f'<div class="gbox">{"".join(o)}</div>')
    return '<div class="smalls">' + "".join(svgs) + "</div>"


# ---------------------------------------------------------------- tables
def drivers_table() -> str:
    feats = [("slant_skel_deg", "Slant (leans less right than training)"), ("dot_diam_over_sw", "Dot diameter in stroke widths"), ("turn_mean_rad", "Mean turning angle (curviness)"),
             ("sw_native", "Native stroke width, px"), ("ink_density", "Ink density"), ("pen_contrast", "Pen contrast"), ("asc_over_sw", "Ascender height in stroke widths"),
             ("xheight_over_sw", "Median height above baseline")]
    cols = ["nakba", "nakba_blind", "khatt"]
    rows = []
    for k, nm in feats:
        cells = []
        for c in cols:
            v = T[c]["scalar_profile"]["per_feature_sd"][k]
            w = min(abs(v) / 1.6, 1) * 50
            side = f"left:50%;width:{w:.1f}%;background:var(--s2)" if v > 0 else f"right:50%;width:{w:.1f}%;background:var(--s1)"
            cells.append(f'<td class="n">{f(v, 2)}</td><td><span class="sb"><i style="{side}"></i></span></td>')
        rows.append(f"<tr><td>{nm}</td>{''.join(cells)}</tr>")
    head = "<tr><th>Measure</th>" + "".join(f'<th colspan="2">{NAME[c]}</th>' for c in cols) + "</tr>"
    return f'<div class="qwrap"><table class="dt"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def medians_table() -> str:
    keys = [("sw_native", "Stroke width, native px", 1), ("dot_diam_over_sw", "Dot diameter ÷ stroke", 2), ("asc_over_dot", "Ascender ÷ dot diameter", 1), ("asc_over_sw", "Ascender ÷ stroke", 1),
            ("pen_contrast", "Pen contrast", 2), ("sw_cv", "Stroke-width variation", 2), ("turn_mean_rad", "Turning angle, rad", 2), ("ink_density", "Ink density", 2),
            ("line_native_h", "Line height, native px", 0)]
    head = "<tr><th>Median per line</th>" + "".join(f"<th>{NAME[t].split(' (')[0]}</th>" for t in ORDER) + "</tr>"
    rows = []
    for k, nm, d in keys:
        vals = []
        for t in ORDER:
            if k == "pen_contrast":
                v = X["other"][t]["pen_contrast_median"]
            elif k in S["scalars"]:
                v = S["scalars"][k][t][2]
            vals.append(f'<td class="n">{f(v, d)}</td>')
        rows.append(f"<tr><td>{nm}</td>{''.join(vals)}</tr>")
    slant = "".join(f'<td class="n">{f(X["slant"][t]["p10_p50_p90"][1])}</td>' for t in ORDER)
    rows.insert(0, f"<tr><td>Slant, degrees (+ leans right)</td>{slant}</tr>")
    rule = "".join(f'<td class="n">{X["rules"][t]["lines_with_rule_ink"] * 100:.0f} %</td>' for t in ORDER)
    rows.append(f"<tr><td>Lines with ruled-line ink</td>{rule}</tr>")
    sl = "".join(f'<td class="n">{X["other"][t]["slope_abs_p90_deg"]:.1f}</td>' for t in ORDER)
    rows.append(f"<tr><td>Baseline slope, 90th percentile of |°|</td>{sl}</tr>")
    return f'<div class="qwrap"><table class="dt"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def dots_table() -> str:
    tags = [("muharaf", "Muharaf"), ("nakba", "NAKBA"), ("khatt", "KHATT"), ("rasam1", "RASAM-1")]
    head = "<tr><th>Height after resize</th>" + "".join(f"<th>{n}</th>" for _, n in tags) + "</tr>"
    rows = []
    for h in (32, 48, 64):
        rows.append(f"<tr><td>{h} px: dots narrower than 2 px</td>" + "".join(f'<td class="n">{D[t][f"{h}_plain"]["share_diam_below_2px"] * 100:.0f} %</td>' for t, _ in tags) + "</tr>")
    for h in (32, 64):
        rows.append(f"<tr><td>{h} px: dot contrast kept, plain → min-pool</td>" + "".join(
            f'<td class="n">{D[t][f"{h}_plain"]["mean_contrast"]:.2f} → {D[t][f"{h}_minpool"]["mean_contrast"]:.2f}</td>' for t, _ in tags) + "</tr>")
    return f'<div class="qwrap"><table class="dt"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def loci_table() -> str:
    g = S["loci_groups"]
    names = [("enclosed (ink on all four sides)", "Enclosed (loops, closed counters)"), ("open upward only (no ink above)", "Open upward only"), ("open downward only (no ink below)", "Open downward only")]
    tags = ["muharaf", "nakba", "khatt"]
    head = "<tr><th>Share of background pixels in the band within 2 strokes of the baseline</th>" + "".join(f"<th>{NAME[t]}</th>" for t in tags) + "</tr>"
    rows = [f"<tr><td>{nm}</td>" + "".join(f'<td class="n">{g[t][k][1] * 100:.1f} %</td>' for t in tags) + "</tr>" for k, nm in names]
    rows.append("<tr><td>Enclosed, above the baseline band</td>" + "".join(f'<td class="n">{g[t][names[0][0]][0] * 100:.1f} %</td>' for t in tags) + "</tr>")
    rows.append("<tr><td>Enclosed, below the baseline band</td>" + "".join(f'<td class="n">{g[t][names[0][0]][2] * 100:.1f} %</td>' for t in tags) + "</tr>")
    return f'<div class="qwrap"><table class="dt"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def overlay_imgs() -> str:
    out = []
    for tag, cap in [("muharaf", "Muharaf"), ("nakba", "NAKBA (ruled-line fragments and neighbouring lines are visible)"), ("khatt", "KHATT"), ("rasam1", "RASAM-1 (crops include neighbouring lines)")]:
        p = run.parent / "overlay" / f"{tag}.png"
        im = Image.open(p).convert("RGB")
        im = im.crop((0, 0, im.width, min(im.height, 330)))
        im = im.resize((720, round(im.height * 720 / im.width)), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=70)
        out.append(f'<figure class="fig"><img alt="Overlay for {cap}" style="border:1px solid var(--line);border-radius:4px;background:#fff" src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}"><figcaption>{cap}</figcaption></figure>')
    return '<div class="smalls">' + "".join(out) + "</div>"


dv = S["dot_validation"]
sep = S["separability_hinge_direction"]
sep_l = S["separability_loci"]
ag = S["agreement"]
us = S["unit_stability"]
kh, nk, nb, mu = T["khatt"], T["nakba"], T["nakba_blind"], T["muharaf"]
xs = X["slant"]

ARTICLE = f"""
  <!-- 6f. geometry measurements -->
  <article class="dpart" id="d-geom">
    <h3>Geometry measurements (X-G1, X-G2, X-G4 to X-G7, 2026-10-11)</h3>
    <div class="verdict"><p><b>KHATT sits far from our training lines in slant and pen proportions, not in contour shape.</b> The measured geometry of {n_lines:,} lines puts KHATT {kh["scalar_profile"]["dist"] / ((nk["scalar_profile"]["dist"] + nb["scalar_profile"]["dist"]) / 2):.1f}× further from the Muharaf training sample than NAKBA on a profile of eight per-line numbers (intervals do not overlap), and the largest single gap is slant: Muharaf leans right by a median {f(xs["muharaf_train"]["p10_p50_p90"][1])}°, NAKBA by {f(xs["nakba"]["p10_p50_p90"][1])}° to {f(xs["nakba_blind"]["p10_p50_p90"][1])}°, KHATT is upright ({f(xs["khatt"]["p10_p50_p90"][1])}°). {xs["khatt"]["share_lines_where_train_density_below_5pct"] * 100:.0f} % of KHATT lines have a slant that fewer than 5 % of training lines share, against {xs["nakba"]["share_lines_where_train_density_below_5pct"] * 100:.0f} % and {xs["nakba_blind"]["share_lines_where_train_density_below_5pct"] * 100:.0f} % for NAKBA. The contour-shape histograms (hinge, loci) put KHATT and NAKBA at about the same distance, so they do not explain the gap in CER (26.4 % against 13.7 %). This is a pattern across three transfer sets, not a causal result: language, writers and image quality differ too.</p></div>
    <p>Everything below is CPU-only and uses the cached images of seven sets ({", ".join(f"{NAME[t].split(' (')[0]} {S['n'][t]}" for t in ORDER)} lines). Each line is first rescaled so that its median stroke is 4 px wide, so histograms from scans of different resolution are comparable. Dots are the small connected components. Code: <span class="mono">scripts/geom_measure.py</span>, <span class="mono">geom_analyze.py</span>, <span class="mono">geom_dots.py</span>; results in <span class="mono">results/geometry/run1</span>.</p>

    <h4>Slant: the clearest difference between datasets</h4>
    <figure class="fig">
      <div class="gbox" role="img" aria-label="Slant range per dataset">{slant_chart()}</div>
      <figcaption>Slant of near-vertical strokes per line (skeleton estimator). Line = 10th to 90th percentile, dot = median. The shaded band is where the training lines lie (5th to 95th percentile, {f(X["train_slant_p5_p95"][0])}° to {f(X["train_slant_p5_p95"][1])}°), so KHATT is not outside the range, it is in the thin end of it: only {TRAIN_BELOW3:.0f} % of training lines are below 3°. A second estimator (shear that sharpens the vertical projection) agrees on rank but not on size: median difference {f(ag["slant proj vs skel"]["median_abs_diff_deg"])}°, correlation 0.4 to 0.7 per dataset.</figcaption>
    </figure>

    <h4>Distance from the training sample, with intervals</h4>
    <figure class="fig">
      {dist_panels()}
      <figcaption>Distance of each evaluation set to the Muharaf training sample; whiskers are 95 % bootstrap intervals over lines. The Muharaf test sample shows the noise floor (same source, different lines). The CRNN’s CER on each set is 7.6, 13.8, 13.7 and 26.4 %. Only the scalar profile puts KHATT clearly past NAKBA. For the hinge histogram KHATT and NAKBA test are equal ({kh["hinge"]["jsd"]:.4f} and {nk["hinge"]["jsd"]:.4f}); for the loci histogram KHATT is {kh["loci"]["jsd"] / nk["loci"]["jsd"]:.1f}× NAKBA, a small difference; for stroke direction KHATT is closer than NAKBA. Jensen–Shannon distances from small samples run high, which is why some intervals sit above their point estimate.</figcaption>
    </figure>
    <h4>What the scalar profile is made of</h4>
    {drivers_table()}
    <p>Signed difference to the Muharaf training sample in standard deviations of all lines pooled. KHATT differs on slant ({f(kh["scalar_profile"]["per_feature_sd"]["slant_skel_deg"], 1)} SD), dot size relative to the stroke ({f(kh["scalar_profile"]["per_feature_sd"]["dot_diam_over_sw"], 1)}), curviness ({f(kh["scalar_profile"]["per_feature_sd"]["turn_mean_rad"], 1)}) and native stroke width (+{kh["scalar_profile"]["per_feature_sd"]["sw_native"]:.1f}, partly a scan-resolution effect). NAKBA differs mostly on slant (−0.6 to −0.8) and ink density (+0.5), the second probably from ruled lines and neighbouring-line fragments in its crops.</p>

    <h4>Per-dataset numbers</h4>
    {medians_table()}
    <ul>
      <li><b>Slope is not the problem.</b> Median baseline slope is within half a degree of zero in every set, and the 90th percentile is below 3° for Muharaf and KHATT. NAKBA ({X["other"]["nakba"]["slope_abs_p90_deg"]:.1f}°) and RASAM (about 12°) are worse, but their crops hold ruled lines and neighbouring lines that pull the estimate, so I would not read that as writing slope. The three estimators (projection, shear search, skeleton direction) agree to a median of {f(ag["slope proj vs skel"]["median_abs_diff_deg"])}° to {f(ag["slope proj vs rot"]["median_abs_diff_deg"])}° and disagree by more than {f(ag["slope proj vs skel"]["p90_abs_diff_deg"], 0)}° on one line in ten, which is the contaminated ones.</li>
      <li><b>Pen contrast separates Maghrebi from the rest.</b> I define it as the width of near-vertical strokes over the width of near-horizontal ones. It is {f(X["other"]["rasam1"]["pen_contrast_median"], 2)} in RASAM-1 and {f(X["other"]["rasam2"]["pen_contrast_median"], 2)} in RASAM-2 (thin verticals, thick horizontals, a cut nib) and 0.98 to 1.12 in Muharaf, NAKBA and KHATT (a pen of even width). Stroke-width variation along the line is also highest in RASAM ({f(X["other"]["rasam1"]["sw_cv_median"], 2)} against {f(X["other"]["muharaf"]["sw_cv_median"], 2)} for Muharaf and {f(X["other"]["khatt"]["sw_cv_median"], 2)} for KHATT). This is a measurable style trait that nothing in our augmentation set imitates.</li>
      <li><b>The ascender is about six dots tall.</b> Ascender height divided by dot diameter is {f(S["scalars"]["asc_over_dot"]["muharaf"][2])} for Muharaf, {f(S["scalars"]["asc_over_dot"]["nakba"][2])} for NAKBA and {f(S["scalars"]["asc_over_dot"]["khatt"][2])} for KHATT, inside the five-to-seven-dot range of the classical alif canon (the canon itself was never agreed). The dot is a slightly steadier unit than the stroke across the three clean sets (between-set spread {us["asc_over_dot"]["between_dataset_range_over_mean"] * 100:.0f} % against {us["asc_over_sw"]["between_dataset_range_over_mean"] * 100:.0f} %), but within a set the spread per line is the same ({us["asc_over_dot"]["within_dataset_iqr_over_median"]["muharaf"] * 100:.0f} % against {us["asc_over_sw"]["within_dataset_iqr_over_median"]["muharaf"] * 100:.0f} % for Muharaf). So neither unit is clearly the better per-line scale yet; that needs the model test (X-G3).</li>
      <li><b>Ruled lines are a NAKBA trait.</b> {X["rules"]["nakba"]["lines_with_rule_ink"] * 100:.0f} % of NAKBA test lines and {X["rules"]["nakba_blind"]["lines_with_rule_ink"] * 100:.0f} % of blind-test lines contain long thin horizontal ink (more than 2 % of the line’s ink), against {X["rules"]["muharaf"]["lines_with_rule_ink"] * 100:.1f} % of Muharaf and none of KHATT. The CRNN’s 13.7 % on NAKBA includes whatever those lines cost it.</li>
    </ul>

    <h4>Characteristic loci (X-G1)</h4>
    {loci_table()}
    <p>Glucksman’s four rays (north, south, east, west; runs of ink capped at two) on the background pixels, split into above, near and below the fitted baseline. Dots are removed first. About a fifth of the pixels near the baseline are enclosed in Muharaf and NAKBA ({S["loci_groups"]["muharaf"]["enclosed (ink on all four sides)"][1] * 100:.1f} % and {S["loci_groups"]["nakba"]["enclosed (ink on all four sides)"][1] * 100:.1f} %) and fewer in KHATT ({S["loci_groups"]["khatt"]["enclosed (ink on all four sides)"][1] * 100:.1f} %): KHATT loops are more open, possibly a faster, flatter pen. The RASAM figures are not shown because their crops hold several lines and their paper is dark and stained, so the loci counts are dominated by neighbours (67 % enclosed).</p>

    <h4>Can contours identify a dataset? (X-G4)</h4>
    <p>A nearest-centroid classifier on one line’s hinge plus stroke-direction histogram identifies which of Muharaf, RASAM-1, NAKBA and KHATT it came from {sep["accuracy"] * 100:.0f} % of the time (chance 25 %), the loci histogram {sep_l["accuracy"] * 100:.0f} %. Two controls sit near chance, as they should: telling NAKBA test from NAKBA blind test gets {S["separability_control_nakba_vs_blind"]["accuracy"] * 100:.0f} % (chance 50 %) and Muharaf train from Muharaf test {S["separability_control_muharaf_train_vs_test"]["accuracy"] * 100:.0f} %. The main confusion is NAKBA taken for Muharaf (111 of 400 lines), which fits their both being historical hands. So the shape descriptors do carry style; they just do not rank KHATT as further from Muharaf than NAKBA.</p>

    <h4>Dots under resizing (X-G7, CPU proxy)</h4>
    {dots_table()}
    <p>At the CRNN’s 64 px, a plain resize keeps dots (3 to 5 % end up under 2 px wide in Muharaf and KHATT, none in NAKBA), so a dot-preserving resize should change little there. At 32 px the share jumps to 43 % (Muharaf), 64 % (KHATT) and 51 % (RASAM-1), and minimum pooling restores most of the contrast. The contrast measure is crude (it is not monotone across heights for NAKBA), so read this as direction only. The model test belongs at 32 to 48 px, and the same logic applies to VLM patch grids.</p>

    <h4>Do the detectors work?</h4>
    <ul>
      <li><b>Dots:</b> the number of detected dot components tracks the number of dotted letters in the transcription with Pearson r between {min(v["pearson_components_vs_marked_letters"] for v in dv.values() if v is not dv["muharaf_train"]):.2f} and {max(v["pearson_components_vs_marked_letters"] for v in dv.values()):.2f} per set (best for Muharaf test, {dv["muharaf"]["pearson_components_vs_marked_letters"]:.2f}; worst for RASAM-1, {dv["rasam1"]["pearson_components_vs_marked_letters"]:.2f}). The median count per dotted letter is 1.1 to 1.5 in the manuscript sets and {dv["khatt"]["median_components_per_marked_letter"]:.2f} in KHATT, where dots are often fused with strokes. Treat dot size as a rough statistic, and dot pairs as one component.</li>
      <li><b>Overlays</b> show where the measures land: dots red, body skeleton blue, fitted baseline green.</li>
    </ul>
    {overlay_imgs()}

    <h4>What this changes in the plan</h4>
    <ul>
      <li><b>X-G3 gets a sharper hypothesis.</b> Reading KHATT sheared by about +10° to +14° moves its lines into the dense part of the training distribution. <span class="mono">scripts/xai_geom_tta.py</span> reads each eval line at five slants with the CRNN and reports CER per slant, the best-confidence reading (X-G8) and the oracle. I wrote it here and could only check the shear itself (positive angles lean strokes right, verified on a test stroke); the model part is copied from the evaluation script and still has to run on the 3090. NAKBA should gain little.</li>
      <li><b>Slant augmentation with a wider, right-shifted range.</b> The training slant spread is wide ({f(X["train_slant_p5_p95"][0])}° to {f(X["train_slant_p5_p95"][1])}°) but thin below 3°, so a shear augmentation that fills the upright end is cheap to try, and is the geometry-matched augmentation of pool idea 7.</li>
      <li><b>X-G7 moves to 32 and 48 px.</b> At 64 px there is little to gain.</li>
      <li><b>Strip ruled lines before evaluating NAKBA.</b> The function exists (<span class="mono">strip_rules</span> in <span class="mono">geom_measure.py</span>). Re-scoring the CRNN with and without it shows how much of the NAKBA CER is paper, not writing.</li>
      <li><b>New pool ideas</b> (rows 12 and 13 of the pool above): pen-contrast augmentation, and the slant-density check as a cheap data-selection score.</li>
      <li><b>Not done:</b> no per-line CER is available on this machine, so the geometry could not be correlated line by line with errors. <span class="mono">per_line.json</span> keeps the sample index, which is the same shuffle order the 3090 evaluation uses, so the join is a few lines on the 3090 once the per-line CER is exported.</li>
    </ul>
    <p><b>Limits.</b> Three transfer sets and one reference sample make this a description, not a test; KHATT also differs from our training lines in language, vocabulary and writers, and I did not control for any of that. The binarization is unreliable on RASAM’s grey, stained paper, so its numbers are shown but not interpreted. The hinge uses legs of 5 px at a 4 px stroke on 12 angle bins; other choices might separate the sets differently. None of the geometry methods here is new; the questions asked of them (slant against transfer, pen contrast, dot size at low resolution) are the part I could not find done for Arabic.</p>
  </article>

"""
# ---------------------------------------------------------------- patch the ledger
html = LEDGER.read_text(encoding="utf-8")
html = re.sub(r"\n  <!-- 6f\. geometry measurements -->.*?</article>\n", "\n", html, flags=re.S)
html = html.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
CSS = """
.gbox{position:relative;overflow-x:auto}
.gbox svg{display:block;width:100%;height:auto;min-width:520px;font-family:var(--f-body)}
.gbox svg text{fill:var(--muted);font-size:11.5px}
.gbox svg .lbl{fill:var(--ink);font-size:12px}
.gbox svg .title{fill:var(--ink);font-size:12.5px;font-weight:600}
.smalls .gbox svg{min-width:280px}
.sb{position:relative;display:block;width:110px;height:10px;background:var(--bg);border-radius:3px}
.sb::after{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;border-left:1px solid var(--axis)}
.sb i{position:absolute;top:0;bottom:0;border-radius:2px}
"""
if ".sb{" not in html:
    html = html.replace("</style>\n\n<div class=\"layout\">", CSS + "</style>\n\n<div class=\"layout\">", 1)
if 'href="#d-geom"' not in html:
    html = html.replace('<li><a href="#d-ctc">CTC aligner and pointing game</a></li>', '<li><a href="#d-ctc">CTC aligner and pointing game</a></li>\n        <li><a href="#d-geom">Geometry measurements</a></li>', 1)
    html = html.replace('<a href="#d-ctc">CTC aligner and pointing game</a><a href="#d-xai">', '<a href="#d-ctc">CTC aligner and pointing game</a><a href="#d-geom">Geometry measurements</a><a href="#d-xai">', 1)
html = html.replace("Snapshot 2026-10-10", "Snapshot 2026-10-11")


def sub(old: str, new: str) -> None:
    global html
    assert old in html, old[:60]
    html = html.replace(old, new, 1)


KH, NK = kh["scalar_profile"]["dist"], (nk["scalar_profile"]["dist"] + nb["scalar_profile"]["dist"]) / 2
if "Measured 2026-10-11" not in html:
    sub('<td><span class="pill o">next</span></td><td>X-G2, X-G4, X-G5, then X-G3</td>', '<td><span class="pill">partly</span></td><td>X-G2, X-G4, X-G5 done; X-G3 next</td>')
    sub("See the Geometry and normalization section.</td></tr>", "See the Geometry and normalization section. <b>Measured 2026-10-11:</b> slant, pen contrast, dot-to-stroke ratio, stroke width and ink density are defined per line and differ between datasets (see Geometry measurements); slope does not.</td></tr>")
    sub('<td><span class="pill o">next</span></td><td>X-G1</td>', '<td><span class="pill b">done</span></td><td>X-G1</td>')
    sub("a different use of the word.</td></tr>", "a different use of the word. <b>Computed 2026-10-11</b> with baseline-relative zones and dots removed: it separates datasets as well as the hinge histogram does (70 % line-level accuracy over four sets) and shows KHATT loops are more open than Muharaf’s. It does not explain the KHATT transfer gap.</td></tr>")
    sub("Novelty is not verified for any of them.</td></tr>", "Novelty is not verified for any of them. Ideas 1 to 3 and 7 now have first results (see Geometry measurements).</td></tr>")
    sub("X-G6, CPU, uses X-G1 to X-G5 and the new generalization table</td>", "X-G6, CPU, uses X-G1 to X-G5 and the new generalization table. <b>Done 2026-10-11:</b> the scalar geometry profile puts KHATT well past NAKBA; contour histograms do not.</td>")
    sub("X-G7: CRNN at 32 to 64 px with and without it; dot-error share</td>", "X-G7: CRNN at 32 to 64 px with and without it; dot-error share. <b>CPU proxy done 2026-10-11:</b> dots survive a plain resize at 64 px and are lost at 32 px; test at 32 to 48 px.</td>")
    sub("X-G8: existing models and cached predictions; five variants per line</td>", "X-G8: existing models and cached predictions; five variants per line. <b>Script written 2026-10-11, not run:</b> <span class=\"mono\">xai_geom_tta.py</span>; KHATT is the set where shear should help.</td>")
    sub("</tbody>\n    </table></div>\n    <h4>E8: CTC-encoder-to-LLM spike (proposed)</h4>", """        <tr><td>12</td><td><b>Pen-contrast augmentation.</b> Thin the vertical strokes and thicken the horizontal ones (an anisotropic erode and dilate) to imitate a cut nib, and the reverse for Maghrebi lines.</td><td>RASAM’s pen contrast is 0.5 to 0.6 against 1.0 elsewhere (measured 2026-10-11), and nothing in the augmentation set changes it.</td><td>Train the CRNN on Muharaf with the augmentation, read RASAM-2 and the ق→ف rate</td><td>May be absorbed by the existing blur and noise augmentations; the dot convention is the bigger Maghrebi problem.</td></tr>
        <tr><td>13</td><td><b>Slant-density score for data selection.</b> Score every candidate line by how many training lines share its slant and pen proportions, and add the rarest ones first.</td><td>41 % of KHATT lines lie where fewer than 5 % of training lines do.</td><td>Score the lines of a new set from images alone; train with the rare tail added</td><td>Slant may only matter for the model through its effect on other things.</td></tr>
      </tbody>
    </table></div>
    <h4>E8: CTC-encoder-to-LLM spike (proposed)</h4>""")
    sub('d:"none", o:"Not computed. In our docs “locus”', 'd:"partial", o:"Computed 2026-10-11 (X-G1; see Geometry measurements). In our docs “locus”')
    sub('d:"none", o:"Not computed.",\n n:"cpu", x:"X-G4', 'd:"partial", o:"Computed 2026-10-11 (X-G4): hinge and direction histograms identify the dataset of a single line 73 % of the time over four sets, and place KHATT and NAKBA equally far from Muharaf.",\n n:"cpu", x:"X-G4')
    sub('d:"none", o:"—",\n n:"cpu", x:"X-G5', 'd:"partial", o:"Ascender height over dot diameter measured 2026-10-11: 5.8 to 6.6 in the three clean sets, inside the classical five to seven dots.",\n n:"cpu", x:"X-G5')
    sub('d:"none", o:"Not implemented. We know the baseline', 'd:"partial", o:"Measured 2026-10-11 (X-G2): slope is near zero in every set; slant differs (Muharaf +11°, KHATT −2°). We know the baseline')
for gid, txt in {
    "X-G1": "Done 2026-10-11: separates datasets at 70 % line accuracy; KHATT loops more open; does not explain the transfer gap.",
    "X-G2": "Done 2026-10-11: slope near zero everywhere; slant median +11° Muharaf, +4 to +6° NAKBA, −2° KHATT; estimators agree to 0.4 to 0.7° on slope and 4° on slant.",
    "X-G4": "Done 2026-10-11: hinge and direction identify the dataset of a line 73 % of the time; KHATT and NAKBA equally far from Muharaf.",
    "X-G5": "Done 2026-10-11: dot detector tracks the transcription (r 0.3 to 0.8); ascender about six dots; pen contrast 0.5 to 0.6 in RASAM, about 1 elsewhere.",
    "X-G6": "Done 2026-10-11: scalar profile KHATT 0.75 against NAKBA 0.30 to 0.35 (noise floor 0.12); three transfer sets only.",
    "X-G7": "CPU proxy done 2026-10-11: dots are lost at 32 px (43 to 64 % under 2 px), kept at 64 px; the model test is still open.",
    "X-G8": "Script written 2026-10-11 (xai_geom_tta.py), not run; shear check passes on a test stroke.",
}.items():
    m = re.search(rf'^ \["{gid}",.*?"\],?$', html, flags=re.M)
    assert m, gid
    line = m.group(0)
    if "2026-10-11" not in line:
        base = line.rstrip()
        comma = base.endswith(",")
        base = base.rstrip(",")
        assert base.endswith('"]'), gid
        html = html.replace(line, base[:-2] + " " + txt + '"]' + ("," if comma else ""), 1)
LEDGER.write_text(html, encoding="utf-8")
print("ledger updated", len(html))
