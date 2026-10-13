"""Ledger block "Per-line and per-letter evidence" from the analysis bundle (plan 2026-10-13, §3.7).

    .geomvenv/bin/python scripts/build_evidence_section.py [--a results/analysis/A] [--b results/analysis/B] [--d results/analysis/D]

Reads analyze_lines.py, analyze_letters.py and run_detectors.py outputs plus docs/research/CORRECTIONS.md, and replaces
the <!-- 6l. evidence --> article in docs/research/paper-ledger.html. Images for the hard-core gallery come from the
panel pickle. Bucket bars use a validated 5-step ordinal blue ramp (dataviz validate_palette.js --ordinal, light and dark).
"""
from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from change_gallery import b64, marked  # noqa: E402

LEDGER = W / "docs/research/paper-ledger.html"
GROUPS = [("exact", ["exact"]), ("≤ 5 % (≥ 95 % right)", ["≤2%", "≤5%"]), ("≤ 25 %", ["≤10%", "≤25%"]), ("≤ 50 %", ["≤50%"]), ("> 50 % (failing)", ["≤100%", ">100%"])]
pc = lambda x: f"{100 * x:.1f}".replace("-", "−")  # noqa: E731
sg = lambda x: f"{100 * x:+.2f}".replace("-", "−")  # noqa: E731
STYLE = """<style>
:root{--q1:#86b6ef;--q2:#5598e7;--q3:#2a78d6;--q4:#1c5cab;--q5:#0d366b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--q1:#184f95;--q2:#256abf;--q3:#3987e5;--q4:#6da7ec;--q5:#b7d3f6}}
:root[data-theme="dark"]{--q1:#184f95;--q2:#256abf;--q3:#3987e5;--q4:#6da7ec;--q5:#b7d3f6}
.bk{display:flex;flex-direction:column;gap:6px}.bk .r{display:grid;grid-template-columns:minmax(150px,260px) minmax(0,1fr) 52px;gap:10px;align-items:center;font-size:13px}
.bk .bar{display:flex;height:14px;gap:2px}.bk .bar span{display:block;height:100%}.bk .bar span:first-child{border-radius:4px 0 0 4px}.bk .bar span:last-child{border-radius:0 4px 4px 0}
.bk .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.bkl{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--muted)}.bkl i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-2px}
#d-evidence{min-width:0}#d-evidence>*{min-width:0;max-width:100%}#d-evidence .qwrap{max-width:100%}
@media (max-width:720px){.bk .r{grid-template-columns:110px minmax(0,1fr) 44px}}
</style>"""


def full_b64(img) -> str:
    """The whole line image (change_gallery.b64 keeps only the right-most 640 px, which hides the end of long lines)."""
    import base64
    import io

    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def table(head, rows):
    return '<div class="qwrap"><table class="dt"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(r) + "</tr>" for r in rows) + "</tbody></table></div>"


def md_table(md: str) -> str:
    rows = [ln for ln in md.splitlines() if ln.startswith("|") and not re.match(r"^\|[-| ]+\|$", ln)]
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in rows]
    fmt = lambda s: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", re.sub(r"`([^`]+)`", r'<span class="mono">\1</span>', html.escape(s)))  # noqa: E731
    return table([fmt(c) for c in cells[0]], [[f"<td>{fmt(c)}</td>" for c in r] for r in cells[1:]])


def bucket_chart(A1: dict, specs: list[tuple[str, str]]) -> str:
    legend = '<div class="bkl">' + "".join(f'<span><i style="background:var(--q{i + 1})"></i>{html.escape(g)}</span>' for i, (g, _) in enumerate(GROUPS)) + "</div>"
    rows, trows = [], []
    for label, key in specs:
        if key not in A1:
            continue
        b = A1[key]["crnn_native"]["buckets"] if key.startswith("crnn") else A1[key]["no_diacritics"]["buckets"]
        cer = A1[key]["crnn_native" if key.startswith("crnn") else "no_diacritics"]["micro"]
        shares = [sum(b[x] for x in members) for _, members in GROUPS]
        segs = "".join(f'<span style="width:{100 * s:.2f}%;background:var(--q{i + 1})" title="{html.escape(GROUPS[i][0])}: {pc(s)} % of lines"></span>' for i, s in enumerate(shares) if s > 0)
        rows.append(f'<div class="r"><span>{html.escape(label)}</span><div class="bar" role="img" aria-label="{html.escape(label)}: ' + ", ".join(f"{g} {pc(s)} %" for (g, _), s in zip(GROUPS, shares)) + f'">{segs}</div><span class="v">{pc(cer)} %</span></div>')
        trows.append([f"<td>{html.escape(label)}</td>", f'<td class="n">{A1[key]["crnn_native" if key.startswith("crnn") else "no_diacritics"]["lines"]}</td>'] + [f'<td class="n">{pc(b[k])}</td>' for k in b])
    head = ["Run", "Lines"] + list(next(iter(A1.values()))["crnn_native"]["buckets"])
    return (legend + '<div class="bk">' + "".join(rows) + "</div>"
            + '<details class="tbl"><summary>Show the table (all eight buckets, % of lines)</summary>' + table(head, trows) + "</details>")


def patch_key_findings(h: str, A1: dict, A2: dict) -> str:
    """Rewrite the impact cells of the key-findings rows that the full panel changes (all 200 KHATT lines, full NAKBA)."""
    def pr(name, ds):
        v = A2.get(f"{name}|{ds}")
        return (v["crnn_native"], v["lines"]) if v and v.get("pack") == "panel-2026-10-13" else (None, None)

    crop, n = pr("ink crop (test time)", "kitab_khatt")
    if crop:
        runs = [k for k in A1 if k.startswith("crnn/") and k.endswith("|panel-2026-10-13|kitab_khatt") and "@inkcrop|" in k and "std" not in k]
        lo = min(A1[k]["crnn_native"]["micro"] for k in runs)
        hi = max(A1[k]["crnn_native"]["micro"] for k in runs)
        bn = crop["beyond_noise"]
        cell = (f"<td><b>Largest gain found (fixed-height models only; VLMs unaffected):</b> KHATT, all {n} lines, {pc(crop['cer_A'])} → {pc(crop['cer_B'])} % "
                f"({sg(crop['delta'])} points [{sg(crop['ci95'][0])}, {sg(crop['ci95'][1])}]; {bn['better']} lines better and {bn['worse']} worse beyond seed noise) "
                f"with no retraining; every one of {len(runs)} CRNNs lands at {pc(lo)}–{pc(hi)} % cropped; NAKBA unchanged. "
                "(Earlier 26.3 → 23.2 % was on 183 lines that excluded the 17 worst crops, CORRECTIONS C3.)</td>")
        h = re.sub(r"<td><b>Largest gain found \(fixed-height models only; VLMs unaffected\):</b>.*?</td>", lambda _: cell, h, count=1, flags=re.S)
    aug, _ = pr("slant/stroke augmentation", "nakba_lines")
    augb, _ = pr("slant/stroke augmentation", "nakba_blind")
    augk, _ = pr("slant/stroke augmentation", "kitab_khatt")
    if aug and augb and augk:
        cell = (f"<td><b>Hurts on unseen sets</b> (full panel): NAKBA test {sg(aug['delta'])} [{sg(aug['ci95'][0])}, {sg(aug['ci95'][1])}], "
                f"blind {sg(augb['delta'])} [{sg(augb['ci95'][0])}, {sg(augb['ci95'][1])}], KHATT {sg(augk['delta'])} points</td>")
        h = re.sub(r"<td>None on the unseen sets \(KHATT 26\.4 → 26\.9 %, NAKBA 13\.8 → 14\.4 %\)</td>|<td><b>Hurts on unseen sets</b>.*?</td>", lambda _: cell, h, count=1, flags=re.S)
    return h


def extra_sections(a) -> str:
    """Sections 7–11: image characteristics per set, joint validity model, TTA gate, PAW-gap intervals, sound vs look."""
    import numpy as np
    import pyarrow.parquet as pq
    out = []
    fp = W / "results/store/features.parquet"
    if fp.exists():
        F = [f for f in pq.read_table(fp).to_pylist() if f["ok"]]
        rows = []
        for ds, name in (("muharaf_train", "Muharaf train (reference)"), ("kitab_khatt", "KHATT"), ("nakba_lines", "NAKBA test"), ("nakba_blind", "NAKBA blind"),
                         ("muharaf_lines", "Muharaf test"), ("rasam1", "RASAM-1"), ("rasam2", "RASAM-2")):
            rs = [f for f in F if f["dataset"] == ds]
            if not rs:
                continue
            med = lambda k: np.nanmedian([f[k] if f[k] is not None else np.nan for f in rs])  # noqa: E731
            share = lambda k: 100 * np.mean([bool(f[k]) for f in rs])  # noqa: E731
            rows.append([f"<td>{name}</td>", f'<td class="n">{len(rs)}</td>', f'<td class="n">{med("slant_deg"):+.1f}°</td>', f'<td class="n">{med("stroke_px64"):.2f}</td>',
                         f'<td class="n">{med("pen_contrast"):.2f}</td>', f'<td class="n">{med("ink_px_per_char"):.1f}</td>',
                         f'<td class="n">{share("slant_outlier"):.0f} %</td>', f'<td class="n">{share("stroke_outlier"):.0f} %</td>', f'<td class="n">{share("pen_outlier"):.0f} %</td>',
                         f'<td class="n">{share("ruled"):.0f} %</td>', f'<td class="n">{share("low_res"):.0f} %</td>'])
        out.append("<h4>7 · Image characteristics of every line, against the CRNN's training lines</h4><p>Measured on the native image of all 7,406 panel lines "
                   "and 3,000 Muharaf training lines (<span class=\"mono\">store_features.py</span>, the X-G1–X-G5 measures). Outlier = outside the training "
                   "lines' 2.5–97.5 percentile band (5 % by construction for a set like the training data). KHATT stands out in slant and stroke width at the "
                   "64 px input, the Maghrebi RASAM sets in pen contrast and pixels per character, NAKBA in ruled lines.</p>"
                   + table(["Set", "Lines", "Median slant", "Stroke px at 64", "Pen contrast", "Px per char", "Slant outlier", "Stroke outlier", "Pen outlier", "Ruled", "Low res"], rows))
    vp = W / "results/analysis/V/validity.json"
    if vp.exists():
        V = json.loads(vp.read_text(encoding="utf-8"))
        tg = list(V["targets"])
        rows = []
        for f in V["features"]:
            cells = []
            for t in tg:
                b, lo, hi = V["targets"][t]["coef"][f]
                sig = lo > 0 or hi < 0
                cells.append(f'<td class="n">{"<b>" if sig else ""}{sg(b)}{"</b>" if sig else ""} <span class="pill">[{sg(lo)}, {sg(hi)}]</span></td>')
            rows.append([f"<td>{html.escape(f)}</td>"] + cells + [f"<td>{'<b>yes</b>' if V['survival'][f]['survives'] else ''}</td>"])
        fit = "; ".join(f"{t}: in-sample ρ {v['spearman_in_sample']:.2f}, leave-one-set-out ρ {min(v['lodo_spearman'].values()):.2f} to {max(v['lodo_spearman'].values()):.2f}" for t, v in V["targets"].items())
        out.append("<h4>8 · Which characteristics predict a line's error, jointly and across families</h4><p>Ridge regression of line CER (no-diacritics rung, "
                   "capped at 1) on standardised characteristics; cells are points of line CER per standard deviation, with a line-bootstrap 95 % interval; bold = "
                   "interval excludes 0. “Survives” = significant with the same sign in at least two system families (<span class=\"mono\">analyze_validity.py</span>). "
                   "Loose crops and slant hurt the CRNN but not the fine-tuned VLMs; pen contrast hurts only the fine-tuned VLMs (the Maghrebi sets). What hurts "
                   "every family: few pixels per character, short lines, unseen words; long words help everyone. PAWs per word looked strong until letters per word "
                   f"was added as a control. The model explains little line-to-line variance and transfers poorly across sets ({html.escape(fit)}): most of a line's "
                   "difficulty is not in these characteristics.</p>"
                   + table(["Characteristic"] + [html.escape(t) for t in tg] + ["Survives"], rows))
    gp = W / "results/analysis/V/tta_gate.json"
    if gp.exists():
        G = json.loads(gp.read_text())
        rows = []
        for run, v in G.items():
            for ds, name in (("kitab_khatt", "KHATT"), ("nakba_lines", "NAKBA test"), ("nakba_blind", "NAKBA blind")):
                x = v[ds]
                rows.append([f"<td>{run}</td>", f"<td>{name}</td>", f'<td class="n">{x["lines"]}</td>', f'<td class="n">{pc(x["upright"])} %</td>',
                             f'<td class="n">{pc(x["tta_all"])} %</td>', f'<td class="n"><b>{pc(x["gated"])} %</b></td>', f'<td class="n">{pc(x["share_gated_on"])} %</td>'])
            rows.append([f"<td colspan=7>{run}: threshold {v['threshold_deg']:.0f}° from the training median; KHATT gain kept {pc(v['khatt_gain_kept'])} %, worst NAKBA change {sg(v['nakba_cost'])} points → <b>{'keep' if v['keep_gate'] else 'reject'}</b></td>"])
        out.append("<h4>9 · Slant TTA only where the slant is unusual</h4><p>Offline from the stored readings: the confidence-selected shear is used only on lines whose "
                   "measured slant is at least T degrees from the training median; T chosen on one half (KHATT half A, even NAKBA ids), reported on the other. "
                   "The gate removes the NAKBA cost; whether it keeps enough of the KHATT gain depends on the model. Held-out KHATT is 91 lines of the old pack; "
                   "the full-panel TTA with every shear stored is queued on the 3090.</p>"
                   + table(["Model", "Set (held-out half)", "Lines", "Upright", "TTA everywhere", "Gated", "Lines given TTA"], rows))
    pp = W / "results/analysis/B/paw_gaps.json"
    if pp.exists():
        P = json.loads(pp.read_text(encoding="utf-8"))
        s = P["summary"]
        top = sorted(P["pairs"].items(), key=lambda t: -t[1]["ratio"])
        pick = top[:6] + top[-4:]
        rows = [[f"<td>{html.escape(k.split('|')[0])}</td>", f"<td>{k.split('|')[1]}</td>", f'<td class="n">{v["spaces_inserted"]}</td>',
                 f'<td class="n"><b>{pc(v["at_paw_gap"])} %</b> [{pc(v["ci95"][0])}, {pc(v["ci95"][1])}]</td>', f'<td class="n">{pc(v["chance"])} %</td>', f'<td class="n">×{v["ratio"]:.2f}</td>'] for k, v in pick]
        out.append(f"<h4>10 · Word splits at PAW gaps, with intervals</h4><p>Above chance with the interval clear of it in <b>{s['above_chance']} of {s['pairs']}</b> system × set "
                   "pairs (lines a system trained on and whole-line failures excluded): every family splits words where the pen lifts, 1.2 to 2.9 times as often as "
                   "chance. The six strongest and four weakest pairs are shown (<span class=\"mono\">analyze_paw_gaps.py</span>).</p>"
                   + table(["System", "Set", "Spaces inserted in words", "At a PAW gap [95 %]", "Chance", "Ratio"], rows))
    hp = W / "results/analysis/B/phonetic.json"
    if hp.exists():
        H = json.loads(hp.read_text(encoding="utf-8"))
        fam = defaultdict(list)
        for k, v in H["runs"].items():
            run = k.split("|")[0]
            f_ = "CRNN" if run.startswith("crnn/") else "CRNN → LLM (E8)" if run.startswith("crnn+llm") else "VLM fine-tuned" if ("-ft" in run or "baseer" in run) else "VLM / OCR base"
            fam[f_].append(v)
        rows = [[f"<td>{f_}</td>", f'<td class="n">{len(vs)}</td>'] + [f'<td class="n">{pc(np.median([v[c] for v in vs]))} %</td>' for c in ("look-only", "sound-only", "both", "other")] for f_, vs in fam.items()]
        rows.append(["<td>chance (no preference)</td>", "<td></td>"] + [f'<td class="n">{pc(H["chance"][c])} %</td>' for c in ("look-only", "sound-only", "both", "other")])
        out.append("<h4>11 · Do systems confuse letters that sound alike, or that look alike?</h4><p>Every letter substitution classed as look-only (same dotless "
                   "skeleton in that position), sound-only (same place of articulation, makhraj, or a dialect merger such as ث/س, ذ/ز, ظ/ض, ق/ء, with a different "
                   "skeleton), both, or other; median share per family (<span class=\"mono\">analyze_phonetic_confusions.py</span>). Look-alike confusions run at "
                   "4–15× chance in every family; sound-alike ones at chance, even for the LLM decoder. Recognition errors are visual; articulation explains none of them.</p>"
                   + table(["Family", "Runs", "Look-only", "Sound-only", "Both", "Other"], rows))
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=W / "results/analysis/A")
    ap.add_argument("--b", type=Path, default=W / "results/analysis/B")
    ap.add_argument("--d", type=Path, default=W / "results/analysis/D")
    ap.add_argument("--panel", type=Path, default=W / "results/panel/panel.pkl")
    a = ap.parse_args()
    A1 = json.loads((a.a / "summary.json").read_text(encoding="utf-8"))
    A2 = json.loads((a.a / "pairs.json").read_text(encoding="utf-8"))
    MX = json.loads((a.a / "matrix_kitab_khatt.json").read_text(encoding="utf-8"))
    B = json.loads((a.b / "letters.json").read_text(encoding="utf-8"))
    BP = json.loads((a.b / "pairs_letters.json").read_text(encoding="utf-8"))
    D = json.loads((a.d / "detectors.json").read_text(encoding="utf-8"))
    corr = md_table((W / "docs/research/CORRECTIONS.md").read_text(encoding="utf-8"))

    def k1(spec, pack, ds="kitab_khatt"):
        return f"{spec}|{pack}|{ds}"
    best_pack = lambda spec, ds="kitab_khatt": next((k for k in (k1(spec, p, ds) for p in ("panel-2026-10-13", "inkcrop_eval", "eval_data", "inkcrop_eval/fp16-bs1", "bench")) if k in A1), "")  # noqa: E731
    specs = [("CRNN, lines as stored", best_pack("crnn/base@none")), ("CRNN, ink-cropped", best_pack("crnn/base@inkcrop")),
             ("Combo CRNN, ink-cropped", best_pack("crnn/combo@inkcrop")), ("Combo CRNN, cropped + slant TTA", best_pack("crnn/combo@inkcrop+tta_conf")),
             ("E8 (CRNN → Qwen3-0.6B)", k1("crnn+llm/e8-qwen3-0.6b@none", "eval_data+heldout")),
             ("Qwen3-VL-8B base", k1("vlm/qwen3-vl-8b-bf16@none", "bench")), ("Qwen3-VL-8B fine-tuned", k1("vlm/suite-v1-qwen3vl8b-ft-bf16@none", "bench")),
             ("Fanar fine-tuned", k1("vlm/suite-v1-fanar2-oryx-ft-bf16@none", "bench")), ("Gemini Flash", k1("vlm/gemini-flash@none", "bench"))]
    buckets = ("<h4>1 · How the lines are distributed, not just the mean</h4><p>Share of KHATT lines by line CER (CRNN rows at the CRNN-native rung, "
               "VLM rows at the no-diacritics rung; the number on the right is the corpus CER). The crop moves lines out of the failing bucket; "
               "no system reads a KHATT line exactly more than a few times.</p>" + bucket_chart(A1, specs))

    prow = []
    for k, v in A2.items():
        name, ds = k.split("|")
        if ds not in ("kitab_khatt", "nakba_lines"):
            continue
        r = v["crnn_native"]
        bn = r.get("beyond_noise", {})
        src = max(r["net_change_by_start_bucket"].items(), key=lambda t: abs(t[1]))[0] if r["net_change_by_start_bucket"] else "—"
        prow.append([f"<td>{html.escape(name)}</td>", f"<td>{'KHATT' if ds == 'kitab_khatt' else 'NAKBA test'}</td>", f'<td class="n">{v["lines"]}</td>',
                     f'<td class="n"><b>{sg(r["delta"])}</b> [{sg(r["ci95"][0])}, {sg(r["ci95"][1])}]</td>',
                     f'<td class="n">{bn.get("better", "—")} / {bn.get("worse", "—")}</td>', f"<td>{html.escape(src)}</td>"])
    pairs = ("<h4>2 · Every comparison against the seed-noise yardstick</h4><p>“Beyond noise” = the line's CER moved by more than the 90th "
             "percentile of the moves between two identical trainings (base vs replicate), within the same line-length tercile. "
             "By construction the replicate itself shows about 10 % of lines on each side; an intervention is believable when its "
             "better / worse counts are lopsided, not when its mean moves.</p>"
             + table(["Comparison", "Set", "Lines", "ΔCER, points [95 %]", "Lines better / worse beyond noise", "Net change comes mostly from lines starting at"], prow))

    lab = MX["label_counts"]
    hard = MX["labels"].get("hard_core", [])
    rows = {r["id"]: r for r in pickle.load(open(a.panel, "rb"))["rows"] if r["dataset"] == "kitab_khatt" and r["id"] in hard} if a.panel.exists() else {}
    cards = []
    for k in hard[:10]:
        if k not in rows:
            continue
        cell = MX["cells"][k]
        best = min(cell, key=cell.get)
        cards.append(f'<div class="gcard worse"><img src="{full_b64(rows[k]["img"])}" alt="KHATT line {k}"><div class="gtxt">'
                     f'<div class="r"><span class="k">truth</span><bdi>{html.escape(rows[k]["label_raw"])}</bdi></div>'
                     f'<div class="r"><span class="k">best of {len(cell)} systems</span><span class="e">{pc(cell[best])} %</span><span class="id">{html.escape(best)}</span></div>'
                     f'<div class="r"><span class="k">median CER</span><span class="e">{pc(MX["consensus_difficulty"][k])} %</span><span class="id">{html.escape(k)} · ink extent {rows[k]["ink_extent"]:.2f}</span></div></div></div>')
    matrix = (f"<h4>3 · Which lines nobody reads: the line × system matrix (KHATT)</h4><p>{MX['lines']} lines read by at least 80 % of "
              f"{len(MX['columns'])} systems ({len(MX['crnn_columns'])} CRNN readings, {len(MX['vlm_columns'])} VLM runs; AIN, PaddleOCR and TrOCR "
              f"left out, see corrections C5–C6). <b>{lab.get('hard_core', 0)}</b> lines are hard for every system (line CER above 25 % "
              f"everywhere), {lab.get('crnn_only_failure', 0)} fail only for the CRNNs and {lab.get('vlm_only_failure', 0)} only for the VLMs. "
              f"The CRNN's line difficulty correlates only weakly with the median VLM's (Spearman {MX['spearman_crnn_vs_median_vlm']:.2f}): "
              "most hard lines are hard for one family, not for both. The hard core is the first queue for the label review (R4.1).</p>"
              + ('<details class="gal"><summary>Show the hard-core lines</summary><div class="gl">' + "".join(cards) + "</div></details>" if cards else ""))

    det_rows = []
    for spec, name in (("crnn/base@none", "CRNN"), ("vlm/qwen3-vl-8b-bf16@none", "Qwen3-VL-8B base"), ("vlm/suite-v1-qwen3vl8b-ft-bf16@none", "Qwen3-VL-8B fine-tuned")):
        for ds in ("kitab_khatt", "nakba_lines"):
            v = D["runs"].get(f"{spec}|{ds}")
            if not v:
                continue
            top = sorted(((d, x.get("excess_edits_explained", 0), x.get("share_lines", 0)) for d, x in v["detectors"].items()), key=lambda t: -t[1])[:3]
            det_rows.append([f"<td>{name}</td>", f"<td>{'KHATT' if ds == 'kitab_khatt' else 'NAKBA test'}</td>", f'<td class="n">{pc(v["cer"])} %</td>',
                             "<td>" + "<br>".join(f"{html.escape(d)}: {pc(sh)} % of lines, {pc(x)} % of edits in excess" for d, x, sh in top) + "</td>",
                             f'<td class="n">{v["residual_lines"]} ({pc(v["residual_share_of_top20"])} %)</td>'])
    detectors = ("<h4>4 · Pattern detectors: what the errors are made of</h4><p>Each detector is a documented rule on the image, the text or the "
                 "reading (<span class=\"mono\">run_detectors.py</span>, 17 so far). “Edits in excess” is how much of a run's error sits on the flagged lines "
                 "beyond what those lines would cost at the run's normal error rate. The residual is the worst fifth of lines that no detector "
                 "explains, the feed for new detectors. The CRNN's KHATT errors are dominated by loose crops, truncation and whole-line failures; "
                 "the VLMs' by diacritics they add and fluent rewrites.</p>"
                 + table(["System", "Set", "CER", "Top detectors", "Unexplained worst-fifth lines"], det_rows))

    paw_rows = []
    for k, v in sorted(B.items()):
        spec, ds = k.split("|")
        if v["lines"] < 150 or v.get("flagged") or ds not in ("kitab_khatt", "nakba_lines", "rasam1", "muharaf_lines") or not v["space_in_word_per_100_gaps"]:
            continue
        if not (spec.startswith("vlm/") or spec in ("crnn/base@none", "crnn/base@inkcrop")):
            continue
        paw_rows.append([f"<td>{html.escape(spec.replace('@none', '').replace('@inkcrop', ', ink-cropped'))}</td>", f"<td>{ds}</td>", f'<td class="n">{v["space_in_word_per_100_gaps"]:.2f}</td>',
                         f'<td class="n"><b>{pc(v["space_at_paw_gap_share"])} %</b></td>', f'<td class="n">{pc(v["paw_gap_chance"])} %</td>'])
    above = sum(1 for r in paw_rows if float(re.sub(r"<.*?>", "", r[3]).replace("%", "").replace("−", "-")) > float(re.sub(r"<.*?>", "", r[4]).replace("%", "").replace("−", "-")))
    paw = ("<h4>5 · Spaces inserted inside words fall at PAW gaps</h4><p>When a system splits a word with a space, does it do so at a gap "
           "between parts of the word (after a letter that cannot join forward) more often than chance? Chance is the share of "
           f"word-internal letter boundaries that are PAW gaps. Above chance in <b>{above} of {len(paw_rows)}</b> system × set rows.</p>"
           + '<details class="tbl"><summary>Show the table</summary>' + table(["System", "Set", "Spaces inserted per 100 word-internal gaps", "At a PAW gap", "Chance"], paw_rows) + "</details>")

    lrows = []
    for k, v in BP.items():
        name, ds = k.split("|")
        if ds != "kitab_khatt":
            continue
        top = [m for m in v["cells"] if m["excludes_0"]][:4]
        lrows.append([f"<td>{html.escape(name)}</td>", f'<td class="n">{v["n_cells_better_ci"]} / {v["n_cells_worse_ci"]}</td>',
                      "<td>" + ", ".join(f'<bdi class="ar">{m["cell"].split("|")[0]}</bdi> {m["cell"].split("|")[1]} {sg(m["delta"])}' for m in top) + "</td>"])
    letters = ("<h4>6 · Which letters move (KHATT)</h4><p>Letter × positional-form cells (≥ 30 occurrences) whose error changes with a line-bootstrap "
               "interval that excludes zero. The replicate row is the noise yardstick: one or two cells move by chance.</p>"
               + table(["Comparison", "Cells better / worse", "Largest moves with intervals clear of zero (points)"], lrows))

    extra = extra_sections(a)
    ARTICLE = f"""
  <!-- 6l. evidence -->
  <article class="dpart" id="d-evidence">
    <h3>Per-line and per-letter evidence; corrections (2026-10-13)</h3>
    {STYLE}
    <div class="verdict"><p><b>A new standard analysis for every experiment.</b> Every reading of every line by every system (CRNN runs, TTA, E8 and 128 VLM model × set pairs, {sum(1 for _ in A1)} run × set summaries) now sits in one per-line store, keyed by the dataset row, with every stored CRNN edit reproduced exactly. Each result gets a line-level view (buckets, shifts beyond seed noise, the line × system matrix), a letter-level view and pattern detectors. The problems found in earlier numbers are listed first.</p></div>
    <h4>Corrections</h4>
    {corr}
    {buckets}
    {pairs}
    {matrix}
    {detectors}
    {paw}
    {letters}
    {extra}
  </article>
"""
    h = LEDGER.read_text(encoding="utf-8")
    h = patch_key_findings(h, A1, A2)
    h = re.sub(r"\n  <!-- 6l\. evidence -->.*?</article>\n(?=\n  <!-- 6b)", "\n", h, flags=re.S)
    h = h.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
    if 'href="#d-evidence"' not in h:
        h = h.replace('<li><a href="#d-day4">Stacking, noise, E8 1.7B</a></li>', '<li><a href="#d-day4">Stacking, noise, E8 1.7B</a></li>\n        <li><a href="#d-evidence">Per-line evidence, corrections</a></li>', 1)
        h = h.replace('<a href="#d-day4">Stacking, noise, E8 1.7B</a><a href="#d-xai">', '<a href="#d-day4">Stacking, noise, E8 1.7B</a><a href="#d-evidence">Per-line evidence, corrections</a><a href="#d-xai">', 1)
    LEDGER.write_text(h, encoding="utf-8")
    print("ledger", len(h))


if __name__ == "__main__":
    main()
