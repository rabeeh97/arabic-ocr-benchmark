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
        cards.append(f'<div class="gcard worse"><img src="{b64(rows[k]["img"])}" alt="KHATT line {k}"><div class="gtxt">'
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
        paw_rows.append([f"<td>{html.escape(spec.split('@')[0])}</td>", f"<td>{ds}</td>", f'<td class="n">{v["space_in_word_per_100_gaps"]:.2f}</td>',
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

    ARTICLE = f"""
  <!-- 6l. evidence -->
  <article class="dpart" id="d-evidence">
    <h3>Per-line and per-letter evidence; corrections (2026-10-13)</h3>
    {STYLE}
    <div class="verdict"><p><b>A new standard analysis for every experiment.</b> Every reading of every line by every system (CRNN runs, TTA, E8 and 128 VLM model × set pairs, {sum(1 for _ in A1)} run × set summaries) now sits in one per-line store, keyed by the dataset row, with every stored CRNN edit reproduced exactly. Each result gets a line-level view (buckets, shifts beyond seed noise, the line × system matrix), a letter-level view and pattern detectors. Eight problems in earlier numbers are listed first.</p></div>
    <h4>Corrections</h4>
    {corr}
    {buckets}
    {pairs}
    {matrix}
    {detectors}
    {paw}
    {letters}
  </article>
"""
    h = LEDGER.read_text(encoding="utf-8")
    h = re.sub(r"\n  <!-- 6l\. evidence -->.*?</article>\n(?=\n  <!-- 6b)", "\n", h, flags=re.S)
    h = h.replace("\n  <!-- 6b. attribution spike -->", ARTICLE + "\n  <!-- 6b. attribution spike -->", 1)
    if 'href="#d-evidence"' not in h:
        h = h.replace('<li><a href="#d-day4">Stacking, noise, E8 1.7B</a></li>', '<li><a href="#d-day4">Stacking, noise, E8 1.7B</a></li>\n        <li><a href="#d-evidence">Per-line evidence, corrections</a></li>', 1)
        h = h.replace('<a href="#d-day4">Stacking, noise, E8 1.7B</a><a href="#d-xai">', '<a href="#d-day4">Stacking, noise, E8 1.7B</a><a href="#d-evidence">Per-line evidence, corrections</a><a href="#d-xai">', 1)
    LEDGER.write_text(h, encoding="utf-8")
    print("ledger", len(h))


if __name__ == "__main__":
    main()
