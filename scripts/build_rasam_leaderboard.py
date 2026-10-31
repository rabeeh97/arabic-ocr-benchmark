"""Build the local RASAM leaderboard page: docs/research/rasam_leaderboard.html (standalone, no network needed).

    PYTHONPATH=src python scripts/build_rasam_leaderboard.py

Four parts: (1) the strict board, systems scored on the exact official Hugging Face test lines (RASAM-1 734, RASAM-2 372, TariMa 229), whether we
ran them or the authors reported them on those lines; (2) published numbers on other splits, with why they are not comparable and why we could not
reproduce them; (3) our ideas and their measured effect; (4) why the scores differ and why papers made their own splits.
Numbers are micro CER in %, raw profile (diacritics kept) unless marked. External models we ran are re-scored from their prediction files in
results/rasam/external/ (<model>_<set>__test.json) when those exist, so the page picks them up after each run.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

EXT = Path("results/rasam/external")
OUT = Path("docs/research/rasam_leaderboard.html")
LINKS = {
    "AMIDDA card": "https://huggingface.co/calfa-ai/amidda1.0Line4B",
    "AMIDDA paper": "https://enc.hal.science/hal-05693582",
    "HAFITH": "https://huggingface.co/mdnaseif/hafith",
    "Phoenix": "https://arxiv.org/abs/2608.19385",
    "TariMa": "https://github.com/calfa-co/tarima",
    "HATFormer AraMS-28k": "https://huggingface.co/Archatext/hatformer-arams28k",
    "RASAM 2021": "https://halshs.archives-ouvertes.fr/halshs-03430697",
    "Lucas 2022": "https://arxiv.org/abs/2211.16147",
    "CHR 2024": "https://ceur-ws.org/Vol-3834/paper35.pdf",
}


def ext_cer(model: str) -> dict:
    """Micro CER per set from our run of an external model, or {} when it has not been run."""
    res = {}
    for s in ("r1", "r2", "tm"):
        f = EXT / f"{model}_{s}__test.json"
        if not f.exists():
            continue
        lines = json.loads(f.read_text())["lines"]
        full = {"r1": 734, "r2": 372, "tm": 229}[s]
        if len(lines) < full:
            res[s] = f"running ({len(lines)}/{full})"
            continue
        e = n = 0
        for l in lines:
            c = char_edit_counts(normalize(l["ref"], "raw").strip(), normalize(l["hyp"], "raw").strip())
            e += c["sub"] + c["del"] + c["ins"]
            n += c["ref_len"]
        res[s] = round(e / max(1, n) * 100, 2)
    return res


# ---- 1. the strict board: exact official test lines -------------------------------------------------------------------------------
# kind: ours | rescored (we ran their public model) | reported-same-lines (authors' number on these lines; not re-run by us)
hat = ext_cer("hatformer")
amidda = ext_cer("amidda")
STRICT = [
    {"sys": "Calfa CTC, one CRNN per corpus (AMIDDA paper baseline)", "kind": "reported-same-lines", "r1": 3.6, "r2": 4.9, "tm": 3.9,
     "src": "AMIDDA card", "note": "The AMIDDA test split contains exactly the official test lines (all 734 / 372 / 229 images identical by hash, our check). "
     "Not reproduced: weights not public; training data, line height and CER averaging not stated on the card (the paper PDF is behind a bot check)."},
    {"sys": "Ours: 2-model hypothesis ensemble (60-epoch EMA + pretrained) + char LM + AraGPT2", "kind": "ours", "r1": 3.72, "r2": 7.71,
     "tm": 5.67, "src": "", "note": "Union of both models' 10-best lists scored by both models' CTC, the in-domain + OpenITI char LM mix and AraGPT2; "
     "weights tuned on validation (design §22)."},
    {"sys": "Ours: char-CRNN, 60 epochs + char LM + AraGPT2 rescoring", "kind": "ours", "r1": 4.12, "r2": 8.92, "tm": 7.04,
     "src": "", "note": "RASAM-1 + repaired RASAM-2 + TariMa train, 96 px, 2.4 M params; LM and rescoring weights tuned on validation only. One seed (seed noise 0.1-0.2)."},
    {"sys": "Ours: same, 60 epochs + char LM", "kind": "ours", "r1": 4.33, "r2": 9.13, "tm": 7.35, "src": "", "note": ""},
    {"sys": "Ours: 30 epochs + char LM (3-seed mean)", "kind": "ours", "r1": 4.38, "r2": 9.30, "tm": 7.67, "src": "", "note": "seed spread about 0.1-0.2"},
    {"sys": "Ours: 30 epochs, greedy (3-seed mean)", "kind": "ours", "r1": 5.24, "r2": 11.79, "tm": 9.27, "src": "", "note": "no language model"},
    {"sys": "HAFITH (SigLIP2 + RoBERTa, 642 M)", "kind": "rescored", "r1": 6.27, "r2": 17.56, "tm": None, "src": "HAFITH",
     "note": "Contaminated on RASAM-1: 362 of the 734 test lines are in its train/val data; on the 372 clean lines it scores 12.34 % (ours 5.72 % with an older model). Never trained on RASAM-2."},
    {"sys": "Gemini 3 Flash, 25-shot in-context", "kind": "reported-same-lines", "r1": 16.5, "r2": 29.0, "tm": 25.6, "src": "AMIDDA card",
     "note": "Reported by Calfa on the same lines; not re-run (paid API)."},
    {"sys": "Phoenix (Kraken, 5 M)", "kind": "rescored", "r1": 23.1, "r2": 18.7, "tm": None, "src": "Phoenix",
     "note": "Reports 7.67 % on its own manuscript-independent split; on the HF crops it reads much worse (crop geometry and its own split differ)."},
    {"sys": "AMIDDA 1.0 Line 4B (Calfa, Qwen3.5-VL-4B LoRA)", "kind": "reported-same-lines" if not amidda else "rescored",
     "r1": amidda.get("r1", 25.0), "r2": amidda.get("r2", 27.6), "tm": amidda.get("tm", 24.2), "src": "AMIDDA card",
     "note": "Calfa's generic VLM, trained on AMIDDA train (includes RASAM-1, RASAM-2, TariMa train). Public; our re-run on a Colab GPU is pending budget."
     if not amidda else "Re-run by us on a Colab GPU; card numbers 25.0 / 27.6 / 24.2."},
    {"sys": "HATFormer, AraMS-28k (TrOCR-base, 334 M)", "kind": "rescored", "r1": hat.get("r1", "pending"), "r2": hat.get("r2", "pending"),
     "tm": hat.get("tm", "pending"), "src": "HATFormer AraMS-28k",
     "note": "Never saw RASAM or TariMa (zero-shot reference; its own unseen Maghrebi book: 37.9 %). Run on the Mac CPU with the card's preprocessing."},
    {"sys": "TariMa Kraken manuscript model", "kind": "rescored", "r1": 42.6, "r2": 29.9, "tm": None, "src": "TariMa", "note": "RASAM unseen."},
]

# ---- 2. published on other splits ------------------------------------------------------------------------------------------------
OTHER = [
    ("RASAM-1", 3.23, "Lucas et al. 2022, meta-word CRNN", "Lucas 2022", "Own page-level split (32 test pages) and labels standardised from 54 to 38 classes.",
     "Model and test page list not public. Our best under the same standardisation and other adjustments: 3.65 %."),
    ("RASAM-1", 4.8, "RASAM paper, word-based CRNN", "RASAM 2021", "Split not described; predates the HF split by 5 years.", "Model not public, test lines unknown."),
    ("RASAM-1", 4.95, "HAFITH, self-reported", "HAFITH", "Own 906-line RASAM test split, about half of the official test lines in its train/val.",
     "Reproduced on the official lines instead: 6.27 % (12.34 % on clean lines)."),
    ("RASAM-1", 5.50, "CHR 2024, word-based CRNN on RASAM-1 + 2", "CHR 2024", "Own random sample, no split defined.", "Model not public."),
    ("RASAM-2", 6.79, "CHR 2024, word-based CRNN on RASAM-1 + 2", "CHR 2024", "Own random sample, no split defined.", "Model not public."),
    ("TariMa", 7.67, "Phoenix (Kraken)", "Phoenix", "Own manuscript/page-independent split.", "Re-scored on the official lines: see strict board (RASAM)."),
    ("TariMa", 9.7, "CHR 2024 RASAM-2 model applied to TariMa", "CHR 2024", "Own TariMa sample, out of domain.", "Model not public."),
]

# ---- 3. our ideas and their effect -----------------------------------------------------------------------------------------------
IDEAS = [
    ("Train on RASAM-1 + RASAM-2 together (A3) instead of each alone", "RASAM-2 greedy 18.47 % -> 12.84 %; RASAM-1 6.82 -> 6.55 %", "kept"),
    ("Character 8-gram LM in CTC beam search (weights tuned on validation)", "-1.2 to -2.5 points everywhere", "kept"),
    ("Line height 96 px instead of 64", "RASAM-1 5.35 -> 4.58 %, RASAM-2 10.45 -> 9.66 % (with LM)", "kept"),
    ("Line height 128 px", "worse in a partial run", "dropped"),
    ("Add TariMa (another Maghrebi corpus) to training", "RASAM-1 4.58 -> 4.32 %, RASAM-2 9.66 -> 9.41 %", "kept"),
    ("Add all of RASAM-1 (more lines of the same manuscripts)", "no gain: diversity beats volume", "dropped"),
    ("Full-width model (4x the parameters)", "no consistent gain", "dropped"),
    ("Automatic data cleaning: 250 crops re-cut, 202 bad lines dropped from training (RS4j)", "no measurable gain: greedy 4.73 / 10.97 / 8.45 vs 4.77 / 10.79 / 8.31 %, paired differences within ±0.6", "dropped"),
    ("Re-cut the 80 broken MS 1936 crops (data repair)", "repaired RASAM-2 test 8.94 -> 8.42 %; small, mixed effect in training", "kept"),
    ("Neighbour-ink + fading augmentation", "mixed, under seed noise", "dropped"),
    ("Darkest RGB channel as input", "worse (RASAM-1 4.33 -> 5.14 %)", "dropped"),
    ("Standardised labels (Lucas-style)", "no real gain once scored the same way (4.04 vs 4.00 % diacritics-insensitive)", "dropped"),
    ("60 epochs instead of 30", "greedy -0.4 to -1.0 points; with LM -0.05 to -0.45", "kept"),
    ("EMA of weights / average of the last 10 epochs", "nothing consistent (the one-cycle schedule already anneals)", "dropped"),
    ("AraGPT2 rescoring of the 10-best", "-0.2 to -0.3 points on every set", "kept"),
    ("Consensus ensemble of seeds", "RASAM-1 4.23 % (older models)", "optional"),
    ("Specialise per corpus (10 epochs on RASAM-2 only / TariMa only)", "RASAM-2 9.10 vs 9.00 % for the same epochs on all sets; TariMa 6.90 vs 7.34 %", "optional"),
    ("Keep more vertical detail (our CRNN at height / 8)", "no gain (4.38 / 9.28 / 7.47 %)", "dropped"),
    ("Calfa's architecture (3 conv blocks, 4 x BiLSTM 512, 120 px, 53 M)", "collapsed at lr 1e-3; at 3e-4 trained as long as it improved (~56 epochs): 4.49 / 9.61 / 8.11 %, behind ours (4.33 / 9.13 / 7.35 %)", "dropped"),
]

WHY_DIFFER = [
    ("Different test lines.", "Each paper scored its own sample, so the numbers are on different, unknown lines of different difficulty. RASAM-2 has 15 manuscripts of very unequal difficulty: 7 broken MS 1936 crops alone hold 12 % of our RASAM-2 errors."),
    ("Leakage between train and test.", "The HF split is random by line, so every test page also has lines in train; HAFITH's own split even contains half of the official RASAM-1 test lines in its training data. Models that memorise lines look better than they read."),
    ("Seen vs new manuscripts (measured).", "Our leave-manuscript-out test on RASAM-2 (5 folds x 3 manuscripts): on the same official test lines, a model that never saw the manuscript scores 9.87 % vs 6.69 % for the same recipe trained on the random line split (14 manuscripts, MS 1936 excluded): +3.2 points, about half as many errors again. All 14 regular manuscripts get worse. Every RASAM-2 number on a random split, ours included, measures reading inside known manuscripts."),
    ("Different references.", "Lucas et al. folded the character set from 54 to 38 classes; other works keep diacritics, hamza forms and placeholders as written. On our predictions the same model moves from 4.32 % to 3.95 % on RASAM-1 just by standardising the references."),
    ("Different averaging.", "Micro CER (all errors / all characters) and mean per-line or per-page CER differ by about 1 point on the same predictions; most papers do not say which they use."),
    ("Different crops.", "Kraken models (Phoenix, TariMa) were trained on their own polygon crops; the HF crops are rectangles cut differently, which costs them a lot (Phoenix 7.67 % on its split vs 23.1 % on the HF crops)."),
    ("Different training data.", "Specialised per-corpus models (Calfa CTC) vs generic models over many scripts (AMIDDA VLM, HAFITH, HATFormer). On RASAM the specialised models win by a wide margin."),
]

WHY_SPLITS = [
    "The first RASAM papers (2021, 2022, 2024) predate the Hugging Face splits (created 2026-05-20) by two to five years; the original GitHub release had no split, so each paper made one, and none described it well enough to reproduce.",
    "Later papers chose their own split on purpose: Phoenix argues that a random line split mixes lines of the same page and manuscript into train and test and so overstates how well a model reads a new manuscript; it used a manuscript/page-independent split instead.",
    "HAFITH built one combined benchmark across Muharaf, KHATT and RASAM with its own 60/20/20 split, which overlaps the official test lines.",
    "Calfa's own AMIDDA paper (2026) is the first to report on the official HF test lines, which is why its per-corpus CTC numbers are the only published ones that sit on our strict board.",
]

KIND = {"ours": ("ours", "Ours"), "rescored": ("rescored", "Re-scored by us"), "reported-same-lines": ("reported", "Reported on these lines")}
BENCH = Path("results/rasam/clean/bench.json")
VIEWS = (("official", "official"), ("drop-only", "drop only"), ("clean", "clean"), ("clean-indep", "clean, people-found drops only"))


def ci_cell(x):
    if not x:
        return '<td class="na">-</td>'
    return f'<td>{x["cer"]:.2f}<span class="ci"> {x["ci95"][0]:.1f}–{x["ci95"][1]:.1f}</span></td>'


def clean_section() -> str:
    """Section 1b from results/rasam/clean/bench.json (scripts/clean_bench.py): every re-runnable system on the official, drop-only, clean and
    clean-indep views with page-bootstrap 95 % intervals, plus the 200-line VLM panel."""
    if not BENCH.exists():
        return ""
    b = json.loads(BENCH.read_text())
    n = b["views"]
    head = "".join(f'<th colspan="3">{html.escape(lab)}<br><span class="ci">{n[v]["r1"]} / {n[v]["r2"]} / {n[v]["tm"]} lines</span></th>' for v, lab in VIEWS)
    sub = "".join("<th>R-1</th><th>R-2</th><th>TM</th>" for _ in VIEWS)
    rows = []
    order = sorted(b["systems"].items(), key=lambda kv: kv[1].get("official/r2", {}).get("cer", 999))
    for name, r in order:
        cls, lab = KIND[r["kind"]] if r["kind"] in KIND else ("rescored", r["kind"])
        cells = "".join(ci_cell(r.get(f"{v}/{t}")) for v, _ in VIEWS for t in ("r1", "r2", "tm"))
        rows.append(f'<tr class="{cls}"><td><span class="tag {cls}">{lab}</span> {html.escape(name)}'
                    f'<div class="note">{html.escape(r.get("note", ""))}</div></td>{cells}</tr>')
    panel = []
    pn = sorted(b.get("panel", {}).items(), key=lambda kv: kv[1].get("official/r1", {}).get("cer", 999))
    for name, r in pn:
        if "ft" not in name and not name.startswith(("Ours", "AMIDDA", "HAFITH")):
            continue  # base (not fine-tuned) VLMs: 40-150 % CER, listed in the record, not here
        panel.append(f"<tr><td>{html.escape(name.replace('VLM fine-tune: ', ''))}</td>"
                     + "".join(ci_cell(r.get(f"{v}/{t}")) for v in ("official", "drop-only") for t in ("r1", "r2")) + "</tr>")
    pairs = "".join(f"<li>{html.escape(k)}: <b>{v['diff']:+.2f}</b> points (95 % {v['ci95'][0]:+.2f} to {v['ci95'][1]:+.2f})</li>"
                    for k, v in b["paired"].items() if "| clean/" in k)
    return f"""<h2>1b. Clean test view, with error bars</h2>
<p class="lead">The same official test lines, three ways. <b>Drop only</b> removes the {n['official']['r1'] - n['drop-only']['r1']} / {n['official']['r2'] - n['drop-only']['r2']} / {n['official']['tm'] - n['drop-only']['tm']}
test lines whose reference does not match the image. <b>Clean</b> also replaces the 25 broken crops (23 RASAM-2, 2 RASAM-1) with crops re-cut from the page, re-read
by every system. <b>People-found drops only</b> removes only the lines found by people (PR #7 of the dataset repo, our by-eye list), in case the automatic rules favour
CRNN-like readers. Small grey numbers: 95 % interval from resampling pages. The corrected lists and crops are released in
<code>release/rasam-clean/</code>. Systems we cannot re-run (Calfa CTC, Gemini) have no clean number.</p>
<div class="wrap"><table><thead><tr><th rowspan="2">System</th>{head}</tr><tr>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="lead">Paired differences on the clean view (same resampled pages for both systems):</p><ul>{pairs}</ul>
<h3>200-line panel: fine-tuned vision-language models</h3>
<p class="lead">Our suite_v1 VLM fine-tunes (LoRA r16, 1 epoch, vision tower frozen; 22k mixed lines) were only run on test lines 0–199 of RASAM-1 / RASAM-2, so they
are compared there. They have no re-cut readings, so this panel shows the official and drop-only views. A few VLM lines errored out; those cells use the lines that
were read (at least 150 of 200).</p>
<div class="wrap"><table><thead><tr><th rowspan="2">System</th><th colspan="2">official</th><th colspan="2">drop only</th></tr>
<tr><th>R-1</th><th>R-2</th><th>R-1</th><th>R-2</th></tr></thead><tbody>{''.join(panel)}</tbody></table></div>"""


def cell(v):
    if v is None:
        return '<td class="na">-</td>'
    if isinstance(v, str):
        return f'<td class="na">{html.escape(v)}</td>'
    return f"<td>{v:.2f}</td>"


def link(key):
    return f'<a href="{LINKS[key]}">{html.escape(key)}</a>' if key in LINKS else ""


def page() -> str:
    def best(col):
        vals = [r[col] for r in STRICT if isinstance(r[col], (int, float))]
        return min(vals) if vals else None
    rows = []
    for r in sorted(STRICT, key=lambda r: r["r2"] if isinstance(r["r2"], (int, float)) else 999):
        cls, lab = KIND[r["kind"]]
        cells = "".join(cell(r[c]).replace("<td>", '<td class="best">', 1) if isinstance(r[c], (int, float)) and r[c] == best(c) else cell(r[c]) for c in ("r1", "r2", "tm"))
        rows.append(f'<tr class="{cls}"><td><span class="tag {cls}">{lab}</span> {html.escape(r["sys"])}</td>{cells}'
                    f'<td class="src">{link(r["src"])}</td><td class="note">{html.escape(r["note"])}</td></tr>')
    other = "".join(f"<tr><td>{t}</td><td>{c:.2f}</td><td>{html.escape(s)}</td><td class='src'>{link(k)}</td><td class='note'>{html.escape(w)}</td>"
                    f"<td class='note'>{html.escape(rep)}</td></tr>" for t, c, s, k, w, rep in OTHER)
    ideas = "".join(f"<tr><td>{html.escape(i)}</td><td>{html.escape(e)}</td><td><span class='tag {v}'>{v}</span></td></tr>" for i, e, v in IDEAS)
    differ = "".join(f"<li><b>{html.escape(h)}</b> {html.escape(t)}</li>" for h, t in WHY_DIFFER)
    splits = "".join(f"<li>{html.escape(t)}</li>" for t in WHY_SPLITS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RASAM Leaderboard</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e3e0d8;--card:#ffffff;--ours:#0f766e;--resc:#7c3aed;--rep:#b45309;--best:#dcfce7;--bestfg:#14532d}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151513;--fg:#ecebe6;--mut:#a3a19a;--line:#33322e;--card:#1d1d1a;--ours:#2dd4bf;--resc:#a78bfa;--rep:#f59e0b;--best:#14532d;--bestfg:#dcfce7}}}}
:root[data-theme="dark"]{{--bg:#151513;--fg:#ecebe6;--mut:#a3a19a;--line:#33322e;--card:#1d1d1a;--ours:#2dd4bf;--resc:#a78bfa;--rep:#f59e0b;--best:#14532d;--bestfg:#dcfce7}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:24px 16px 64px}}
main{{max-width:1180px;margin:0 auto}} h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 8px}}
.lead{{color:var(--mut);margin:0 0 16px;max-width:70ch}} .wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);white-space:nowrap}} td:nth-child(n+2):nth-child(-n+4){{font-variant-numeric:tabular-nums;white-space:nowrap}}
td.best{{background:var(--best);color:var(--bestfg);font-weight:600}} td.na{{color:var(--mut)}} td.note{{color:var(--mut);font-size:13px;min-width:260px}} td.src a{{color:inherit}}
.tag{{display:inline-block;font-size:11px;font-weight:600;padding:1px 7px;border-radius:99px;border:1px solid currentColor;margin-right:6px;white-space:nowrap}}
.tag.ours,.tag.kept{{color:var(--ours)}} .tag.rescored{{color:var(--resc)}} .tag.reported,.tag.optional{{color:var(--rep)}} .tag.dropped{{color:var(--mut)}}
tr.ours td:first-child{{font-weight:600}} .ci{{color:var(--mut);font-size:11px;font-weight:400}} div.note{{color:var(--mut);font-size:12px;font-weight:400}} h3{{font-size:16px;margin:24px 0 6px}} ul{{max-width:80ch;padding-left:20px}} li{{margin:6px 0}}
</style></head><body><main>
<h1>RASAM leaderboard</h1>
<p class="lead">Character error rate (CER, %, micro-averaged, diacritics kept; lower is better) on Maghrebi Arabic manuscript lines. The strict board only holds
systems scored on the exact official Hugging Face test lines: RASAM-1 (734 lines), RASAM-2 (372), TariMa (229). Best value per column highlighted.</p>
<h2>1. Strict board: the exact official test lines</h2>
<div class="wrap"><table><thead><tr><th>System</th><th>RASAM-1</th><th>RASAM-2</th><th>TariMa</th><th>Source</th><th>Notes</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="lead"><span class="tag ours">Ours</span> our models. <span class="tag rescored">Re-scored by us</span> a public model we ran on these lines with the
same metric. <span class="tag reported">Reported on these lines</span> the authors' own number on the same lines; not re-run by us (reason in the notes).</p>
{clean_section()}
<h2>2. Published on other splits (not comparable)</h2>
<div class="wrap"><table><thead><tr><th>Test</th><th>CER</th><th>System</th><th>Source</th><th>Why not comparable</th><th>Reproduction</th></tr></thead>
<tbody>{other}</tbody></table></div>
<h2>3. Our ideas and what they did</h2>
<div class="wrap"><table><thead><tr><th>Idea</th><th>Effect on the official tests</th><th>Verdict</th></tr></thead><tbody>{ideas}</tbody></table></div>
<h2>4. Why the scores differ</h2><ul>{differ}</ul>
<h2>5. Why papers made their own splits</h2><ul>{splits}</ul>
</main></body></html>"""


def main() -> None:
    OUT.write_text(page(), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB); HATFormer {hat or 'not run'}; AMIDDA {amidda or 'not run'}")


if __name__ == "__main__":
    main()
