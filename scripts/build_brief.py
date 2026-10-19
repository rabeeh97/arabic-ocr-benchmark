"""Build the one-page visual brief of the research (docs/research/brief.html).

A short, phone-friendly companion to the full ledger: the findings that matter, their numbers, and how each
relates to the papers we have read. No images, no libraries. Content lives in this file; rerun to rebuild.
Usage: .geomvenv/bin/python scripts/build_brief.py
"""
from __future__ import annotations

import html
from pathlib import Path

LEDGER = "https://claude.ai/artifact/T6y934CtFfiBGAnySZRT7e"
BIB = {  # key -> (label, link or None)
    "alazzawi2026gap": ("Al-Azzawi 2026 · gap", "https://arxiv.org/abs/2606.18884"),
    "alazzawi2026labelerr": ("Al-Azzawi 2026 · label errors", "https://arxiv.org/abs/2601.16713"),
    "alazzawi2026crossling": ("Al-Azzawi 2026 · cross-lingual", "https://arxiv.org/abs/2605.02089"),
    "alazzawi2026seqmodel": ("Al-Azzawi 2026 · sequence model", "https://arxiv.org/abs/2605.05900"),
    "alshaibani2025dotless": ("Alshaibani 2025 · dotless Arabic", None),
    "heakl2025kitab": ("KITAB-Bench 2025", None),
    "farazi2026vlmocr": ("Farazi 2026 · VLMs on manuscripts", "https://arxiv.org/abs/2608.22366"),
    "kohut2023finetuning": ("Kohút 2023 · fine-tuning", None),
    "vidalgorene2024rasam2": ("Vidal-Gorène 2024 · RASAM 2", None),
    "retsinas2022best": ("Retsinas 2022 · HTR practices", None),
    "wigington2017augment": ("Wigington 2017 · augmentation", None),
    "luo2020learn": ("Luo 2020 · learned augmentation", None),
    "eltay2022adaptive": ("Eltay 2022 · adaptive augmentation", None),
    "mueller2012letter": ("Müller 2012 · letter identification", None),
    "alluhaybi2020connect": ("Alluhaybi 2020 · connectedness", None),
    "zeghli2025bigram": ("Zeghli 2025 · open bigrams", None),
    "doctor2022graphemic": ("Doctor 2022 · normalisation", "https://arxiv.org/abs/2210.12273"),
    "hegghammer2021noisy": ("Hegghammer 2021 · OCR benchmarking", None),
    "baek2019wrong": ("Baek 2019 · model comparisons", None),
    "zhang2020ran": ("Zhang 2020 · radical analysis", None),
    "swayamdipta2020cartography": ("Swayamdipta 2020 · cartography", None),
    "ye2013diqa": ("Ye 2013 · image quality", None),
    "shannon1951": ("Shannon 1951 · entropy", None),
    "mahmoud2014khatt": ("Mahmoud 2014 · KHATT", None),
}
THEMES = {
    "m": "Are the numbers right?",
    "s": "Shapes and dots",
    "g": "Spacing and geometry",
    "l": "Language",
    "w": "Models and writers",
}
# (theme, grade, headline, metric, detail, papers)
FINDINGS = [
    ("m", "A", "Cropping the line to its ink is the biggest single gain", "KHATT 31.0 → 23.3 % CER",
     "All 14 CRNNs land at 22–24 % after cropping, with no retraining. Base VLMs don't care, but three fine-tuned VLMs lose 3–4 points on loose crops. The full KHATT set is already tight, so the crop changes nothing there.",
     "retsinas2022best"),
    ("m", "A", "Scores depend on mark and spelling conventions", "τ 0.87–0.89 with marks kept",
     "Keeping diacritics re-orders leaderboards on KITAB-KHATT, Muharaf and RASAM-2, and only 35.8 % of Muharaf test lines carry marks. Muharaf writes the hamza on 4 % of alifs, NAKBA 16 %. The headline is the no-diacritics score.",
     "doctor2022graphemic alshaibani2025dotless"),
    ("m", "A", "Test lines leak into training, unevenly", "7.5 % to 95 % by set",
     "NAKBA-trained systems gain 20.1 points more on leaked NAKBA-blind lines, so that headline uses clean lines. 86.5 % of one KITAB subset sits word for word in Muharaf train.",
     "heakl2025kitab"),
    ("m", "A", "Pipeline bugs hid inside the scores", "PaddleOCR 85.8 → 62.9 % CER",
     "A flag reversed its text. Several models were cut off at a token limit (gemini-flash dropped the tail of 44–46 % of lines), and page-scale sets hit a 160-token cap.",
     "hegghammer2021noisy baek2019wrong"),
    ("m", "A", "Seed noise sets the bar for every claim", "±1 point on a 91-line half",
     "Two identical trainings differ by +0.4 [−0.6, +1.4]. 761 lines change between seeds of one recipe, and after multiple-comparison control the KHATT replicate's surviving letter cells go from 2 to 0.", ""),
    ("m", "A", "Label noise does not change the rankings", "9.4 % of Muharaf test lines flagged",
     "Dropping the 126 flagged lines moves CER by −0.4 to +0.9 points across 13 models and changes no rank.", "alazzawi2026labelerr"),
    ("s", "A", "Unusual letter shapes predict errors best", "error 1–5 % → 30–53 %",
     "Letter error climbs from the most typical fifth of shapes to the most novel band in every dataset, and survives the confound checks (odds ratio about 16 for CRNNs, 4.4 for fine-tuned VLMs).",
     "alazzawi2026gap zhang2020ran"),
    ("s", "B", "Adding missing shapes helps only where they are missing", "−0.8 points [−1.6, +0.1]",
     "Shape-targeted repeats beat a random control on held-out KHATT lines, but not on NAKBA, where the gain is plain repetition.", "swayamdipta2020cartography"),
    ("s", "A", "Skeletons, not dots, are the problem", "≈ 90 % of error survives without dots",
     "Lines that are right except for their dots are rare (0.1–4 %). The dot tax is a median 1.8 CER points, 8 % of error.", "alshaibani2025dotless"),
    ("s", "B", "Teeth are the weak spot", "+6–21 % over baseline, 6 of 6 families",
     "Letters with teeth are misread more than their letter baseline in every model family. Among one-tooth letters, error rises with run length.", ""),
    ("s", "B", "Confusions are visual, and shared by every model family", "ρ 0.63–0.90 between families",
     "Pairs that differ only in dots are 3–18 times more confusable than other pairs. Models confuse letters that look alike, not letters that sound alike.", "mueller2012letter"),
    ("s", "B", "Dots return from context for known words, not for unseen ones", "95 % restored by the top candidate",
     "A quarter to a third of words share their dotless form with another word, yet the most frequent word restores 95 % of the known ones.", "alshaibani2025dotless"),
    ("g", "B", "Word gaps and word-part gaps look alike", "P = 0.68 (CRNN), 0.63 (VLMs)",
     "Models insert spaces inside words at part gaps far more than chance (87 of 89 pairs). The space tax is 2–11 CER points against 1–4 for dots.", "alluhaybi2020connect zeghli2025bigram"),
    ("g", "B", "Slant matters a little, and a test-time fix is nearly free", "KHATT 26.3 → 25.3 %",
     "Slant and stroke augmentation in training hurts on unseen sets (+0.6 to +0.8 points).", "wigington2017augment luo2020learn eltay2022adaptive"),
    ("g", "N", "Tone, paper noise and ruled lines do not matter", "no measurable CER effect",
     "Measured clearly and found inert for the CRNN. Kept only as dataset descriptors.", "ye2013diqa"),
    ("g", "B", "Dots vanish at low resolution", "43–64 % under 2 px at 32 px",
     "At the CRNN's 64 px dots survive. Pixels per character and line length predict errors in every family.", ""),
    ("l", "A", "A free-running LLM decoder makes recognition worse", "KHATT 26.4 → 40.0 %",
     "NAKBA goes 13.8 → 20.8 %, and a model three times larger is no better. Fluent but wrong words are 41–44 % of errors, against 24 % for the CRNN.", "farazi2026vlmocr"),
    ("l", "B", "A word list alone cannot restore dots", "20–23 % of dot swaps accepted",
     "A lexicon covers 95.8–99.4 % of the panel's words, but it also accepts a fifth of single dot-swap errors.", ""),
    ("l", "B", "KHATT's text is far from its training text", "4.0 vs 3.4 vs 2.9 bits per character",
     "5.9 % of its word parts never occur in training, and unseen words hurt every family.", "shannon1951"),
    ("w", "A", "Fine-tuning on Muharaf rescues VLMs across scripts", "KHATT 77 → 30 % CER",
     "Base Qwen3-VL-4B scores 77–119 % on cursive sets. 8,000 Muharaf lines remove 47–67 points everywhere, even on Maghrebi manuscripts (119 → 52 %).",
     "farazi2026vlmocr kohut2023finetuning heakl2025kitab"),
    ("w", "B", "Maghrebi conventions must be learned", "ق read as ف in 23–27 % of cases",
     "A CRNN without Maghrebi training reads RASAM at about 47 % CER instead of about 12 %.", "vidalgorene2024rasam2"),
    ("w", "B", "CRNNs and VLMs fail on different lines", "KHATT 14.3 % vs 17.4 / 18.1 %",
     "The per-line better of the two is far below either, so a cross-family combination is a strong candidate.", ""),
    ("w", "B", "On KHATT, the text matters more than the writer, until you train on it", "R² 0.152 vs 0.088",
     "KHATT is one fixed paragraph, so it works as a natural experiment. For the KHATT-trained model it flips: writer 0.26, text 0.03.", "mahmoud2014khatt"),
    ("w", "B", "Models look at the right letters and still misread them", "about 3× chance",
     "Attribution maps point at a token's own letters three times above chance, but wrong tokens look in the right place as often as correct ones.", ""),
]
# (paper key, what the paper did, what we found, relation)
PAPERS = [
    ("alazzawi2026gap", "Why Arabic-script handwriting trails Latin: cleaning labels halves the gap, but 4.4 points remain at full data.",
     "We reproduce their substitution metric exactly. On the same lines, VLMs make as many similar-letter errors per character, and letter-shape novelty is the strongest error predictor.", "Extends"),
    ("alazzawi2026labelerr", "Human review finds 6–9 % label errors in Muharaf and Ajami.",
     "Their review logs flag 9.4 % of Muharaf test lines, and dropping them changes no rank in our comparisons.", "Confirms"),
    ("alazzawi2026crossling", "Arabic, Urdu and Persian help each other at 100–1,000 lines, and the gain comes with the BiLSTM.",
     "Neither paper tests VLMs. We show a single Muharaf fine-tune carries a VLM across scripts.", "Complements"),
    ("heakl2025kitab", "A multi-domain Arabic OCR benchmark that most of our test sets come from.",
     "We audited it: one subset leaks 86.5 % into Muharaf train, and loose crops cost fixed-height models 7.5 points.", "Audits"),
    ("farazi2026vlmocr", "OCR-conditioned correction helps on clean print and hurts on cursive manuscripts.",
     "Our free-running LLM decoder also hurts on handwriting, so language help has to be held to the image.", "Consistent"),
    ("alshaibani2025dotless", "Studies Arabic text with the dots removed.",
     "Without dots about nine tenths of the recogniser's error remains, and context restores 95 % of known words.", "Consistent"),
    ("mueller2012letter", "A choice model of letter identification from perceivability, similarity and bias.",
     "We use it on model confusions: they are shared across families and follow shape, not sound.", "Method"),
    ("alluhaybi2020connect", "How letter connectedness shapes human reading of Arabic words.",
     "Recognisers show the same ambiguity: word gaps and part gaps overlap, and spaces land at part gaps.", "Parallel"),
    ("retsinas2022best", "Preprocessing and training practices for handwriting recognition.",
     "Cropping to the ink before resizing is the largest single gain we measured.", "Same family"),
    ("vidalgorene2024rasam2", "The RASAM 2 Maghrebi manuscript set.",
     "We use it as the Maghrebi testbed: the qāf/fāʾ convention and few-shot behaviour.", "Uses"),
]
STATE = [
    ("Done", "Two GPU nights of recognisers, plus the full KHATT set, characterisation of dots, spacing, shapes and writers."),
    ("Running", "A VLM night: how much the models rely on the image, geometry and resolution probes, and shape targeting for NAKBA."),
    ("Next", "Decoding with a lexicon, writer adaptation curves, and a combination of the two model families."),
]


def e(s: str) -> str:
    return html.escape(s, quote=True)


def chips(keys: str) -> str:
    out = []
    for k in keys.split():
        label, url = BIB[k]
        out.append(f'<a class="chip" href="{e(url)}" target="_blank" rel="noopener">{e(label)}</a>' if url
                   else f'<span class="chip">{e(label)}</span>')
    return "".join(out)


def bars(rows, vmax, unit="", cls=""):
    out = [f'<div class="bars {cls}">']
    for label, v, text, tone in rows:
        out.append(f'<div class="bar"><span class="bl">{e(label)}</span><span class="bt"><i class="{tone}" style="--w:{v / vmax * 100:.1f}%"></i></span>'
                   f'<b>{e(text)}{unit}</b></div>')
    out.append("</div>")
    return "".join(out)


def charts() -> str:
    crop = bars([("Stored crops", 31.0, "31.0", "hi"), ("Ink-cropped", 23.3, "23.3", "lo")], 35, " %")
    leak = bars([("RASAM", 1.1, "≤ 1.1", "lo"), ("Muharaf", 7.5, "7.5", "lo"), ("NAKBA blind", 16.3, "16.3", "mid"),
                 ("NAKBA test", 18.3, "18.3", "mid"), ("KITAB-KHATT", 44.5, "44.5", "hi"), ("Full KHATT", 95.0, "95", "hi")], 100, " %")
    dec = bars([("KHATT, CRNN", 26.4, "26.4", "lo"), ("KHATT, + LLM", 40.0, "40.0", "hi"),
                ("NAKBA, CRNN", 13.8, "13.8", "lo"), ("NAKBA, + LLM", 20.8, "20.8", "hi")], 45, " %")
    comp = ('<div class="stack" role="img" aria-label="About 92 percent of error is skeleton, 8 percent dots">'
            '<i class="sk" style="--w:92%"><b>skeleton ≈ 92 %</b></i><i class="dt" style="--w:8%"></i></div>'
            '<div class="stackcap"><span>skeleton</span><span>dots 8 %</span></div>')
    cards = [
        ("Crop to the ink", "KHATT, CER on 200 lines", crop),
        ("Leaked test lines", "share of lines whose text or image is in their own train set", leak),
        ("What the error is made of", "median over 91 model × dataset pairs", comp),
        ("Adding an LLM decoder", "CER, lower is better", dec),
    ]
    return "".join(f'<figure class="chart"><figcaption><b>{e(t)}</b><span>{e(s)}</span></figcaption>{b}</figure>' for t, s, b in cards)


def finding_cards() -> str:
    out = []
    for th, g, head, metric, detail, papers in FINDINGS:
        out.append(
            f'<article class="card" data-theme-key="{th}">'
            f'<div class="ctop"><span class="tag">{e(THEMES[th])}</span><span class="g g{g}" title="evidence grade">{g}</span></div>'
            f'<h3>{e(head)}</h3><p class="metric">{e(metric)}</p><p>{e(detail)}</p>'
            + (f'<div class="chips">{chips(papers)}</div>' if papers else '') + '</article>')
    return "".join(out)


def paper_rows() -> str:
    out = []
    for k, they, we, rel in PAPERS:
        label, url = BIB[k]
        name = f'<a href="{e(url)}" target="_blank" rel="noopener">{e(label)}</a>' if url else e(label)
        out.append(f'<div class="prow"><div class="pn">{name}<span class="rel r-{rel.split()[0].lower()}">{e(rel)}</span></div>'
                   f'<p><i>They</i> {e(they)}</p><p><i>We</i> {e(we)}</p></div>')
    return "".join(out)


CSS = """
:root{--bg:#f4f6f8;--surface:#ffffff;--ink:#13212c;--muted:#5a6b78;--line:#dbe2e8;--accent:#0b5d7e;--accent-soft:#dcecf3;
--ga:#0e7a55;--gb:#2a66b8;--gc:#a66a00;--gn:#6b7782;--lo:#2f8f7a;--mid:#c9962a;--hi:#c2503a;
--f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;--f-ar:"Noto Naskh Arabic","Geeza Pro",serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e161d;--surface:#16212a;--ink:#e6edf2;--muted:#93a4b1;--line:#27363f;--accent:#62b6d6;--accent-soft:#16323f;
--ga:#4cc59a;--gb:#7aa7e6;--gc:#e0a84a;--gn:#9aa7b1;--lo:#4fb59f;--mid:#d9aa45;--hi:#e27a63;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e161d;--surface:#16212a;--ink:#e6edf2;--muted:#93a4b1;--line:#27363f;--accent:#62b6d6;--accent-soft:#16323f;
--ga:#4cc59a;--gb:#7aa7e6;--gc:#e0a84a;--gn:#9aa7b1;--lo:#4fb59f;--mid:#d9aa45;--hi:#e27a63;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--f-body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0}
.wrap{max-width:1080px;margin:0 auto;padding-block:28px 56px}
header{display:grid;grid-template-columns:1fr auto;gap:20px;align-items:end;padding-bottom:22px;border-bottom:1px solid var(--line)}
.eyebrow{font:500 11.5px var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
h1{font-size:clamp(26px,5vw,40px);line-height:1.12;margin:0 0 10px;text-wrap:balance;font-weight:600;letter-spacing:-.01em}
.lede{margin:0;max-width:62ch;color:var(--muted);font-size:16px}
.motif{font:700 clamp(40px,9vw,72px)/1 var(--f-ar);direction:rtl;color:var(--accent);text-align:center;margin:0}
.motif small{display:block;font:400 11px var(--f-body);direction:ltr;color:var(--muted);margin-top:6px;letter-spacing:.02em}
h2{font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:38px 0 14px;font-weight:600;font-family:var(--f-mono)}
.points{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:12px;margin-top:22px}
.points p{margin:0;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:8px;font-size:14px;min-width:0}
.points b{display:block;font-size:12px;font-family:var(--f-mono);color:var(--accent);margin-bottom:4px;letter-spacing:.04em;text-transform:uppercase}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.chart{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.chart figcaption{display:flex;flex-direction:column;gap:2px;margin-bottom:12px}
.chart figcaption b{font-size:14px}.chart figcaption span{font-size:12px;color:var(--muted)}
.bars{display:flex;flex-direction:column;gap:7px}
.bar{display:grid;grid-template-columns:92px 1fr 56px;align-items:center;gap:8px;font-size:12.5px}
.bl{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bt{height:10px;background:var(--bg);border-radius:5px;overflow:hidden}
.bt i{display:block;height:100%;width:var(--w);border-radius:5px;background:var(--lo)}
.bt i.mid{background:var(--mid)}.bt i.hi{background:var(--hi)}
.bar b{font:500 12.5px var(--f-mono);text-align:right;font-variant-numeric:tabular-nums}
.stack{display:flex;height:34px;border-radius:6px;overflow:hidden;background:var(--bg)}
.stack i{display:flex;align-items:center;padding:0 10px;font-style:normal;width:var(--w)}
.stack .sk{background:var(--hi)}.stack .dt{background:var(--mid)}.stack b{color:#fff;font:500 12px var(--f-mono);white-space:nowrap}
.stackcap{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:6px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.filters button{font:500 13px var(--f-body);color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:99px;padding:6px 13px;cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.filters button:focus-visible,.chip:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,320px),1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:15px 16px 14px;display:flex;flex-direction:column;gap:6px;min-width:0}
.card[hidden]{display:none}
.ctop{display:flex;justify-content:space-between;align-items:center;gap:10px}
.tag{font:500 11px var(--f-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.g{font:600 12px var(--f-mono);width:22px;height:22px;border-radius:50%;display:inline-grid;place-items:center;color:var(--bg)}
.gA{background:var(--ga)}.gB{background:var(--gb)}.gC{background:var(--gc)}.gN{background:var(--gn)}
.card h3{font-size:16.5px;line-height:1.3;margin:2px 0 0;font-weight:600;text-wrap:balance}
.metric{margin:2px 0 0;font:500 14px var(--f-mono);color:var(--accent);font-variant-numeric:tabular-nums}
.card p{margin:0;font-size:14px;color:var(--ink)}.card .metric{color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chip{font-size:11.5px;color:var(--muted);border:1px solid var(--line);border-radius:99px;padding:2px 9px;text-decoration:none;background:var(--bg)}
a.chip{color:var(--accent);border-color:var(--accent-soft);background:var(--accent-soft)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--muted);margin:0 0 14px}
.legend span{display:inline-flex;align-items:center;gap:6px}.legend .g{width:16px;height:16px;font-size:10px}
.papers{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),1fr));gap:14px}
.prow{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.pn{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px;font-weight:600;font-size:14.5px}
.pn a{color:var(--accent);text-decoration:none}.pn a:hover{text-decoration:underline}
.rel{font:500 11px var(--f-mono);text-transform:uppercase;letter-spacing:.05em;padding:2px 8px;border-radius:99px;background:var(--accent-soft);color:var(--accent);white-space:nowrap}
.prow p{margin:4px 0 0;font-size:13.5px}.prow i{font:500 11px var(--f-mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-style:normal;margin-right:6px}
.state{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px}
.state div{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;font-size:14px;min-width:0}
.state b{display:block;font:500 12px var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--accent);margin-bottom:3px}
footer{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
footer a{color:var(--accent)}
@media (max-width:560px){header{grid-template-columns:1fr}.motif{text-align:left}.bar{grid-template-columns:78px 1fr 50px}}
@media (prefers-reduced-motion:no-preference){.bt i,.stack i{transition:width .5s ease}}
"""

JS = """
(function(){var btns=document.querySelectorAll('.filters button'),cards=document.querySelectorAll('.card');
function show(k){cards.forEach(function(c){c.hidden=!(k==='all'||c.getAttribute('data-theme-key')===k)});
btns.forEach(function(b){b.setAttribute('aria-pressed',b.getAttribute('data-k')===k?'true':'false')});}
btns.forEach(function(b){b.addEventListener('click',function(){show(b.getAttribute('data-k'))})});})();
"""


def render() -> str:
    filters = '<button type="button" data-k="all" aria-pressed="true">All</button>' + "".join(
        f'<button type="button" data-k="{k}" aria-pressed="false">{e(v)}</button>' for k, v in THEMES.items())
    points = [
        ("Measurement first", "Many apparent model gaps were crops, conventions, leaked lines or bugs. We fixed those before comparing models."),
        ("Skeletons, not dots", "Most recognition error is a misread letter skeleton. Dots, spaces and language each add a smaller cost."),
        ("Shape novelty", "The more unusual a letter's shape compared with training, the more likely it is misread, in every dataset."),
        ("Language must obey the image", "A free-running language model hurts. Priors only help when the image constrains them."),
    ]
    legend = "".join(f'<span><i class="g g{g}">{g}</i>{t}</span>' for g, t in
                     [("A", "strong, replicated"), ("B", "solid, one gap"), ("C", "suggestive"), ("N", "measured, little effect")])
    return (
        '<title>Arabic OCR Brief</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Naskh+Arabic:wght@700&display=swap">\n'
        f"<style>{CSS}</style>\n"
        '<div class="wrap"><header><div><p class="eyebrow">Arabic handwriting recognition · research brief</p>'
        "<h1>Where recognisers fail on Arabic script, and why</h1>"
        '<p class="lede">The findings that matter from our measurements, with their numbers, and how each relates to the papers we have read. The full evidence is in the ledger.</p></div>'
        '<p class="motif" aria-hidden="true">ب ت ث<small>one skeleton, three dot patterns</small></p></header>'
        f'<div class="points">{"".join(f"<p><b>{e(a)}</b>{e(b)}</p>" for a, b in points)}</div>'
        f'<h2>Four numbers worth remembering</h2><div class="charts">{charts()}</div>'
        f'<h2>Findings</h2><div class="filters" role="group" aria-label="Filter findings by theme">{filters}</div>'
        f'<p class="legend">Evidence grade: {legend}</p><div class="grid">{finding_cards()}</div>'
        f'<h2>The papers, and where we stand</h2><div class="papers">{paper_rows()}</div>'
        f'<h2>Where the work is</h2><div class="state">{"".join(f"<div><b>{e(a)}</b>{e(b)}</div>" for a, b in STATE)}</div>'
        f'<footer>Source: the research ledger, which holds every table, gallery and decision rule. <a href="{LEDGER}" target="_blank" rel="noopener">Open the full ledger</a>.</footer></div>'
        f"<script>{JS}</script>\n"
    )


if __name__ == "__main__":
    out = Path("docs/research/brief.html")
    out.write_text(render(), encoding="utf8")
    print("wrote", out, round(out.stat().st_size / 1024, 1), "KB;", len(FINDINGS), "findings,", len(PAPERS), "papers")
