"""Build the one-page visual brief of the research (docs/research/brief.html).

A phone-friendly companion to the full ledger: for each finding, the question, what we did, what we found, a
visual, and the papers it relates to. No libraries; the only images are a few tiny real handwriting crops stored
in docs/research/brief_assets.json (extracted from the ledger). Content lives in this file; rerun to rebuild.
Usage: .geomvenv/bin/python scripts/build_brief.py
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

LEDGER = "https://claude.ai/artifact/T6y934CtFfiBGAnySZRT7e"
ASSETS = json.loads(Path("docs/research/brief_assets.json").read_text(encoding="utf8"))
BIB = {  # key -> (label, link or None)
    "alazzawi2026gap": ("Al-Azzawi 2026 · gap", "https://arxiv.org/abs/2606.18884"),
    "alazzawi2026labelerr": ("Al-Azzawi 2026 · label errors", "https://arxiv.org/abs/2601.16713"),
    "alazzawi2026crossling": ("Al-Azzawi 2026 · cross-lingual", "https://arxiv.org/abs/2605.02089"),
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


def e(s) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- visual building blocks
def pairbars(rows, vmax, unit=" %", lo_is_good=True, la="before", lb="after"):
    """rows: (label, a, b). Two bars per row on one scale: grey for a, coloured for b."""
    out = ['<div class="pb">']
    for label, a, b in rows:
        good = (b < a) == lo_is_good
        out.append(f'<div class="pbr"><span class="bl">{e(label)}</span><div>'
                   f'<div class="bt"><i class="gr" style="--w:{a / vmax * 100:.1f}%"></i><b>{a:g}{unit}</b></div>'
                   f'<div class="bt"><i class="{"lo" if good else "hi"}" style="--w:{b / vmax * 100:.1f}%"></i><b>{b:g}{unit}</b></div></div></div>')
    out.append(f'<p class="key"><i class="gr"></i>{e(la)}<i class="{"lo" if lo_is_good else "hi"}"></i>{e(lb)}</p></div>')
    return "".join(out)


def bars(rows, vmax, unit=" %"):
    """rows: (label, value, text, tone)."""
    out = ['<div class="pb">']
    for label, v, text, tone in rows:
        out.append(f'<div class="pbr"><span class="bl">{e(label)}</span><div class="bt"><i class="{tone}" style="--w:{v / vmax * 100:.1f}%"></i><b>{e(text)}{unit}</b></div></div>')
    out.append("</div>")
    return "".join(out)


def rangebars(rows, vmax, unit=" %"):
    """rows: (label, lo, hi, tone): a floating bar from lo to hi."""
    out = ['<div class="pb">']
    for label, lo, hi, tone in rows:
        txt = f"{lo:g}–{hi:g}" if hi != lo else f"{lo:g}"
        out.append(f'<div class="pbr"><span class="bl">{e(label)}</span><div class="bt"><i class="{tone}" style="--l:{lo / vmax * 100:.1f}%;--w:{max(hi - lo, vmax * 0.012) / vmax * 100:.1f}%;margin-left:var(--l)"></i><b>{txt}{unit}</b></div></div>')
    out.append("</div>")
    return "".join(out)


def forest(rows, lo, hi, unit="", note=""):
    """rows: (label, estimate or None, low, high). Zero line, interval, point; one shared axis."""
    pos = lambda x: (x - lo) / (hi - lo) * 100
    out = ['<div class="fo">']
    for label, est, a, b in rows:
        iv = f'<i class="iv" style="left:{pos(a):.1f}%;width:{max(pos(b) - pos(a), 0.8):.1f}%"></i>' if a is not None else ""
        pt = f'<u class="pt" style="left:{pos(est):.1f}%"></u>' if est is not None else ""
        txt = (f"{est:+g}" if est is not None else "") + (f" [{a:+g}, {b:+g}]" if a is not None and est is not None and (a != est or b != est)
                                                           else (f"{a:+g} to {b:+g}" if a is not None and est is None else ""))
        out.append(f'<div class="for"><span class="bl">{e(label)}</span><div class="tr"><s style="left:{pos(0):.1f}%"></s>{iv}{pt}</div><b>{e(txt)}{unit}</b></div>')
    out.append(f'<div class="for ax"><span></span><div class="tr"><em style="left:0">{lo:+g}</em><em style="left:{pos(0):.1f}%">0</em><em style="right:0;left:auto">{hi:+g}</em></div><b></b></div>')
    if note:
        out.append(f'<p class="key">{e(note)}</p>')
    out.append("</div>")
    return "".join(out)


def stack(parts):
    """parts: (label, percent, tone)."""
    segs = "".join(f'<i class="{t}" style="--w:{p}%"><b>{e(l)} {p:g} %</b></i>' for l, p, t in parts)
    return f'<div class="stack" role="img">{segs}</div>'


def textdemo(items):
    return '<div class="td">' + "".join(
        f'<div><span class="ar{" big" if big else ""}" dir="rtl" lang="ar" style="{e(style)}">{e(ar)}</span><small>{e(cap)}</small></div>'
        for ar, cap, big, style in items) + "</div>"


def chips_plain(items):
    return '<div class="chips plain">' + "".join(f'<span class="chip">{e(i)}</span>' for i in items) + "</div>"


def crop_pairs():
    out = ['<div class="cp">']
    for c in ASSETS["crop"]:
        out.append(f'<figure><img alt="line as stored" src="{c["stored"]}" loading="lazy"><img alt="line cropped to the ink" src="{c["cropped"]}" loading="lazy">'
                   f'<figcaption>as stored: <b class="hi">{c["err_stored"]} errors</b> · cropped: <b class="lo">{c["err_cropped"]} errors</b></figcaption></figure>')
    out.append("</div>")
    return "".join(out)


def letter_grid(items, label):
    out = [f'<p class="key">{e(label)}</p><div class="lg">']
    for f in items[:4]:
        out.append(f'<figure class="{"bad" if f["cls"] == "bad" else "ok"}"><div><img alt="letter in KHATT" src="{f["letter_img"]}" loading="lazy">'
                   f'<img alt="nearest training letter" src="{f["nearest_img"]}" loading="lazy"></div>'
                   f'<figcaption><span class="ar" lang="ar">{e(f["char"])}</span> {e(f["form"])} · distance {f["novelty"]:g}</figcaption></figure>')
    out.append("</div>")
    return "".join(out)


def gap_curves():
    def pts(mean, sd, color):
        xs = [i * 3 for i in range(0, 101)]
        ys = [math.exp(-0.5 * ((x / 100 * 3 - mean) / sd) ** 2) for x in xs]
        d = " ".join(f"{x:.0f},{90 - y * 70:.1f}" for x, y in zip(xs, ys))
        return f'<polygon points="0,90 {d} 300,90" fill="var({color})" fill-opacity=".38" stroke="var({color})" stroke-width="1.4"/>'
    return ('<svg viewBox="0 0 300 118" role="img" aria-label="Two overlapping distributions of gap width" class="gc">'
            + pts(0.85, 0.42, "--lo") + pts(1.45, 0.55, "--hi")
            + '<line x1="0" y1="90" x2="300" y2="90" stroke="var(--line)"/>'
            '<text x="4" y="108" class="sv">narrow</text><text x="296" y="108" class="sv" text-anchor="end">wide</text>'
            '<text x="52" y="22" class="sv" fill="var(--lo)">gap inside a word</text><text x="196" y="14" class="sv" fill="var(--hi)">gap between words</text></svg>'
            '<p class="key">Schematic. Measured: a word gap is wider than a part gap only 68 % of the time (CRNN), 63 % (VLMs).</p>')


# ---------------------------------------------------------------- findings
F = []


def add(theme, grade, head, metric, q, did, found, papers, viz):
    F.append(dict(theme=theme, grade=grade, head=head, metric=metric, q=q, did=did, found=found, papers=papers, viz=viz))


add("m", "A", "Cropping the line to its ink is the biggest single gain", "KHATT 31.0 → 23.3 % CER",
    "Do models read worse when the writing fills only a thin strip of a tall image?",
    "Cropped each line to the band holding its ink before resizing it to the model's input height, then re-read the same 200 KHATT lines. 14 CRNNs, 3 seeds, no retraining.",
    "10 % of KHATT lines have writing in under 35 % of the image height, and those averaged 85 % CER. After cropping, every CRNN lands at 22–24 %. Base VLMs don't care, but three fine-tuned VLMs lose 3–4 points on loose crops. The full KHATT set is already tight, so cropping changes nothing there.",
    "retsinas2022best",
    crop_pairs() + pairbars([("KHATT, 200 lines", 31.0, 23.3)], 35, la="as stored", lb="cropped"))
add("m", "A", "Scores depend on mark and spelling conventions", "τ 0.87–0.89 with marks kept",
    "Do the datasets spell and mark Arabic the same way, so that their scores can be compared?",
    "Counted hamza and diacritic use in every label set, then re-scored every system with marks kept, with marks stripped, and with hamza folded into plain alif.",
    "Keeping diacritics re-orders leaderboards on KITAB-KHATT, Muharaf and RASAM-2, and only 35.8 % of Muharaf test lines carry marks. NAKBA writers do not draw the hamza that its labels contain, worth about 2 CER points. The headline score is therefore the no-diacritics one.",
    "doctor2022graphemic alshaibani2025dotless",
    textdemo([("كَتَبَ", "with marks", True, ""), ("كتب", "marks stripped", True, ""), ("أ إ آ ا", "hamza folded to alif", True, "")])
    + bars([("Muharaf", 4, "4", "lo"), ("NAKBA", 16, "16", "mid")], 20, " %") + '<p class="key">share of alifs whose label writes a hamza</p>')
add("m", "A", "Test lines leak into training, unevenly", "7.5 % to 95 % by set",
    "Are some test lines also in a model's training data?",
    "Matched each test line against its own training split by normalised text (MinHash) and by image hash, checked on 89 known KITAB/KHATT image pairs, then compared systems on leaked and clean lines.",
    "NAKBA-trained systems gain 20.1 points more on leaked NAKBA-blind lines, so that headline uses clean lines only. 86.5 % of one KITAB subset sits word for word in Muharaf train, and the full KHATT test is a fixed paragraph.",
    "heakl2025kitab",
    bars([("RASAM", 1.1, "≤ 1.1", "lo"), ("Muharaf", 7.5, "7.5", "lo"), ("NAKBA blind", 16.3, "16.3", "mid"), ("NAKBA test", 18.3, "18.3", "mid"),
          ("KITAB-KHATT", 44.5, "44.5", "hi"), ("Full KHATT", 95.0, "95", "hi")], 100, " %") + '<p class="key">share of test lines also found in the same set\'s training split</p>')
add("m", "A", "Pipeline bugs hid inside the scores", "PaddleOCR 85.8 → 62.9 % CER",
    "Are some low scores caused by the harness rather than the model?",
    "Compared each model's output with the ground truth read backwards, split errors by position in the line, and checked output lengths against the token limits.",
    "A flag reversed PaddleOCR's text. Several models were cut off at a token limit (gemini-flash dropped the tail of 44–46 % of lines), and page-scale sets hit a 160-token cap.",
    "hegghammer2021noisy baek2019wrong",
    textdemo([("مرحبا", "correct order", True, ""), ("ابحرم", "characters reversed, as the flag left it", True, "")])
    + pairbars([("PaddleOCR, KHATT", 85.8, 62.9)], 100, la="as stored", lb="order fixed"))
add("m", "A", "Seed noise sets the bar for every claim", "±1 point on a 91-line half",
    "How much does a score move when nothing changes but the random seed?",
    "Retrained the same recipe with another seed and compared line by line, and counted the lines that flip between identical trainings.",
    "Two identical trainings differ by +0.4 points. 761 lines change between seeds of one recipe. After multiple-comparison control the KHATT replicate's surviving letter cells fall from 2 to 0, so claims smaller than about 1 point need a second seed.",
    "",
    forest([("Seed noise", 0.4, -0.6, 1.4), ("Shape targeting", -0.8, -1.6, 0.1), ("Ink crop", -7.5, -10.6, -4.9)], -12, 3, note="CER change in points, with 95 % interval. Only the crop clearly clears the noise."))
add("m", "A", "Label noise does not change the rankings", "9.4 % of Muharaf test lines flagged",
    "Do wrong labels in the test set change which model wins?",
    "Used the human review logs of Al-Azzawi's CER-HV study to drop the flagged Muharaf test lines, and re-scored 13 models.",
    "126 of 1,334 test lines are flagged. Dropping them moves CER by −0.4 to +0.9 points and changes no rank.",
    "alazzawi2026labelerr",
    stack([("flagged", 9.4, "hi"), ("kept", 90.6, "lo")]) + forest([("CER change", None, -0.4, 0.9)], -2, 2, note="points, across 13 models; 0 rank changes"))
add("s", "A", "Unusual letter shapes predict errors best", "error 1–5 % → 30–53 %",
    "Are letters that look unlike the training letters misread more often?",
    "Aligned every line to its letters (CTC forced alignment), gave each letter a feature vector, and measured its distance to the nearest training letter of the same character and position.",
    "Letter error climbs from the most typical fifth of shapes to the most novel band in every dataset, and survives the confound checks (odds ratio about 16 for CRNNs, 4.4 for fine-tuned VLMs). KHATT's letters sit far from Muharaf's: 21 % are beyond Muharaf's own 95th percentile, against 10 % for NAKBA and 5 % for Muharaf.",
    "alazzawi2026gap zhang2020ran",
    letter_grid(ASSETS["novel"], "Most novel KHATT letters (left) beside the nearest training letter (right). All misread.")
    + letter_grid(ASSETS["typical"], "Typical letters, read correctly.")
    + rangebars([("Most typical fifth", 1, 5, "lo"), ("Most novel band", 30, 53, "hi")], 60, " % wrong"))
add("s", "B", "Adding the missing shapes helps only where they are missing", "−0.8 points [−1.6, +0.1]",
    "If we add training lines that contain the missing shapes, does recognition improve?",
    "Chose lines by shape novelty using half of the data, repeated them three times in training (random lines as the control), and tested on the held-out half.",
    "Shape-targeted repeats beat the random control on held-out KHATT lines, but the interval touches zero. On NAKBA the gain is plain repetition, not shape.",
    "swayamdipta2020cartography",
    forest([("KHATT, held-out", -0.8, -1.6, 0.1)], -3, 2, note="CER change vs random repeats, points"))
add("s", "A", "Skeletons, not dots, are the problem", "≈ 90 % of error survives without dots",
    "How much of the error is the dots?",
    "Re-scored every reading after removing all dots (the dotless skeleton) and split each line's errors into dot errors and skeleton errors.",
    "Lines that are right except for their dots are rare (0.1–4 %). The dot tax is a median 1.8 CER points, about 8 % of error. Three-dot letters are the most dot-error-prone.",
    "alshaibani2025dotless",
    textdemo([("ب ت ث", "one skeleton, one to three dots", True, ""), ("ٮ ٮ ٮ", "the dotless skeleton", True, "")])
    + stack([("skeleton", 92, "hi"), ("dots", 8, "mid")]) + '<p class="key">what the error is made of, median over 91 model × dataset pairs</p>')
add("s", "B", "Teeth are the weak spot", "+6–21 % over baseline, 6 of 6 families",
    "Which parts of a letter fail: loops, teeth, tails?",
    "Labelled each letter by its skeleton features and compared its error with the same letter's average error.",
    "Letters with teeth are misread more than their baseline in every model family, and among one-tooth letters the error rises with the length of the run of teeth. Loops are above baseline in four of six families; tails and verticals are below.",
    "",
    textdemo([("ستين", "a run of teeth: س alone has three", True, "")]) + rangebars([("Letters with teeth", 6, 21, "hi")], 30, " % more errors"))
add("s", "B", "Confusions are visual, and shared by every model family", "ρ 0.63–0.90 between families",
    "Which letters get confused, and is it by look or by sound?",
    "Built a confusion matrix from every aligned substitution and compared the matrices across model families with a similarity-choice model.",
    "Pairs that differ only in dots are 3–18 times more confusable than other pairs. Models confuse letters that look alike, not letters that sound alike, and the pattern is a property of the script, not of one model.",
    "mueller2012letter",
    '<div class="pairs">' + "".join(f'<span class="ar big" lang="ar" dir="rtl">{a}</span>' for a in ["ب ت ث", "ج ح خ", "د ذ", "ر ز", "ف ق", "س ش", "ص ض", "ع غ"]) + '</div><p class="key">letters that differ only in their dots</p>')
add("s", "B", "Dots return from context for known words, not for unseen ones", "95 % restored by the top candidate",
    "Could a language prior put the dots back?",
    "Counted how many words share their dotless form with another word, and how often the most frequent such word is the right one.",
    "A quarter to a third of words share their dotless form, yet the most frequent word restores 95 % of the known ones. The dots matter for the fifth of words that are unseen.",
    "alshaibani2025dotless",
    rangebars([("Words sharing a dotless form", 25, 33, "mid")], 100, " %") + bars([("Known words restored by the top candidate", 95, "95", "lo")], 100, " %"))
add("g", "B", "Word gaps and word-part gaps look alike", "P = 0.68 (CRNN), 0.63 (VLMs)",
    "Is the space between words visibly wider than the space inside a word?",
    "Measured pixel gaps at word boundaries and between word parts on aligned lines, and counted the spaces models insert inside words.",
    "Six letters (ا د ذ ر ز و) never join the next one, so words split into parts with small gaps. Models insert spaces inside words at those gaps far more than chance (87 of 89 pairs). The space tax is 2–11 CER points, against 1–4 for dots.",
    "alluhaybi2020connect zeghli2025bigram",
    textdemo([("مد ر سة", "one word, three parts (مدرسة)", True, "word-spacing:.35em")]) + gap_curves())
add("g", "B", "Slant matters a little, and a test-time fix is nearly free", "KHATT 26.3 → 25.3 %",
    "Does writing slant explain part of the gap between datasets?",
    "Measured slant from stroke orientation, sheared lines toward another dataset's slant and re-scored, and trained with slant and stroke augmentation.",
    "Datasets lean differently. Shearing KHATT toward Muharaf's slant lowers CER from 26.3 to 25.3 %. Slant and stroke augmentation in training hurts on unseen sets by 0.6 to 0.8 points.",
    "wigington2017augment luo2020learn eltay2022adaptive",
    forest([("KHATT", -2, None, None), ("NAKBA", 5, 4, 6), ("Muharaf", 11, None, None)], -5, 15, "°", note="median slant; positive leans right")
    + textdemo([("ما شاء الله", "upright, like KHATT", False, ""), ("ما شاء الله", "leaning right, like Muharaf", False, "display:inline-block;transform:skewX(-11deg)")]))
add("g", "N", "Tone, paper noise and ruled lines do not matter", "no measurable CER effect",
    "Do paper tone, scan noise, ruled lines or baseline slope change recognition?",
    "Measured each on every line and regressed it against the line's error, and removed ruled lines to see what changed.",
    "Datasets differ a lot in tone and ruled lines, and contour histograms identify a line's dataset 70–73 % of the time. None of it tracks the CRNN's errors, so we keep them only as dataset descriptors.",
    "ye2013diqa",
    chips_plain(["paper tone", "paper noise", "ruled lines", "baseline slope", "contour histograms"]) + '<p class="key">all measured clearly, none moved CER</p>')
add("g", "B", "Dots vanish at low resolution", "43–64 % under 2 px at 32 px",
    "Do dots survive when the line is shrunk?",
    "Shrank lines to 64 and 32 px high and counted dot components smaller than 2 px.",
    "At the CRNN's 64 px dots survive. At 32 px, 43–64 % fall under 2 px. Pixels per character and line length predict errors in every family.",
    "",
    rangebars([("At 32 px high", 43, 64, "hi")], 100, " % of dots under 2 px") + '<p class="key">at 64 px most dots survive</p>')
add("l", "A", "A free-running LLM decoder makes recognition worse", "KHATT 26.4 → 40.0 %",
    "Would a language model reading the CRNN's features fix the recognition errors?",
    "Passed the frozen CRNN's features through a small projector into Qwen3-0.6B (and 1.7B) and fine-tuned with LoRA.",
    "Recognition gets worse on every set, and a model three times larger is no better. The output is more fluent, but the wrong words are fluent ones: 41–44 % of wrong words, against 24 % for the CRNN.",
    "farazi2026vlmocr",
    pairbars([("KHATT", 26.4, 40.0), ("NAKBA", 13.8, 20.8)], 45, lo_is_good=True, la="CRNN", lb="CRNN + LLM")
    + bars([("Wrong words that are fluent: CRNN", 24, "24", "lo"), ("with the LLM", 41, "41", "hi")], 50, " %"))
add("l", "B", "A word list alone cannot restore dots", "20–23 % of dot swaps accepted",
    "Could a dictionary catch the dot errors?",
    "Assembled lexicons (Ayaspell, aspell-ar, the Pecina list, Wikidata and GeoNames names, Arabic Wikipedia) and tested what they accept.",
    "The lexicon covers 95.8–99.4 % of the words in our test sets, but it also accepts a fifth of single dot-swap errors. KHATT's unseen words are mostly label typos.",
    "",
    rangebars([("Real words found", 95.8, 99.4, "lo"), ("Dot-swap errors accepted", 20, 23, "hi")], 100, " %"))
add("l", "B", "KHATT's text is far from its training text", "4.0 vs 3.4 vs 2.9 bits per character",
    "Is the text we ask models to read unusual compared with what they trained on?",
    "Computed character 5-gram cross-entropy under each training text and counted unseen words and unseen word parts.",
    "KHATT's text is the farthest, and unseen words hurt every model family.",
    "shannon1951",
    bars([("KHATT", 4.0, "4.0", "hi"), ("NAKBA", 3.4, "3.4", "mid"), ("Muharaf", 2.9, "2.9", "lo")], 4.5, " bits/char")
    + '<p class="key">5.9 % of KHATT\'s word parts never occur in training, against 3.6–3.9 % elsewhere</p>')
add("w", "A", "Fine-tuning on Muharaf rescues VLMs across scripts", "KHATT 77 → 30 % CER",
    "Can a modest amount of handwriting data turn a general vision-language model into a reader?",
    "Fine-tuned Qwen3-VL-4B with LoRA on 8,000 Muharaf lines and tested it on sets it never saw.",
    "The base model scores 77–119 % CER on cursive sets. One epoch removes 47–67 points everywhere, even on Maghrebi manuscripts that look nothing like Muharaf.",
    "farazi2026vlmocr kohut2023finetuning heakl2025kitab",
    pairbars([("KHATT", 77, 30), ("NAKBA", 89, 29), ("RASAM-1", 119, 52), ("RASAM-2", 102, 49), ("Muharaf", 90, 36)], 120, la="base model", lb="fine-tuned"))
add("w", "B", "Maghrebi conventions must be learned", "ق read as ف in 23–27 % of cases",
    "Does Maghrebi writing need its own training?",
    "Scored a CRNN with and without Maghrebi training on RASAM and tracked how often ق was read as ف.",
    "Maghrebi writers dot qāf and fāʾ differently from the Mashriqi convention, so a model trained elsewhere confuses them. Without Maghrebi training a CRNN reads RASAM at about 47 % CER instead of about 12 %.",
    "vidalgorene2024rasam2",
    pairbars([("CRNN on RASAM", 47, 12)], 50, la="no Maghrebi training", lb="with it")
    + rangebars([("Base model reads ق as ف", 23, 27, "hi")], 40, " % of qāf"))
add("w", "B", "CRNNs and VLMs fail on different lines", "KHATT 14.3 % vs 17.4 / 18.1 %",
    "Do the two model families get the same lines wrong?",
    "Compared per-line errors of the CRNN and the fine-tuned VLMs, and scored the per-line better of the two.",
    "Their difficulty correlates weakly, and the per-line better of the two is far below either, so a cross-family combination is a strong candidate.",
    "",
    pairbars([("KHATT", 17.4, 14.3), ("NAKBA test", 10.2, 4.1)], 20, la="CRNN alone", lb="better of CRNN and VLM")
    + '<p class="key">fine-tuned VLM alone: 18.1 % (KHATT), 18.7 % (NAKBA test)</p>')
add("w", "B", "On KHATT, the text matters more than the writer, until you train on it", "R² 0.152 vs 0.088",
    "How much of a line's error is the writer, and how much is the text?",
    "KHATT's lines are pieces of one fixed paragraph written by many people, so the same words appear in many hands. Split word error into text and writer shares (partial R²).",
    "For models not trained on KHATT the text explains more than the writer. For the KHATT-trained model it flips: once the paragraph's words are learned, what remains is the writer.",
    "mahmoud2014khatt",
    pairbars([("General CRNN", 0.152, 0.088), ("KHATT-trained CRNN", 0.03, 0.26)], 0.3, unit="", lo_is_good=False, la="text share", lb="writer share"))
add("w", "B", "Models look at the right letters and still misread them", "about 3× chance",
    "When a VLM misreads a letter, was it looking at the wrong place?",
    "Computed attention and gradient maps for Qwen3-VL-4B and checked whether they point at the token's own letters (a pointing game against the aligned letter spans).",
    "The maps point at a token's own letters about three times above chance, but wrong tokens look in the right place as often as correct ones. Errors are misreadings, not misdirected looking.",
    "",
    bars([("Chance", 1, "1×", "gr"), ("Attribution maps", 3, "3×", "lo")], 4, ""))

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
    ("Done", "Two GPU nights of recognisers, the full KHATT test set, and characterisation of dots, spacing, shapes and writers."),
    ("Running", "A VLM night: how much the models rely on the image, geometry and resolution probes, and shape targeting for NAKBA."),
    ("Next", "Decoding with a lexicon, writer adaptation curves, and a combination of the two model families."),
]
GLOSS = [
    ("CER", "Character error rate: the share of characters that are wrong against the ground truth. 25 % means about one letter in four."),
    ("CRNN", "A small recognition network (about 10 million parameters) trained on handwriting lines. Our baseline."),
    ("VLM", "A large vision-language model (Qwen3-VL, Fanar, AIN) that reads an image and writes the text."),
    ("Skeleton and dots", "Arabic letters share base shapes and differ by dots: ب ت ث are one skeleton with one, two or three dots."),
    ("PAW", "A part of a word: a run of joined letters. ا د ذ ر ز و don't join the next letter, so words break into parts."),
    ("Marks and hamza", "Diacritics and the hamza are written inconsistently across datasets, so how we score them changes the numbers."),
    ("[a, b]", "A 95 % interval from resampling lines. If it contains zero, the effect may be noise."),
]
DATASETS = [
    ("KHATT", "Modern handwriting from many writers. We use a 200-line panel and the full 1,038-line test set, most of which is one fixed paragraph."),
    ("Muharaf", "Ruq'ah handwriting stored at 60 px line height. The hardest set for every model, and our main training data."),
    ("NAKBA", "Degraded 20th-century memoirs by one writer: a 2,095-line test split and a 2,671-line blind set."),
    ("RASAM", "Maghrebi-script historical manuscripts (RASAM-1 and RASAM-2), with different dot conventions."),
    ("KITAB-Bench", "A benchmark collection that several of our sets come from."),
]


def chips(keys: str) -> str:
    out = []
    for k in keys.split():
        label, url = BIB[k]
        out.append(f'<a class="chip" href="{e(url)}" target="_blank" rel="noopener">{e(label)}</a>' if url else f'<span class="chip">{e(label)}</span>')
    return "".join(out)


def finding_cards() -> str:
    out = []
    for f in F:
        out.append(
            f'<article class="card" data-theme-key="{f["theme"]}"><div class="ctop"><span class="tag">{e(THEMES[f["theme"]])}</span>'
            f'<span class="g g{f["grade"]}" title="evidence grade">{f["grade"]}</span></div><h3>{e(f["head"])}</h3><p class="metric">{e(f["metric"])}</p>'
            f'<div class="cbody"><dl><dt>The question</dt><dd>{e(f["q"])}</dd><dt>What we did</dt><dd>{e(f["did"])}</dd><dt>What we found</dt><dd>{e(f["found"])}</dd></dl>'
            f'<div class="viz">{f["viz"]}</div></div>' + (f'<div class="chips">{chips(f["papers"])}</div>' if f["papers"] else "") + "</article>")
    return "".join(out)


def paper_rows() -> str:
    out = []
    for k, they, we, rel in PAPERS:
        label, url = BIB[k]
        name = f'<a href="{e(url)}" target="_blank" rel="noopener">{e(label)}</a>' if url else e(label)
        out.append(f'<div class="prow"><div class="pn">{name}<span class="rel">{e(rel)}</span></div><p><i>They</i> {e(they)}</p><p><i>We</i> {e(we)}</p></div>')
    return "".join(out)


CSS = """
:root{--bg:#f4f6f8;--surface:#ffffff;--ink:#13212c;--muted:#5a6b78;--line:#dbe2e8;--accent:#0b5d7e;--accent-soft:#dcecf3;
--ga:#0e7a55;--gb:#2a66b8;--gc:#a66a00;--gn:#6b7782;--lo:#2f8f7a;--mid:#c9962a;--hi:#c2503a;--gr:#aab6bf;
--f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;--f-ar:"Noto Naskh Arabic","Geeza Pro",serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e161d;--surface:#16212a;--ink:#e6edf2;--muted:#93a4b1;--line:#27363f;--accent:#62b6d6;--accent-soft:#16323f;
--ga:#4cc59a;--gb:#7aa7e6;--gc:#e0a84a;--gn:#9aa7b1;--lo:#4fb59f;--mid:#d9aa45;--hi:#e27a63;--gr:#5d6e79;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e161d;--surface:#16212a;--ink:#e6edf2;--muted:#93a4b1;--line:#27363f;--accent:#62b6d6;--accent-soft:#16323f;
--ga:#4cc59a;--gb:#7aa7e6;--gc:#e0a84a;--gn:#9aa7b1;--lo:#4fb59f;--mid:#d9aa45;--hi:#e27a63;--gr:#5d6e79;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--f-body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0}
.wrap{max-width:1080px;margin:0 auto;padding-block:28px 56px}
header{display:grid;grid-template-columns:1fr auto;gap:20px;align-items:end;padding-bottom:22px;border-bottom:1px solid var(--line)}
.eyebrow{font:500 11.5px var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
h1{font-size:clamp(26px,5vw,40px);line-height:1.12;margin:0 0 10px;text-wrap:balance;font-weight:600;letter-spacing:-.01em}
.lede{margin:0;max-width:62ch;color:var(--muted);font-size:16px}
.motif{font:700 clamp(40px,9vw,72px)/1 var(--f-ar);direction:rtl;color:var(--accent);text-align:center;margin:0}
.motif small{display:block;font:400 11px var(--f-body);direction:ltr;color:var(--muted);margin-top:6px;letter-spacing:.02em}
h2{font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:38px 0 14px;font-weight:600;font-family:var(--f-mono)}
.sub{margin:-6px 0 14px;color:var(--muted);font-size:14px;max-width:70ch}
.points,.gloss{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:12px}
.points{margin-top:22px}
.points p,.gloss div{margin:0;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:8px;font-size:14px;min-width:0}
.points b,.gloss b{display:block;font-size:12px;font-family:var(--f-mono);color:var(--accent);margin-bottom:4px;letter-spacing:.04em;text-transform:uppercase}
.gloss div{font-size:13.5px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.filters button{font:500 13px var(--f-body);color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:99px;padding:6px 13px;cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.filters button:focus-visible,.chip:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.list{display:flex;flex-direction:column;gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px 18px 15px;display:flex;flex-direction:column;gap:6px;min-width:0}
.card[hidden]{display:none}
.ctop{display:flex;justify-content:space-between;align-items:center;gap:10px}
.tag{font:500 11px var(--f-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.g{font:600 12px var(--f-mono);width:22px;height:22px;border-radius:50%;display:inline-grid;place-items:center;color:var(--bg);font-style:normal}
.gA{background:var(--ga)}.gB{background:var(--gb)}.gC{background:var(--gc)}.gN{background:var(--gn)}
.card h3{font-size:19px;line-height:1.28;margin:2px 0 0;font-weight:600;text-wrap:balance}
.metric{margin:0;font:500 14.5px var(--f-mono);color:var(--accent);font-variant-numeric:tabular-nums}
.cbody{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:22px;margin-top:8px;align-items:start}
dl{margin:0;display:flex;flex-direction:column;gap:3px;min-width:0}
dt{font:500 10.5px var(--f-mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin-top:8px}
dt:first-child{margin-top:0}
dd{margin:0;font-size:14px}
.viz{background:var(--bg);border-radius:8px;padding:14px;min-width:0;display:flex;flex-direction:column;gap:12px}
.key{margin:0;font-size:12px;color:var(--muted)}
.key i{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 5px 0 12px;vertical-align:-1px}.key i:first-child{margin-left:0}
.pb{display:flex;flex-direction:column;gap:8px}
.pbr{display:grid;grid-template-columns:minmax(64px,92px) 1fr;gap:8px;align-items:center;font-size:12.5px}
.pbr>div{display:flex;flex-direction:column;gap:3px}
.bl{color:var(--muted);line-height:1.25}
.bt{display:flex;align-items:center;gap:7px;height:12px}
.bt i{display:block;height:9px;width:var(--w);border-radius:5px;background:var(--lo);flex:none;max-width:calc(100% - 64px)}
.bt i.gr,.key i.gr{background:var(--gr)}.bt i.mid{background:var(--mid)}.bt i.hi,.key i.hi{background:var(--hi)}.bt i.lo,.key i.lo{background:var(--lo)}
.bt b{font:500 12px var(--f-mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.fo{display:flex;flex-direction:column;gap:9px}
.for{display:grid;grid-template-columns:minmax(64px,92px) 1fr auto;gap:8px;align-items:center;font-size:12.5px}
.for b{font:500 11.5px var(--f-mono);white-space:nowrap;font-variant-numeric:tabular-nums;min-width:0}
.tr{position:relative;height:16px;border-radius:4px}
.tr s{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--muted);text-decoration:none}
.tr .iv{position:absolute;top:6px;height:4px;border-radius:3px;background:var(--accent);opacity:.55}
.tr .pt{position:absolute;top:2px;width:11px;height:11px;border-radius:50%;background:var(--accent);margin-left:-5px;text-decoration:none}
.ax .tr{height:14px}.ax em{position:absolute;top:0;font:500 10.5px var(--f-mono);font-style:normal;color:var(--muted);transform:translateX(-50%)}
.ax em:first-child{transform:none}
.stack{display:flex;height:30px;border-radius:6px;overflow:hidden;background:var(--bg)}
.stack i{display:flex;align-items:center;padding:0 8px;font-style:normal;width:var(--w);min-width:0;overflow:hidden}
.stack i.hi{background:var(--hi)}.stack i.lo{background:var(--lo)}.stack i.mid{background:var(--mid)}
.stack b{color:#fff;font:500 11px var(--f-mono);white-space:nowrap}
.td{display:flex;flex-wrap:wrap;gap:6px 20px}
.td div{display:flex;flex-direction:column;gap:2px;align-items:flex-start}
.td small{font-size:11.5px;color:var(--muted)}
.ar{font-family:var(--f-ar);font-size:22px;line-height:1.5}.ar.big{font-size:30px;font-weight:700}
.pairs{display:flex;flex-wrap:wrap;gap:4px 18px}
.cp{display:flex;flex-direction:column;gap:10px}
.cp figure{margin:0;display:flex;flex-direction:column;gap:3px}
.cp img{display:block;width:100%;height:auto;border-radius:4px;background:#fff;border:1px solid var(--line)}
.cp figcaption{font-size:12px;color:var(--muted)}.cp b.hi{color:var(--hi)}.cp b.lo{color:var(--lo)}
.lg{display:grid;grid-template-columns:repeat(auto-fill,minmax(98px,1fr));gap:8px}
.lg figure{margin:0;border:2px solid var(--line);border-radius:6px;padding:4px;background:var(--surface);min-width:0}
.lg figure.bad{border-color:var(--hi)}.lg figure.ok{border-color:var(--lo)}
.lg div{display:flex;gap:4px;justify-content:center;background:#fff;border-radius:3px;padding:2px}
.lg img{height:44px;width:auto;max-width:48%;image-rendering:auto}
.lg figcaption{font-size:11px;color:var(--muted);text-align:center;margin-top:3px}.lg .ar{font-size:15px}
.gc{width:100%;height:auto}.gc .sv{font:500 9px var(--f-body);fill:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{font-size:11.5px;color:var(--muted);border:1px solid var(--line);border-radius:99px;padding:2px 9px;text-decoration:none;background:var(--bg)}
a.chip{color:var(--accent);border-color:var(--accent-soft);background:var(--accent-soft)}
.chips.plain{margin:0}
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
@media (max-width:760px){.cbody{grid-template-columns:1fr;gap:14px}}
@media (max-width:560px){header{grid-template-columns:1fr}.motif{text-align:left}.card{padding:14px 14px 13px}}
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
    gloss = "".join(f"<div><b>{e(a)}</b>{e(b)}</div>" for a, b in GLOSS)
    datasets = "".join(f"<div><b>{e(a)}</b>{e(b)}</div>" for a, b in DATASETS)
    return (
        '<title>Arabic OCR Brief</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Naskh+Arabic:wght@400;700&display=swap">\n'
        f"<style>{CSS}</style>\n"
        '<div class="wrap"><header><div><p class="eyebrow">Arabic handwriting recognition · research brief</p>'
        "<h1>Where recognisers fail on Arabic script, and why</h1>"
        '<p class="lede">What we measured, how, and what it means, with the papers each result relates to. Every finding shows its question, what we did, what we found, and the evidence. The full tables are in the ledger.</p></div>'
        '<p class="motif" aria-hidden="true">ب ت ث<small>one skeleton, three dot patterns</small></p></header>'
        f'<div class="points">{"".join(f"<p><b>{e(a)}</b>{e(b)}</p>" for a, b in points)}</div>'
        f'<h2>Terms in one line</h2><div class="gloss">{gloss}</div>'
        f'<h2>The datasets</h2><div class="gloss">{datasets}</div>'
        f'<h2>Findings</h2><div class="filters" role="group" aria-label="Filter findings by theme">{filters}</div>'
        f'<p class="legend">Evidence grade: {legend}</p><div class="list">{finding_cards()}</div>'
        f'<h2>The papers, and where we stand</h2><div class="papers">{paper_rows()}</div>'
        f'<h2>Where the work is</h2><div class="state">{"".join(f"<div><b>{e(a)}</b>{e(b)}</div>" for a, b in STATE)}</div>'
        f'<footer>Source: the research ledger, which holds every table, gallery and decision rule. <a href="{LEDGER}" target="_blank" rel="noopener">Open the full ledger</a>.</footer></div>'
        f"<script>{JS}</script>\n"
    )


if __name__ == "__main__":
    out = Path("docs/research/brief.html")
    out.write_text(render(), encoding="utf8")
    print("wrote", out, round(out.stat().st_size / 1024, 1), "KB;", len(F), "findings,", len(PAPERS), "papers")
