"""Build the project timeline (docs/research/timeline.html): step by step, why we did each thing, what came out,
with visual evidence. Reuses the brief's chart components. Evidence images come from docs/research/timeline_assets.json
(scripts/extract_timeline_assets.py). Usage: .geomvenv/bin/python scripts/build_timeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import build_brief as B

A = json.loads(Path("docs/research/timeline_assets.json").read_text(encoding="utf8"))
e = B.e
LEDGER, BRIEF = B.LEDGER, "https://claude.ai/artifact/7ZBSXytbDy5EY6s7VY36FB"


# ------------------------------------------------------------------ evidence blocks
def img(src, cap="", alt=""):
    return f'<figure class="ev"><img alt="{e(alt or cap)}" src="{src}" loading="lazy">' + (f"<figcaption>{e(cap)}</figcaption>" if cap else "") + "</figure>"


def line_card(c, labels=None, note=""):
    """A line image with the errors each system made on it."""
    ims = "".join(f'<img alt="{e(i["alt"])}" src="{i["src"]}" loading="lazy">' for i in c["images"][:2])
    errs = c["errs"][:3]
    lo = min(n for _, n in errs) if errs else 0
    rows = "".join(f'<li class="{"win" if n == lo and len(errs) > 1 else ""}"><span>{e(labels[i] if labels else l)}</span><b>{n} errors</b></li>' for i, (l, n) in enumerate(errs))
    return (f'<figure class="lc">{ims}<ul>{rows}</ul>' + (f'<figcaption>{e(note)}</figcaption>' if note else "")
            + (f'<bdi class="tr" dir="rtl" lang="ar">{e(c["truth"][:70])}</bdi>' if c.get("truth") else "") + "</figure>")


def pair_rows(t, label_a, label_b, k=2):
    out = ['<div class="xr">']
    for r in t["rows"][:k]:
        out.append(f'<figure><img alt="{e(label_a)}" src="{r[0]["src"]}" loading="lazy"><figcaption>{e(label_a)}</figcaption></figure>'
                   f'<figure><img alt="{e(label_b)}" src="{r[1]["src"]}" loading="lazy"><figcaption class="new">{e(label_b)}</figcaption></figure>')
    out.append("</div>")
    return "".join(out)


def aug_row(item, k=4):
    ps = [p for p in item["panels"] if p["src"]][:k]
    return (f'<div class="ag"><b>{e(item["name"])}</b><div>' + "".join(
        f'<figure><img alt="{e(p["caption"])}" src="{p["src"]}" loading="lazy"><figcaption>{"original" if p["orig"] else e(p["caption"][:44])}</figcaption></figure>' for p in ps)
        + "</div></div>")


def allo_row(a):
    return (f'<div class="al"><span class="ar" lang="ar">{e(a["char"])}</span><small>{e(a["form"])} · {a["n"]} examples</small><div>'
            + "".join(f'<figure><img alt="cluster" src="{t["src"]}" loading="lazy"><figcaption>{t["pct"]} %</figcaption></figure>' for t in a["tiles"][:6]) + "</div></div>")


def viz(*parts):
    return "".join(parts)


G = A["galleries"]
S = []  # steps


def step(phase, title, why, did, out, ev, papers=""):
    S.append(dict(phase=phase, title=title, why=why, did=did, out=out, ev=ev, papers=papers))


PHASES = [
    ("p1", "Before the academic turn", "We started with a benchmark and a few fine-tunes, to see how well anything reads Arabic handwriting."),
    ("p2", "Turning it into research", "We read the closest papers, characterised the data we had, and probed where models look."),
    ("p3", "Measuring the writing itself", "We measured slant, strokes and letter shapes, then tried fixes and learned which ones survive scrutiny."),
    ("p4", "Making the numbers trustworthy", "We looked line by line, caught our own mistakes, and wrote down a plan with rules for every result."),
    ("p5", "Running the plan", "We built the tooling and ran the characterisation, the GPU nights, and the clean-up."),
]

# ---------------------------------------------------------------- phase 1
step("p1", "Build a benchmark and sweep models",
     "Before fine-tuning anything we needed to know how existing systems read Arabic handwriting, with numbers we could trust.",
     "Built a harness that runs any OCR system on the same lines and scores it on a ladder: raw, light normalisation, no diacritics, and dotless skeleton. Test sets: KHATT, Muharaf, NAKBA, RASAM-1 and RASAM-2, and subsets of KITAB-Bench. Swept paid and free-tier VLMs, local models, PaddleOCR and Tesseract, with caching and cost review before any paid run.",
     "Muharaf was the hardest handwriting set for every model (best single model 34 % CER, against 5.5 % on KHATT). A general vision-language model, Qwen3-VL-4B, was unusable on cursive: 77 to 119 % CER.",
     B.bars([("KHATT", 77, "77", "hi"), ("NAKBA", 89, "89", "hi"), ("RASAM-2", 102, "102", "hi"), ("Muharaf", 90, "90", "hi"), ("RASAM-1", 119, "119", "hi")], 120, " %")
     + '<p class="key">base Qwen3-VL-4B, CER on each set before any fine-tuning</p>', "heakl2025kitab")
step("p1", "Fine-tune a VLM on one handwriting set",
     "If the base model fails, does a modest amount of in-domain handwriting fix it, and does the fix carry to scripts it never saw?",
     "LoRA-fine-tuned Qwen3-VL-4B for one epoch on 8,000 Muharaf lines and tested it on KHATT, NAKBA and the two Maghrebi RASAM sets.",
     "A large, transferable gain: 47 to 67 points on every held-out set, including Maghrebi manuscripts that look nothing like Muharaf.",
     B.pairbars([("KHATT", 77, 30), ("NAKBA", 89, 29), ("RASAM-1", 119, 52), ("RASAM-2", 102, 49), ("Muharaf", 90, 36)], 120, la="base", lb="fine-tuned"),
     "kohut2023finetuning farazi2026vlmocr")
step("p1", "Choose the base model, and unfreeze the vision encoder",
     "Is a model built for Arabic a better starting point, and is the frozen vision encoder holding the fine-tune back?",
     "Ran the same recipe on three bases (Qwen3-VL-4B, the Arabic-specialised Baseer-Nakba, PaliGemma-2-3B), then repeated the Qwen run with the vision encoder trained too.",
     "Specialised was not better: Baseer-Nakba already read NAKBA at 1.0 % (it was trained on it) and got worse after our recipe (3.4 %). Unfreezing the encoder helped everywhere, most on NAKBA, and it fixed letter identity more than dots (the frozen RASAM rows used only 20 lines, so those two are noisy).",
     B.pairbars([("KHATT", 30.0, 25.3), ("NAKBA", 32.6, 20.8), ("RASAM-1", 52.1, 43.2), ("RASAM-2", 49.4, 48.6)], 60, la="encoder frozen", lb="encoder trained"))
step("p1", "Find out which scores were not real",
     "Some numbers looked too good or too bad, and we could not tell whether the model or the harness was responsible.",
     "Audited the pipeline: where in the line errors fall, how long outputs were against the token limits, whether test lines also sat in training, and how images were stored.",
     "Several scores were artefacts. Models were cut off at token limits (gemini-flash dropped the tail of 44 to 46 % of lines), page-scale sets hit a 160-token cap, 86.5 % of one KITAB subset sat word for word in Muharaf train, and the public Muharaf copy is stored at only 60 px line height.",
     B.bars([("gemini-flash", 45, "44–46", "hi"), ("PaddleOCR", 31, "31", "mid")], 50, " % of lines tail-cut")
     + B.bars([("kitab_muharaf in Muharaf train", 86.5, "86.5", "hi")], 100, " %"), "hegghammer2021noisy baek2019wrong")

# ---------------------------------------------------------------- phase 2
step("p2", "Read the closest papers and borrow their yardstick",
     "We wanted to characterise Arabic script the way English had been characterised, and a group at Luleå had published four connected studies on the same data.",
     "Mapped what each paper measures to what we could measure. Recomputed their similar-substitution metric on their own published predictions, then on our models over the same lines.",
     "Their metric reproduced exactly (4.17 %). VLMs did not confuse dots less than the CRNN: per character they made as many similar-letter errors, with a lower share only because they made two to five times more of every other kind. Only 6 % of substitutions fell in our hand-drawn shape groups and 13 % were dot pairs, so the groups need to be learned.",
     B.pairbars([("Gap to Latin, 100 lines", 24.2, 11.8), ("Gap to Latin, full data", 6.0, 4.4)], 26, " pts", la="before label cleaning", lb="after") + '<p class="key">their result: cleaning labels halves the gap at 100 lines, and 4.4 points remain</p>'
     + B.bars([("dot pairs", 13, "13", "mid"), ("our shape groups", 6, "6", "lo"), ("no group", 65, "65", "hi")], 70, " %") + '<p class="key">our check: share of 185,902 letter substitutions</p>',
     "alazzawi2026gap alazzawi2026crossling")
step("p2", "Characterise the data we already had",
     "Before adding experiments, learn what the existing readings can already tell us about why lines are hard.",
     "Re-scored every reading on the [[ladder|normalisation ladder]], [[align|aligned edits]], and [[reg|regressed line difficulty]] on text and image features across complete model blocks.",
     "Dots are a small tax (median 1.8 CER points, 8 % of error). 60 % of spaces wrongly inserted inside words sit at gaps between word parts. Difficulty belongs to the line (split-half reliability 0.76 to 0.81), and image features lift the explained variance from 0.06–0.32 to 0.15–0.63.",
     B.stack([("skeleton", 92, "hi"), ("dots", 8, "mid")]) + '<p class="key">what the error is made of, median over 91 model × dataset pairs</p>'
     + B.rangebars([("Text features alone", 6, 32, "gr"), ("Text + image features", 15, 63, "lo")], 70, " %") + '<p class="key">share of the differences between lines that the features explain</p>', "alshaibani2025dotless")
step("p2", "Check how much label noise matters",
     "If some test labels are wrong, our rankings might be wrong.",
     "Used the published human review logs to drop the flagged Muharaf test lines and re-scored 13 models.",
     "126 of 1,334 lines are flagged, and dropping them moves CER by −0.4 to +0.9 points with no change of rank.",
     B.stack([("flagged", 9.4, "hi"), ("kept", 90.6, "lo")]) + B.forest([("CER change", None, -0.4, 0.9)], -2, 2, note="points, across 13 models; 0 rank changes"), "alazzawi2026labelerr")
step("p2", "Probe where a vision-language model looks",
     "An idea on the table was to protect or freeze the weights that matter for a given letter. Before guessing which weights, check where the model looks for that letter.",
     "On Qwen3-VL-4B with our fine-tuned adapter, computed attention, relative attention and gradient × (input − paper) maps for each generated token, hid the patches a map pointed at, and ran a pointing game against the aligned letter spans. Tried NVIDIA's LocateAnything-3B to find words and letters zero-shot.",
     "Three maps pass: hiding what they point to removes 4.3 to 5.1 nats of the token's log-probability, against 0.8 for random patches, and one puts 9× more attribution on ink than chance. Wrong tokens differ from correct ones in confidence, not in how sharp their maps are, so errors are misreadings of the right place. LocateAnything could not find an Arabic word, letter or dot.",
     viz(img(A["attribution"][0]["src"], "A RASAM-1 line and the maps for one generated token: attention, relative attention, gradient × input"),
         B.bars([("Map patches hidden", 4.7, "4.3–5.1", "lo"), ("Random patches", 0.8, "0.8", "gr")], 6, " nats lost"),
         img(A["attribution"][1]["src"], "LocateAnything zero-shot probe: boxes do not follow words, letters or dots")))

# ---------------------------------------------------------------- phase 3
step("p3", "Measure how the writing differs between datasets",
     "Datasets differ visibly. Which physical properties differ enough to explain why a model trained on one fails on another?",
     "Built detectors for slant (stroke orientation), stroke width and pen contrast, dots, baseline, paper tone, ruled lines and contour loci, and checked each against the transcriptions and with overlays on real lines.",
     "Slant is the clearest difference: KHATT is upright while Muharaf leans right, and 41 % of KHATT lines sit where under 5 % of training lines do. Pen contrast separates Maghrebi (0.5–0.6) from the rest (about 1.0). Tone, noise and ruled lines differ a lot, but contour histograms that identify a dataset 70–73 % of the time do not track the transfer gap. The dot detector tracks dotted letters with r from 0.31 to 0.78.",
     viz(B.forest([("KHATT", -2, None, None), ("NAKBA", 5, 4, 6), ("Muharaf", 11, None, None)], -5, 15, "°", note="median slant of near-vertical strokes; positive leans right"),
         '<div class="evs">' + "".join(img(d["src"], d["caption"]) for d in A["detectors"][:2]) + "</div>"))
step("p3", "Make one dataset look like another, and see if the errors follow",
     "Measuring differences does not show they matter. A causal test is to transform lines until their measurements match another dataset, then re-read them.",
     "Fitted a few knobs (shear, stroke width, sharpening, grey levels, noise) on 100 lines so twelve measures matched the target dataset, with single-step ablations, then re-scored the CRNN on the transformed lines.",
     "The calibration works (mean gap to the target 0.77 SD before, 0.23 after). The effects on errors are modest: shearing Muharaf upright raises CER from 6.2 to 8.2 %, shearing KHATT toward Muharaf's slant lowers it from 26.3 to 25.3 %, and thinning NAKBA's strokes to Muharaf's width lowers it from 13.6 to 12.5 %.",
     viz(pair_rows(A["transfer"]["muharaf_to_khatt"], "Muharaf original", "made to look like KHATT", 2),
         pair_rows(A["transfer"]["nakba_to_muharaf"], "NAKBA original", "strokes made like Muharaf", 1),
         B.pairbars([("Muharaf, sheared upright", 6.2, 8.2), ("KHATT, sheared like Muharaf", 26.3, 25.3), ("NAKBA, strokes thinned", 13.6, 12.5)], 30, la="original", lb="transformed", lo_is_good=True)),
     "wigington2017augment")
step("p3", "Catalogue augmentations and preview them on real lines",
     "The measurements suggested training augmentations, but an augmentation that changes what the line says silently poisons training.",
     "Listed 18 candidates, from the literature and our own, and rendered each on real KHATT, Muharaf and RASAM-1 lines with the label underneath, judging by eye whether the label still fits. Some candidates have a guard that rejects a variant (stroke thickening that would merge dots).",
     "A reviewed catalogue with a verdict per idea. Only two have been trained so far (slant and stroke width). The rest wait for the characterisation to say which are worth a GPU night.",
     '<div class="ags">' + "".join(aug_row(it) for it in A["aug"][:6]) + "</div>", "luo2020learn eltay2022adaptive")
step("p3", "Train with slant and stroke augmentation",
     "If slant and stroke width differ across datasets, a model trained on varied slant and stroke should travel better.",
     "Retrained the CRNN with random shear (−12° to +20°) and fractional stroke thickening and thinning applied on the GPU, and tested it on KHATT, NAKBA and NAKBA-blind.",
     "It became slant-robust (an upright Muharaf line costs +0.6 points instead of +2.0), but it hurt on unseen sets: NAKBA test +0.84, blind +0.79 and KHATT +0.59 points. Kept as a robustness default, not a gain.",
     viz(line_card(G["aug_khatt"][0], ["CRNN", "CRNN + augmentation"], "A line the augmented model reads better"),
         line_card(G["aug_khatt_hurt"][0], ["CRNN", "CRNN + augmentation"], "A line it reads worse"),
         B.forest([("NAKBA test", 0.84, 0.66, 1.01), ("NAKBA blind", 0.79, 0.64, 0.95), ("KHATT", 0.59, None, None)], -0.5, 1.5, note="CER change in points (positive = worse)")),
     "wigington2017augment luo2020learn eltay2022adaptive")
step("p3", "Notice that loose crops hurt, and crop to the ink",
     "A table of the worst KHATT lines showed writing filling a thin strip of a tall image: 10 % of lines had ink in under 35 % of the height, at 85 % CER.",
     "Cropped each line to the band holding its ink before resizing and re-read the lines. Later corrected our first number, which had excluded the 17 worst crops by a length filter.",
     "KHATT 31.0 → 23.3 % on all 200 lines, for every CRNN, with no retraining. The largest gain we found. VLMs are unaffected, though fine-tuned ones lose a little on loose crops.",
     viz(B.crop_pairs(), B.pairbars([("KHATT, 200 lines", 31.0, 23.3)], 35, la="as stored", lb="ink-cropped")), "retsinas2022best")
step("p3", "Ask whether unusual letter shapes explain the errors",
     "After the crop, what remains? The letters themselves might look unlike anything in training.",
     "Aligned each line to its letters with CTC forced alignment, gave each letter a feature vector, and measured its distance to the nearest training letter of the same character and position. Built a bank of 75,182 training letters and clustered it into allographs.",
     "Letter-shape novelty is the strongest error predictor we found: letter error climbs from 1–5 % in the most typical fifth to 30–53 % in the most novel band, in every dataset, confirmed with a second model's features.",
     viz(B.letter_grid(B.ASSETS["novel"], "Most novel KHATT letters (left) beside the nearest training letter (right). All misread."),
         B.letter_grid(B.ASSETS["typical"], "Typical letters, read correctly."),
         '<p class="key">Shapes the model has seen most (clusters of the training bank, with their share)</p>' + "".join(allo_row(a) for a in A["allographs"][:3]),
         B.rangebars([("Most typical fifth", 1, 5, "lo"), ("Most novel band", 30, 53, "hi")], 60, " % wrong")), "alazzawi2026gap zhang2020ran")
step("p3", "Try an LLM on top of the recogniser",
     "KHATT's text is far from the training text, so a language model reading the CRNN's features might repair what the image alone cannot.",
     "Passed the frozen CRNN's features through a small projector into Qwen3-0.6B (later 1.7B) and fine-tuned with LoRA.",
     "It got worse on every set (KHATT 26.4 → 40.0 %, NAKBA 13.8 → 20.8 %), and a model three times larger was no better. The words are fluent but wrong: 41 to 44 % of wrong words, against 24 % for the CRNN.",
     viz(line_card(G["e8_khatt"][0], ["CRNN", "CRNN + LLM"], "A line the LLM helps"), line_card(G["e8_khatt_hurt"][0], ["CRNN", "CRNN + LLM"], "A line it ruins"),
         B.pairbars([("KHATT", 26.4, 40.0), ("NAKBA", 13.8, 20.8)], 45, la="CRNN", lb="CRNN + LLM")), "farazi2026vlmocr")
step("p3", "Find out the hamza is a labelling convention",
     "Many NAKBA errors involved alif with hamza, and the CRNN's reading looked right to a human.",
     "Counted hamza use per label set and re-scored NAKBA with the hamza folded into plain alif.",
     "Muharaf writes the hamza on 4 % of alifs, NAKBA's labels on 16 %, and NAKBA writers do not draw it. Folding it removes about 2 points (13.8 → 11.6 %). The model emits a hamza when its training labels did, regardless of the image.",
     viz(line_card(G["hamza"][0], ["as written", "hamza folded"], "The same reading, scored two ways"),
         B.pairbars([("NAKBA CER", 13.8, 11.6)], 16, la="as written", lb="hamza folded")), "doctor2022graphemic")
step("p3", "Add the missing shapes to training",
     "If novel shapes cause errors, adding training lines that contain them should help.",
     "Chose lines by shape novelty using half of the KHATT data (nothing from the other half), repeated them three times in training, and compared with the same number of random repeats, on the held-out half.",
     "A small gain on held-out KHATT lines (−0.8 points, interval touching zero), about 2 points in the two most novel letter bands, and nothing on NAKBA, where the gain was plain repetition.",
     viz(line_card(G["targeted"][0], ["random repeats", "shape-targeted"], "A held-out line"),
         B.forest([("KHATT, held-out", -0.8, -1.6, 0.1)], -3, 2, note="CER change vs random repeats, points")), "swayamdipta2020cartography")
step("p3", "Run the overnight stacking and noise checks",
     "Single comparisons can fool us. Do the fixes stack, and how much does a score move when nothing changes but the seed?",
     "Trained the same recipe with different seeds, combined crop with shape-targeted repeats, tried slant test-time augmentation gated on measured slant, and retried the LLM decoder at three times the size.",
     "Two identical trainings differ by +0.4 points on a 91-line half, so only larger effects count. Crop and shape targeting each replicated, and the combined recipe (shape-targeted repeats on cropped data) was tested against each alone. Slant test-time augmentation is applied only where a line's slant is unusual, which keeps it nearly free.",
     viz(line_card(G["stack"][0], ["original, ink-cropped", "combined, ink-cropped"], "Crop plus shape-targeted training"),
         line_card(G["tta"][0], ["upright", "confidence-selected shear"], "Slant test-time augmentation"),
         B.forest([("Seed noise", 0.4, -0.6, 1.4), ("Shape targeting", -0.8, -1.6, 0.1), ("Ink crop", -7.5, -10.6, -4.9)], -12, 3, note="CER change in points with 95 % interval")))

# ---------------------------------------------------------------- phase 4
step("p4", "Go line by line, and correct our own mistakes",
     "Mean scores hide how errors are distributed. We needed per-line and per-letter evidence for every claim.",
     "Studied the distribution of lines, every comparison against the seed-noise yardstick, a line × system matrix, 18 pattern detectors, where spaces fall, which letters move, joint predictors of difficulty, and whether confusions follow sound or look. Kept a corrections log.",
     "Twenty lines are hard for every system and feed the label review. Spaces inserted inside words fall at gaps between word parts (87 of 89 pairs). Confusions follow look, not sound. The log recorded that our first crop number hid the worst crops, that PaddleOCR's output was reversed by a flag, and that one model's score (AIN) was set aside until a memorisation check is done.",
     viz('<div class="evs">' + "".join(img(h["src"], f'median CER {h["median_cer"]} %') for h in A["hardcore"]) + "</div>",
         '<p class="key">Lines no system reads: the first queue for label review</p>'), "mueller2012letter alluhaybi2020connect")
step("p4", "Test whether the shape library is feasible",
     "A library of real letter shapes could feed synthesis and a shape-token recogniser, but only if letters can be cut out reliably.",
     "Checked the CTC letter spans against the ink and counted how often a line's ink components match its label's word parts.",
     "Spans are narrow anchors, not cut lines (median 12 px at 64 px height), and handwritten word parts often touch: ink components match the label's parts exactly on only 6–9 % of lines. A library needs alignment and ink together.",
     img(A["contact"]["src"], "Letter crops from the CTC spans, by letter and form: the boxes show how narrow the spans are"))
step("p4", "Write the map, the plan and the rules",
     "By now there were far more threads than we could hold. Each needed a place, a grade, and a rule for what result would change what.",
     "Built a research map of 12 questions and, at last count, 168 items with grades and evidence links (138 at first), a coverage check that every idea is placed, a plan with a decision rule written before each run, and a claims register. Later folded everything into a 105-item master checklist.",
     "A single list to work from, with every unfinished item placed and each result's decision rule fixed in advance.",
     B.chips_plain(["168 items", "12 questions", "coverage check", "decision rules first", "claims register", "105-item checklist"]))

# ---------------------------------------------------------------- phase 5
step("p5", "Fix the standards before running anything new",
     "Many small effects survived only because we looked at many cells at once, and WER and fixed-vs-broken lines were missing from the comparisons.",
     "Added Benjamini–Hochberg control (q = 0.1) to the letter and detector tables, WER and lines fixed or broken to every comparison, and a seed-instability tool with a new detector.",
     "The KHATT replicate's surviving letter cells fall from 2 to 0, while the ink crop keeps 41. 761 lines change between seeds of one recipe, and single-seed 'better' lines are two to three times enriched in them.",
     B.pairbars([("KHATT replicate, surviving cells", 2, 0), ("Ink crop, surviving cells", 41, 41)], 45, " cells", lo_is_good=False, la="before FDR", lb="after FDR"))
step("p5", "Add the full KHATT test set and its structure",
     "The 200-line panel is small, and KHATT's copy turned out to be unusual.",
     "Added the full 1,038-line KHATT v1.0 test set (as stored and cropped), flagged fixed text and overlap with KITAB-KHATT, and rebuilt pseudo-pages by aligning lines to the one paragraph and linking by handwriting style.",
     "Almost every line is a piece of one 77-word paragraph (1,035 of 1,038), so a fixed-text split is not possible. As a natural experiment, text explains more word error than the writer for generic CRNNs; for the KHATT-trained model the writer dominates. The images are already tight crops.",
     B.stack([("fixed text", 99.7, "hi"), ("", 0.3, "lo")]) + B.pairbars([("General CRNN", 0.152, 0.088), ("KHATT-trained CRNN", 0.03, 0.26)], 0.3, unit="", lo_is_good=False, la="text share", lb="writer share"),
     "mahmoud2014khatt")
step("p5", "Assemble a lexicon and text sample",
     "Decoding with a word list needs one that is large, licensed and measured against our data.",
     "Merged Ayaspell, aspell-ar, the Pecina list, Wikidata and GeoNames names and an Arabic Wikipedia vocabulary, recorded licences, and measured coverage of the panel's words.",
     "It covers 95.8 to 99.4 % of the words in our test sets. It also accepts 20 to 23 % of single dot-swap errors, so a lexicon alone cannot restore dots.",
     B.rangebars([("Real words found", 95.8, 99.4, "lo"), ("Dot-swap errors accepted", 20, 23, "hi")], 100, " %"))
step("p5", "Detect more failure patterns",
     "Some error types still had no detector: neighbour-line intrusion, the lam-alef ligature, memorisation and seed instability.",
     "Added four detectors and a gallery of best, worst and random flagged lines for each.",
     "Lam and alef inside the ligature are misread 1.6× as often as elsewhere. AIN-7B reads 17.5 % of KITAB-KHATT lines exactly (17× its nearest peers) and copies 54 % of the transcription's whitespace quirks, which we still treat as unverified.",
     B.bars([("AIN-7B", 17.5, "17.5", "hi"), ("nearest peers", 1.0, "≈ 1", "gr")], 20, " % exact lines"))
step("p5", "Audit the datasets",
     "Scores only mean something if the label sets, splits and difficulty are understood.",
     "Compared rankings with and without diacritics, measured leakage by text and image hash, counted spelling variants, fitted item-response difficulty, and registered challenge subsets.",
     "Marks re-order leaderboards on three sets, so the headline is the no-diacritics score. Leakage runs from under 1 % (RASAM) to 95 % (full KHATT), and memorisation inflates only NAKBA-blind, so its headline uses clean lines.",
     B.bars([("RASAM", 1.1, "≤ 1.1", "lo"), ("Muharaf", 7.5, "7.5", "lo"), ("NAKBA blind", 16.3, "16.3", "mid"), ("NAKBA test", 18.3, "18.3", "mid"),
             ("KITAB-KHATT", 44.5, "44.5", "hi"), ("Full KHATT", 95.0, "95", "hi")], 100, " %"), "doctor2022graphemic")
step("p5", "Characterise skeletons, spacing, confusions and word parts",
     "With the numbers trustworthy, return to the original question: what makes Arabic script hard for a recogniser?",
     "Measured gap widths, teeth, loops and tails, dot-writing styles, vertical stacking, a similarity-choice model of confusions, an inventory of word parts, tokenizer views and root sharing.",
     "Word and part gaps overlap, so spacing is visually ambiguous. Teeth beat the letter baseline in all six model families. Confusability is shared across families and follows shape. An 8,000-type inventory of word parts covers 95.9 to 98.4 % of tokens, and real-word substitutions share roots more than expected.",
     B.gap_curves() + B.rangebars([("PAW inventory coverage", 95.9, 98.4, "lo"), ("Teeth over baseline", 6, 21, "hi")], 100, " %"),
     "mueller2012letter alluhaybi2020connect zeghli2025bigram")
step("p5", "Find the shortest text that covers every letter form",
     "A short calibration text that a writer could copy would let us adapt a model to one hand.",
     "Searched Arabic Wikipedia and KHATT labels for sentences covering every letter in every positional form.",
     "One or two lines cannot do it: final and isolated forms only occur at word ends, so full coverage needs at least 44 words. The best single sentence covers 65 of 104 forms, three cover 93, and seven or eight cover all.",
     B.bars([("1 sentence", 65, "65", "mid"), ("2 sentences", 85, "85", "mid"), ("3 sentences", 93, "93", "lo"), ("7–8 sentences", 104, "104", "lo")], 104, " of 104 forms"))
step("p5", "Fix the harness and raise the token caps",
     "Two harness problems from the first phase were still open: PaddleOCR's reversed output and truncated model outputs.",
     "Wrote patches that stop the reversal and record why each generation stopped, add per-model token caps and token log-probabilities, and re-scored PaddleOCR from its stored readings.",
     "PaddleOCR improves on every set once its output is in the right order, and the report lists which models still hit their caps.",
     B.pairbars([("KHATT", 85.8, 62.9), ("Muharaf", 87.9, 77.1), ("NAKBA", 89.3, 83.4), ("RASAM-1", 90.6, 88.2), ("RASAM-2", 91.2, 85.7)], 100, la="as stored", lb="order fixed"))
step("p5", "Run GPU night 1: recognisers on the full set",
     "The full KHATT set, an in-domain KHATT recogniser, a skeleton-only recogniser, hard-example repeats, writer adaptation and a Maghrebi few-shot curve all needed GPU time.",
     "Queued eight hours of CRNN jobs on the 3090 with checkpoints every ten steps, brought the results back and backed them up.",
     "Done and backed up. Generic CRNNs read the full KHATT test at about 30 to 32 % CER while the in-domain KHATT model reaches about 9 %, which shows how much of the gap is domain. The rest is not analysed yet.",
     B.bars([("shape-targeted, seed 2", 30.7, "30.7", "mid"), ("hard-example, seed 3", 31.4, "31.4", "mid"), ("trained on KHATT", 9.2, "9.2", "lo")], 35, " % CER, full KHATT test"))
step("p5", "Run GPU night 2 and tidy the project",
     "The VLM questions (how much models rely on the image, geometry and resolution checks, in-context writer calibration) and the NAKBA shape-targeting runs were next, and the project had split across several branches.",
     "Built and launched the VLM night with a follow-up queue, merged every branch into one, wrote the ledger's overview, slimmed the ledger for phones, and made this timeline and the brief.",
     "Night 2 was launched and its results are not in yet; the 3090 became unreachable afterwards. All code and results are on one branch.",
     B.chips_plain(["night 2 launched", "follow-up queue waiting", "one branch", "ledger overview", "visual brief", "this timeline"]))


import re as _re

P = json.loads(Path("docs/research/timeline_popups.json").read_text(encoding="utf8"))
TERM = _re.compile(r"\[\[(\w+)\|([^\]]+)\]\]")


def rich(text: str) -> str:
    """Escape text, then turn [[id|label]] into a button that opens the pop-up with that id."""
    return TERM.sub(lambda m: f'<button type="button" class="term" data-pop="{m.group(1)}" aria-haspopup="dialog">{m.group(2)}</button>', e(text))


RUNG_INFO = {
    "raw": ("raw", "As stored. Nothing changed."),
    "canonical": ("canonical", "Unicode clean-up: invisible controls removed, presentation forms folded back to letters, spaces collapsed."),
    "tatweel_digits": ("tatweel and digits", "The stretching mark ـ is removed and Arabic-Indic digits (١٢٣) become 123."),
    "orthographic": ("spelling folds", "Spelling variants are treated as the same: the alef forms, ي and ى, ة and ه, and hamza carriers."),
    "no_diacritics": ("no diacritics", "Every mark (fatha, kasra, damma, shadda and the rest) is removed."),
    "rasm": ("rasm", "Dots are removed: every letter becomes its dotless skeleton."),
    "no_spaces": ("no spaces", "Spaces are removed, so only letter errors remain."),
}


def pop_ladder() -> str:
    ex = P["example"]
    top = max(r["edits"] for r in ex["rungs"]) or 1
    rows = []
    for r in ex["rungs"]:
        name, desc = RUNG_INFO[r["name"]]
        rows.append(
            f'<div class="lr"><div class="lh"><b>{e(name)}</b><span>{e(desc)}</span></div>'
            f'<div class="ls"><span class="ar" dir="rtl" lang="ar">{e(r["gt"])}</span><span class="ar hy" dir="rtl" lang="ar">{e(r["hyp"])}</span></div>'
            f'<div class="le"><i style="--w:{r["edits"] / top * 100:.0f}%"></i><b>{r["edits"]}</b></div></div>')
    st = P["stair"]
    names = {r["name"]: RUNG_INFO[r["name"]][0] for r in st["rungs"]}
    stair = B.bars([(names[r["name"]], r["cer"] * 100, f'{r["cer"] * 100:.1f}', "lo" if r["name"] != "no_spaces" else "gr") for r in st["rungs"]], 35, " %")
    return (
        "<p>The same reading, scored again and again. Each rung removes one more kind of detail from <i>both</i> the reference and the reading, then counts the errors that are left. "
        "Errors that disappear at a rung were only about that detail. What remains at the end is letter identity.</p>"
        f'<h4>One real line</h4><p class="sm">Reference on top, a fine-tuned model\'s reading below, and the number of edits at each rung (line {e(ex["sample"])}). The edits fall from {ex["rungs"][0]["edits"]} to {ex["rungs"][-1]["edits"]}.</p>'
        f'<div class="ladder">{"".join(rows)}</div>'
        f'<h4>The same staircase over a whole set</h4><p class="sm">Corpus CER of a fine-tuned Qwen3-VL-8B on {st["n_lines"]} NAKBA lines, at each rung. '
        "The last bar rises because removing spaces also shortens the reference, so the rate can go up even though fewer errors remain.</p>"
        f"{stair}"
        '<h4>What we use it for</h4><p class="sm">The <b>dot tax</b> is the drop between the no-diacritics rung and the rasm rung: error that only the dots explain. '
        "Here it is 2.2 points (27.7 → 25.5 %), and about nine tenths of the error is still there once the dots are gone.</p>")


def pop_align() -> str:
    ex = P["example"]
    c = ex["counts"]
    ref_len = c["eq"] + c["sub"] + c["del"]
    cols = []
    for op, g, h in ex["ops"]:
        gm = e(g) if g else "–"
        hm = e(h) if h else "–"
        mark = {"eq": "=", "sub": "×", "del": "−", "ins": "+"}[op]
        cols.append(f'<div class="col {op}"><span class="ar">{gm}</span><i>{mark}</i><span class="ar">{hm}</span></div>')
    edits = c["sub"] + c["del"] + c["ins"]
    return (
        "<p>To count errors, the reading is lined up against the reference character by character using the fewest possible changes. "
        "Every change is one of three kinds, and CER is simply the number of changes divided by the length of the reference:</p>"
        '<div class="legend2"><span class="k eq">= match</span><span class="k sub">× substitution: a wrong letter</span><span class="k del">− deletion: a letter the model skipped</span><span class="k ins">+ insertion: a letter the model added</span></div>'
        f'<h4>One real line, aligned</h4><p class="sm">Reference above, reading below. Read right to left, as the Arabic is.</p>'
        f'<div class="scroll"><div class="agrid" dir="rtl">{"".join(cols)}</div></div>'
        f'<p class="eq2"><b>{c["sub"]}</b> substitutions + <b>{c["del"]}</b> deletions + <b>{c["ins"]}</b> insertions = <b>{edits}</b> edits, over <b>{ref_len}</b> reference characters: '
        f'CER = {edits}/{ref_len} = <b>{edits / ref_len * 100:.0f} %</b>.</p>'
        '<h4>Why it matters</h4><p class="sm">Counting is not enough. Because every edit has a place and a kind, we can ask where errors fall (the end of the line, the gaps inside words, dots) and what kind they are. '
        "That is how we found that spaces cost more than dots. It also shows why CER can pass 100 %: insertions have no ceiling.</p>")


def pop_reg() -> str:
    r2 = [("Historical books", 0.32, 0.58), ("KITAB online KHATT", 0.16, 0.43), ("RASAM-1", 0.11, 0.36), ("NAKBA lines", 0.51, 0.63),
          ("RASAM-2", 0.10, 0.25), ("NAKBA blind", 0.17, 0.27), ("KITAB-KHATT", 0.32, 0.39), ("Muharaf", 0.06, 0.15)]
    chart = B.pairbars(r2, 0.7, unit="", lo_is_good=False, la="text features only", lb="text + image features")
    return (
        "<p>Each line has a <b>difficulty</b>: how badly the models read it, taken as the median error over a block of models that all read exactly the same lines. "
        "A <b>regression</b> predicts that number from properties of the line, by fitting a weight for each property. If the properties predict difficulty well, they explain why a line is hard.</p>"
        '<div class="flow"><div><b>Text features</b><span>length, rare letters, unseen words, digits</span></div>'
        '<div><b>Image features</b><span>size, ink density, stroke width, contrast, blur, skew, ink per character</span></div>'
        '<i>→</i><div class="fit"><b>Fitted weights</b><span>one per feature</span></div><i>→</i>'
        '<div class="fit"><b>Predicted difficulty</b><span>compared with the actual line CER</span></div></div>'
        '<h4>What the score means</h4><p class="sm"><b>R²</b> is the share of the differences between lines that the features explain: 0 is none, 1 is all. We fit on ranks (which line is harder than which), so a few extreme lines do not dominate.</p>'
        f'<h4>Real results, by set</h4>{chart}'
        '<p class="sm">Text alone explains 0.06 to 0.32 of the differences (NAKBA lines, 0.51, is the exception). Adding image features lifts it to 0.15 to 0.63, most on the historical books, online KHATT and RASAM-1 (about +0.25), and least on KHATT (+0.07). '
        "Single image coefficients turned out unstable, so we only trust the combined fit, not any one feature on its own.</p>")


POPS = {
    "ladder": ("The normalisation ladder", pop_ladder),
    "align": ("Aligned edits", pop_align),
    "reg": ("Regressed line difficulty", pop_reg),
}

EXTRA_CSS = """
.tl{position:relative;margin-left:14px;border-left:2px solid var(--line);padding-left:22px}
.tl .step{position:relative;margin:0 0 22px}
.tl .step::before{content:"";position:absolute;left:-30px;top:6px;width:14px;height:14px;border-radius:50%;background:var(--accent);border:3px solid var(--bg)}
.ph{margin:34px 0 14px -14px;padding:12px 16px;background:var(--accent-soft);border-radius:8px;border-left:4px solid var(--accent)}
.ph h2{margin:0 0 2px;font:600 17px var(--f-body);text-transform:none;letter-spacing:0;color:var(--ink)}
.ph p{margin:0;font-size:13.5px;color:var(--muted)}
.ph .pn{font:500 11px var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.sn{font:500 11px var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.step .card{background:var(--surface)}
.step h3{margin:2px 0 6px;font-size:18px}
.ev{margin:0;display:flex;flex-direction:column;gap:4px}
.ev img{display:block;width:100%;height:auto;border-radius:4px;border:1px solid var(--line);background:#fff}
.ev figcaption,.lc figcaption,.xr figcaption{font-size:11.5px;color:var(--muted)}
.evs{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.lc{margin:0;display:flex;flex-direction:column;gap:6px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:8px}
.lc img{display:block;max-width:100%;height:auto;border-radius:3px;background:#fff}
.lc ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px;font-size:12px}
.lc li{display:flex;justify-content:space-between;gap:8px;color:var(--muted)}.lc li.win b{color:var(--lo)}.lc li b{font:500 12px var(--f-mono)}
.lc .tr{font:400 15px var(--f-ar);color:var(--ink)}
.xr{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.xr figure{margin:0;display:flex;flex-direction:column;gap:3px}.xr img{width:100%;height:auto;display:block;border-radius:3px;background:#fff;border:1px solid var(--line)}
.xr .new{color:var(--accent)}
.ags{display:flex;flex-direction:column;gap:10px}
.ag b{font:500 11.5px var(--f-mono);color:var(--muted);display:block;margin-bottom:4px}
.ag>div{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:6px}
.ag figure{margin:0}.ag img{width:100%;height:auto;display:block;border-radius:3px;background:#fff;border:1px solid var(--line)}
.ag figcaption{font-size:10.5px;color:var(--muted);line-height:1.25;margin-top:2px}
.al{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;align-items:center;margin-top:6px}
.al .ar{font-size:26px;font-weight:700;grid-row:span 2}.al small{font-size:11px;color:var(--muted)}
.al>div{display:flex;gap:6px;flex-wrap:wrap}.al figure{margin:0;text-align:center}.al img{height:40px;width:auto;background:#fff;border-radius:3px;border:1px solid var(--line)}
.al figcaption{font-size:10.5px;color:var(--muted)}
.why{background:var(--bg);border-radius:6px;padding:8px 10px}
.jump{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0}
.jump a{font:500 13px var(--f-body);color:var(--accent);text-decoration:none;border:1px solid var(--accent-soft);background:var(--accent-soft);border-radius:99px;padding:5px 12px}

.term{font:inherit;color:var(--accent);background:none;border:0;border-bottom:1.5px dotted var(--accent);padding:0;cursor:help}
.term:hover{background:var(--accent-soft)}.term:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.pop{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center;background:rgba(8,14,20,.62);padding:20px max(16px,env(safe-area-inset-right,0px)) 20px max(16px,env(safe-area-inset-left,0px))}
.pop[hidden]{display:none}
.pp{background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:10px;max-width:700px;width:100%;max-height:88vh;overflow:auto;padding:18px 20px 20px;box-shadow:0 12px 40px rgba(0,0,0,.35)}
.pp header{display:flex;justify-content:space-between;align-items:center;gap:12px;border:0;padding:0 0 8px;margin:0}
.pp header h3{margin:0;font-size:19px}
.pp .x{font:500 13px var(--f-body);background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:99px;padding:5px 12px;cursor:pointer;flex:none}
.pp p{margin:8px 0;font-size:14px}.pp p.sm{font-size:13px;color:var(--muted)}
.pp h4{margin:16px 0 4px;font:500 11px var(--f-mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.ladder{display:flex;flex-direction:column;gap:8px}
.lr{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 12px;padding:8px 10px;background:var(--bg);border-radius:6px}
.lh{grid-column:1/-1;display:flex;flex-direction:column}.lh b{font:500 12px var(--f-mono);color:var(--accent)}.lh span{font-size:12px;color:var(--muted)}
.ls{display:flex;flex-direction:column;min-width:0}.ls .ar{font-size:17px;line-height:1.5;overflow-wrap:anywhere}.ls .hy{color:var(--hi)}
.le{display:flex;align-items:center;gap:6px;min-width:70px}.le i{display:block;height:8px;width:var(--w);max-width:60px;border-radius:4px;background:var(--hi)}.le b{font:500 12px var(--f-mono)}
.legend2{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12.5px;margin:8px 0}
.k{display:inline-flex;align-items:center;gap:5px}.k.eq{color:var(--muted)}.k.sub{color:var(--hi)}.k.del{color:var(--mid)}.k.ins{color:var(--gb)}
.scroll{overflow-x:auto;padding-bottom:6px}
.agrid{display:inline-grid;grid-auto-flow:column;gap:2px}
.col{display:grid;grid-template-rows:auto auto auto;justify-items:center;width:27px;padding:2px 0;border-radius:4px}
.col .ar{font-size:19px;line-height:1.4}.col i{font:500 11px var(--f-mono);font-style:normal;color:var(--muted)}
.col.sub{background:color-mix(in srgb,var(--hi) 22%,transparent)}.col.del{background:color-mix(in srgb,var(--mid) 26%,transparent)}.col.ins{background:color-mix(in srgb,var(--gb) 24%,transparent)}
.eq2{font-size:14px}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:10px 0}
.flow>div{flex:1 1 150px;min-width:0;background:var(--bg);border-radius:6px;padding:8px 10px}.flow b{display:block;font-size:13px}.flow span{font-size:12px;color:var(--muted)}
.flow>i{align-self:center;font-style:normal;color:var(--muted);font-size:18px}.flow .fit{flex:0 1 140px}
@media (max-width:560px){.tl{margin-left:6px;padding-left:16px}.tl .step::before{left:-24px}.ph{margin-left:-6px}}
"""



POP_JS = """
(function(){var last=null;
function close(){document.querySelectorAll('.pop').forEach(function(p){p.hidden=true});document.body.style.overflow='';if(last){last.focus();last=null}}
document.querySelectorAll('.term').forEach(function(b){b.addEventListener('click',function(){var p=document.getElementById('pop-'+b.getAttribute('data-pop'));if(!p)return;last=b;p.hidden=false;document.body.style.overflow='hidden';p.querySelector('.x').focus()})});
document.querySelectorAll('.pop').forEach(function(p){p.addEventListener('click',function(ev){if(ev.target===p)close()});p.querySelector('.x').addEventListener('click',close)});
document.addEventListener('keydown',function(ev){if(ev.key==='Escape')close()});})();
"""


def render() -> str:
    parts = [
        '<title>Research Timeline</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Naskh+Arabic:wght@400;700&display=swap">\n'
        f"<style>{B.CSS}{EXTRA_CSS}</style>\n",
        '<div class="wrap"><header><div><p class="eyebrow">Arabic handwriting recognition · project timeline</p>'
        "<h1>How the project unfolded, step by step</h1>"
        '<p class="lede">What we did in order, why we did it, and what came out, with the evidence from each step. Steps run in the order we took them, not by date. '
        f'The findings behind them are in <a href="{BRIEF}" target="_blank" rel="noopener">the brief</a> and every table is in <a href="{LEDGER}" target="_blank" rel="noopener">the ledger</a>.</p></div>'
        '<p class="motif" aria-hidden="true">١ ٢ ٣<small>one step at a time</small></p></header>',
        '<nav class="jump" aria-label="Phases">' + "".join(f'<a href="#{pid}">{e(t)}</a>' for pid, t, _ in PHASES) + "</nav>",
        '<div class="tl">',
    ]
    n = 0
    cur = None
    for s in S:
        if s["phase"] != cur:
            cur = s["phase"]
            pid, pt, pd = next(p for p in PHASES if p[0] == cur)
            idx = [p[0] for p in PHASES].index(cur) + 1
            parts.append(f'<div class="ph" id="{pid}"><span class="pn">Phase {idx}</span><h2>{e(pt)}</h2><p>{e(pd)}</p></div>')
        n += 1
        parts.append(
            f'<section class="step"><div class="card"><span class="sn">Step {n}</span><h3>{e(s["title"])}</h3>'
            f'<div class="cbody"><dl><dt>Why</dt><dd>{rich(s["why"])}</dd><dt>What we did</dt><dd>{rich(s["did"])}</dd><dt>What came out</dt><dd>{rich(s["out"])}</dd></dl>'
            f'<div class="viz">{s["ev"]}</div></div>' + (f'<div class="chips">{B.chips(s["papers"])}</div>' if s["papers"] else "") + "</div></section>")
    parts.append("</div>")
    for pid, (ptitle, fn) in POPS.items():
        parts.append(f'<div class="pop" id="pop-{pid}" role="dialog" aria-modal="true" aria-label="{e(ptitle)}" hidden><div class="pp"><header><h3>{e(ptitle)}</h3><button type="button" class="x">Close</button></header>{fn()}</div></div>')
    parts.append(f'<footer>Built from the research ledger and the brief. <a href="{LEDGER}" target="_blank" rel="noopener">Open the full ledger</a> · <a href="{BRIEF}" target="_blank" rel="noopener">Open the brief</a>.</footer></div>')
    parts.append(f"<script>{POP_JS}</script>")
    return "".join(parts)


if __name__ == "__main__":
    out = Path("docs/research/timeline.html")
    out.write_text(render(), encoding="utf8")
    print("wrote", out, round(out.stat().st_size / 1024), "KB;", len(S), "steps;", out.read_text(encoding="utf8").count("<img"), "images")
