"""Build the ledger's opening synthesis section and inject it before the research map.

The section lists what we have found from the first benchmark to today, with the evidence grade and the
papers we already read. Idempotent: the block lives between <!-- synthesis --> and <!-- /synthesis -->.
Usage: .geomvenv/bin/python scripts/build_synthesis_section.py [--ledger docs/research/paper-ledger.html]
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

BIB = {  # bibkey -> (short label, arXiv id or None)
    "alazzawi2026gap": ("Al-Azzawi et al. 2026, Latin–Arabic gap", "2606.18884"),
    "alazzawi2026labelerr": ("Al-Azzawi et al. 2026, label errors (CER-HV)", "2601.16713"),
    "alazzawi2026crossling": ("Al-Azzawi et al. 2026, cross-lingual", "2605.02089"),
    "alazzawi2026seqmodel": ("Al-Azzawi et al. 2026, sequence modelling", "2605.05900"),
    "alshaibani2025dotless": ("Alshaibani et al. 2025, dotless Arabic", None),
    "alomari2026iconicity": ("Alomari et al. 2026, character iconicity", "2608.02935"),
    "mahmoud2014khatt": ("Mahmoud et al. 2014, KHATT", None),
    "marti2002iam": ("Marti and Bunke 2002, IAM", None),
    "mueller2012letter": ("Müller et al. 2012, letter identification", None),
    "alluhaybi2020connect": ("Alluhaybi et al. 2020, letter connectedness", None),
    "zeghli2025bigram": ("Zeghli et al. 2025, open bigrams", None),
    "shannon1951": ("Shannon 1951, entropy of printed English", None),
    "hegghammer2021noisy": ("Hegghammer 2021, OCR benchmarking", None),
    "baek2019wrong": ("Baek et al. 2019, scene-text comparisons", None),
    "rice1995isri": ("Rice et al. 1995, ISRI OCR test", None),
    "ye2013diqa": ("Ye and Doermann 2013, image quality", None),
    "swayamdipta2020cartography": ("Swayamdipta et al. 2020, dataset cartography", None),
    "rodriguez2021irt": ("Rodriguez et al. 2021, IRT for evaluation", None),
    "zhang2020ran": ("Zhang et al. 2020, radical analysis", None),
    "lehal2013urdu": ("Lehal 2013, Nastalique ligatures", None),
    "doctor2022graphemic": ("Doctor et al. 2022, Perso-Arabic normalisation", "2210.12273"),
    "heakl2025kitab": ("Heakl et al. 2025, KITAB-Bench", None),
    "farazi2026vlmocr": ("Farazi et al. 2026, VLMs on Arabic manuscripts", "2608.22366"),
    "kohut2023finetuning": ("Kohút et al. 2023, fine-tuning as adaptation", None),
    "vidalgorene2024rasam2": ("Vidal-Gorène et al. 2024, RASAM 2", None),
    "li2025htrvt": ("Li et al. 2025, HTR-VT", None),
    "retsinas2022best": ("Retsinas et al. 2022, HTR best practices", None),
    "wigington2017augment": ("Wigington et al. 2017, augmentation", None),
    "luo2020learn": ("Luo et al. 2020, learned augmentation", None),
    "eltay2022adaptive": ("Eltay et al. 2022, adaptive augmentation", None),
    "louradour2014curriculum": ("Louradour and Kermorvant 2014, curriculum for HTR", "1312.1737"),
    "wortsman2022wiseft": ("Wortsman et al. 2022, robust fine-tuning", None),
    "perea2010tl": ("Perea et al. 2010, Arabic letter-position coding", None),
}


def link(key: str) -> str:
    label, ax = BIB[key]
    if ax:
        return f'<a href="https://arxiv.org/abs/{ax}">{html.escape(label)}</a>'
    return f"<span>{html.escape(label)}</span>"


def cite(keys: str) -> str:
    return "; ".join(link(k) for k in keys.split()) if keys else ""


def grade(g: str) -> str:
    cls = {"A": "g-ok", "B": "g-acc", "C": "g-part", "N": "g-none"}[g[0]]
    return f'<span class="g {cls}">{html.escape(g)}</span>'


# (finding, evidence pointer, grade, bibkeys)
ERA1 = [
    ("Off-the-shelf VLMs cannot read cursive handwriting. Base Qwen3-VL-4B scores 77–119 % CER on KHATT, NAKBA and the Maghrebi RASAM sets. "
     "Fine-tuning on 8,000 Muharaf lines alone removes 47–67 points on every held-out set, including the Maghrebi manuscripts (RASAM-1 119 → 52 %).",
     "exp3090 notes, 2026-09-09", "A", "heakl2025kitab farazi2026vlmocr kohut2023finetuning vidalgorene2024rasam2"),
    ("A specialised base is not automatically better. Baseer-Nakba reads NAKBA at 1.0 % CER before any fine-tuning (it was trained on it) and gets worse (3.4 %) "
     "under our recipe; PaliGemma-2-3B and Qwen3-VL-4B both end near 30 % with the same recipe.",
     "exp3090 A/B/C, 2026-09-13", "B", "farazi2026vlmocr"),
    ("Unfreezing the vision encoder (differential learning rate) helps letter identity, not dots: Muharaf 36.3 → 28.7 % CER (−7.6 [−9.1, −6.2]); the dot share of error stays at 7–8 %.",
     "E1 preliminary, one seed", "B", "kohut2023finetuning wortsman2022wiseft"),
    ("Muharaf is the hardest handwriting set (best single model 34 % CER against 5.5 % on KHATT). The public copy is stored at 60 px line height (letter core about 12 px), "
     "so every fine-tune learned from low-resolution lines.",
     "image layer R6.0", "B", "mahmoud2014khatt"),
    ("Dots are a small tax; spaces and diacritic policy are larger ones. Across 91 competent model × dataset pairs the dot tax is a median 1.8 CER points (8 % of error); "
     "60 % of the 10,655 spaces wrongly inserted inside words fall at a gap between word parts (PAWs).",
     "error decomposition P2", "A", "alshaibani2025dotless alluhaybi2020connect"),
    ("Part of every score is pipeline. The last tenth of the line is the worst in 9 of 10 blocks (deletions and overrun); gemini-flash cuts the tail of 44–46 % of lines, PaddleOCR 31 %; "
     "page-scale sets hit a 160-token output cap; a flag reversed PaddleOCR's output (KHATT 62.9 % CER un-reversed against 85.8 % as stored).",
     "R2.8, X6, C5", "A", "hegghammer2021noisy baek2019wrong rice1995isri"),
    ("Leakage can fake a fine-tuning gain: 86.5 % of kitab_muharaf lines (20+ characters) occur word for word in Muharaf train, so fine-tuned rows there measure recall.",
     "profile P1", "A", "heakl2025kitab"),
    ("Difficulty belongs to the line: consensus difficulty has split-half reliability 0.54–0.81, and image features lift rank-regression R² from 0.06–0.51 (text only) to 0.15–0.63.",
     "sample characteristics", "B", "rodriguez2021irt swayamdipta2020cartography ye2013diqa"),
    ("Muharaf label noise (126 of 1,334 test lines flagged by CER-HV) does not move our comparisons: dropping them changes CER by −0.4 to +0.9 points for 13 models and changes no rank.",
     "X2", "A", "alazzawi2026labelerr"),
]

ERA2 = [
    ("The Latin–Arabic gap is large with little data (the gap to Latin sets is +24.2 points at 100 lines for the noisy Arabic sets), halves after label cleaning (+11.8) and is 4.4 points at full data. "
     "A 53 M-parameter transformer matches the 10 M CRNN (KHATT 8.9 vs 8.5 %), so architecture does not close it.",
     "their results; we use their logs", "A", "alazzawi2026gap li2025htrvt"),
    ("Data should be sized in characters per letter-form shape, not in lines: KHATT needs about 150 k characters to reach 10 % CER, IAM about 34 k. Positional labels in the CTC alphabet gave no consistent gain.",
     "their results", "B", "alazzawi2026gap marti2002iam mahmoud2014khatt"),
    ("Their similar-substitution metric reproduces exactly on their predictions (4.17 %). On the same Muharaf lines, VLMs make as many or more similar-letter substitutions per character than the CRNN "
     "(1.7–3.6 % against 1.2–1.5 %); their share is lower only because they make two to five times more of every other substitution.",
     "R2.6", "B", "alazzawi2026gap mueller2012letter"),
    ("Hand-drawn shape groups explain little of the confusions: of 185,902 letter substitutions, 6 % fall in our shape groups, 13 % are dot pairs and 65 % match no group. "
     "Loop and stroke pairs (ا/ل, ل/ن, ع/م) are frequent, so groups should be position-aware and learned from data.",
     "R2.6 follow-up", "B", "alazzawi2026gap"),
    ("Transfer inside the Arabic script (Arabic, Urdu, Persian at 100–1,000 lines) comes with the BiLSTM, not the shared CNN features. Neither group paper tests VLMs, which is where our work complements theirs.",
     "their results", "B", "alazzawi2026crossling alazzawi2026seqmodel doctor2022graphemic"),
    ("Script conventions change what must be learned. Base AIN reads Maghrebi ق as ف in 23–27 % of cases; a CRNN without Maghrebi training reads RASAM at about 47 % CER instead of about 12 %. "
     "Writers spell as they speak (Levantine متل for مثل in Muharaf labels).",
     "R2.9", "B", "vidalgorene2024rasam2"),
]

ERA3 = {
    "Is the number right? Measurement and labels": [
        ("Cropping to the ink band before resizing takes KHATT from 31.0 to 23.3 % CER on all 200 lines (−7.5 [−10.6, −4.9], 3 seeds; 35 lines better, none worse) for every fixed-height CRNN, at test time, with no retraining. "
         "Base VLMs are insensitive to loose crops, but three fine-tuned VLMs lose 3–4 points on them (Fanar-ft 25.5 vs 21.8 %, Qwen3-VL-8B-ft 27.2 vs 22.7 %); PaddleOCR is at 98 %. "
         "The full KHATT set is already tightly cropped, so the crop changes nothing there.",
         "X-S1, 1.9", "A", "retsinas2022best"),
        ("The alif-hamza follows the training labels, not the image. Muharaf writes it on 4 % of alifs, NAKBA 16 %; NAKBA writers do not draw it. Folding it removes about 2 points of NAKBA CER (13.8 → 11.6 %). "
         "NAKBA test and blind differ in hamza (+6.1) and alif maqsura (−7.1), so splits are reported separately.",
         "T11, 2.10", "A", "doctor2022graphemic"),
        ("Diacritics are a scoring rule, not a detail. Keeping marks re-orders leaderboards on KITAB-KHATT (Kendall τ 0.886), Muharaf (0.882) and RASAM-2 (0.867); only 35.8 % of Muharaf test lines carry marks. "
         "The headline is therefore the no-diacritics rung, with the orthographic rung required wherever spelling variants are common (5.6 % of NAKBA-blind tokens).",
         "R2.7, R4.3", "A", "alshaibani2025dotless"),
        ("Leakage differs by set. Share of lines whose text or image is in their own train set: Muharaf 7.5 %, NAKBA test 18.3 %, NAKBA blind 16.3 %, KITAB-KHATT 44.5 %, full KHATT test 95 % (a fixed paragraph), RASAM ≤ 1.1 %. "
         "NAKBA-trained systems gain 20.1 points more on leaked NAKBA-blind lines, which is the only set where memorisation survives FDR; its headline uses clean lines.",
         "R4.2, 2.9", "A", "heakl2025kitab"),
        ("One model carries a memorisation signature: AIN-7B reads 17.5 % of KITAB-KHATT lines exactly (17× its nearest-CER peers) and copies 54 % of the transcription's whitespace quirks (7×). Still marked unverified (C6).",
         "D23", "C", ""),
        ("Seed noise is a yardstick for every claim. Two identical trainings differ by +0.4 points [−0.6, +1.4] on a 91-line KHATT half; 761 lines flip between seeds of one recipe (1 % of KITAB to 17.5 % of Muharaf), "
         "unstable lines are short (median 17 vs 71 letters), and single-seed 'better' lines are two to three times enriched in them.",
         "E3 seeds, D24", "A", ""),
        ("Multiple comparisons matter: after Benjamini–Hochberg at q = 0.1, the KHATT replicate's surviving letter cells fall from 2/2 to 0/0, while the ink crop keeps 41/0.",
         "1.1", "A", ""),
        ("Difficulty has structure we can report: IRT difficulty beats median CER only on NAKBA test; short fragments are worse after FDR on Muharaf and NAKBA; loose crops cost +58 points on KITAB-KHATT and +49 on NAKBA test.",
         "R3.2, R3.3, X5-len", "B", "rodriguez2021irt"),
    ],
    "Letter shapes and skeletons": [
        ("Letter-shape novelty is the strongest predictor of letter errors: error rises from 1–5 % in the most typical fifth to 30–53 % in the most novel band in every dataset, and it survives the confound battery "
         "(within-line odds ratio about 16 for CRNNs, 4.4 for fine-tuned VLMs). KHATT's letters are far from training letters (21 % beyond Muharaf's own 95th percentile, against 10 % and 5 %). Pixel novelty is weak.",
         "X-G10, ledger §14", "A", "alazzawi2026gap zhang2020ran lehal2013urdu"),
        ("Adding the missing shapes helps only where the shapes are missing: shape-targeted repeats beat a random control on KHATT, including held-out lines (−0.8 points [−1.6, +0.1]), but not on NAKBA, where the gain is repetition itself.",
         "X-G11", "B", "swayamdipta2020cartography louradour2014curriculum"),
        ("The positional-form tail is covered: at most 0.2 % of test letters have a letter-in-position seen fewer than 50 times, so the gap at full data is not a missing-form problem.",
         "R1.7", "A", "alazzawi2026gap"),
        ("Teeth are a weak point: letters with teeth are misread 6–21 % more than their letter baseline in all six model families (survives BH). Loops exceed baseline in four of six families; tails and verticals are below. "
         "Among one-tooth letters, error rises with run length (3+ vs 1: 1.84 [1.68, 2.04] for the CRNN).",
         "2.3", "B", ""),
        ("Without dots, about nine tenths of every family's error remains: models misread skeletons, not dots, and lines right except for their dots are rare (0.1–4 %). Three-dot letters are the most dot-error-prone.",
         "no-dots-cer", "A", "alshaibani2025dotless"),
        ("Dots are mostly recoverable from context for known words and not for unseen ones: a quarter to a third of words share their dotless form with another word, yet the most frequent word restores 95 % of known words. "
         "This echoes why dots were added to the script.",
         "dotless ambiguity", "B", "alshaibani2025dotless alomari2026iconicity"),
        ("Confusions are visual and shared across model families (Spearman ρ 0.63–0.90 between families, lower bounds above 0.4); letter pairs that differ only in dots are 3–18× more confusable than other pairs; response bias follows letter frequency. "
         "They follow shape, not sound.",
         "2.13", "B", "mueller2012letter"),
        ("No dot-writing style (dash, caret, displaced, touching) survives the pre-registered rule, and the dot detector fails its own validation, so this stays a low-confidence null. "
         "Vertical stacking of letters is also a null (stacked vs not 0.99 [0.90, 1.09]).",
         "2.4, 2.5", "C", ""),
    ],
    "Spacing, connectivity and geometry": [
        ("Models insert word-internal spaces at PAW gaps far more than chance (87 of 89 pairs), and the space tax (2–11 CER points) exceeds the dot tax (1–4). "
         "Word gaps and split-PAW gaps overlap: the chance that a word gap is wider than a PAW gap is 0.68 [0.67, 0.70] for CRNNs and 0.63 for VLMs, so spacing is visually ambiguous.",
         "T1, 2.1", "B", "alluhaybi2020connect zeghli2025bigram"),
        ("Slant is causal but small: shearing Muharaf upright raises CER 6.2 → 8.2 %, shearing KHATT toward Muharaf lowers it 26.3 → 25.3 %; slant test-time augmentation gated on measured slant is nearly free. "
         "Slant and stroke augmentation in training hurts on unseen sets (NAKBA test +0.84, blind +0.79, KHATT +0.59 points).",
         "X-G8, X-T3", "B", "wigington2017augment luo2020learn eltay2022adaptive"),
        ("Stroke width and pen contrast matter a little: pen contrast separates Maghrebi (0.5–0.6) from the rest (about 1.0); thinning NAKBA's strokes to Muharaf's width lowers its CER 13.6 → 12.5 %.",
         "X-G2", "B", ""),
        ("Ink and paper tone, paper noise, ruled lines and baseline slope have no measurable effect on the CRNN. Contour histograms identify a line's dataset 70–73 % of the time but do not track the transfer gap.",
         "X-G1, X-G4", "N", "ye2013diqa"),
        ("Resolution: dots survive at 64 px, but at 32 px 43–64 % fall under 2 px. Pixels per character and line length predict errors in every family; loose crops and slant only for CRNNs.",
         "X-G7, T5", "B", ""),
    ],
    "Language, decoding and lexicon": [
        ("An unconstrained LLM decoder on CRNN features makes things worse and is closed in this form: KHATT 26.4 → 40.0 %, NAKBA 13.8 → 20.8 %; a 1.7 B model is no better (40.1 %). "
         "Fluent-but-wrong words are 41–44 % of wrong words against 24 % for the CRNN.",
         "E8", "A", ""),
        ("KHATT's text is the farthest from training (4.0 bits per character against 3.4 for NAKBA and 2.9 for Muharaf; 5.9 % of its PAWs never occur in training), and unseen words hurt every family.",
         "T6", "B", "shannon1951"),
        ("A lexicon covers 95.8–99.4 % of the panel's words, but it accepts 20–23 % of single dot-swap errors, so a word list alone cannot restore dots; KHATT's 'unseen' words are mostly label typos.",
         "1.7", "B", ""),
    ],
    "Models, combination and writers": [
        ("CRNN and VLM difficulty correlate weakly, and the per-line better of the two is far below either: KHATT 14.3 % against 17.4 / 18.1 %, NAKBA test 4.1 % against 10.2 / 18.7 %. A cross-family combination is a strong candidate.",
         "T8", "B", "farazi2026vlmocr"),
        ("An in-domain CRNN trained on KHATT reaches 9.2 % CER on the full 1,038-line KHATT test, close to the published CRNN, so the standard-split CRNN is the fair reference.",
         "night 1, 4.6", "B", "alazzawi2026gap mahmoud2014khatt"),
        ("KHATT is almost one fixed 77-word paragraph (1,035 of 1,038 test lines), so it works as a natural experiment: for generic CRNNs the text explains more word error than the writer (partial R² 0.152 vs 0.088), "
         "while for the KHATT-trained model the writer dominates (0.26 vs 0.03). Ink density is the only style feature that predicts error for all readers. Writer IDs are not in the public copy; "
         "rebuilt pseudo-pages are valid as clusters for intervals (page ICC 0.11 [0.06, 0.17]).",
         "2.16, 2.17", "B", "mahmoud2014khatt"),
        ("Covering every letter form takes at least 44 words (final and isolated forms occur only at word ends); the best single real sentence covers 65 of 104 forms and 7–8 sentences cover all.",
         "2.7", "B", ""),
    ],
    "Where models look": [
        ("Attribution maps are valid (9× on ink) and point at the token's own letters about three times above chance, but wrong tokens look in the right place as often as correct ones: errors are misreadings, not misdirected looking. "
         "LocateAnything zero-shot failed.",
         "T9", "B", ""),
    ],
}

TAKEAWAYS = [
    "Much of what first looked like model weakness was measurement: loose crops, hamza and diacritic conventions, leaked test lines, truncated outputs and reversed text each moved scores by points, and several were wrongly pinned on the model.",
    "The errors that remain are visual misreadings of rare or novel letter skeletons. Dots, spaces and language each add a tax, but the largest part is the skeleton, and attention maps say the model looks at the right place.",
    "Fixes that act on the input (ink crop, targeted shapes) help fixed-height recognisers and not the VLMs; language priors help only when constrained by the image, and an unconstrained LLM decoder hurts.",
    "Datasets are not interchangeable: conventions (hamza, marks, spelling), leakage and writer structure differ by set and split, so results are reported per split, with seed noise and FDR as the yardstick.",
    "Open and queued: cross-family combination, a library of real letter shapes, writer adaptation and lexicon-constrained decoding (see the master checklist below).",
]

NOT_YET = [
    "GPU night 1 finished on 2026-10-18; its outputs are not analysed yet: posterior dumps for decoding, the skeleton-only CRNN, the in-domain NAKBA CRNN, writer adaptation, the few-shot Maghrebi curve and the hard-example control.",
    "GPU night 2 (VLM image reliance, geometry and resolution probes, shape targeting for NAKBA) is built but not yet run.",
]


def render() -> str:
    def table(rows):
        out = ['<div class="qwrap"><table class="dt"><thead><tr><th>Finding</th><th>Evidence</th><th>Grade</th><th>Related papers we have read</th></tr></thead><tbody>']
        for text, ev, g, keys in rows:
            out.append(f"<tr><td>{html.escape(text)}</td><td>{html.escape(ev)}</td><td>{grade(g)}</td><td>{cite(keys)}</td></tr>")
        out.append("</tbody></table></div>")
        return "".join(out)

    parts = [
        '<section id="synthesis">',
        '<div class="section-head"><div class="eyebrow">Synthesis · updated 2026-10-18</div>',
        "<h2>What we found, from the first benchmark to today</h2>",
        "<p>Three stretches of work, in order: the benchmark and fine-tuning phase (September 2026), the reading of the Al-Azzawi group's four papers (January–June 2026) "
        "and our own measurements against them, and the academic turn of October 2026, when we began characterising Arabic script the way English had been characterised. "
        "Grades follow the research map: A strong and replicated, B solid with one gap, C suggestive, N negative or closed. "
        "Every row points to its evidence in the sections below and to the papers we have read on the same question.</p></div>",
        "<h3>1 · Benchmark and fine-tuning (September 2026)</h3>",
        table(ERA1),
        "<h3>2 · The Al-Azzawi papers and what we measured against them</h3>",
        table(ERA2),
        "<h3>3 · Characterising the script (October 2026)</h3>",
    ]
    for title, rows in ERA3.items():
        parts += [f"<h4>{html.escape(title)}</h4>", table(rows)]
    parts += ["<h3>What this adds up to</h3><ul>"]
    parts += [f"<li>{html.escape(t)}</li>" for t in TAKEAWAYS]
    parts += ["</ul><h3>Not yet analysed or run</h3><ul>"]
    parts += [f"<li>{html.escape(t)}</li>" for t in NOT_YET]
    parts += ["</ul></section>"]
    return "".join(parts)


def inject(ledger: Path) -> None:
    h = ledger.read_text(encoding="utf8")
    block = "<!-- synthesis -->" + render() + "<!-- /synthesis -->\n"
    h = re.sub(r"<!-- synthesis -->.*?<!-- /synthesis -->\n?", "", h, flags=re.S)
    marker = "<!-- research map -->"
    assert marker in h, "research map marker not found"
    h = h.replace(marker, block + marker, 1)
    toc_item = '<li><a href="#synthesis">0 · What we found, start to finish</a></li>'
    h = h.replace(toc_item, "")
    anchor = '<li><a href="#key-findings">'
    i = h.find(anchor)
    assert i > 0, "contents anchor not found"
    j = h.rfind("<ol", 0, i)
    assert j > 0, "contents list not found"
    k = h.find(">", j) + 1
    h = h[:k] + toc_item + h[k:]
    ledger.write_text(h, encoding="utf8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=Path("docs/research/paper-ledger.html"))
    a = ap.parse_args()
    for g in [t[2] for r in [ERA1, ERA2, *ERA3.values()] for t in r]:
        assert g[0] in "ABCN"
    inject(a.ledger)
    print("injected synthesis:", a.ledger)
