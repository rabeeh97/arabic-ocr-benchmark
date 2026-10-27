# Design: RASAM ceiling — our own CRNN, and re-scoring the published models

Created 2026-10-22. Serves the question "what is the best achievable on RASAM, and does better data give better results
in general?" (the data-quality / data-scaling argument for the wider programme). Plan only: nothing here has been run.

## 0. What the literature says (to be reproduced, not trusted)

| Source | Model | Protocol | CER |
|---|---|---|---|
| RASAM, ICDAR 2021 | word-based CRNN, trained on RASAM-1 | RASAM-1 test, in-domain | 4.8 % |
| Lucas et al., arXiv 2211.16147 | meta-word CRNN, specialised | own split, in-domain | 3.23 % |
| RASAM-2, CHR 2024 (ceur-ws Vol-3834 paper35) | word-based CRNN, trained on RASAM-2 | RASAM-2 test / RASAM-1 test | 6.79 % / 5.50 % |
| same | RASAM-1 model | RASAM-2 test (unseen manuscripts) | 30.91 % |
| Phoenix/Athar, arXiv 2608.19385 | Kraken CNN-BiLSTM-CTC (5 M params) | 1,789-line RASAM "development guard" | 7.67 % (its own data, RASAM is in training) |
| Ours | Fanar-2-Oryx LoRA, CER-light | 200-line panel | 11 % (RASAM-1), 17 % (RASAM-2) |

Not comparable yet: different test lines, unknown diacritic handling in the papers, our 200-line panels.

## 1. Data we have (HF, Apache-2.0, ungated)

| Set | train | validation | test | Notes |
|---|---|---|---|---|
| `calfa-ai/RASAM-1` | 5,860 | 733 | 734 | 3 manuscripts, 17th–19th c. |
| `calfa-ai/RASAM-2` | ~2,980 | 372 | 372 | 15 manuscripts |

- Local cache holds RASAM-1 train + test and RASAM-2 test. **RASAM-2 train/validation and RASAM-1 validation must be
  fetched to the SSD** (`~/.cache/huggingface` is symlinked to the SSD; memory `downloads-go-to-ssd`). Check the RASAM-1 train
  parquet: shard `train-00004-of-00005` was not in the listing.
- Splits are probably random by line inside each manuscript, so scores are in-domain (the papers' are too). The HF cards do not
  say. **No manuscript IDs in the HF copy**: R0.4 must recover them from `calfa-co/rasam-dataset` (`list-images.tsv`, PAGE-XML)
  before any leave-manuscript-out claim.
- **Do not reuse our existing aligner CRNNs for RASAM.** `xai_prepare_aligner_data.py` default mode trains on RASAM-1 *test*
  lines (CORRECTIONS C4), and `--standard-split` has no RASAM at all. The new run below uses the published train splits.

## 2. Part A — our own CRNN on RASAM-like data

Goal: the RASAM number a plain CTC line recogniser gets with the whole train split, and how it scales with data.

**Model.** Start from the existing `CRNN` in `scripts/xai_train_aligner.py` (VGG-style, height 64, 3-layer BiLSTM, ~tens of M
params, `--width` scales it down for the laptop). The published best is a *word-class* CRNN; ours is character CTC, so a gap
there is a finding, and a word-class variant is an optional late arm (A8).

**Recipe is unknown, so treat it as a small sweep, not a guess.** Fix everything else, vary one thing at a time on the RASAM-1
validation split (never on test), 2 seeds each, then freeze and run 3 seeds on test:

| Axis | Values |
|---|---|
| line height | 64 (current), 96 |
| geometric aug (`--geom-aug`: slant, stroke) | off, on |
| width / capacity | 0.5, 1.0 |
| epochs | 30, 60 (checkpoint every epoch; memory `training-checkpoint-granularity`) |
| label policy | raw (diacritics kept), diacritics stripped |
| lr | 1e-3, 2e-3 |

Not swept: optimizer, scheduler (keep the script's), colour handling (grey as now; the paper links coloured lines to high CER, so
log CER on coloured vs black lines instead).

**Arms** (each scored on the full RASAM-1 test 734 and RASAM-2 test 372, not a panel):

| ID | Train on | Mirrors |
|---|---|---|
| A1 | RASAM-1 train | RASAM paper 4.8 % |
| A2 | RASAM-2 train | CHR 6.79 % |
| A3 | RASAM-1 + RASAM-2 train | combined |
| A4 | A1 → RASAM-2 test, A2 → RASAM-1 test | CHR Table 4 cross cells (30.91 / 5.50) |
| A5 | data-scaling: A3 with 0.5k, 1k, 2k, 4k, all lines (`--max-train`, 3 seeds) | the "better data" curve |
| A6 | manuscript-diversity: fixed 2k lines drawn from 1, 3, 15 manuscripts | RASAM-1 → RASAM-2 effect. **Needs R0.4** |
| A7 | Muharaf pretrain (our existing train shards) → RASAM fine-tune | does non-Maghrebi data help |
| A8 | (optional) word-class head | the published recipe |

**Metrics.** CER raw (diacritics kept, NFC) and CER-light, per `metrics.py`; bootstrap 95 % CI by line; CER by length band (X5) and
by coloured/black line; ق→ف rate. Predictions written in the canonical prediction format so the characterisation pipeline
(`canonical_runs.json`, per-sample figures) consumes them without a special case. Remember reading-order pitfalls
(CORRECTIONS C1, C5): the CRNN emits visual order, so assert direction on a few lines before scoring.

**Compute.** RASAM is small (≤ 9k lines). 3090 over SSH, or the Mac (`--device mps`, width 0.5) for the sweep. A1 on the 3090
should be well under an hour.

**Gates.**
- G1 (after A1–A3): if A1 raw CER ≤ 7 % we are at the published level with a plain char CRNN, and Fanar's 11 % is the thing to beat.
- G2 (A5): CER falls monotonically in data; report the fitted slope. A flat curve kills the "more data" claim for RASAM.
- G3 (A6): at equal line count, more manuscripts beats fewer. This is the diversity half of the claim.

## 3. Part B — re-scoring the published models ourselves

| Candidate | Downloadable? | Trained on | What we can verify |
|---|---|---|---|
| `factlogic/phoenix-arabic-manuscript-htr` (Kraken `.mlmodel`, 5 M params, CC BY-NC-SA 2.0) | **yes** (HF) | RASAM-1/2, TariMa, Muharaf, Agapet, Omar | its claims on **Omar/NAKBA (11.84 %)** and **TariMa (10.72 %)**, plus RASAM-1/2 test. RASAM is training data, so those RASAM scores are *not* held-out. Report them labelled "seen". Muharaf in its training data too: our Muharaf test is contaminated for it |
| TariMa Kraken models, Zenodo 7810571 (manuscript model, 16 MB, CC BY 4.0) | **yes** | TariMa only | RASAM as a clean cross-corpus Maghrebi baseline (RASAM was *not* in training) |
| Calfa word-based CRNN (4.8 / 6.79 / 3.23 %) | **no** weights found (no release on HF, Zenodo, calfa-co GitHub, Calfa open-science page) | RASAM | cannot replicate. Ask the authors (see below); otherwise A8 is our re-implementation |
| Baseer (Misraj), reported 53 % on RASAM by Phoenix | check HF | — | optional: we have VLM harness results already |
| Our VLM fine-tunes | yes (local) | RASAM-1/2 subset | re-score on the **full** test sets, not the 200-line panel |

**Protocol (identical for every model).**
1. Same line images: the HF RASAM test splits as stored, no extra crops. Kraken runs recognition-only, with a single box covering the
   whole image as the segmentation (no layout model), so only the recogniser is tested.
2. Same ground truth and normalisation as Part A; both raw and CER-light.
3. Same `metrics.py` CER, same bootstrap. Log model SHA-256 and Kraken version in the run manifest.
4. Check text direction on 10 lines per model before scoring.
5. Verification verdict per claim: reproduced (within 1 point and inside the CI), or not, with the likeliest reason (different
   lines, different normalisation, seen/unseen).

**Setup.** Kraken in its own venv on the SSD (not the project venv, which has no Kraken; pin the version, because the `.mlmodel`
format has changed across releases). Weights to `/Volumes/PortableSSD/...`. CPU inference on the laptop is enough (~1,100 lines).

**Ask Calfa.** A short email to the RASAM authors (Vidal-Gorène et al., corresponding author on the HAL record) asking whether the
word-based CRNN weights or the exact test line lists can be shared. Costs nothing, and settles the 4.8 / 6.79 comparison. Draft
only: sending is the user's call.

## 4. Order of work

1. Fetch missing RASAM splits; confirm counts (734 / 733 / 5,860 and 372 / 372 / ~2,980) and write a `rasam_split_check.json`.
2. Set up Kraken venv; run Phoenix + TariMa model on the RASAM test lines and on NAKBA (Part B). CPU, no GPU, same day.
3. Recover manuscript IDs from the GitHub repo (R0.4 for RASAM).
4. Part A sweep on validation, then A1–A4, then A5–A7.
5. Re-score our VLM fine-tunes on the full RASAM test sets.
6. One table: published claim, reproduced value, ours (CRNN, VLM), seen/unseen; update `NEXT_STEPS.md`, `progress.json`, Atlas.

## 5. Risks

- Splits may leak neighbouring lines of one page into train and test, inflating every in-domain number including the papers'. Only
  R0.4 (page IDs) can quantify that.
- A diacritics policy mismatch with the papers could shift CER by a few points either way; hence both variants everywhere.
- Phoenix's licence is NC; fine for research, so keep outputs out of anything redistributed commercially.
- The Kraken `.mlmodel` may not load in the newest Kraken; fall back to the version Phoenix states.

## 6. Results so far (2026-10-22)

**RS1 / RS3 (done).** All splits cached; counts in `results/rasam/rasam_split_check.json` (RASAM-1 5,864 / 733 / 734; RASAM-2
2,976 / 372 / 372). The parquet carries `manuscript_name` and `image_name`, so no GitHub recovery was needed
(`results/rasam/rasam_meta.json`). **The splits are random by line: every test page also occurs in train** (260 of 260 RASAM-1
pages, 140 of 140 RASAM-2 pages); only 1.0 % / 0.5 % of test lines repeat a train line verbatim. RASAM-1 has 3 manuscripts, RASAM-2
has 15 (all 15 in every split). Every in-domain RASAM score, ours and the papers', is therefore page-level in-domain. A
manuscript-held-out test needs our own split by `manuscript_name` (RASAM-2 has 15 units).

**RS2 (done for Phoenix and TariMa).** Kraken 7.0.3 + torch 2.4.1 in `~/.venvs/kraken-rasam` (internal disk: the SSD is exFAT with
1 MB clusters, so a venv there costs about 15x its real size). Models: `phoenix/model.mlmodel` (SHA-256 matches the card),
`tarima/model-manuscript_best.mlmodel`. Recognition only, whole image as one line box, `scripts/rasam_kraken_score.py`.

| Input | Phoenix, RASAM-1 test, first 60 lines under 100 px tall |
|---|---|
| raw colour crop | 73.8 % CER |
| grey | 73.8 % |
| black mask filled with paper median | 71.7 % |
| **Otsu-binarised (used)** | **13.6 %** |

The HF crops carry black rotation masks and (for the tall ones) neighbouring-line ink; the models expect Kraken-extracted,
binarised lines. The binarisation variant was chosen by looking at those 60 test lines (effect size makes selection bias
negligible, but disclosed). Full test sets, binarised, CER (raw / light / diacritic-insensitive barely differ):

| Model | RASAM-1 test (734) | RASAM-2 test (372) |
|---|---|---|
| Phoenix (RASAM is in its training data) | 23.1 % | 18.7 % |
| – lines under 100 px tall (clean crops) | 14.7 % | 17.1 % |
| – lines 100 px or taller (neighbour ink) | 37.1 % | 18.9 % |
| TariMa manuscript model (RASAM unseen) | 42.6 % | 29.9 % |

Verdicts: Phoenix's reported 7.67 % on a 1,789-line RASAM guard is **not reproduced** on the HF test crops (best case 14.7 % on the
cleanest half); the guard lines are different and were presumably Kraken-extracted, so this is a protocol gap, not proof the claim
is wrong. TariMa transfers to RASAM at 30–43 %, consistent with the RASAM paper's cross-manuscript numbers (31 % for a RASAM-1
model on RASAM-2).

**RS4 (in progress).** Data prep (`scripts/rasam_prepare_crnn_data.py`, raw labels, mask filled, height 64), test scorer
(`scripts/rasam_crnn_eval.py`) and a `--train-sets` option in `xai_train_aligner.py` work end to end. Compute: the Mac has 8.6 GB
RAM, MPS ran out of memory at batch 8 and 16 (about 6 GB "other allocations"), and the 3090 (`10.100.0.3`) timed out (VPN?). A1 at half
width (2.4 M params) on CPU takes about 550 s per epoch, so 30 epochs is about 4.5 h. The full-width model and the sweep need the 3090.

### A1 result (2026-10-23): a 2.4 M-parameter character CRNN trained on RASAM-1 train only

Config: `xai_train_aligner.py` CRNN, `--width 0.5` (2.4 M params), height 64, mask filled, raw labels (diacritics kept), `--geom-aug`,
30 epochs, bs 16, seed 0, CPU, `--train-sets r1` (5,860 train lines; 733 validation lines for model selection, never test).
Checkpoint = lowest validation CER (epoch 28, 6.75 %). Scored with `rasam_crnn_eval.py` on the full published test sets, bootstrap by line
(`scripts/rasam_result_summary.py`). One seed, so the interval covers lines, not training randomness.

| Test set | micro CER raw | 95 % CI | light | diac-insensitive |
|---|---|---|---|---|
| RASAM-1 test (734, same 3 manuscripts) | **6.82 %** | 6.30–7.39 | 6.79 | 6.47 |
| (final-epoch checkpoint) | 6.88 % | | | |
| RASAM-2 test (372, 15 manuscripts, unseen) | **24.6 %** | 23.0–26.3 | 24.4 | 23.8 |

RASAM-1 detail: manuscript 417 4.0 %, 1977 6.3 %, 609 8.4 %; main text 6.7 %, marginalia 15.8 %, catchwords 59.6 % (n = 10); lines of at
most 20 characters 17.5 % (n = 49); exact-line match 9.4 %. RASAM-2 detail: easiest 13–15 % (MS 1960, 45b, 9), hardest MS 1936 79.7 % (the
heavily vocalised manuscript the CHR paper also flags) and MS 1944 42 %; exact-line match 0.8 %.

How it compares, with the caveats that matter:
- Published word-based CRNN, RASAM-1 on RASAM-1: 4.8 % (their test lines, protocol unknown). Ours is 2 points higher with a *character* model a
  fifth the size of the aligner's full-width variant; not the same lines, so no claim of parity.
- Published RASAM-1 model on RASAM-2: 30.9 %; ours 24.6 %. Different lines and a different recogniser, directionally the same cliff: a
  3-manuscript model loses 18 points on unseen manuscripts. This is the evidence that **diversity of manuscripts, not line count,** limits RASAM-1.
- Our best VLM fine-tune on RASAM-1 was 11 % (Fanar-2-Oryx, CER-light, 200-line panel) and its RASAM-2 17 % is in-domain (it trained on 2k RASAM-2
  lines), so it is not comparable to the 24.6 % cross cell above. On RASAM-1 the small CRNN is ahead of the VLM on a stricter, larger test.
- Phoenix (RASAM in its training data) re-scored at 23.1 % on the same test set, because the HF crops are not what it was trained on.

Open before this is a result to quote: A2 (RASAM-2 train) and A3 (both), a second seed, and the recipe sweep (full width needs the 3090).

## 7. Error analysis of A1 and what to change (2026-10-23)

Scripts: `rasam_error_analysis.py` (Levenshtein alignment, edge whitespace stripped like the benchmark metric), `rasam_crop_analysis.py`,
`rasam_lexicon_postedit.py`; outputs in `results/rasam/analysis/`. A1 on RASAM-1 test: 3,430 edits over 50,322 characters (6.82 %).

**Why not 4.8 %.** (a) Not the same lines and probably not the same crops: the paper scored Kraken-style page extraction; we score the HF crops.
(b) The aggregation is unspecified ("average CER"): on the same predictions we get micro 6.82 %, mean per page 7.57 %, median per page 5.71 %,
mean per manuscript 6.24 %, so convention alone moves the number by about ±1 point. (c) The published model is word-based; our test says a
naive lexicon does **not** replicate that benefit: 16.8 % of test words are out of the training lexicon, and snapping hypothesis words to the
nearest training word within 1–2 edits makes CER **worse** (8.1–9.8 % vs 6.82 %; 276 fixes, 609 breakages). A soft character LM is the fair
test (Phoenix's order-8 LM gained 0.7 points on RASAM). (d) Untuned: 2.4 M parameters, height 64 (median crop is 3.6× taller, so downscaled),
one seed, no sweep.

**What the errors are** (A1, RASAM-1 test; ops: 51 % substitutions, 31 % deletions, 19 % insertions):
1. **Word spacing, 16 % of errors**: 285 merged words and 175 split words. In cursive script the gap between PAWs and the gap between words
   are close; spaces are 18 % of reference characters and the model is wrong on 547 of them.
2. **One-dot confusions, 24 % of substitutions (12 % of all errors)**, and the Maghrebi-specific pairs lead the list: **ب→ف 42 (+ ف→ب 17)**, **ن→ق 31**
   (in Maghrebi script fā' takes its dot below, so it looks like bā', and qāf takes one dot above, so it looks like nūn), then ب↔ي, ت↔ن, ج↔ح.
   These pairs are ambiguous in the image itself; only context (language) can separate them.
3. **Thin letters lost or invented, about 12 % of errors**: ل م ا are deleted 95 / 85 / 80 times and inserted 45 / 39 / 53 times. These are
   thin vertical strokes, exactly what a 3.6× downscale and a 4× width reduction in the CNN erase.
4. **Dense lines are harder**: crops 300 px or taller score 8.8 % against 6.0 % for short ones; the quartile with the least horizontal pixels per
   character (at 64 px height) scores 8.4 % against 5.6 % in the loosest quartile. The last tenth of the line is worst (8.5 % vs about 5 %), as in
   the VLM end-of-line finding.
5. **Fragments**: catchwords 59.6 % (n = 10), marginalia 15.8 %, lines of at most 20 characters 17.5 % (n = 49); 8 of the 12 lines with CER above 0.5
   are catchwords or one- and two-word lines. Only 1 of the 12 is a crop that shows a neighbouring line (test 185, whose reading is the reference
   of test 587), so crop contamination is **not** a major cause.
6. **Rare letters and marks**: ئ 28 %, غ 22 %, ء 15 %; fatha 73 % wrong but marks are 0.2 % of characters and 3 % of errors.

**RASAM-2 (unseen manuscripts, 24.6 %)** has the same pattern (spaces 16 %, dots 17 % of substitutions) at a higher level, plus an annotation
convention gap: أ is wrong 100 % of the time (n = 45) because RASAM-1 writes it 91 times against 437 in RASAM-2. But أ→ا and ي→ى together are
about 3 % of RASAM-2 substitutions, so handwriting shift, not convention, is the main cost.

**Label conventions differ between the two sets** (counted directly from the HF parquet transcriptions): diacritics in 11.9 % of RASAM-1 lines vs 0.3 %
of RASAM-2 lines (RASAM-2 has no sukun); hamza: RASAM-1 uses bare ء 1,705 times and أ 91, RASAM-2 uses أ 437, إ 102, آ 43; final ى is 22 % of
RASAM-1 yā' vs 16 % of RASAM-2; both use a `#` placeholder (100 and 121 times) and stray RLM / RLE controls (83 lines in RASAM-2).

### Proposed dataset modifications (to be tested, each as a retraining arm)
- **D1 splits:** page-disjoint (RASAM-1) and manuscript-disjoint (RASAM-2: for example 10 / 2 / 3 manuscripts, three fixed folds). Today every test page
  also occurs in train, so in-domain numbers are optimistic; the cross cell (24.6 %) is the honest one.
- **D2 crops:** re-extract lines from the PAGE-XML polygons and baselines on the original pages (XML downloaded, pages downloading), masked and
  deskewed, tight to the ink band, at one fixed height. Aim is resolution and consistency, not neighbour removal (see item 5); it also makes the lines
  match what the published Kraken models were trained on, so the Phoenix replication becomes fair.
- **D3 label layers:** base text (diacritics stripped, hamza and ya normalised by a documented rule) for training and the headline CER, and a separate
  diacritic layer scored on its own; map `#` to an explicit "illegible" tag and drop those characters from the loss; remove RLM / RLE controls.
- **D4 tags:** region (main, marginalia, catchword), line-length class, crop height; report CER stratified and keep catchwords out of the headline.
- **D5 larger, harmonised pool:** add TariMa (2.4 k lines) and the four Maghrebi AraMS books (about 8.9 k lines) under the same conventions, with
  manuscript-held-out evaluation. The point is manuscript diversity, which is what the cross cell says is missing.

### Retraining arms
| ID | Change | Expected effect | Where |
|---|---|---|---|
| A2, A3 | RASAM-2 train; RASAM-1 + RASAM-2 train (running, half width, CPU) | A3 should fix the cross cell if diversity is the lever | Mac |
| B1 | D3 labels (base layer) on A3 | removes convention noise; fewer wasted outputs | Mac |
| B2 | D1 manuscript-disjoint RASAM-2 split | true generalisation gap, comparable across runs | Mac |
| B3 | D2 crops | tests the crop/resolution effect | needs pages |
| B4 | height 96 / 128, full width | thin letters, dense lines | 3090 |
| B5 | + TariMa + AraMS Maghrebi (D5) | the data-diversity curve | 3090 |
| B6 | soft character LM beam search (OpenITI text), tuned on validation | space and dot-convention errors that need context | CPU, offline |

### A2 result and a correction to the comparison (2026-10-24)

A2 = same recipe as A1 (half-width char CRNN, 30 epochs, raw labels, geom-aug, seed 0) trained on RASAM-2 train only (2,923 lines), best checkpoint
by RASAM-2 validation (18.8 %): **RASAM-2 test 18.47 % raw** (in-domain), **RASAM-1 test 24.86 %** (cross). With A1 (RASAM-1 train: 6.82 % on RASAM-1,
24.6 % on RASAM-2) the cross cells are symmetric at about 25 %: a model trained on one set loses about 18 points on the other, in both directions.

RASAM-2 in-domain is much worse than RASAM-1 in-domain (18.5 % vs 6.8 %) with the same model. RASAM-2 has 15 manuscripts with about 195 training lines each
(RASAM-1: 3 manuscripts with 300 to 2,700 lines each), so the in-domain set is both smaller and far more varied.

**Comparison correction.** The CHR 2024 table reports for the "RASAM 2" model 5.50 % on RASAM-1 and 6.79 % on RASAM-2, in-domain on both. That
only makes sense if that model was trained on RASAM-1 and RASAM-2 together. The like-for-like arm for our model is therefore **A3 (train on both)**, not
A1 or A2, and the published reference numbers for it are 5.50 % and 6.79 %. A1 vs 4.8 % and A2 vs 6.79 % were not matched comparisons.

### B6 first result (2026-10-24): a character n-gram LM in the beam search helps substantially

`scripts/rasam_lm_decode.py`: CTC prefix beam search (beam 12) over A1's posteriors with a Witten-Bell character 6-gram trained on the RASAM-1 TRAIN
labels only, weight alpha and per-character bonus beta tuned on 150 RASAM-1 VALIDATION lines (grid alpha 0.3 / 0.6 / 1.0, beta 0 / 0.5 / 1.0; validation greedy 7.01 %,
best 5.66 % at alpha 0.3, beta 1.0). Model unchanged (A1 best checkpoint). Test, micro CER, raw profile:

| Test set | greedy | with LM | diac-insensitive greedy → LM |
|---|---|---|---|
| RASAM-1 test (734) | 6.82 % | **5.50 %** | 6.47 → 5.13 |
| RASAM-2 test (372, unseen manuscripts, RASAM-1 LM) | 24.59 % | **21.97 %** | 23.75 → 21.25 |

So the LM removes about a fifth of the errors on the seen manuscripts and about a tenth on the unseen ones, with no retraining and seconds of decoding. It is
much closer to the published 4.8 % than the greedy number, and, unlike the naive lexicon snapping (which hurt), it works because it weighs context
against the image evidence instead of overriding it. The best setting sits on the grid edge (largest beta), so a finer grid and other orders
are being tried (`A1_lm_fine`); the grid is chosen on validation only.
Caveat: one seed of one model; the LM text is in-domain (same three manuscripts), which is part of why it helps most there.

Finer sweep (`A1_lm_fine`, order-8 LM, 300 validation lines, alpha 0.2 to 0.5 x beta 1.0 to 2.5): the optimum is interior, alpha 0.4 with beta 1.0 to 1.5
(validation 5.41 % vs greedy 6.83 %). Test with the chosen alpha 0.4, beta 1.0: **RASAM-1 5.46 %** (greedy 6.82), **RASAM-2 21.59 %** (greedy 24.59). So tuning moved the
result by under 0.1 point on RASAM-1 and 0.4 point on RASAM-2; the gain is robust to the LM order and settings and saturates near 5.5 % for this acoustic model.

## 8. B2 result: a manuscript-disjoint RASAM-2 split (2026-10-25) — and a correction

B2 = A2's recipe trained on 12 RASAM-2 manuscripts (2,357 train lines), with MS 1936, MS 1960 and MS 6 held out (every line of them is test; rule fixed in advance, see
`rasam_make_msplit.py`). Single seed, half width, raw labels, no LM.

| Test | n | CER (raw) |
|---|---|---|
| B2 on the 3 unseen manuscripts (all lines) | 703 | **31.07 %** (95 % CI 29.1 to 33.1); MS 1936 71.5 %, MS 1960 15.5 %, MS 6 21.3 % |
| B2 on the published test lines of the 12 training manuscripts | 304 | 14.97 % |
| Same-line comparison, unseen manuscripts: lines that are in A2's published test split | 68 | A2 (trained on those manuscripts) 32.76 % vs B2 (never saw them) **32.98 %** |
| Same-line comparison, the other 12 manuscripts | 304 | A2 15.35 % vs B2 14.97 % |

**Correction.** I said earlier that the random line split makes in-domain scores optimistic because every test page also occurs in train. For RASAM-2 and this
recipe the evidence says it barely matters: a model that saw 195 lines of a manuscript is no better on that manuscript's test lines than one that never saw it
(0.2 point difference on 68 lines, well inside the noise). Difficulty is intrinsic to the manuscript (MS 1936 is at 71 % either way), not leakage. The claim
may still hold for RASAM-1 (3 manuscripts with up to 2,700 training lines each), which B2 does not test, and page-level overlap is not isolated here.
Caveats: one seed, a small comparison set, a wide CI.

## 9. Operations log (2026-10-25)
- The SSD disconnected twice (2026-10-24 about 21:00 and 2026-10-25 overnight); the queue master was relaunched after each. B2 finished before the second drop; A3 was
  at epoch 7 when it was cut. Everything the Colab supervisor needs now lives on the internal disk (`~/colab_staging`, small files) and is copied to the SSD afterwards.
- Colab Pro on this account (balance 200 units on 2026-10-24): **2 GPU sessions can run at once** (the third allocation was refused), **4 CPU sessions** ran at once.
  Measured billing: T4 1.07 units/hour, CPU 0.08 units/hour per session. Units spent on the Colab tests so far: 1.51 of the 10-unit budget.

## 10. A3 result (2026-10-25): training on RASAM-1 + RASAM-2 together (Colab T4, weights verified)

Same recipe as A1/A2 (half-width char CRNN, height 64, raw labels, geom-aug, 30 epochs, bs 16, seed 0), trained on both train splits (8,783 lines) on a
Colab T4 in 30 x 29 s; best checkpoint by the mixed validation split. Scored on the Mac on the full published test sets; character LM (order 8, train labels of
both sets) tuned on 300 validation lines of each set separately.

| Model | RASAM-1 test: greedy -> LM | RASAM-2 test: greedy -> LM |
|---|---|---|
| A1, RASAM-1 train only | 6.82 % -> 5.46 % | 24.59 % -> 21.59 % |
| A2, RASAM-2 train only | 24.86 % | 18.47 % |
| **A3, both** | **6.55 % -> 5.35 %** | **12.84 % -> 10.45 %** |
| published word-based CRNN trained on both (CHR 2024) | 5.50 % | 6.79 % |

What it shows:
- **More (and different) data helps the smaller set most:** adding RASAM-1's 5.9 k lines to RASAM-2's 2.9 k cuts RASAM-2 test CER from 18.5 % to 12.8 % (-5.7 points,
  -31 % relative), although RASAM-1 comes from 3 other manuscripts. RASAM-1 also gains a little from RASAM-2 (6.82 -> 6.55 %). This is the clearest evidence so far for
  the "more data gives better results" argument.
- With the LM, RASAM-1 reaches 5.35 %, below the published 5.50 % (caveats: unknown aggregation and lines in the paper, one seed); RASAM-2 at 10.45 % is still well behind
  the published 6.79 %, which is where the remaining levers (resolution R2, capacity, seeds) should be tested.
- Diacritic-insensitive: 4.98 % (RASAM-1) and 9.95 % (RASAM-2) with the LM.

## 11. R2 result (2026-10-25): input line height 96 px beats 64 px (Colab T4, verified weights)

A3 recipe (RASAM-1+2 train, half width, 30 epochs, seed 0) with the line image at 96 and 128 px instead of 64. The 128 px run was cut by its unit cap at about
epoch 21 (best checkpoint kept), so it is not a fair comparison. Scored on the Mac, full official test sets; character LM tuned on validation only.

| Line height | RASAM-1 test greedy -> LM | RASAM-2 test greedy -> LM | s / epoch (T4) |
|---|---|---|---|
| 64 px (A3) | 6.55 % -> 5.35 % | 12.84 % -> 10.45 % | 29 |
| **96 px** | **5.39 % -> 4.58 %** | **12.14 % -> 9.66 %** | 56 |
| 128 px (partial, ~21 epochs) | 5.85 % -> 4.68 % | 13.51 % -> 10.49 % | 93 to 101 |

- 96 px removes about a sixth of the RASAM-1 errors (thin strokes and dense lines: section 7 found 12 % of errors were lost or invented thin letters).
- **RASAM-1 with the LM at 4.58 % is below the best published 4.8 %**, on the official test split (one seed; the paper's own test lines are unknown).
- RASAM-2 improves less (9.66 % vs the published 6.79 %); more training data (TariMa, all of RASAM-1) is the next lever, running now.

## 12. Hard-line analysis of the best model (96 px + LM), 2026-10-25 (Mac CPU)

Scripts: `rasam_hard_lines.py` (features, error concentration, hard-for-everyone check, consensus ensemble, image gallery in
`results/rasam/hard_lines/hard_lines_{r1,r2}.html`), `rasam_broken_crops.py`.

**Errors are concentrated.** RASAM-1: the worst 10 % of lines hold 41 % of the character errors (RASAM-2: 46 %). Lines that fail are hard for every model:
of the 16 RASAM-1 lines above 30 % CER, 14 are above 30 % for all four of our systems (RASAM-2: 24 of 33), so they point at the data more than at the model.

**What makes a line hard (best system, micro CER).**
- RASAM-1: catchwords 51 % (n = 10), marginalia 8.1 % vs main text 4.5 %; lines with `#` (illegible placeholder) 13 %; the tallest crops (307 to 634 px,
  neighbouring lines inside) 6.5 % vs 3.5 to 4.2 %; the most red/blue-ink lines 6.3 % vs about 4.2 %; MS 609 6.0 % vs 4.0 % / 3.5 %.
- RASAM-2: faint lines (lowest dark-ink fifth) 17.5 % vs 5.6 %; low paper contrast 14 % vs 6 %; very short or very sparse lines worse; **MS 1936 66 %**.
- Length, dotted-letter share and diacritics have little effect.

**Data defects found.**
- **80 RASAM-2 crops of MS 1936 are degenerate (1 to 7 px tall): 66 in train, 7 in validation, 7 in test.** No other set has any. The 7 test crops hold 12 % of our
  RASAM-2 errors (9.66 % -> **8.59 %** without them); the 66 training crops teach the model to emit filler ("الا الا ان"). MS 1936 also contains Berber (Tamazight)
  written in Arabic script, which neither our Arabic training data nor the Arabic LM covers.
- RASAM-1 MS 609: two test crops show the neighbouring line (each reading is the other line's reference).
- Some references look garbled where our reading is fluent Arabic (e.g. reference "لنه افتممما عليه فقالالطه طلع البلتتجبى…", ours "حتى اجتمعوا عليه فقال لهم طلع
  الليلة…"): either label noise or a deliberately diplomatic transcription of scribal slips; needs a manual review.

**LM effect.** RASAM-1: 327 lines better, 106 worse, 301 unchanged; RASAM-2: 233 better, 50 worse.

**Free ensemble.** Taking, per line, the reading that agrees most with the other systems (medoid of 4 readings, no reference used): RASAM-1 **4.23 %** (best single
4.58 %), RASAM-2 9.57 % (9.66 %). The per-line oracle (choosing with the reference) is 3.57 % / 8.11 %, so better selection or rescoring has room.

### Recommendations
Data (highest value, also a dataset contribution):
1. Re-cut the 80 MS 1936 lines from the original pages and PAGE-XML (both on the SSD); train without the broken crops; report the official test and a
   "repaired" test side by side.
2. Manual label review of the hardest 60 lines (gallery ready): mark crop swaps, garbled references, diplomatic vs normalised transcriptions.
Augmentation (each targets a measured weakness):
3. Neighbour-ink augmentation: paste fragments of other lines above and below and jitter the vertical crop (tall crops 6.5 % vs 4 %).
4. Colour and contrast: feed the darkest RGB channel instead of grey and add contrast/fading jitter (red/blue ink 6.3 %; faint RASAM-2 lines 17.5 %).
5. Short lines: cut word-level sub-lines from long training lines using our CTC alignment (catchwords, marginalia, fragments).
Models and decoding:
6. N-best rescoring: export the beam's n-best and rescore with a stronger Arabic LM (a pretrained Arabic causal LM on a T4), restricted to the n-best so it cannot
   hallucinate; plus the consensus ensemble. Cheap, and the oracle gap says it can pay.
7. VLM fine-tuning (Qwen-VL / Fanar, as in suite_v1) is not worth the units now: our CRNN is far ahead of the old VLM fine-tunes (11 % / 17 %) and a full RASAM fine-tune on
   an A100 would cost about 10 to 15 units.

## 13. More data at 96 px: TariMa helps, more of the same manuscripts does not (2026-10-25, Colab T4, verified)

Same recipe (96 px, half width, 30 epochs, seed 0). "Repaired" = RASAM-2 test with the 7 broken MS 1936 crops re-cut from the page images (the model is
the same; it was trained on the published, partly broken data). Character LM trained on each model's own training text, tuned on validation only.

| Training data | RASAM-1 test | RASAM-2 official | RASAM-2 repaired | TariMa test |
|---|---|---|---|---|
| RASAM-1 + RASAM-2 train (baseline, section 11) | 5.39 -> 4.58 % | 12.14 -> 9.66 % | 11.04 -> 8.94 % | - |
| + TariMa train (1,828 lines, 26 documents, manuscripts and lithographs) | 5.19 -> **4.32 %** | 11.92 -> **9.41 %** | 10.83 -> **8.42 %** | 8.76 -> **7.33 %** |
| + RASAM-1 validation and test (all of RASAM-1; RASAM-2 scored only) | - | 12.32 -> 9.77 % | 11.25 -> 8.92 % | - |

(greedy -> with LM; micro CER, raw profile)

- **Diversity, not volume.** About 2,000 lines from 26 new documents (TariMa) improve every test set; about 1,470 more lines from the 3 RASAM-1 manuscripts the model
  already knows do nothing for RASAM-2. Together with A1 -> A3 (adding RASAM-1 cut RASAM-2 error by a third) and the RASAM-2 paper's own cross-manuscript cliff,
  this is the strongest evidence so far for the "better (more diverse) data -> better results" argument.
- RASAM-1 with TariMa and the LM: **4.32 %, below the best published 4.8 %** on the official test (one seed; the paper's own test lines are unknown).
- TariMa test 7.33 % (published: 9.70 % for the RASAM-2 model applied out of domain, CHR 2024; Phoenix 10.72 % on its own TariMa split).

## 14. Capacity: full width does not pay (2026-10-25)

Same as the 96 px baseline but width 1.0 (about 9.4 M parameters instead of 2.4 M); 143 s/epoch on a T4 (2.5x the half-width model); cut by its 1.4-unit cap at
about epoch 26 (best checkpoint), so slightly under-trained.

| 96 px, greedy -> LM | RASAM-1 test | RASAM-2 official | RASAM-2 repaired |
|---|---|---|---|
| width 0.5 (baseline) | 5.39 -> 4.58 % | 12.14 -> 9.66 % | 11.04 -> 8.94 % |
| width 1.0 (about 26 epochs) | 5.22 -> 4.47 % | 12.16 -> 10.18 % | 11.08 -> 9.21 % |

No consistent gain at 2.5x the compute: the final recipe keeps width 0.5 and spends the units on data and seeds instead.

Runner change the same day: when the HF copy already holds every weight and config file of the VM (sha256), the VM is stopped immediately and the Mac copy is
fetched from HF (CPU self-test: early stop verified, 6/6 files verified). This removes the slow VM-to-Mac download from the billed time.

## 15. Repaired data and augmentations (2026-10-25, Colab T4, verified; one seed each)

RASAM-1 + RASAM-2 train at 96 px, half width, 30 epochs; RASAM-2 train/validation use the 80 re-cut MS 1936 crops. Augmentations from section 12:
neighbour ink (p = 0.3: strips of other lines pasted above and below), fade (p = 0.3: ink pulled towards the paper), darkest RGB channel as input.

| greedy -> LM | RASAM-1 test | RASAM-2 official | RASAM-2 repaired |
|---|---|---|---|
| published data, no extra aug (section 11) | 5.39 -> 4.58 % | 12.14 -> 9.66 % | 11.04 -> 8.94 % |
| repaired data | 5.18 -> **4.33 %** | 12.30 -> 10.03 % | 11.13 -> 8.96 % |
| repaired + neighbour + fade | 5.40 -> 4.43 % | 12.84 -> **9.65 %** | 11.69 -> **8.94 %** |
| repaired + neighbour + fade + darkest channel | 6.44 -> 5.14 % | 13.05 -> 10.12 % | 11.99 -> 9.43 % |

- The darkest channel is clearly worse (dropped).
- Repairing the training data helps RASAM-1 a little and is neutral on RASAM-2; the neighbour/fade augmentation is mixed. Both effects are under half a point,
  which is the size of the run-to-run noise we have not measured yet. **Next: several seeds of the leading recipes before any further recipe claim.**

## 16. Seeds, 60 epochs, weight averaging, label standardisation, pretrained-LM rescoring (2026-10-26, Colab T4 + Mac CPU, verified)

Recipe: RASAM-1 + repaired RASAM-2 + TariMa train, 96 px, half width, geometric augmentation. Official HF test sets unless marked; micro CER, raw profile
unless marked. Character 8-gram LM weights tuned on validation only.

**Run-to-run noise (3 seeds, 30 epochs, raw labels).** Greedy: RASAM-1 5.32 / 5.18 / 5.23 % (mean 5.24), RASAM-2 11.83 / 11.79 / 11.74 % (11.79),
TariMa 9.26 / 9.32 / 9.22 % (9.27), repaired RASAM-2 10.68 / 10.63 / 10.53 % (10.61). With the LM: 4.38 / 9.30 / 7.67 / 8.50 % (means; spread about
0.1-0.2 points). Differences under about 0.2 points between single runs are noise.

**Standardised labels do not make a better model.** The standardised-label recipe (3 seeds, LM) scores 4.04 / 8.82 / 7.02 / 7.88 % against standardised
references. The raw-label models scored diacritic-insensitively (the same simplification) give 4.00 / 8.78 / 7.17 / 7.97 %. The earlier "gain" of
standardised labels (section 15 follow-up) is the easier reference, not a better reader. Keep raw labels and report both profiles.

**60 epochs instead of 30 (one run, seed 0).** The one-cycle schedule had flattened the 30-epoch curve only because the learning rate reaches zero.

| test | 30 ep, 3-seed mean greedy -> LM | 60 ep best ckpt | 60 ep final | 60 ep EMA best (ep 48) | 60 ep EMA final | 60 ep avg of last 10 |
|---|---|---|---|---|---|---|
| RASAM-1 | 5.24 -> 4.38 | 4.74 -> 4.33 | 4.81 | 4.94 -> 4.28 | 4.77 | 4.79 -> **4.24** |
| RASAM-2 | 11.79 -> 9.30 | 10.91 -> **9.13** | 10.74 | 11.30 -> 9.14 | 10.79 | 10.89 -> 9.19 |
| TariMa | 9.27 -> 7.67 | 8.34 -> 7.35 | 8.37 | 8.74 -> 7.35 | 8.31 | 8.32 -> **7.19** |
| RASAM-2 repaired | 10.61 -> 8.50 | 9.77 -> **8.05** | 9.61 | 10.14 -> 8.20 | 9.64 | 9.74 -> 8.13 |

- 60 epochs cut greedy CER by 0.4-1.0 points (well beyond seed noise); after the LM the gain shrinks to 0.05-0.45 points. Validation flattens from
  about epoch 46, so more than 60 epochs is unlikely to pay.
- EMA (decay 0.999) and averaging the last 10 epochs give nothing consistent over the plain final weights: the one-cycle schedule already anneals to a
  near-zero learning rate, which is what averaging would otherwise provide. Not worth adding to the recipe.

**Pretrained-LM rescoring (AraGPT2-base on the char-LM 10-best; weights w, v tuned on validation).** For the +TariMa model of section 13:
RASAM-1 4.32 -> **4.15 %**, RASAM-2 9.41 -> **9.02 %**, repaired RASAM-2 8.42 -> **8.04 %** (oracle in the 10-best 3.28 / 7.84 / 6.92 %).
Gains 0.2-0.4 points, consistent on all three sets. The repaired list is tuned on the official RASAM-2 validation N-best (same lines except 7 re-cuts).

**Next:** 60 epochs + AraGPT2 rescoring together (N-best from the 60-epoch model, LM scoring on a Colab CPU session, about 0.1 units); 3-seed
60-epoch runs only if the combination is the headline number.

**60 epochs + AraGPT2 rescoring (RS4b, 2026-10-26).** 10-best lists from the 60-epoch model (model_best), AraGPT2-base scores on two Colab CPU
sessions (about 12 min per list on CPU; the TariMa lists went to a second session because of the 120-minute VM limit), w and v tuned on validation:

| test | 60 ep + char LM | + AraGPT2 rescoring | oracle in the 10-best |
|---|---|---|---|
| RASAM-1 | 4.33 % | **4.12 %** | 3.18 % |
| RASAM-2 | 9.13 % | **8.92 %** | 7.66 % |
| TariMa | 7.35 % | **7.04 %** | 5.84 % |
| RASAM-2 repaired | 8.05 % | **7.83 %** | 6.61 % |

The two gains add up only partly: 60 epochs alone took RASAM-2 from 9.30 % (30-epoch mean) to 9.13 %, rescoring adds another 0.2-0.3 points everywhere.
Our best single-model numbers on the official HF test splits: **RASAM-1 4.12 %, RASAM-2 8.92 %, TariMa 7.04 %** (one seed; seed noise about 0.1-0.2).

## 17. Why is Calfa's per-corpus CRNN so much better? Specialisation and architecture tests (2026-10-27, Colab T4 + Mac CPU, verified)

The AMIDDA paper (HIP 2026, read in full) describes its per-corpus CTC baselines (3.6 / 4.9 / 3.9 % on exactly our official test lines) as
"pre-existing production models", one per corpus, "trained on the same material that AMIDDA inherits": greyscale lines at 120 px, three conv
blocks (64/128/256, GroupNorm) that keep 30 rows (height / 4), four BiLSTM layers of 512 units, dropout 0.3, CTC; CER with jiwer, no
normalisation (the same metric as ours). It does not say whether those models were trained before the current split existed.

Test CER with the char LM (raw):

| model | RASAM-1 | RASAM-2 | TariMa | RASAM-2 repaired |
|---|---|---|---|---|
| 60-epoch shared model (starting point, section 16) | 4.33 | 9.13 | 7.35 | 8.05 |
| + 10 epochs on all three sets (control) | 4.24 | **9.00** | 7.34 | **7.97** |
| + 10 epochs on RASAM-2 only (specialised) | - | 9.10 | - | 8.01 |
| + 10 epochs on TariMa only (specialised) | - | - | **6.90** | - |
| tall CRNN (our layers, height / 8: 12 rows), 30 epochs | 4.38 | 9.28 | 7.47 | 8.57 |
| Calfa-architecture CRNN (52.9 M, 120 px), 15 epochs, lr 3e-4 | 5.16 | 10.81 | 9.84 | 10.14 |

- **Specialisation does not explain the gap.** RASAM-2-only fine-tuning is no better than the same 10 epochs on all sets; TariMa-only gains
  about 0.4 points (one seed). Calfa's lead is 4 points on RASAM-2 and 3 on TariMa.
- **Vertical resolution alone does not either.** Keeping 12 rows instead of 3 in our model changes nothing beyond noise.
- **Their architecture, as described, is worse in our budget.** At lr 1e-3 it collapsed (CTC blank plateau, stopped at epoch 4); at 3e-4 it
  learns faster per epoch than ours early on but ends 15 epochs well behind (it is 22x larger and 6x slower per epoch). Their training
  length, augmentation and stopping rule are not published.
- What remains: longer training of the large model, unpublished training details, or training data that overlapped today's test lines.
  The draft request to Calfa (`docs/research/drafts/calfa-request.md`, not sent) asks exactly this.

Budget: specialisation 1.63 units, tall CRNN 1.53, Calfa architecture 3.01 (+ about 0.9 for the two stopped attempts).

## 18. Leave-manuscript-out on RASAM-2: how much the random line split flatters RASAM-2 (2026-10-27, Colab T4 + Mac CPU, verified)

RASAM-2's 15 manuscripts (about 250 lines each) in 5 folds of 3 (`make_lomo_specs.py`, `results/rasam/lomo_folds.json`). Each fold trains our
standard recipe (96 px, half width, 30 epochs, seed 0, repaired crops) on RASAM-1 + the other 12 RASAM-2 manuscripts + TariMa; every line of the
3 held-out manuscripts (train, validation and test) is unseen. Scored with the char LM (tuned on the remaining RASAM-2 validation lines).
Comparison on the SAME lines: each held-out manuscript's official test lines, read by the fold model vs the in-split models of the same recipe
(3 seeds, random line split, so they saw other lines of the same pages). Report: `rasam_lomo_report.py` -> `results/rasam/lomo_report.json`.

| | lines | unseen-manuscript model | in-split model (3 seeds) | gap |
|---|---|---|---|---|
| all 15 manuscripts, all lines (greedy / LM) | 3,720 | 14.17 / 12.20 % | - | - |
| all 15, official test lines | 372 | 12.01 % | 9.30 % (9.29 / 9.13 / 9.47) | +2.71 |
| without MS 1936 (broken crops), official test lines | 350 | **9.87 %** | **6.69 %** (6.69 / 6.58 / 6.80) | **+3.18 (+48 % relative)** |

Per manuscript (official test lines, LM): the gap is positive for all 14 regular manuscripts, from +0.16 (MS 1960) to +7.09 points (MS 1944;
MS 1943 +6.41, MS 1983 +4.26). MS 1936 is the outlier (46 % on all its lines even with the re-cut crops; its in-split figure is on the original
broken crops).

**Finding.** On RASAM-2, a model that has seen other lines of the same pages reads the official test lines about a third better than a model
facing a new manuscript. Every published RASAM-2 number on a random or undocumented split (including ours on the official split, and Calfa's
4.9 % if its training saw these pages) measures in-manuscript reading, not generalisation to a new manuscript. This is the gap Phoenix's
manuscript-independent split was designed to expose; this is its first measurement on RASAM.

Cost: two T4 jobs, about 3.5 units of their own (the supervisor's per-job figures include another project's sessions on the same account).
