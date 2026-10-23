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
