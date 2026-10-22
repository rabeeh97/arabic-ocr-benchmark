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
