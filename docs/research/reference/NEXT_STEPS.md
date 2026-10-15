# Next steps: the research backlog

> **The** place for what to do next. Last updated 2026-10-09.
> - Read [`GUIDELINES.md`](GUIDELINES.md) first: it holds the rules, context and conventions.
> - Phase status for the live Atlas page lives in [`progress.json`](progress.json).
>   When a task here changes state, update `progress.json` too and rebuild the
>   Atlas (see GUIDELINES §6).
> - Each task has an ID, the gate or decision it serves, its inputs and outputs,
>   and a **done when** line. Tick tasks off in place: `[x]` plus the date.

## Now (re-ordered 2026-10-07 after the per-sample findings)

Why this order: per-sample analysis (findings: "What makes a line hard") showed
that line difficulty is reliable (split-half up to 0.81), but text features
explain only 6–32 % of it. So the **image layer** is the biggest gap. Two
effects, end-of-line errors and short-line failures, may be pipeline
artefacts rather than script properties, and must be checked before anything
is built on them. All block datasets are in the local HF cache, so everything
below except E1 runs offline on CPU.

**CPU track**
1. [x] (2026-10-07) **R2.8 End-of-line check (D3).** The last tenth of the line is the worst
   in 9 of 10 blocks. Split the end-of-line excess into deletions
   (truncation, early stop) and substitutions, and relate it to image aspect
   ratio and to downscaling (`max_pixels`, the 1024 px cap). Also check
   whether GT lines are cut at the image edge.
   *Done when:* the excess is attributed to a cause (or ruled an artefact), and
   if it is the pipeline, a fix is proposed. *Done:* pipeline artefact, namely
   model-specific early stopping and output-token caps (gemini-flash truncated on
   74 % of lines); no line ends are cropped. Fix → X6.
2. [x] (2026-10-07) **R6.0 Image layer, first pass.** All 10 blocks: R² from
   0.06–0.51 (text) to 0.15–0.63 (text + image). Ink pixels per character is
   significant within RASAM-1 and NAKBA-blind; the public Muharaf copy is stored at
   60 px. Next: writer IDs (R0.4) for the remaining variance; R9 resolution probe.
3. [ ] **X5 Length-stratified reporting (D4).** Short lines (≤ 20 characters)
   fail disproportionately and drive whole-line failures. Report CER per length
   band on the leaderboard and Atlas, and flag fragment lines (digits, single
   words, marginalia) as their own subset.
   *Done when:* the leaderboard shows CER for fragments vs full lines.
4. [ ] **R3.3 Challenge subsets.** Using the now-reliable consensus difficulty:
   top-quintile-hard, dot-dense, OOV-heavy, medial-dense and fragment subsets,
   registered as dataset filters.
5. [ ] **R4.1 Label-noise review.** Rank lines by consensus difficulty and
   model agreement in every block; hand-review the top 30 per dataset. CER-HV's
   "script mismatch" flags are *not* label noise for VLMs (correct labels the
   CRNN charset could not read); segmentation and non-text flags are (AUC 0.94,
   n = 7). *Done when:* an estimated label-error rate per dataset.
6. [ ] **R2.7 Diacritic-policy metric decision (D4)**. (R2.6 done 2026-10-09.)

**GPU track (Colab or 3090; checked 2026-10-07)**

The 3090 is reachable over SSH and idle (187 GB free), but all merged models and
cached base models were deleted in clean-up; `a-experiment` holds only the
AIN-NAKBA LoRA checkpoints, char-CTC and Paddle (3.6 GB). E1 arm (c) needs only
its **142 MB LoRA adapter** (`suite-v1-artifacts/qwen3vl4b_visft_difflr/runs/exp3090-qwen3vl4b/adapter_model.safetensors`)
on the public Qwen3-VL-4B base, so either machine works. **Colab plan:** upload the
adapter to a private Hub repo, then one crash-safe notebook (src-layout path setup,
predictions checkpointed to Drive per batch) loads the base, applies the adapter, and
scores `e1.bench.yaml` (4 sets × 200 lines; fits a T4). Arm (b) trains in the
same notebook on an L4/A100 with step checkpoints mirrored to the Hub.
7. [ ] **E1.1 (~1 h):** bench condition (c) on the E1 panel. Copy the merged
   weights from `suite-v1-artifacts/qwen3vl4b_visft_difflr/merged/exp3090-qwen3vl4b/`
   (8.3 GB) to `<dir>\merged\suite-v1-qwen3vl4b-visft-difflr\` on the box, then
   `python scripts/run_3090_benchmark.py --dir <dir> --experiment suite-v1-qwen3vl4b-visft-difflr --datasets-file configs/training/suite_v1/e1.bench.yaml --no-full-accumulate --detach`.
8. [ ] **E1.2 (~10 h):** train + bench condition (b):
   `python scripts/run_3090_pipeline.py --dir <dir> --base qwen3vl4b_suite_visft --detach`.
9. [ ] **E1.3:** compare the three arms. **Primary outcome changed:** letter error
   on medial and dot-ambiguous letters (where the per-sample analysis puts the
   failures), not the dot share of CER, which the in-domain result showed does
   not move. Secondary: RASAM CER and the end-of-line excess. Paired bootstrap;
   a second seed if the gap is under 3 points.

**From the paper ledger (2026-10-08, `paper-ledger.html`): tests we can run now**
- [ ] **P-L1 Dot-restoration ceiling (D3)** (`alshaibani2025dotless`, `shannon1951`).
  A character n-gram on Muharaf train restores i'jam on predictions whose rasm
  is right but dots wrong; also report H(dots | rasm, context). CPU.
  *Done when:* share fixed vs broken (rare words separately) and the
  conditional entropy are in findings. The gain is capped at the dot tax (~1.8 pts).
- [ ] **P-L2 PAW-space repair (D3)** (`alluhaybi2020connect`). Merge a predicted
  space after a non-joiner when the joined form is in the Muharaf-train lexicon
  and the parts are not. CPU. *Done when:* WER/CER before and after on every
  competent pair.
- [ ] **P-L3 PAW-inventory coverage curve (D2)** (`lehal2013urdu`). Distinct PAWs
  and rasm-PAWs needed for 95 % / 99 % coverage per dataset. CPU, text only.
  *Done when:* the table is on the Atlas.
- [x] (2026-10-09, same-definition part) **P-L4 Similarity-choice fit (R2.6 extension)** (`mueller2012letter`,
  `alazzawi2026gap`). Similar-shape confusions as a share of *substitutions*,
  plus similarity vs bias per model family. *Done when:* a same-definition
  comparison with the paper's 30 % / 15 %.
- [ ] **P-L5 Checkpoint weight averaging (GPU)** (Baseer-Nakba). Average the
  AIN-7B NAKBA adapter over early and late checkpoints (400…1936 steps).
  *Done when:* NAKBA CER of the averaged vs final adapter.
- [ ] **P-L6 Per-sample loss logging in E1.2** (`swayamdipta2020cartography`).
  *Done when:* hard-to-learn train lines are compared with the CER-HV train flags.

**From the Arabic-script HTR deep dive (2026-10-09, ledger section "Arabic-script HTR studies")**
- [ ] **R4.1 via the review tool** (https://claude.ai/artifact/FAEdbYULz3g88KQhXoKiPQ):
  458 candidates (all 218 CER-HV-reviewed Muharaf test lines, their verdict hidden until
  you decide, plus the top 25 consensus suspects per block). Verdicts are stored per
  reviewer in the artifact's database. Export them with ArtifactData (`decisions`
  collection) to `results/label_review/`. Rebuild candidates:
  `PYTHONPATH=src .venv/bin/python scripts/build_label_review.py`.
  *Done when:* a label-error rate per dataset and κ with CER-HV on the 218 lines.
- [ ] **R1.6 Lexical diversity** with `alazzawi2026crossling` Table 3 definitions (unique
  lines, duplication, mean max Jaccard, unique words, TTR) per dataset and per K-subset. CPU.
- [ ] **R1.7 Train/test shape tail** (`alazzawi2026gap` Fig. 6): the share of test letters whose
  (letter, form) has fewer than 10 / 50 training examples, and an iso/fin-merged shape count. CPU.
- [ ] **E3′** Run E3 on their K grid (100, 500, 1k, 2k, 4k, full) and report it in training
  characters too, so our LoRA curve sits next to their CRNN curve (Muharaf, KHATT).
- [ ] **E5 CRNN reference.** Their CRNN (Retsinas best practices; CER-HV `src/`) trained on our
  datasets on the 3090. *Done when:* CRNN CER and the R2.6 profile on every block.
- [ ] **E6 Maghrebi adaptation.** Arms: a prompt hint (qāf one dot above, fā one dot below);
  RASAM train lines (5,864) in the mix; dot-convention augmentation. *Done when:* ق→ف and
  RASAM CER per arm.
- [ ] **E7 Multilingual pre-training with a freeze sweep.** CRNN pretrained on Latin (IAM,
  READ, NorHand), on Arabic-script (KHATT, NUST, PHTD) and on both; fine-tune at K = 100, 500,
  1000 with no freezing, CNN layers 1–3 frozen, all CNN frozen, or only the BiLSTM trained.
  Prior: `aradillas2021boosting` (freeze ≤ 3 CNN layers), `alazzawi2026seqmodel` (transfer
  goes with the BiLSTM). *Done when:* ΔCER vs joint training per setting.
- [ ] **A1 Augmentation ablation.** Catalogue items 13 (kashida stretch), 14 (PAW-gap
  jitter, only with alignment or ≤ ±20 %), 16 (dot minimal-pair renders) and 17 (Maghrebi
  dot swap) against the standard 1–4 baseline. Each must move its target error without
  raising total CER. Previews and verdicts: ledger augmentation catalogue.
- [ ] **X9 Contact the Luleå authors** about the CER-HV licence (X3), the cross-lingual code
  release, and our VLM-side numbers. Ask the user first; this is outward-facing.

**Dropped or demoted by the findings**
- Ensembles / ROVER: the per-line oracle beats the best single model by only
  0.8–3.6 CER points.
- PAWs per word as a difficulty predictor: it was word length. (PAW-gap
  *space* errors, R2.4, still stand.)
- Augmentations that break image–label agreement (decided 2026-10-09 from the
  previews): dot dropout (item 15: a missing dot leaves the label describing
  another word, and it rewards language-prior guessing) and pixel span masking
  (item 6). Not to be used for training.
- R5 language layer: OOV share is already measured and is the strongest text
  predictor. The rest of R5 waits until R6 says how much is left to explain.

### Done 2026-10-07
- **H1** research layer committed on `research-characterization`.
- **X1** `kitab_muharaf` tagged in-train; fine-tune scores on it are never quoted.
- **R0.3 + R2.1–R2.4** canonical predictions and the error decomposition.
- **X2** cleaned Muharaf: public flag logs (CER-HV, no licence). Removing the
  flagged test lines moves CER < 1 point and changes no rank.
- **E1.0** E1 configs; preliminary in-domain result −7.6 CER points.
- **R3.1 / R3.2-lite** per-sample characteristics over complete blocks
  (`characterize/samples.py`, `scripts/sample_characteristics.py`, Atlas §10).

---

## R: characterisation track (see `characterization-plan.md` §Revised phases)

### R0 Inventory (P0)
- [x] R0.1 GT for 18 datasets recovered from 489 prediction files (2026-10-07)
- [x] R0.2 Muharaf train text from the HF cache (2026-10-07)
- [x] **R0.3** (2026-10-07) Pick one canonical prediction run per (model, dataset): the
  latest complete run with a known prompt version. Write it to
  `results/characterize/canonical_runs.json`.
  *Done when:* every leaderboard row maps to exactly one predictions file.
  *Done, adapted:* leaderboard rows are already unions of runs, so the rule is
  per sample (`characterize/runs.py`): standard prompt only (the prompt-study
  ablations are excluded), latest successful reading, `smoke/` runs skipped (one
  wrote a different checkpoint under an existing model name). The manifest lists
  every source file per pair. 161 of 162 pairs with the same n reproduce the
  leaderboard's diacritic-insensitive CER exactly.
- [ ] **R0.4** Recover writer, page and document IDs into `Sample.meta` in the
  adapters (Muharaf, KHATT via KITAB, NAKBA, RASAM). *Done when:* the profiler
  can report lines per writer.

### R1 Text profiler (P1)
- [x] R1.1 Positional-form (locus) analyzer + position-aware rasm map + PAWs, 13 tests
- [x] R1.2 `scripts/profile_datasets.py` over all sets, exact-line leakage
- [ ] **R1.3** Add a profiler output: the share of lines whose diacritics are
  *inconsistent* within a dataset (the same word with and without marks).
- [ ] **R1.4** Write a datasheet-card stub per dataset into the Atlas (number,
  policy flags, warnings).

### R2 Error decomposition (P2): R2.1–R2.4 done 2026-10-07
Code: `characterize/errors.py`, `scripts/decompose_errors.py` →
`results/characterize/<stamp>/errors.json`, tests `tests/test_characterize_errors.py`,
Atlas §9. Findings in `../experiments/findings-and-notes.md`.
- [x] **R2.1 Normalisation ladder.** Score every canonical prediction at the
  rungs raw → canonical → +tatweel/digits → +orthographic folds →
  −diacritics → rasm → −spaces. The ladder functions go in
  `characterize/errors.py`; reuse the steps in `normalize.py`.
  *Done when:* there is a per-(model, dataset) table of CER at each rung plus
  the share of error owed to each rung, and the Atlas shows it.
- [x] **R2.2 rasm-CER and dot tax.** Map both sides with `rasm_of`. Dot tax =
  CER(diac-insensitive) − rasm-CER.
- [x] **R2.3 Diacritic-mismatch tax.** Count marks inserted where the GT has
  none, and stray GT marks dropped. Compare against each dataset's policy.
- [x] **R2.4 Aligned error taxonomy.** Levenshtein with backtrace; classify
  each operation as dot / shape-similar / PAW-space / diacritic / ligature /
  orthographic / other / whole-line hallucination. Add a unit test on
  hand-built pairs: ب→ت, a dropped fatha, a split lam-alef, an inserted space
  at a PAW gap.
- [ ] **R2.5 Fluency-hallucination rate**: unaligned predicted words that are
  frequent MSA words.
- [x] (2026-10-09) **R2.6** Split the "other" bucket (~58 % of competent-pair edits): letter
  substitutions outside the dot/shape groups vs letter insertions vs deletions,
  and check the shape-similar groups against the real confusion matrix
  (`analysis/charstats.py`) rather than the hand-written list.
  *Done when:* "other" is under 30 % or its parts are named.
  *Done:* `scripts/substitution_analysis.py`. The parts are letter substitutions 25 %, deletions
  17 %, insertions 12 % and non-letters 4 % of edits. 65 % of letter substitutions match no
  group. With the paper's metric on the same Muharaf lines, VLMs make as many similar-letter
  errors per character as the CER-HV CRNN (findings: "R2.6").
- [x] (2026-10-09) **R2.9 Style-convention error.** ق→ف and ف→ق per GT occurrence, per dataset
  and paired per model. Maghrebi RASAM: 23–27 % for base AIN-7B vs 2–3 % elsewhere.
  Next: add the rate to the Atlas, and check AraMS (same asymmetry) for Maghrebi hands.
- [ ] **R2.7** Decide the headline-metric question the diacritic tax raises (D4):
  should CER·light keep harakat on datasets whose GT is undiacritised?
  Re-score with marks stripped on `none`-policy sets and compare model rankings.
  *Done when:* there is a rank-correlation table and a recommendation.

### R3 Consensus difficulty (P3)
- [x] **R3.1** (2026-10-07) ISRI-style quality groups: per sample, the median CER across
  models, split into quintiles per dataset. *Done* over complete blocks, with
  split-half reliability (0.54–0.81 on line sets) and a rank regression on text
  features (R² 0.06–0.32).
- [ ] **R3.2** IRT (2PL) difficulty and discrimination per sample, with models
  as respondents.
- [ ] **R3.3** Challenge subsets: see "Now" #4 (high-PAW replaced by
  medial-dense and OOV-heavy, per the findings).

### R4 Annotation audit (P4)
- [x] R4.0 Exact-line leakage vs Muharaf train (2026-10-07)
- [ ] **R4.1** Label-noise flags: every strong model fails *and* the
  predictions agree with each other. Hand-review the top 50 per dataset;
  report an estimated label-error rate.
- [ ] **R4.2** Near-duplicate leakage (MinHash on normalised text) and a
  perceptual image hash, within and across sets and splits.
- [ ] **R4.3** Spelling-inconsistency report: the same word in several
  spellings within one dataset (hamza, ta-marbuta, ya).
- [ ] **R4.4** A per-dataset conventions table, from the dataset papers and
  confirmed against the data.

### R5–R11
- [ ] **R5** Language layer: register/dialect, OOV, entropy, number and name
  density. Reference corpora (Arabic Wikipedia, Tashkeela, IAM, English
  Wikipedia). Tokenizer view: fertility, whether dot-pairs share tokens.
  CAMeL Tools as an optional extra.
- [ ] **R6** Image layer (first pass is "Now" #2, R6.0): px per x-height, dot size in px, blur, noise,
  contrast, skew/slant, overlap. Script-style labels on 50 lines per dataset.
- [ ] **R7** Validity analysis (gate V2): mixed-effects model of per-sample
  CER plus leave-one-dataset-out and leave-one-model-family-out. Output: a
  ranked list of the characteristics that survive.
- [ ] **R8** Decision experiments (gate V5):
  - (a) choose normalisation and post-processing from the ladder, re-scored
    on existing predictions;
  - (b) coverage-balanced vs random fine-tuning data at equal size (GPU);
  - (c) binarise/deskew per model family (GPU).
- [ ] **R9** Causal probes (gate V3): synthetic renders that change one factor
  each (dots off, diacritics on/off, isolated forms, x-height sweep).
- [ ] **R10** Cross-script (gate V4): Persian PHTD, Urdu NUST-UHWR, Pashto, IAM.
- [ ] **R11** Release (gate V6): datasheet cards, the library, a write-up.

## L: literature track (template: `_TEMPLATE.md`, bibliography: `references.bib`)
- [x] L0 `script-characteristics.md` (Wave 0) · [x] `characteristics-framework.md`
- [ ] **L1 A1 Diacritics:** harakat in OCR/HTR, Unicode ambiguity,
  diacritisation policy, metrics. Feeds R2.3.
- [ ] **L2 A2 Annotation schema:** GT conventions, PAGE-XML/ALTO/TEI, input
  schema. Feeds R4.4 and S3.
- [ ] **L3 A3 Script types:** HTR vs print vs historical fonts vs layouts.
  Feeds R6 and E1.
- [ ] **L4 B1 Architectures:** CTC → seq2seq → VLM, vision-encoder freezing,
  tokenizers. Feeds E1 and E2.
- [ ] **L5 B2 Arabic datasets taxonomy**, including demographic metadata.
  Start from `../datasets-catalog.md` §5 and the 2025 ACM survey.
- [ ] **L6 B3 Other languages and transfer:** IAM, RIMES, READ, CASIA, Urdu,
  Persian. Also `alazzawi2026crossling`.
- [ ] **L7** Verify the items marked *(unverified)* in `references.bib`
  (Hijja venue, ISRI/DIQA details, Lehal figures).

## E: experiment track (GPU; 3090 by default, Colab as an option)
- [ ] **E1 Frozen vs unfrozen ViT, controlled.** Base Qwen3-VL-4B, 8k Muharaf,
  same seed and steps. Conditions:
  - (a) frozen: the existing run;
  - (b) unfrozen at the same LR: `exp3090_qwen3vl4b_visft.yaml`;
  - (c) unfrozen with a differential `vision_lr`: new `_visft_difflr` config;
  - a format-only prompt control.

  2 seeds or bootstrap CIs. **Primary outcome: the dot-error share (from
  R2.4).** Secondary: Maghrebi (RASAM) CER. Exclude `kitab_muharaf`.
- [ ] **E2 Tokenizer.** Read the char-CTC results. Then test whether fertility
  predicts per-sample CER (a join only, no training) and whether harakat
  encoding variants split into different tokens. Train only if there is an
  effect.
- [ ] **E3 Data scaling** at 1k/2k/4k/8k/full lines, measured in characters
  per (letter, position) shape (the R2 coverage curve).
- [ ] **E4 Baseer replay.** Lower rank, fewer steps, and some NAKBA-domain lines
  in the mix, to avoid forgetting (findings-and-notes).

## C: curriculum track: letters first, then words (P12, opened 2026-10-09)

The question: can a recogniser learn Arabic the way children do, letters
first (in their positional forms, grouped by shape), then PAWs, words and
lines? A second idea: give each letter its own slice of the weights, freezing
earlier slices as letters are added. Status: **parked** until the E-track
GPU work is done; the C1 and C2 arms run on the 1–2M-param char-CTC CRNN
(`charctc/`), which trains on CPU.

**What is already known**
- Easy-to-hard ordering is old (bengio2009curriculum). For HTR,
  louradour2014curriculum trained RNN-CTC on words then lines, Arabic
  OpenHaRT included, and gained mostly in convergence speed.
- wu2021curricula: curricula help mainly with a small training budget or
  noisy labels, which is our regime (8k Muharaf lines; label errors per
  CER-HV and R4).
- Domain-level ordering already works here: Baseer-Nakba's general → Muharaf
  → NAKBA chain beat a joint mix (0.16 vs 0.18 CER).
- The Arabic schoolbook (hijā'ī) letter order already groups shapes by rasm
  family (ب ت ث, ج ح خ, د ذ, ر ز, …), so "similar letters next to each
  other" is how children are actually taught.
- Freezing per-letter weight slices is continual learning: progressive
  networks (rusu2016progressive), PackNet masks (mallya2018packnet), and EWC
  as the soft version (kirkpatrick2017ewc). Those methods exist for data you
  can't revisit. We have every letter all the time, so joint training with
  replay is the upper bound they approach, not a baseline they beat.

**Why per-letter weight slices probably won't help as stated**
1. CNN features are parts, not letters. One "dot above" detector serves ن
   ت ث ف ق ز ذ خ ض ظ غ ش. R2 already shows letters factor into rasm ×
   dots, so cutting the weights by letter duplicates shared parts.
2. The hard part is cursive context, not isolated identity. Deletions and
   insertions are 28 % of edits in R2.6 (segmentation), and isolated-letter
   benchmarks are near ceiling.
3. 28 letters × up to 4 forms means about 100 stages, each with its own
   freeze ratio and order to tune.

Two reframings keep the intuition: weights split along real factors (C3),
and the incremental run used as a measuring tool, not a training method (C2).

- [ ] **C0 Data and baseline.** Render isolated letters in all positional
  forms (tatweel/ZWJ for medial and final), single PAWs and corpus words,
  using `training/synth.py` fonts and its degradation stack. Optional
  handwritten letters: Hijja (children aged 7–12, positional forms; check
  the licence). Get a char-CTC baseline that trains past the blank collapse
  in `charctc-runbook.md` on Muharaf 8k, evaluated on KHATT and Muharaf test.
  *Done when:* the baseline has a CER with a bootstrap CI and the stage
  datasets exist with manifests.
- [ ] **C1 Curriculum vs controls** (the main test). Equal total steps and 3
  seeds per arm:
  - (a) lines only;
  - (b) all stages shuffled into one mix, the key control for "more data,
    not order";
  - (c) letters → PAWs → words → lines, with replay of earlier stages;
  - (d) reverse order (anti-curriculum);
  - (e) letter order inside stage 1: hijā'ī shape families vs abjadī vs
    random.

  Run at 1k, 2k and 8k real lines. Outcomes: CER, steps to reach a fixed
  CER, and dot share (R2.4).
  *Prediction, written in advance:* (c) beats (a) clearly and (b) slightly,
  mostly at 1k–2k lines; (e) is within noise.
  *Done when:* the table has CIs and the prediction is confirmed or refuted.
- [ ] **C2 Letter-incremental probe.** Add shape families one at a time,
  three ways: replay only, PackNet-style frozen masks, and EWC. Record the
  **forgetting matrix**: how much learning family j hurts family i. That
  gives a measured feature-sharing map, to set against the R2.6 confusion
  pairs and the rasm groups.
  *Prediction:* replay ≥ EWC > frozen masks on final CER.
  *Done when:* the matrix and its rank correlation with the R2.6 confusions
  are in findings.
- [ ] **C3 Factorised head.** Shared trunk with two CTC heads, rasm skeleton
  and dot pattern, recombined at decoding, vs the plain character head. This
  keeps the "a set of weights per set of letters" intuition, along the
  script's real factors. Classic segmentation-based Arabic OCR handled dots
  separately; whether end-to-end CTC does it too is not searched yet (L
  task). *Done when:* CER and dot share for both heads with CIs.
- [ ] **C4 Checkpoint averaging** (from Baseer-Nakba's epoch-1 + epoch-5
  average). Not new: it is checkpoint averaging/SWA (izmailov2018swa), and
  averaging an early with a late model is close to WiSE-FT
  (wortsman2022wiseft). Cheap tests: the "Checkpoint weight averaging" row
  in the ledger queue (AIN-7B NAKBA adapters) and, for free, every C1 arm
  (average the stage-end checkpoints).
- Link to S: a children's dataset (S0) would record letter-learning stages
  in real handwriting, the natural data for C1 stage 1.

## X: fixes and follow-ups from findings
- [ ] **X6** Output-token caps: add `max_tokens` (and a minimal thinking budget)
  to `gemini-flash`, `openrouter-dots3-note-free`, `glm-5.3-flash-paid` and the
  other models that pile up at their cap; flag their current leaderboard rows;
  re-score within free quota. *Done when:* no model has > 5 % of lines at its cap.
- [ ] **X7** Full-resolution Muharaf: are the original scans and line polygons
  available above the public copy's 60 px line height? If so, re-crop and
  re-score a few models; pair with an R9 probe (KHATT downscaled to 60 px).
- [ ] **X3** Retrain without the 1,424 CER-HV-flagged Muharaf train lines
  (add an `exclude_ids` option to the training data spec, fed from
  `rescore_muharaf_clean.flagged("train")`). Low priority: the test-side effect
  is < 1 CER point. Check the licence with the authors before redistributing
  anything derived from their logs.
- [ ] **X4** Model-name hygiene: a run launched under one BASES entry with
  another entry's config writes outputs under the wrong model name (happened
  for visft-difflr). Make `run_3090_experiment.py` take the experiment name
  from the config, not the BASES entry, or refuse when they disagree.
- [ ] **X1** `kitab_muharaf` is 86.5 % in Muharaf train: exclude it or label
  it in fine-tune comparisons.
- [ ] **X2** Cleaned Muharaf: availability, licence, re-score, retrain.
- [ ] **X3** Check the ~6 % of `muharaf_lines` test lines that also occur in
  train (formulaic openings?) before quoting in-domain gains.
- [ ] **X4** Diacritic policy: a third of Muharaf lines carry sparse marks.
  Measure whether fine-tunes now *add* marks on undiacritised sets
  (R2.3), and decide whether to strip marks from the training targets.
- [ ] **X5** Add a line-level printed Arabic set, and a page prompt/track for
  the page-scale KITAB sets.
- [ ] **X6** Backfill `kitab_khatt` for Baseer-Nakba.

## S: own dataset (stretch; living design guide in `dataset-design.md`)
- [ ] **S-guide (ongoing)** Keep `dataset-design.md` current: every finding
  that changes what the dataset needs updates a rule there (with why and
  source) and its change log. Rewritten 2026-10-09 with text blocks T1–T11.
- [ ] **S0** Identify the school/partner contact and the ethics-approval route
  (minors: IRB, parental consent, child assent, licence). This blocks
  everything else in S. Start early.
- [ ] **S1** Content design: copy-text balanced over (letter × position) and
  dot-group minimal pairs (use the R1 analyzer to verify coverage), paired
  diacritised/undiacritised text, free writing, digits and form fields.
- [ ] **S2** Sampling design: grades either side of grade 3–4, gender,
  handedness. Other demographics only if R7/L5 show they matter.
- [ ] **S3** Annotation protocol from L1/L2, double annotation of a subset,
  an automated label check before release.
- [ ] **S4** Pilot with 20–30 writers; size the full collection from E3/R2.
