# Next steps: the research backlog

> **The** place for what to do next. Last updated 2026-10-07.
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

**RASAM track (added 2026-10-22; design: [`designs/rasam-crnn-and-replication.md`](designs/rasam-crnn-and-replication.md))**
- [x] (2026-10-22) **RS1** Fetch missing RASAM splits (RASAM-2 train/val, RASAM-1 val) to the SSD; verify counts.
- [x] (2026-10-22, Phoenix and TariMa; claim not reproduced, see design §6) **RS2** Kraken venv on the SSD; re-score Phoenix and the TariMa manuscript model on full RASAM-1/2 test and on NAKBA (CPU).
  *Done when:* each published claim has reproduced / not-reproduced with a reason.
- [x] (2026-10-22, columns already in the HF parquet) **RS3** Recover RASAM manuscript/page IDs from `calfa-co/rasam-dataset` (part of R0.4).
- [ ] **RS4** (A1 done 2026-10-23: 6.82 % RASAM-1 test, 24.6 % RASAM-2 test, half-width CPU, see design §6) Our own char-CRNN on RASAM train splits: validation sweep, arms A1–A4, then data-scaling A5 and diversity A6.
  - [x] (2026-10-26, design §16) 3 seeds of the repaired + TariMa recipe (raw and standardised labels): seed noise about 0.1–0.2 points; standardised labels give no real gain once scored the same way.
  - [x] (2026-10-26, design §16) 60 epochs vs 30: greedy −0.4 to −1.0 points, after the LM −0.05 to −0.45; EMA and last-10 averaging add nothing consistent (dropped).
  - [x] (2026-10-26, design §16) AraGPT2 N-best rescoring: −0.2 to −0.4 points on RASAM-1, RASAM-2 and repaired RASAM-2.
  - [ ] **RS4b** Combine 60 epochs + AraGPT2 rescoring (N-best from the 60-epoch model, LM scoring on a Colab CPU session, about 0.1 units); seeds only if it becomes the headline.
- [ ] **RS5** Re-score our VLM fine-tunes on the full RASAM test sets (not the 200-line panel).
- [ ] **RS6** Draft (do not send) a request to the RASAM authors for the word-based CRNN weights or test line lists.

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
6. [ ] **R2.7 Diacritic-policy metric decision (D4)**, then **R2.6** (split
   "other").

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

**Dropped or demoted by the findings**
- Ensembles / ROVER: the per-line oracle beats the best single model by only
  0.8–3.6 CER points.
- PAWs per word as a difficulty predictor: it was word length. (PAW-gap
  *space* errors, R2.4, still stand.)
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
- [ ] **R2.6** Split the "other" bucket (~58 % of competent-pair edits): letter
  substitutions outside the dot/shape groups vs letter insertions vs deletions,
  and check the shape-similar groups against the real confusion matrix
  (`analysis/charstats.py`) rather than the hand-written list.
  *Done when:* "other" is under 30 % or its parts are named.
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

## S: own dataset (stretch; spec in `dataset-design.md`)
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
