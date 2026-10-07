# Next steps: the research backlog

> **The** place for what to do next. Last updated 2026-10-07.
> - Read [`GUIDELINES.md`](GUIDELINES.md) first: it holds the rules, context and conventions.
> - Phase status for the live Atlas page lives in [`progress.json`](progress.json).
>   When a task here changes state, update `progress.json` too and rebuild the
>   Atlas (see GUIDELINES §6).
> - Each task has an ID, the gate or decision it serves, its inputs and outputs,
>   and a **done when** line. Tick tasks off in place: `[x]` plus the date.

## Now (in this order)

1. [x] (2026-10-07, branch `research-characterization`) **H1:** commit the research layer (docs/research/*, characterize/, the
   profiler, the Atlas builder, the tests, the findings entry) on a branch, so
   nothing lives only in the working tree.
2. [x] (2026-10-07) **X1:** stop quoting fine-tuned `kitab_muharaf` scores. Mark the set
   "in-train" in the leaderboard and fine-tune panel. *Done:* `make_leaderboard.py`
   reads `overlaps_source`, masks fine-tune cells on in-train sets and never counts
   them as held-out (`tests/test_leaderboard_in_train.py`). No such score had been
   published: the 3090 runners already ban the set.
3. [x] (2026-10-07) **R2.1–R2.4:** error decomposition: the normalisation ladder, rasm-CER,
   the diacritic-mismatch tax and the taxonomy. This is the next research
   deliverable.
4. [x] (2026-10-07) **X2:** find out whether the cleaned Muharaf release
   (`alazzawi2026labelerr`) is public. If it is, plan a re-score and a
   retraining. *Done:* public as human-verified flag logs (no corrected text;
   the paper removes flagged lines) at github.com/SanaNGU/CER-HV, **no licence
   file**, so nothing is vendored. IDs map 1:1 onto `muharaf_lines/test/*`.
   Re-score (`scripts/rescore_muharaf_clean.py`): dropping the 126 flagged test
   lines moves CER·light by < 1 point and changes no rank. Retraining → X3.
5. [x] (2026-10-07) **E1.0:** prepare the controlled ViT-unfreeze runs so the GPU works
   while P2–P4 run on CPU. *Done:* condition (c) had already trained on the box
   (2026-09-24) under the wrong BASES entry; its config is recovered as
   `configs/training/suite_v1/qwen3vl4b_visft_difflr.yaml`, condition (b) is
   `qwen3vl4b_visft.yaml`, both identical to the frozen suite recipe except the
   vision settings, with BASES entries and the shared panel `e1.bench.yaml`.
   Preliminary in-domain result in findings. GPU steps → E1.1–E1.3.

## Next (in this order)

1. [ ] **E1.1 (GPU, ~1 h):** bench condition (c) on the E1 panel. Copy the merged
   weights from `suite-v1-artifacts/qwen3vl4b_visft_difflr/merged/exp3090-qwen3vl4b/`
   (8.3 GB) to `<dir>\merged\suite-v1-qwen3vl4b-visft-difflr\` on the box, then
   `python scripts/run_3090_benchmark.py --dir <dir> --experiment suite-v1-qwen3vl4b-visft-difflr --datasets-file configs/training/suite_v1/e1.bench.yaml --no-full-accumulate --detach`.
   The new name keeps its rows apart from the old frozen `exp3090-qwen3vl4b-ft-bf16`.
2. [ ] **E1.2 (GPU, ~10 h):** train + bench condition (b):
   `python scripts/run_3090_pipeline.py --dir <dir> --base qwen3vl4b_suite_visft --detach`.
3. [ ] **E1.3:** re-run `scripts/decompose_errors.py` and compare the three arms
   on the panel: CER, rasm-CER, dot share (primary), RASAM CER. Paired bootstrap
   on shared lines. A second seed for the winner if the gap is under 3 points.
4. [ ] **R2.6 / R2.7** (CPU): split the "other" error bucket; decide the
   diacritic-policy metric question.
5. [ ] **R3.1–R3.3** consensus difficulty, then **R4.1** label-noise flags, which
   can now be validated against the CER-HV Muharaf flags as ground truth.

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
- [ ] **R3.1** ISRI-style quality groups: per sample, the median CER across
  models, split into quintiles per dataset.
- [ ] **R3.2** IRT (2PL) difficulty and discrimination per sample, with models
  as respondents.
- [ ] **R3.3** Challenge subsets (dot-dense, diacritised, long, high-PAW, top
  difficulty quintile) registered as dataset filters.

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
- [ ] **R6** Image layer: px per x-height, dot size in px, blur, noise,
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
