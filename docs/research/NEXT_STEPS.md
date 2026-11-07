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
  - [x] (2026-10-27, design §17) Why Calfa's CTC (3.6 / 4.9 / 3.9 % on the same lines) leads: per-corpus specialisation, taller CRNN and their architecture tested; none explains the gap. Draft question to Calfa in `drafts/calfa-request.md` (not sent).
  - [x] (2026-10-27, design §18) **RS4c** Leave-manuscript-out on RASAM-2: unseen-manuscript 9.87 % vs in-split 6.69 % on the same official test lines (without MS 1936); +3.2 points.
  - [x] (2026-10-26, design §16: RASAM-1 4.12 %, RASAM-2 8.92 %, TariMa 7.04 %) **RS4b** Combine 60 epochs + AraGPT2 rescoring (N-best from the 60-epoch model, LM scoring on a Colab CPU session, about 0.1 units); seeds only if it becomes the headline.
- [x] (2026-10-28, design §19) **RS4d** AraMS-28k Maghrebi books as unseen manuscripts: book_03 13.94 % (paper: Kraken 32.71, HATFormer 37.88); books 20 / 24 use eastern fa / qaf dotting (15-19 % of fa read as qaf).
- [x] (2026-10-28, design §20) **RS4e** Few-shot adaptation to an unseen RASAM-2 manuscript: 10 lines close a third of the unseen-manuscript gap, 100 lines about 80 %, the whole pool (~220 lines) all of it (9.87 -> 6.63 % vs in-split 6.69 %).
- [x] (2026-10-29, design §21) **RS4f** OpenITI classical-Arabic char 8-gram mixed into the decoder: -0.2 to -0.4 points in-split, about -1.1 on the AraMS Maghrebi books (16.82 -> 15.67 %). Next: stack with AraGPT2 rescoring.
- [x] **RS4g** Pretrain our CRNN on ~40k extra Arabic lines, then fine-tune (2026-10-29, design §22): not better alone (5.21 / 11.58 / 8.29 greedy) but a hypothesis-level ensemble with E60 + OpenITI LM reaches **3.72 / 7.92 / 5.91** (`scripts/push_combo.py`). Checkpoint averaging/EMA: no gain; frame-level logit averaging hurts.
- [x] **RS4i** (done 2026-10-31 via RS4j–RS4l: re-cuts, drop list, clean benchmark, release, defect gallery) RASAM / TariMa data defects (design §22): published crops that show another place of the page (MS 1936, beyond the 80 degenerate ones), crops with neighbouring lines, vertical margin notes; references that are uncorrected HTR (whole pages: MS 1977 0071/0158, RES_MON_8_5560 0169 — MS 1977 already reported in calfa-co/rasam-dataset#2, 2021, closed unfixed), shifted between lines (MS 609 p. 00203), doubled, or cut short. Screens: `scripts/rasam_data_audit.py`, `scripts/rasam_crop_check.py`; examples tab in `results/rasam/push/both_fail.html`. Next: finish a page-level review, then send Calfa/OpenITI a list of affected lines (draft only; the user sends it). Official numbers stay on the published test set.
- [x] **RS4j** (done 2026-10-29, E60 recipe only; greedy) Clean retrain is within seed noise of the original on every test view (clean test 4.19 / 7.65 / 8.39 vs 4.24 / 7.79 / 8.24); the test-set fixes are what move scores (RASAM-2 10.79 → 7.81 re-cut). 250 crops re-cut (`rasam_recut_v2.pkl`), 202 lines dropped (`results/rasam/clean/drop.json`, rules in `scripts/rasam_clean_lists.py`, PR #7 manual-invalid included). Open: LM decode of both models on the three views; PT recipe not rerun. Original plan: Retrain on cleaned training data. (1) Repair: re-cut every crop whose published image does not match its PAGE-XML polygon (extend `rasam_recut_broken.py` from h < 8 px to the crop-check list). (2) Find bad training references with *cross-fitted* readings — a model never trained on that line (5 folds by page, or the LOMO packs), since E60/PT have memorised the train lines — and drop lines whose reference is far less plausible than the reading (garbled), shifted, doubled or truncated; drop vertical/blank crops. (3) Rerun the best recipes (E60 96 px w0.5 60 ep EMA; PT external pretrain + fine-tune) on the cleaned pack, then the N-best ensemble + OpenITI LM. (4) Report on the official test, a repaired-crop test, and a cleaned test (defective lines removed) side by side. Colab cost ≈ 2 T4 runs (≈ 6–8 units) plus CPU for cross-fitting; check the budget floor first.
- [ ] **RS4h** Fine-tune HATFormer on RASAM / TariMa (idea 3, later).
- [x] **RS4k** (done 2026-10-31) Clean test benchmark with error bars (`scripts/clean_bench.py` → `results/rasam/clean/bench.json`): official / drop-only / clean (25 re-cut test crops re-read by every runnable system, 21 bad-reference lines removed) / clean-indep (only people-found drops) views, 95 % page-bootstrap intervals, paired differences, 200-line VLM panel. Best system clean 3.17 / 5.47 / 5.59 (clean-indep 2.99 / 5.49 / 5.48); RASAM-2 interval 5.8–10.0 → 4.7–6.4; no ranking changes. Re-cut re-reads: ours/HATFormer/Kraken on the Mac, AraGPT2 + HAFITH + AMIDDA on Colab (`colab_recut_rerun.py`; outputs must be named `preds_*` or `colab_job.py` does not sync them — first CPU run lost them).
- [x] **RS4l** (done 2026-10-31) Release `release/rasam-clean/` (manifest of 447 lines, 250 re-cut crops, drop list, standalone scorer verified against clean_bench) and draft issue `docs/research/drafts/rasam-issue.md` for calfa-co/rasam-dataset (refs #2, #6, PR #7). **User to send** the issue and `drafts/calfa-request.md`.
- [x] **RS4m** (done 2026-11-01, [`dataqa.md`](dataqa.md)) Generic automatic data-QA checks for any line dataset (`src/arabic_ocr_bench/dataqa.py`, `scripts/dataqa_scan.py`, `dataqa_overlap.py`). There are 17 checks in five tiers: text (empty, repeated, char runs, off-script); image (degenerate, vertical, blank, multi-line, neighbour clip); joint (length mismatch); reader (GT truncated, image truncated, doubled, shifted, unreadable); dataset (duplicate image/GT, train-test GT overlap). On RASAM test they catch 13/17 of the by-eye non-garbled defects and find a new doubled reference (RASAM-1/test/000387). Scanned 9 local sets. AMIDDA test = the union of the RASAM-1/2/TariMa tests, and AMIDDA train shares none of them. KHATT test shares 67 % of its GTs with train. Next: an LM-based `garbled` check with a pluggable char LM; run the reader tier on Muharaf/NAKBA with our readings.
- [x] **RS5a** (done 2026-11-02) Toolkit plan written: assessment, error taxonomy v0.1 (71 types: 18 built, 49 planned, 4 future; fix classes A0/A1/S/H), toolkit spec, paper plan and roadmap. A website mock-up with real flagged lines is a private artifact. **The plan docs now live outside this repo** in the git-ignored local folders `tadqiq/` (toolkit; docs in `tadqiq/docs/`) and `tadqiq-platform/` (product). Each will get its own repo; the name is not final.
- [x] **RS5b** (done 2026-11-02) The user answered the open questions: name still open, licence must be compatible with the public datasets' licences, toolkit and platform split, v1 = line-level OCR/HTR, human verification later, 3090 soon plus ≤ 20 Colab credits for VLM readings.
- [ ] **RS5c** (in progress 2026-11-02) Implement the taxonomy's detectors in `tadqiq/src/tadqiq/` and scan every cached line dataset (`tadqiq/scripts/scan_all.sh`). Measure precision against reader readings (`eval_readers.py`) and recall against reported defects (`compare_known.py`: CER-HV Muharaf labels, RASAM cleaning lists). First full run done (16 datasets, 196k lines; summary in progress.json). Findings and the review queue are in the private mock-up artifact (Review tab, shared verdicts). **2026-11-03:** outside error reports compared (`tadqiq/docs/08`), and new detectors raise Muharaf CER-HV recall from 61 % to 92 % (77 % without script mismatch) at the same control rate. A local review app is in `tadqiq-platform/app`. Next (`tadqiq/docs/08` §9): readings for Muharaf, AMIDDA and NAKBA, plus the R08 CTC score; a per-dataset scope policy; segmentation checks from page geometry; a signature/region classifier; R07 orientation. Multilingual work is parked (`tadqiq/docs/05-roadmap.md`). Future modules (preparation advisor, coverage analytics, acquisition advice, dataset maintenance) are planned in `tadqiq/docs/09-future-modules.md`. **2026-11-03 precision sprint:** per-detector precision measured (`tadqiq/scripts/precision_proxy.py`); adapted-LM garbled, hygiene demotions and a scope policy cut Muharaf false alarms on valid lines 28.1 % → 18.0 % at 92 % recall. Word-context LMs (trigram, CAMeLBERT) are too weak to ship; odd_word (per-word char LM) is experimental. Figures + review page: `tadqiq/results/figures/muharaf/index.html`. Open: KHATT garbled 20 → 240 (fold leakage through repeated paragraphs; decide on grouping duplicates into one fold); the user's review of the precision sample in the app. **2026-11-04:** convention-aware false-alarm fixes (Muharaf false alarms 18 % → 13 % at 90 % recall; `tadqiq/scripts/experiments/demote_ablation.py`, `fafix_check_readers.py`). Open: orientation (upside-down pages) needs a stronger reader or a self-supervised orientation model (user to decide); signature-crop check; stacked-date segmentation; Xiaomi-OCR-0 (0.8B, no Arabic yet) fine-tune is a candidate run. **2026-11-04 (later):** garbled and letter_position experimental (Muharaf false alarms 13 % → 9 %); D01n near-duplicate images, P04 page outliers and a garbled reader gate built and tested (`tadqiq/tests/test_dataset_checks.py`); signature, stacked-text and orientation-window checks tried and dropped (`tadqiq/docs/10` §3). **Done (2026-11-04):** RASAM-1 re-check + text reader gate + three-tab public audit page (tadqiq/docs/10 §4). **Done (2026-11-05):** human RASAM-1 re-check (earlier Claude claims withdrawn), I10 rotated_crop, review-yield and model-impact evidence, fourth audit tab (tadqiq/docs/10 §5). **Done (2026-11-05, later):** RASAM-1 re-check closed (no more batches); tiered review deck (203 cards); leakage counts + model check (`tadqiq/scripts/leak_impact.py`); cross-dataset summary (`dataset_summary.py`, local 'Across datasets' tab); synthetic certain errors on Muharaf (`synthetic_inject.py`/`synthetic_eval.py`). **Done (2026-11-05, evening):** tiered review answered by the user (203 cards, 17 min; projections in `dataset_summary.json`); RASAM-1 synthetic run on Colab CPU (`configs/colab/synth_rasam1_cpu.json`; shift/truncate 99 %, merged text 80 %, dropped word 20 %, typo 10 %; `tadqiq/results/synthetic/rasam1_recall.json`). Review verdicts mean "transcription matches the image" (wrong = confirmed data error); projections count each line once and mechanical flags are no longer added as certain. Next: Iskandar rotated_crop needs 2+ more definite answers; structural findings (rotated crops, near-duplicate text) need their own question if we want their precision; a flip/orientation check (synthetic recall 0 %); word-level recall (drop_word, typo) is the weak spot.
- [ ] **RS5d** (started 2026-11-06) Evidence that cleaning helps, and where detection is blind.
  - [x] (2026-11-06) Review deck for the uncorrected datasets (`tadqiq/scripts/build_expect_deck.py`, `review_findings.py --deck expect`, port 8792): 101 definite + 96 possible cards; question = does the transcription match the image.
  - [x] (2026-11-06) Muharaf lines CER-HV did not check (`tadqiq/scripts/muharaf_unchecked.py`): 920 of 21,209 carry a flag; ~721 (663–764) errors expected from measured precisions (hard-line bias: optimistic); ranked cleaning list `results/muharaf-public/cleaning_list.jsonl`.
  - [ ] Dirty vs clean training on Muharaf (`scripts/muharaf_clean_exp.py`, lists `tadqiq/scripts/muharaf_clean_lists.py`): arms D / HV (−1,424 CER-HV errors) / TK (−2,008 toolkit flags, checks ≥ 80 % precision) / HVTK (−2,150) / RND (−2,150 random), 3 seeds, same CRNN recipe (h96, w0.5, 30 epochs, EMA), full and CER-HV-clean test. Colab T4, one session at a time (~7.5 h per seed, ~25 units total); resumable from HF.
    - **Seed 0 (2026-11-07):** full-test CER D 14.67 / HV 13.82 / TK 13.46 / HVTK 13.16 %; CER-HV-clean test D 12.52 / HV 11.25 / TK 10.85 / HVTK 10.50 %. RND (the control) was cut at epoch 19 by the unit cap; it is being completed. Seeds 1–2 are paused until the stronger-CRNN pilot below reports, because the cleaning gain may shrink with a stronger model (CER-HV saw only 0.2 points).
  - [ ] **Compete with CER-HV on Muharaf (main effort, 2026-11-07).** CER-HV (arXiv 2601.16713) uses the same split. Comparing their released predictions to ours on the same terms (whitespace collapsed): full test, them 9.67 vs our best 13.19 %; clean test, them 7.93 vs 10.54 %. The gap is the recognizer. Pilot (`configs/colab/muh_bp_pilot_t4.json`, `make_muh_clean_spec.py pilot`): their ResNet CRNN (`--arch bp`: 7×7 stem, 2/4/4 ResNet blocks, column max-pool, 3×BiLSTM-256, CTC shortcut ×0.1) plus `--warp-aug` (small rotation + elastic), arm D, seed 0, 40 epochs, 6-unit cap. If it reaches ~10 % full: run D/HV/TK/HVTK/RND × 3 seeds (~45–75 units) and report their way (mean ± std, paired t-test). Then beam search + char LM, and a seed ensemble.
    - **Later plan, not now (user, 2026-11-07): three alternative recipes** that change what CER-HV did, based on results reported elsewhere. Each runs as a single-seed arm-D pilot against the bp baseline, and only the winner gets the cleaning arms.
      - **Recipe A, modern optimisation (same network).** AdamW with weight decay 0.05, 1–2 epoch warmup then cosine decay. Their recipe uses constant-LR Adam with early stopping at up to 800 epochs. EMA weights evaluated and checkpointed. SWA or checkpoint averaging over the last ~10 % of training. bf16/AMP with batch 32–64 and LR scaled to match. Stochastic depth (drop-path ~0.1) in the ResNet blocks. Dropout between LSTM layers raised to 0.3–0.5. Gradient clipping. A fixed budget (~150–200 epochs) instead of 800, with early stopping on the EMA validation CER. Expected effect: same or better CER for a quarter of the compute; the main risk is CTC instability at high LR (warmup handles it).
      - **Recipe B, HTR-specific augmentation (same network, same optimiser).** Keep their affine, elastic and noise. *Add:* random erasing / Cutout on line strips (common in English HTR, e.g. IAM recipes); stroke-width morphology (erode/dilate) for pen variation; SpecAugment-style time and height masking on the CNN feature sequence; random crop-boundary jitter, plus pasting a few rows of ink from neighbouring lines (Muharaf's real failure mode is neighbour-line clipping); paper texture, bleed-through and background-intensity synthesis for scanned manuscripts; slant/shear (±15°, to match Ruq'ah and right-to-left slant). *Drop or reduce:* large rotations (beyond ±3° they make labels wrong and we now have the orientation detector); grid distortion at high magnitude (it breaks dots; dots carry Arabic letter identity, cf. our R2.4 dot-error share). *Tune:* one RandAugment-style global magnitude knob, not per-op hand tuning. Expected effect: the biggest gain on the CER-HV-clean test, where most residual errors are writer/style errors.
      - **Recipe C, architecture + data (bigger change).** Replace the BiLSTM with a 4–6 layer Conformer or Transformer encoder over the CNN features, keep CTC, and add an attention-decoder auxiliary loss (CTC/attention multitask, as in speech and HATFormer/TrOCR-style HTR). Wider ResNet or ConvNeXt-T stem. Synthetic pre-training on rendered Arabic lines (OpenITI/Arabic Wikipedia text × historical-looking fonts), then fine-tune. Self-training: pseudo-label the 2,000+ lines our toolkit flags, plus unlabelled Muharaf/NAKBA pages, at a high-confidence threshold. Test-time: beam search with a char n-gram LM and horizontal-scale TTA. Expected effect: largest potential (this is where the 8–9 % systems sit), but highest cost; run it only after A and B.
  - [x] (2026-11-06) Synthetic grid v1 (`tadqiq/scripts/synthetic_grid.py`, Colab CPU): 9 datasets × 20 cells (text: shift, random text, merge, empty, truncate 10/25/50 %, drop 1/2 words, typo 1/2/3; image: crop 10/25/50 %, two lines, rot180, rot90, exact / near leaks). Whole-line text errors are caught 85–100 % only where model readings exist (RASAM-1/2, TariMa); rot180 and crops ~0–25 % everywhere; rot90, leaks, empty ~95–100 %.
  - [ ] Grid v2 (running): fresh baseline scan per dataset (AMIDDA's stored scan predates check changes: 12.7 % of controls gained flags), leak / copy partners excluded from controls.
  - [ ] (PARKED 2026-11-06 by the user: too many credits for now) New detectors for the blind cells: cross-fitted reader per dataset (2 folds by page, Colab GPU) so reader checks run everywhere; orientation from reader confidence on image vs 180° rotation; crop from reading length vs GT length plus left-edge ink; then rerun the grid with readings of the injected images.
- [ ] **RS5** Re-score our VLM fine-tunes on the full RASAM test sets (not the 200-line panel). AMIDDA 1.0 Line 4B (25.0 / 27.6 / 24.2 %) sits where our suite_v1 Qwen3-VL-4B LoRA was (27 / 33 %); Fanar (11 / 17 %) and AIN are only scored on the 200-line panel. Same-200-line comparison (2026-10-29): Fanar FT 11.2 / 17.2, AIN FT 18.0 / 24.0, AMIDDA 22.7 / 28.8, Qwen3-VL-4B FT 27.3 / 32.1. Needs the 3090 (merged weights are only on the SSD; a 15 GB transfer otherwise).
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
