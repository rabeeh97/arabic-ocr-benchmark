# Master checklist (2026-10-15)

**This is the single list we work from.** It holds every item that remains or is planned, from every source:
- the research map (`research_map.json`, which itself places all 323 IDs from the ledger, NEXT_STEPS, the plans and this
  project's chats);
- the plans;
- the design documents.

`scripts/check_master_checklist.py` proves every unfinished map item appears here. When something new comes up, add it to the
map and here, then re-run the check.

**How to read an item**
- *CPU* / *GPU* / *human* / *you*: what it needs.
- *after*: what must come first.
- *done when*: the exit test.
- → the document that explains it.
- `map:` the map IDs it covers.

**Rules for every item that produces a number:** [`designs/evaluation-standards.md`](designs/evaluation-standards.md). These
cover the panel, store, seeds, intervals, FDR, WER and lines fixed, galleries, the ledger, and writing the decision rule before
the run.

**Order:** phases follow dependencies and value. Inside a phase, items are independent unless "after" says otherwise. GPU nights
are batched so the 3090 runs unattended.

---

## Phase 0 — From you (decisions and permissions; nothing else is blocked by these except where noted)
- [ ] **0.1 Go-ahead for phases 1–5** (laptop work and the first CRNN night) or changes to them. *you*
- [ ] **0.2 Permission for main-checkout changes.** This session cannot write there. *you* → [data-and-label-audit §7](designs/data-and-label-audit.md) · map: `main-checkout-sync`
  - [x] 0.2a Apply `patches/0001-paddleocr-reverse-output.diff` and re-score PaddleOCR.
  - [x] 0.2b Add the output-token-cap fix (X6) and token log-probabilities (`vlm-logprobs`) to the benchmark runner. Needed before phase 7.
  - *2026-10-15:* 0.2a/0.2b **applied to the main checkout on 2026-10-17 (uncommitted there; `patch --fuzz=0`, 28 tests pass / 2 skipped)** (`patches/0001` rebuilt; it had a bad hunk header; `0002` runner token cap + logprobs + `token_cap_report.py`; `0003` X6 caps with `cache_tag: x6`; glm-5.3 paid → cost review before any run). PaddleOCR CER from the stored readings, un-reversed: KHATT 85.8→62.9, Muharaf 87.9→77.1, NAKBA 89.3→83.4, RASAM-1 90.6→88.2, RASAM-2 91.2→85.7, hindawi 100→100.
  - [ ] 0.2c Refresh `docs/research/NEXT_STEPS.md` from this checklist; update the Atlas tracker (`progress.json`) and republish the Atlas.
- [ ] **0.3 Label review, about 2 h**, in the review tool, including "can a human read it?" for the 20 KHATT hard-core lines. Can happen any time before phase 14. *human* → [data-and-label-audit §1](designs/data-and-label-audit.md) · map: `R4.1`
- [ ] **0.4 Outward contacts, only if you want them:** the Luleå authors (CER-HV licence, code, our numbers); the KHATT authors for writer IDs (low priority). *you* · map: `X9`
- [ ] **0.5 Native-speaker check** of the calibration-sentence candidates (after 2.7). *you*

## Phase 1 — Standards and tooling first (CPU; before any new number)
- [x] **1.1 FDR control** in `analyze_letters.py` and `run_detectors.py` tables (Benjamini–Hochberg q = 0.1). *done when:* the ledger tables mark surviving cells. → [evaluation-standards](designs/evaluation-standards.md) · map: `fdr`
- [x] **1.2 WER and lines fixed or broken** in every pair and seed comparison. → [evaluation-standards](designs/evaluation-standards.md) · map: `wer-lines`
  - *2026-10-15:* `scripts/eval_stats.py` (BH, bootstrap p, WER, lines fixed/broken, ΔWER CI); A/B/D outputs and ledger tables regenerated. **Grade-relevant:** the KHATT replicate's surviving letter cells go 2/2 → 0/0 after FDR; ink crop on KHATT keeps 41/0. B/D outputs were stale (pre-Oct-14 store), so some ledger counts moved for non-FDR reasons. Ledger prose not yet reconciled (→ 2.20). Letter-cell p-values use a normal approx. to the bootstrap SE (e.g. 67 vs 64 cells for E8 vs CRNN on NAKBA blind).
- [ ] **1.3 Decision rules and claims register** kept current. Every item below states its rule (in its design document); update the register in `PLAN-2026-10-15-experiments.md` whenever a grade changes. · map: `decision-rules`, `paper-groupings`
- [x] **1.4 Seed-instability tool:** characterise lines that flip between identical trainings (3 seeds); add detector D24 (numeric and seed instability). → [evaluation-standards](designs/evaluation-standards.md) · map: `seed-instability`
- [x] **1.5 Remaining detectors:** D4 neighbour-line intrusion, D21 lam-alef ligature, D23 memorisation signature (also the C6 AIN check, kept unverified). *done when:* they appear in the detector table. → [data-and-label-audit §9](designs/data-and-label-audit.md) · map: `detectors-open`, `D21`, `C6`
  - *2026-10-17:* D4/D21/D23/D24 in `run_detectors.py` (+`detectors_extra.py`, `seed_instability.py`); gallery `results/analysis/D/gallery_new_detectors.html`. Run×dataset rows with hits / BH survivors: D4 393/64 (heuristic, ~2/3 precision by eye), D21 398/185 (lam/alef in the ligature misread 1.6× as often), D23 322/172 (AIN-7B on KITAB: 17.5 % exact lines, 17× peers; C6 still unverified), D24 398/248 (761 lines flip across seeds; single-seed "better" lines are 2–3× enriched in unstable lines, so require all-seed agreement for line-level claims).
- [x] **1.6 Full KHATT v1.0 in the panel** (1,038 test lines, as stored and cropped, fixed-text flag, overlap with KITAB-KHATT marked). → [data-and-label-audit §8](designs/data-and-label-audit.md) · map: `khatt-full`
  - *2026-10-15:* `scripts/khatt_full_panel.py` → `results/panel/khatt_full_{panel,train}.pkl` (gitignored) + flags parquet + manifest; 3090 handoff in [KHATT-FULL-3090.md](KHATT-FULL-3090.md). **Corrections to the design:** (a) the Teklia copy is almost all fixed text (1,035 of 1,038 test lines; vocabulary of 1,061 words), so a fixed-text vs unique-text split can't be done. Split by `text_in_train` instead (713 of 1,038). (b) 89 of the 200 KITAB-KHATT texts occur in v1.0, not 127 (12 test lines; *not* in-train for the aligner CRNNs, which train on Muharaf + RASAM only; corrected 2026-10-17). `store_build.py` now knows `kitab_khatt_full`.
- [x] **1.7 Lexicon assembly** on the SSD: Ayaspell, aspell-ar, the Pecina list if obtainable, Wikidata and GeoNames names, an Arabic Wikipedia vocabulary, ANETAC. Licences recorded; coverage report. → [lexicon](designs/lexicon.md) · map: `lexicon`
- [x] **1.8 Modern Arabic text sample** for language models (an Arabic Wikipedia subset) on the SSD. → [decoding-and-restoration §1](designs/decoding-and-restoration.md) · map: `lm-decoding`
  - *2026-10-17:* `scripts/lexicon/` → `/Volumes/PortableSSD/data/lexicon` (405 MB; merged `build/lexicon.tsv.gz`, 9.9M forms, hamza variants linked) and `data/text/arwiki` (one 2023-11-01 shard: 174k articles, 25.7M tokens). Licences, manifest and coverage are in `results/lexicon/`. Coverage of the panel's ground-truth words: 95.8 % (Muharaf) to 99.4 % (NAKBA). **Caveat:** the lexicons accept 20–23 % of dot-swap errors, so a lexicon constraint alone can't restore dots (matters for 5.1/5.3). Not obtained: the Alotaibi & Lee gazetteer; the skeleton and PAW indexes are still to be built.
- [x] **1.9 Fix `vlm_ink_extent.py`** (group by model; re-run with the un-reversed PaddleOCR) and escape newlines in gallery readings. → [data-and-label-audit](designs/data-and-label-audit.md) · map: `fix-vlm-ink-extent`
  - *2026-10-15:* grouping by model, `--paddle-order stored|unreverse` and the ⏎ newline escapes are done and re-run. *2026-10-17:* no PaddleOCR re-run needed. The client's `reverse_output` is a plain `text[::-1]` on a deterministic model's output, so the un-reversed stored readings are exactly what a re-run would give. Regenerated with `--paddle-order unreverse`: PaddleOCR 63.2 % CER overall, 98.2 % on loose crops, 58.3 % on tight crops.

## Phase 2 — Characterisation on existing data (CPU)
- [x] **2.1 PAW gap widths:** split vs unsplit PAW gaps against word gaps. → [dots-and-skeleton §1](designs/dots-and-skeleton-studies.md) · map: `PAW-gap-width`
  - *2026-10-18:* `dots_analyze.py` (rules pre-registered in `results/characterize/dots/prereg.json`). P(word gap is wider than a split PAW gap) = 0.68 [0.67, 0.70] for CRNN and 0.63 for VLMs: **word and PAW gaps overlap, so spacing is visually ambiguous.** Grade B−.
- [ ] **2.2 Novel-shape clusters (atlas v0).** → [shape-programme step 1](designs/shape-programme.md) · map: `novel-clusters`
- [x] **2.3 Skeleton features:** loops, teeth, tails; tooth-count detector. → [dots-and-skeleton §3](designs/dots-and-skeleton-studies.md) · map: `loops-teeth`
  - *2026-10-18:* Teeth beat the letter baseline in all 6 families (1.06–1.21, survive BH), so a **T3 finding, grade B**. The minim-style tooth-count test fails as specified (س counts as 3 teeth). Among one-tooth letters, errors rise with run length (CRNN 3+ vs 1 = 1.84×), grade C+.
- [x] **2.4 Dot-writing styles:** dash, caret, displaced, touching. → [dots-and-skeleton §2](designs/dots-and-skeleton-studies.md) · map: `dot-styles`
  - *2026-10-18:* No dot style survives the rule (grade D/C). Low confidence: dot detection on spans failed its validation (2.18). Re-run after a body-assigned dot detector.
- [x] **2.5 Vertical stacking** and reading-order errors. → [dots-and-skeleton §4](designs/dots-and-skeleton-studies.md) · map: `vertical-stacking`
  - *2026-10-18:* **Null result** (stacked vs non-stacked about 1.0 after stratification; CRNN 0.99 [0.90, 1.09]). Grade C. The ink-centroid stacking detector isn't hand-validated.
- [ ] **2.6 PAW-inventory coverage curve** (distinct PAWs for 95 / 99 %). → [shape-programme, related](designs/shape-programme.md) · map: `P-L3`
- [x] **2.7 Calibration-sentence candidates** (set cover over letter × form). *after:* 1.7 → [writer-adaptation C](designs/writer-adaptation.md) · map: `calibration-sentence`
  - *2026-10-18:* `calib_sentence_cover.py`. Full coverage needs at least 44 words (22 letters × final/isolated forms), so one or two lines can't cover everything. Best single sentence 65/104 forms; 7–8 real sentences reach 104/104; 6 short sets plus patch words reach 104/104 (shortest 51 chars). **Review page for 0.5: `results/calibration/review.html`.**
- [x] **2.8 Diacritic scoring rule** (rank correlation with and without marks; includes the sparse-marks note). → [data-and-label-audit §4](designs/data-and-label-audit.md) · map: `R2.7`
  - *2026-10-17:* `audit_diacritics.py`. Marks re-rank the leaderboard on KITAB-KHATT (τ 0.886), Muharaf (0.882) and RASAM-2 (0.867). **Rule: no_diacritics is the headline rung everywhere.** Muharaf has marks on 35.8 % of test lines (the sparse-marks note, confirmed). Grade A−.
- [x] **2.9 Leakage and duplicates** (MinHash plus image hash; the 6 % Muharaf test-in-train lines). → [data-and-label-audit §2](designs/data-and-label-audit.md) · map: `R4.2`
  - *2026-10-17:* `audit_leakage.py` (image hash recovers all 89 KITAB/KHATT pairs). Share of each set leaked: Muharaf 7.5 % (formulaic short lines), NAKBA test 18.3 %, NAKBA blind 16.3 %, KITAB 44.5 %, full KHATT 95 %, RASAM ≤ 1.1 %. Memorisation survives FDR **only on NAKBA blind** (NAKBA-trained systems gain 20.1 points more on leaked lines), so **its headline goes on clean lines**. Duplicate images: 16 in NAKBA blind and 7 in RASAM-2, all to R4.1. Grade A. Still to do: join `leakage_flags` into store_build.
- [ ] **2.10 Spelling inconsistency (R4.3) and the conventions table (R4.4).** → [data-and-label-audit §3](designs/data-and-label-audit.md) · map: `R4.3`, `R4.4`
  - *2026-10-17:* `audit_spelling.py`. Minority-variant tokens: NAKBA blind 5.6 %, Muharaf 2.3 %, NAKBA test 2.1 %, KHATT 1.7 %, RASAM 0.7–0.8 %. **The orthographic rung is required in the headline everywhere except RASAM; report the NAKBA test/blind and RASAM-1/2 splits separately, never pooled.** Still to do: the documented-conventions column from the dataset papers.
- [ ] **2.11 Profiler outputs and datasheets** (R1.3 diacritic policy share; R1.4 datasheet cards). → [data-and-label-audit §5](designs/data-and-label-audit.md) · map: `R1.3-R1.4`
  - *2026-10-17:* `audit_datasheets.py`: 7 datasheet cards (`results/audit/datasheets.html`). Still to do: put them on the Atlas.
- [x] **2.12 IRT difficulty and discrimination** (R3.2); **challenge subsets** as dataset filters (R3.3); **length-stratified reporting** (X5-len). → [data-and-label-audit §6](designs/data-and-label-audit.md) · map: `R3.2`, `R3.3`, `X5-len`
  - *2026-10-17:* `audit_difficulty.py`. IRT difficulty beats median CER only on NAKBA test, so keep median CER elsewhere. 12 challenge subsets registered (loose crops cost +58 points on KITAB, +49 on NAKBA test). Fragments are worse after FDR on Muharaf, NAKBA test and NAKBA blind, so they get a separate leaderboard subset there. 627 lines with negative discrimination go to R4.1. Still to do: a `subset_keys` loader and fragment columns in `make_leaderboard`.
- [x] **2.13 Similarity-choice model** of confusions (Müller-style; human vs model confusability). → [dots-and-skeleton](designs/dots-and-skeleton-studies.md) · map: `P-L4`
  - *2026-10-18:* Confusability is shared across families (Spearman ρ 0.63–0.90, all lower bounds above 0.4), so it is **a script property, grade B**. Dot-only pairs are 3–18× more confusable. Response bias follows letter frequency. The human-vs-model comparison is still open (no human confusion matrix).
- [ ] **2.14 Tokenizer view** (dot-pair token sequences, fertility vs CER). → [decoding-and-restoration §6](designs/decoding-and-restoration.md) · map: `E2`
- [ ] **2.15 Root-sharing of real-word substitutions** (CAMeL Tools). → [decoding-and-restoration §6](designs/decoding-and-restoration.md) · map: `perea-root`
- [x] **2.16 KHATT fixed-text natural experiment** (same text, several writers). *after:* 1.6 → [writer-adaptation D](designs/writer-adaptation.md) · map: `P1.8`
  - *2026-10-18:* `writer_khatt_fixed_text.py` on the full-KHATT readings (20 runs). Generic CRNNs: **text explains more of the word errors than writer** (crnn/base partial R² 0.152 [0.144, 0.169] vs 0.088 [0.076, 0.102]; all 18 non-KHATT runs agree). **The in-domain KHATT CRNN flips it** (writer 0.26 vs text 0.03; CER 0.092; but 713 of its test texts are in train). Only ink density survives FDR for all readers (denser ink, fewer errors). The full-KHATT ink-cropped rows equal the stored ones (binarised tight crops), so they add nothing.
- [x] **2.17 Writer and page IDs** (R0.4): recover what exists; otherwise document page clusters. → [writer-adaptation D](designs/writer-adaptation.md) · map: `R0.4`
  - *2026-10-18:* No writer or page IDs survive in the Teklia copy (the original file names need a licence request). **Pseudo-pages reconstructed** by aligning each line to the 77-word paragraph and linking by handwriting style (`writer_khatt_pages.py`): test 263 pages (88 complete). Validated: the page ICC of residual error is 0.11 [0.06, 0.17] against a null of about 0, so the pages are **usable as clusters for intervals**.
- [ ] **2.18 Geometry literature follow-ups:** second baseline estimator, AHTID/MW check, polygon-curvature descriptor, codebook paper, FATF curves, dots validated on spans. → [dots-and-skeleton §11](designs/dots-and-skeleton-studies.md) · map: `geometry-lit`
  - *2026-10-18:* First slice: the Abu-Ain-style sub-word baseline beats the band estimator (+0.026 [0.023, 0.030]), so **adopt it**. Dots-on-spans fails validation (hit rate 0.34–0.77). Still to do: reading AHTID/MW, Abdi, Ahmed and Parvez.
- [ ] **2.19 Detector residual review:** the worst lines no detector explains; write the next detector. Repeat after every phase that adds results. → [data-and-label-audit §9](designs/data-and-label-audit.md) · map: `detector-residual`
- [ ] **2.20 Ledger and map update** with phase-2 results (panels, grades, checklist ticks).

## Phase 3 — Human review (any time; see 0.3) · map: `R4.1`

## Phase 4 — GPU night 1: CRNN runs (about 8 h; one queue, detached, with a live-view command)
- [ ] **4.1 CTC posterior dump:** base, combo and standard-split CRNNs on the panel plus the full KHATT test (15–30 min). *after:* 1.6 → [decoding-and-restoration §0](designs/decoding-and-restoration.md) · map: `lm-decoding`, `P-L2`, `P-L1`
- [ ] **4.2 Skeleton-only CRNN** (standard recipe, dotless labels; 1 seed, about 1 h). → [decoding-and-restoration §4](designs/decoding-and-restoration.md) · map: `two-stage-reader`
- [ ] **4.3 CRNN writer adaptation** on the RASAM-2 manuscripts and NAKBA: K × {random, letter coverage, novelty} × {full fine-tune, top layers} (about 2 h). → [writer-adaptation A](designs/writer-adaptation.md) · map: `writer-adapt-crnn`
- [ ] **4.4 Few-shot Maghrebi curve** (RASAM-1 manuscripts, K = 16 / 100 / 300; ق→ف tracked; about 1.5 h). → [writer-adaptation A](designs/writer-adaptation.md) · map: `E6`
- [ ] **4.5 In-domain NAKBA CRNN** (1 seed now, a second later; about 1 h each). → [training-and-transfer](designs/training-and-transfer.md) · map: `nakba-indomain`
- [ ] **4.6 Full KHATT:** all existing CRNNs on the KHATT v1.0 test, plus an in-domain KHATT CRNN on its train (about 1.5 h). *after:* 1.6 → [training-and-transfer](designs/training-and-transfer.md) · map: `khatt-full`
- [ ] **4.7 Hard-example control** (repeat the highest-loss lines; 2 seeds). → [shape-programme, related](designs/shape-programme.md) · map: `hard-example-control`
- [ ] **4.8 Shape targeting for NAKBA:** selection on blind half A; targeted and random, 2 seeds each. → [shape-programme, related](designs/shape-programme.md) · map: `shape-nakba`
  - *2026-10-17:* **moved to night 2.** Night 1 part B (`queue_n1b.bat`, ~8.2 h) runs 4.1, 4.2, 4.5, 4.3, 4.4 and 4.7, and would have run to ~13 h with 4.8 in it.
- [ ] **4.9 Back up the new weights to the SSD; trim regenerable files on the 3090.** → [evaluation-standards](designs/evaluation-standards.md) · map: `box-cleanup`

## Phase 5 — Analysis of night 1 (CPU)
- [ ] **5.1 Language-model decoding** (Muharaf, NAKBA, modern text; lexicon-constrained). *after:* 4.1, 1.7, 1.8 → [decoding-and-restoration §1](designs/decoding-and-restoration.md) · map: `lm-decoding`
- [ ] **5.2 PAW-aware spacing** (rule and decoder versions). *after:* 4.1 → [decoding-and-restoration §2](designs/decoding-and-restoration.md) · map: `P-L2`
- [ ] **5.3 Dot-restoration ceiling and the two-stage reader result** (known vs unseen words). *after:* 4.1, 4.2 → [decoding-and-restoration §3–4](designs/decoding-and-restoration.md) · map: `P-L1`, `two-stage-reader`
- [ ] **5.4 Writer-adaptation curves**, including the no-training prototype memory (method iii, CPU). *after:* 4.3 → [writer-adaptation A](designs/writer-adaptation.md) · map: `writer-adapt-crnn`
- [ ] **5.5 Maghrebi curve; in-domain diagnostics** (crop, novelty, PAW splits, hamza on the in-domain model); full-KHATT re-check of every KHATT claim; hard-example and NAKBA-targeting results. *after:* 4.4–4.8 · map: `E6`, `nakba-indomain`, `khatt-full`, `hard-example-control`, `shape-nakba`
- [ ] **5.6 TTA selector study** (agreement across shears vs confidence, on the stored all-shear readings) and the decision whether gated TTA joins the pipeline. → [dots-and-skeleton](designs/dots-and-skeleton-studies.md) · map: `tta-selector`
- [ ] **5.7 Ledger, map and claims-register update.**

## Phase 6 — The allograph library (CPU, a few days)
- [ ] **6.1 Extraction pipeline** (alignment + ink skeleton, cut at baseline joins, dots attached, dot style classified, join points recorded). *after:* 2.2, 2.4 → [shape-programme step 2](designs/shape-programme.md) · map: `shape-library`
- [ ] **6.2 Quality gate** (cycle consistency; ≥ 30 % of letters pass; ≥ 20 glyphs per letter × form from ≥ 2 sources). *done when:* the gate passes or the failure is diagnosed.
- [ ] **6.3 Atlas page in the ledger** (shapes per letter × form per dataset; the missing ones).

## Phase 7 — GPU night 2: VLMs (about 6–8 h)
- [ ] **7.1 VLM setup on the 3090:** download the base Qwen3-VL-8B (and 4B) from the Hub on the box; copy the fine-tuned adapters; multi-image prompt support. *after:* 0.2b → [writer-adaptation B](designs/writer-adaptation.md) · map: `vlm-logprobs`
- [ ] **7.2 In-context writer calibration** (reference line plus label, then the test line; controls: other writer, label only). *after:* 2.7 → [writer-adaptation B](designs/writer-adaptation.md) · map: `vlm-incontext-writer`
- [ ] **7.3 VLM geometry check** (cropped and slant-sheared KHATT). → [dots-and-skeleton §9](designs/dots-and-skeleton-studies.md) · map: `vlm-geometry`
- [ ] **7.4 Image-reliance check** (log-probability with and without the image). → [attribution-track](designs/attribution-track.md) · map: `X-A3`
- [ ] **7.5 E1.1: bench the diff-LR arm** on the held-out sets. → [training-and-transfer](designs/training-and-transfer.md) · map: `E1`
- [ ] **7.6 Resolution probe, VLM part** (KHATT at 60 px and three noise levels). → [dots-and-skeleton §8](designs/dots-and-skeleton-studies.md) · map: `R9`
- [ ] **7.7 Back up and trim.** · map: `box-cleanup`

## Phase 8 — Cross-family combination (CPU)
- [ ] **8.1 Combiners:** per-line confidence choice, ROVER across families, constrained VLM correction of the CRNN. *after:* 7.1 log-probabilities → [decoding-and-restoration §5](designs/decoding-and-restoration.md) · map: `cross-family-ensemble`
- [ ] **8.2 Ledger, map and register update.**

## Phase 9 — Synthesis (CPU to build, GPU to test)
- [ ] **9.1 Composer v1** (glyph joining at connection points, re-dotting, measured gaps, kashida, geometry and background variation; text from training texts, lexicon and coverage-balanced text). *after:* 6.2, 1.7 → [shape-programme step 3](designs/shape-programme.md) · map: `synthesis-v1`, `X-G11`, `aug-9`, `aug-13`, `aug-19`
- [ ] **9.2 Realism checks** (real-vs-synthetic classifier ≤ 75 %, geometry within real ranges, CER in the real range).
- [ ] **9.3 Training experiment** (K × sampling policy, 3 seeds; controls: real repeats, hard-example, StackMix-style cut, font renders). → map: `synthesis-v1`, `aug-5-7-8-9-12-13`
- [ ] **9.4 Augmentations through the composer:** 10 rare-character oversampling, 14 PAW-gap jitter, 16 dot minimal pairs, 17 Maghrebi dot swap, 18 harakat on/off, 20 fragment crops, 21 style mixing, pen contrast. → [dots-and-skeleton §10](designs/dots-and-skeleton-studies.md), [data-and-label-audit §10](designs/data-and-label-audit.md) · map: `aug-10`, `aug-14`, `aug-16`, `aug-17`, `aug-18`, `aug-20`, `aug-21`, `pen-contrast-aug`
- [ ] **9.5 Augmentation ablation A1** (13, 14, 16, 17 against the 1–4 baseline). · map: `A1`
- [ ] **9.6 Render, then refine** (only if seams are visible or exploited). → [shape-programme step 4](designs/shape-programme.md) · map: `render-refine`
- [ ] **9.7 Personalised synthesis** (a writer's own glyphs from K lines). *after:* 9.3, 5.4 → [shape-programme step 6](designs/shape-programme.md) · map: `personalised-synthesis`

## Phase 10 — Shape-token recognition (GPU + CPU)
- [ ] **10.1 Vocabularies:** dotless class × form, discovered clusters, shape BPE (vs letters). *after:* 2.2, 6.2, 2.6 → [shape-programme step 5](designs/shape-programme.md) · map: `shape-tokens`
- [ ] **10.2 Constrained decoders** (lexicon / n-gram WFST → transducer → constrained small LLM; joining grammar; top-k beams). *after:* 5.1
- [ ] **10.3 Factorised head** (skeleton CTC + dot head; also curriculum C3). → [dots-and-skeleton §6](designs/dots-and-skeleton-studies.md) · map: `X-G9`

## Phase 11 — Normalisation, resolution, architecture (GPU)
- [ ] **11.1 Normalisation ablation** (height 48/64/96 × pad vs resize, deslant, binarisation; 2 seeds). → [dots-and-skeleton §8](designs/dots-and-skeleton-studies.md) · map: `X-G3`
- [ ] **11.2 Retsinas-recipe CRNN** as a second architecture (2 seeds) and the key diagnostics on it. · map: `E5-retsinas`
- [ ] **11.3 Dot-preserving resize and dot channel at 32–48 px.** · map: `X-G7`
- [ ] **11.4 Resolution probe, CRNN part.** · map: `R9`
- [ ] **11.5 Geometry long tail:** pretext pretraining (X-G12), geometry-matched augmentation, slant-density selection; learned normalisation only if 11.1 shows normalisation helps. → [dots-and-skeleton §11](designs/dots-and-skeleton-studies.md) · map: `X-G12`, `geom-matched-aug`, `slant-density`, `pix2pix`

## Phase 12 — Training and transfer (GPU; larger runs)
- [ ] **12.1 E1.2 / E1.3** (train arm b; compare arms). *after:* 7.5 gains ≥ 3 points on 2 of 4 sets → [training-and-transfer](designs/training-and-transfer.md) · map: `E1`
- [ ] **12.2 DoRA / rsLoRA arm** (after E1 says encoder or decoder). · map: `ketaba-dora`
- [ ] **12.3 Multilingual pre-training with a freeze sweep, and cross-dataset mixing** (needs downloads). · map: `E7`, `aug-11`
- [ ] **12.4 Data scaling in characters** (E3 / E3′). · map: `E3`
- [ ] **12.5 VLM training tricks:** checkpoint averaging (P-L5), per-sample loss logging (P-L6), Baseer replay (E4). · map: `P-L5`, `P-L6`, `E4`
- [ ] **12.6 Retrain without CER-HV-flagged lines** (X3); check for full-resolution Muharaf (X7); output-token caps and re-score (X6, after 0.2b). → [data-and-label-audit §7](designs/data-and-label-audit.md) · map: `X3-retrain`, `X7`, `X6`
- [ ] **12.7 Printed Arabic set and page track.** · map: `X5-print`
- [ ] **12.8 Second seed of the in-domain NAKBA CRNN.** · map: `nakba-indomain`

## Phase 13 — Attribution and localisation (parked behind characterisation; order inside the track)
- [ ] **13.1 Maps on dots (X-A6), other models including the CRNN (X-A7), letter-family sample (X-A2b).** → [attribution-track](designs/attribution-track.md) · map: `X-A6`, `X-A4-A5-A7`, `X-A2b`
- [ ] **13.2 Per-module gradient probe (X-D0)**, then **freeze-by-letter (X-D1)** if family-specific. · map: `X-D0`, `X-D1`
- [ ] **13.3 Chefer relevance maps and map sensitivity (X-A4, X-A5).** · map: `X-A4-A5-A7`
- [ ] **13.4 LocateAnything fine-tune (X-C2)**, gated on span quality and a baseline. · map: `X-C2`
- [ ] **13.5 Grounding literature** (Q-Mask, PositionOCR, InstructOCR, segmentation-free spotting, VML-HD licence and X-B1 validation, character segmentation). · map: `grounding-lit`

## Phase 14 — Generalisation, gates, write-ups
- [ ] **14.1 Cross-script check (gate V4):** Persian PHTD, Urdu NUST-UHWR, Pashto, and IAM as the Latin analogue. → [training-and-transfer](designs/training-and-transfer.md) · map: `R10`
- [ ] **14.2 Decision experiments (gate V5)** consolidated: coverage, novelty and selection results vs random; transformation choice. · map: `R8`
- [ ] **14.3 Validation gates V3–V6** per characteristic; image layer second pass (R6); release package and datasheets (R11). → [evaluation-standards](designs/evaluation-standards.md) · map: `gates`, `R6-R11`
- [ ] **14.4 Write-ups per the claims register** (audit; characterisation across families; methods). · map: `paper-groupings`

## Phase 15 — Own dataset, literature, parked tracks
- [ ] **15.1 Own dataset (S0 partner and ethics, S1 content design with the calibration sentence and coverage text, S2 sampling, S3 annotation protocol, S4 pilot)** and the script-styles inventory. → [`reference/dataset-design.md`](reference/dataset-design.md), [`reference/NEXT_STEPS.md`](reference/NEXT_STEPS.md) §S · map: `S-track`, `S1-styles`, `srihari`
- [ ] **15.2 Literature tasks L1–L7** (diacritics, annotation schema, script types, architectures, dataset taxonomy, other languages, verifying unverified references). → [`reference/NEXT_STEPS.md`](reference/NEXT_STEPS.md) §L · map: `L-track`
- [ ] **15.3 Language layer (R5):** register, dialect, entropy, numbers and names (partly done through OOV and the lexicon). · map: `R5`
- [ ] **15.4 Curriculum track C0–C4** (revisit after 9.3). → [training-and-transfer](designs/training-and-transfer.md) · map: `C-track`

## Parked long shots (kept on the list, not scheduled)
- [ ] **P.1 Pen-trajectory recovery** with sigma-lognormal primitives. · map: `pen-trajectory`
- [ ] **P.2 Bigram study** (no test planned). · map: `zeghli`
- [ ] **P.3 Separate dots head in the RAN style** (subsumed by 10.3 unless that fails). · map: `zhang2020ran`
- [ ] **P.4 AIN-7B contamination:** unverified; handled by detector D23 (1.5) only. · map: `C6`

## Dropped (recorded so they are not re-proposed)
- **D.1 Dot dropout augmentation:** breaks image–label agreement. · map: `aug-15`
- **D.2 Pixel span masking:** masked letters stay in the label. · map: `aug-6`
- **D.3 Pointing game as a training signal:** wrong tokens already look at the right place. · map: `pointing-loss`

## Continuous (after every phase)
- Ledger evidence panel and gallery; research-map item status and grade; this checklist ticked; corrections log
  (`CORRECTIONS.md`) for anything that changes an earlier number; claims register; SSD backup and 3090 trim.
