# Plan 2026-10-13: what the results support, how every experiment gets analysed, and what to run next

> Written 2026-10-13 on branch `worktree-xai-attribution-spike`. This is a plan; no GPU job was run and no file other than
> this one was changed. Paths: `W/` = this worktree, `M/` = the main checkout `/Volumes/PortableSSD/repos/arabic_ocr`.
> Every number marked *(recomputed)* was recomputed today on the laptop from the per-line files named next to it. Unmarked
> numbers are quoted from the ledger (`W/docs/research/paper-ledger.html`, v19), `M/docs/experiments/findings-and-notes.md`
> or the result JSONs.

## 0. Summary

1. **These results would survive a strict reviewer today: the benchmark-validity facts.**
   - `kitab_muharaf` is 86.5 % inside Muharaf train.
   - KITAB-KHATT ships loose crops. 35 of its 200 lines have writing in under 35 % of the image height. Fixed-height
     recognisers lose 3 points or more on them, and the CRNN's own evaluation quietly drops the 17 worst.
   - There are token-cap truncation artefacts.
   - There is a diacritic-policy tax.
   - NAKBA has a hamza annotation convention.
   - **AIN-7B shows a contamination signature on KITAB-KHATT (new today).** Its CER is 6.4 %, against ≥ 20.7 % for every
     non-AIN model, and it copies the ground truth's whitespace quirks on 67 of 96 lines.
   - Together these make one solid benchmark-audit paper.
2. **The ink-band crop is the strongest intervention, but it is a pipeline/benchmark artefact, not a script
   characteristic.**
   - KHATT −3.1 points, CI [−5.0, −1.5].
   - Replicated in 7 separately trained CRNNs (−2.7 to −3.3).
   - At the line level the effect is concentrated: 18 lines improve beyond seed noise and 1 gets worse.
3. **The best Arabic-specific findings each need one or two named additions:**
   - spaces inserted at PAW gaps: 60 % of wrong word-internal spaces, against 35 % by chance;
   - the Maghrebi qāf convention: base AIN reads ق as ف on 23–27 % of RASAM qāfs, against 2–3 % elsewhere;
   - slant as a small causal factor for the CRNN: −1.1 and +2.0 points.
4. **Letter-shape novelty is the most original idea and currently the weakest evidence.**
   - Three confounds are not ruled out: alignment by the same model, loose-crop size, and line-level difficulty.
   - The targeted-data intervention gives −0.76 [−1.57, +0.09] on 91 lines, with one seed and halves that share text.
   - At the line level it cannot be told apart from a plain reseed: 8 lines better and 3 worse beyond the noise band,
     against 10 and 8 for a reseed.
5. **Several ledger numbers mix different line sets or normalisations. Fix these first (§1.5):**
   - two different 600-line "NAKBA blind" samples;
   - CRNN versus VLM alef folding;
   - X-T2's Muharaf lines, about 80 % of which are in the CRNN's training data;
   - a TTA gallery that shows reversed readings.
6. **The analysis framework (§3) runs on every experiment from now on.**
   - One per-line store holds every CRNN JSON, the TTA and E8 outputs and the 289 canonical VLM pairs, all scored on the
     R2.1 ladder.
   - **A** (sample level): accuracy buckets, transition matrices, a line × model matrix, and flips calibrated against
     seed noise.
   - **B** (letter, PAW and word level): letter × positional form, confusions, PAWs, words, and the join with letter
     novelty.
   - 24 pattern detectors with an "explained edits" table and a gallery of unexplained lines.
   - KHATT is the set where every CRNN and 18 VLM runs read the same lines: 15 runs read all 183, and 3 read the subset
     inside rows 20–199. The ground-truth join is exact.
7. **Next steps, in order (§5):**
   - **P0:** the store, the fixes and a harmonised CRNN evaluation (about 1.5 CPU days and 2–3 GPU hours).
   - **P1:** CPU analyses that test each CRNN claim on other model families.
   - **P2:** seeds and a factorial design for the targeted data, a standard-split CRNN (E5), and a VLM geometry check.
   - Everything else is gated behind these.

---

## 1. Outcome inventory

### 1.1 Grading rubric

**Grades:**
- **A**: publication-grade as stated (only write-up needed).
- **B**: real and likely to hold, but needs the 1–2 named additions before submission.
- **C**: suggestive; noise or a confound is not excluded, so a redesigned or replicated test is needed.
- **D**: descriptive, negative or diagnostic; supporting material only.

**Criteria, applied to every row:**
- effect size and CI;
- replication: seeds, a second model, an independent check;
- generality: across datasets and across model families;
- mechanism clarity;
- novelty against the ledger papers and `references.bib`;
- evaluation size.

**Two weaknesses common to the CRNN rows:**
- It is one architecture, with single unseeded runs.
- It is one setting: zero-shot transfer from Muharaf + RASAM to KHATT and NAKBA.

### 1.2 CRNN, geometry and attribution track (2026-10-10 to 10-13)

| # | Finding | Evidence (n, effect, CI) | Replication / generality | Mechanism / novelty | Grade | What makes it publication-grade |
|---|---|---|---|---|---|---|
| T1 | **Ink-band crop before resizing** (X-S1) | <ul><li>KHATT 26.3 → 23.2 %, −3.1 [−5.0, −1.5] (183 lines); NAKBA test and blind ±0.0 (600 / 600)</li><li>18 / 183 lines have ink in < 35 % of the height and average 85 % CER (`W/results/geometry/ink_extent.json`)</li><li>Bucket view *(recomputed)*: lines at 50–100 % CER 10.9 → 2.2 %, empty readings 9 → 0, readings under half the GT length 13 → 0</li><li>Beyond the seed-noise band *(recomputed)*: 18 lines better, 1 worse</li></ul> | <ul><li>7 CRNN trainings, −2.7 to −3.3: base, replicate, random control, shape, combo, augmented, crop-trained (all `*_inkcrop.json`, *recomputed*)</li><li>Training on cropped lines adds nothing (23.5 vs 23.2 %)</li><li>Fixed-height models only. Per model *(recomputed)*, VLM loose − tight is −21 to +4.5 points on about 35 loose lines; PaddleOCR 98.5 vs 84.6 %</li></ul> | <ul><li>Trivially clear: writing shrinks to a few pixels at 64 px</li><li>Novelty low as a method (cropping to the text box is standard)</li><li>New as a measured defect of a public benchmark</li></ul> | **A** (as a benchmark or pipeline finding) | <ul><li>Score all 200 lines: the 17 lines the CTC-length filter drops all have ink extent 0.06–0.18 (§1.5 #2)</li><li>Run PaddleOCR on cropped lines (a second fixed-height family)</li><li>Report crop looseness for every benchmark set</li><li>Frame it as an artefact, not as an Arabic property</li></ul> |
| T2 | **Slant**: datasets differ; a causal but small factor (X-G2, X-T2, X-G8, X-T3) | <ul><li>Median slant: Muharaf train +11.2°, KHATT −1.8°, NAKBA +3.9 / +5.5°; 41 % of KHATT lines sit where < 5 % of training lines lie</li><li>Slant-only shear of KHATT toward Muharaf: −1.1 [−1.7, −0.4]</li><li>Muharaf sheared upright: +2.0 [+1.6, +2.5] (300 lines)</li><li>Confidence-selected TTA: KHATT −0.9, but NAKBA test +0.45 worse</li><li>Slant augmentation: sensitivity +0.6 instead of +2.0, yet KHATT +0.5 and NAKBA +0.6 worse</li></ul> | <ul><li>Same direction in the augmented model (−1.3) and the shape model (−0.9, `night/shape_transfer.json`); one seed each</li><li>CRNN only</li><li>About 239 of the 300 Muharaf transfer lines are CRNN training lines (§1.5 #3)</li></ul> | <ul><li>Clear sensitivity; about 1 of KHATT's 20-point gap</li><li>Deslanting is classical (`vinciarelli2001norm`); no Arabic study of a slant shift between datasets with a causal shear test was found</li></ul> | **B−** | <ul><li>Redo the Muharaf arm on its 61 held-out lines</li><li>Shear test on 2–3 VLMs</li><li>Second seed of the augmented model</li><li>Slant-gated TTA (CPU simulation first, §5)</li></ul> |
| T3 | **Stroke width** (NAKBA thinning) | <ul><li>−1.1 [−1.7, −0.6] (300 lines)</li><li>KHATT and Muharaf ±0.1–0.3</li></ul> | Augmented model −1.2; CRNN only; one dataset | The "stroke" step also includes blur, so the cause is not isolated | **C+** | <ul><li>Stroke normalisation as a test-time step on all sets</li><li>Separate stroke from blur</li><li>VLM check</li></ul> |
| T4 | **Tone, paper noise, ruled lines and slope do nothing** | <ul><li>KHATT tone / noise +0.1 [−0.1, +0.4]</li><li>NAKBA ruled lines painted out +0.1 [−0.0, +0.1]</li><li>Median slope within 0.5° of zero</li></ul> | Tight intervals for this model; CRNN only (it normalises grey levels per line) | Expected for this preprocessing | **B** (as a null) | Use as negative controls in the geometry paper; state that the scope is this model |
| T5 | **Geometry descriptors** (X-G1 to X-G7: pen contrast, dot units, loci, hinge) | <ul><li>7 sets of 200–400 lines</li><li>Pen contrast: RASAM 0.49 / 0.63 vs about 1.0 elsewhere</li><li>Ascender ≈ 6 dots</li><li>Loci and hinge histograms identify a line's dataset 70–73 % of the time</li><li>Scalar profile: KHATT 0.75 SD vs NAKBA 0.30–0.35 (noise floor 0.12)</li></ul> | <ul><li>Descriptive</li><li>The two slant estimators correlate only 0.4–0.7</li><li>Three transfer sets</li></ul> | Methods borrowed (Glucksman loci, Bulacu hinge); the Arabic measurements are new | **B−** (datasheet) / **D** (explanation) | <ul><li>Validate the estimators</li><li>Join to per-line errors through the store</li><li>Measure all lines, not samples</li></ul> |
| T6 | **Letter-shape novelty predicts letter errors** (X-G10) | <ul><li>Error rises from 1–5 % (most typical fifth) to 30–53 % (beyond Muharaf's 95th percentile) in all 4 sets (9k–31k letters per set)</li><li>Letters beyond the 95th percentile: KHATT 21 %, NAKBA 10–11 %, Muharaf 5 %</li><li>Pixel (model-free) novelty is weaker: Muharaf 3.7 → 10.8 %, KHATT's top two pixel bands 15 → 34 %</li><li>CNN vs pixel novelty: ρ = 0.31 (`letters/summary.json`)</li></ul> | <ul><li>A second model's features give the same pattern</li><li>But `--feat-run` swaps only the features; spans and right/wrong labels still come from the evaluated model (§1.5 #11)</li><li>CRNN only</li></ul> | <ul><li>Plausible mechanism, three confounds open (below)</li><li>The method is a deep k-NN out-of-distribution score (Sun et al., ICML 2022; cited from memory, not checked) per letter × positional form</li><li>No application to Arabic allographs across datasets was found</li></ul> | **C** (B if P1.2 passes) | <ul><li>Run the P1.2 confound battery (§5.2):<ol><li>novelty on ink-cropped lines: the most novel KHATT letters come from loose-crop lines such as `kitab_khatt/000028`</li><li>within-line and within-letter models</li><li>do VLM letter errors rise with CRNN novelty? (cross-family test, CPU)</li><li>spans from a second model</li></ol></li><li>The 24 KHATT "letter-list" lines (isolated letter drills) need a separate stratum</li></ul> |
| T7 | **Shape-targeted training repeats** (X-G11 lite) | <ul><li>Half B, targeted − random: −0.76 [−1.57, +0.09] *(recomputed)*; half A −1.05 [−2.02, −0.09]</li><li>Replicate − base on half B: +0.44 [−0.57, +1.39]</li><li>Cropped half B: −0.46 [−1.26, +0.33]</li><li>NAKBA test −0.21 [−0.57, +0.15], NAKBA blind −0.13 (no effect, so specific)</li><li>Novelty bands 4–5 about 2 points lower than the control</li><li>Line level: 8 better / 3 worse beyond the noise band, against 10 / 8 for a plain reseed</li></ul> | <ul><li>One seed per arm; 91 lines</li><li>The halves are a row-parity split: 44 of 91 half-B lines share a text cluster with half A (§1.5 #10)</li></ul> | <ul><li>Selection uses KHATT's unlabelled distribution, i.e. unsupervised domain adaptation; legitimate, but say so</li><li>No HTR prior found for writer-similarity data selection; the closest is `kohut2023choose` (which pretraining data to choose)</li></ul> | **C** | <ul><li>3 seeds per arm</li><li>A text-cluster-disjoint KHATT split</li><li>A second target (select on NAKBA blind half A, test on half B)</li><li>A hard-example-mining control (§5, P2.1)</li></ul> |
| T8 | **Stacked fixes** (X-S2: crop-trained + targeted repeats, then TTA) | <ul><li>Cropped KHATT: combo 22.3 % vs base 23.2 %, −0.95 [−1.64, −0.22]; vs replicate −1.26 [−1.96, −0.56]; vs random control −1.10 [−1.66, −0.51] *(recomputed)*</li><li>With TTA: 21.5 %; NAKBA 13.2 %</li><li>Line level: 13 better / 12 worse beyond the noise band</li></ul> | <ul><li>One seed</li><li>No factorial control: crop-trained + random repeats was never run</li><li>Crop-training alone is +0.3</li></ul> | Unclear why crop-training would add to the targeted repeats | **C** | 2 × 2 factorial (crop-train × targeted / random) × 3 seeds |
| T9 | **Slant and stroke augmentation in training** (X-T3) | <ul><li>Robust to slant (+0.6 vs +2.0)</li><li>Unseen sets slightly worse: KHATT 26.4 → 26.9, NAKBA 13.8 → 14.4, NAKBA blind 13.7 → 14.6 (sample A)</li></ul> | One seed | A clean null for transfer | **C** (B with a second seed) | Second seed |
| T10 | **E8 / E8b**: CRNN features → Qwen3-0.6B / 1.7B | <ul><li>Worse on every set by 4–14 points (KHATT 40.0 / 40.1 vs 26.4)</li><li>Fluent-but-wrong share of wrong words: 41 / 44 % vs 24 %</li><li>*(recomputed)*: 4.7 % of NAKBA lines exceed 100 % CER (loops) vs 0.8 % for the CRNN; 20 readings longer than 1.5× the GT</li></ul> | One configuration, 3 epochs, uncropped lines | <ul><li>The language prior overrides the image</li><li>Consistent with `greekvlm2026`</li></ul> | **D** (negative) | Keep as validation for the fluent-rewrite detector (§3.5 D15); appendix or workshop note |
| T11 | **NAKBA hamza is an annotation convention** | <ul><li>Labels put a hamza on 15.7 % of alifs vs 3.6 % in Muharaf train</li><li>Of 776 ground-truth hamza-alifs, the CRNN writes bare ا at 634 and matches the label at 52</li><li>Folding the hamza: 13.8 → 11.6 % (600 lines); blind 12.9 → 11.5 % (118 lines) (`hamza.json`)</li></ul> | <ul><li>CRNN only</li><li>The project's CER-light already folds alef forms for VLMs (`M/src/arabic_ocr_bench/normalize.py`, "light")</li></ul> | <ul><li>Supported by images, but not yet counted</li><li>Alef folding is standard; the NAKBA-specific fact is new</li></ul> | **B** | <ul><li>Counted manual audit: 100 GT hamza sites, drawn or not (about 1 h)</li><li>How often NAKBA-trained models emit a hamza, against Muharaf-trained ones (CPU)</li></ul> |
| T12 | **Text distance** (KHATT farthest) | <ul><li>Character 5-gram cross-entropy: KHATT 4.00 vs NAKBA 3.37–3.39 vs Muharaf 2.92 bits per character</li><li>Unseen PAWs: 5.9 % vs 3.6–3.9 %</li><li>Unseen words: 33.8 % vs 31.7–32.2 % (`text_shift.json`, `paw_coverage.json`)</li></ul> | Descriptive; 3 targets | <ul><li>Prior: `garridomunoz2025` (CVPR, Latin only, 336 out-of-distribution cases): textual divergence correlates 0.62 with out-of-distribution CER; visual divergence about −0.3</li><li>Untested for Arabic</li></ul> | **D** now | <ul><li>Per-line, within-dataset regression on text cross-entropy with image covariates (P1.7)</li><li>A text-only intervention: CTC decoding with a character n-gram LM (P2.8)</li></ul> |
| T13 | **Dots at low resolution** (X-G7 CPU proxy) | <ul><li>At 32 px: 19–64 % of dots fall under 2 px</li><li>At 64 px: 0–5 %</li></ul> | No model test | | **D** | Model test at 32 / 48 px, if ever |
| T14 | **CRNN generalises better than our VLM LoRA** (X-E1) | <ul><li>CRNN: 13.8 / 13.7 / 26.4 %</li><li>Qwen3-VL-4B LoRA: 29 / 30 %</li></ul> | Not like-for-like: <ul><li>3× the training data, including most Muharaf test lines</li><li>No alef fold vs CER-light</li><li>183 vs 200 lines</li></ul> | | **C** | Same lines, same ladder rung, standard split (E5) |
| T15 | **Attribution maps** (X-A1, X-B2) | <ul><li>Deletion: 4.3–5.1 nats vs 0.8 for random patches</li><li>Gradient map: 9× more mass on ink than chance</li><li>Pointing game about 3× chance (296 tokens, 43 lines)</li><li>Wrong tokens look in the right place as often as correct ones</li></ul> | One VLM; spans are in-sample | Diagnostic | **D** | Only with X-A3 (image reliance) and more lines |
| T16 | **LocateAnything-3B zero-shot** | Whole-image boxes on 42 / 42 lines | | | **D** (negative) | |
| T17 | **خ weakness** | 7 of 10 tokens wrong | n = 10 | | **D** | X-A2b, if ever |

### 1.3 VLM benchmark and characterisation track (2026-09 to 10-09)

| # | Finding | Evidence | Grade | What it needs |
|---|---|---|---|---|
| V1 | `kitab_muharaf` leakage | 86.5 % of lines (≥ 20 characters) appear word for word in Muharaf train; every other set ≤ 6 % | **A** | Already masked; write it up |
| V2 | Token-cap truncation artefacts | <ul><li>`gemini-flash`: 74 % of outputs stop at 252 tokens</li><li>`dots3`: 100 % at 256</li><li>`glm-5.3`: 39 % at its cap</li></ul> | **A** | X6 re-score; flag the rows |
| V3 | Diacritic-policy tax | <ul><li>Base Qwen3-VL-4B inserts 10.5 marks per 100 letters on PATS</li><li>CER 18.8 % raw vs 7.1 % with marks folded</li><li>Claude Opus pays 8 points on HistoryAr</li></ul> | **B+** | R2.7: rank correlation with and without marks, and a scoring rule |
| V4 | The dot tax is small | Median 1.8 points, or 8 % of diacritic-insensitive error, over 91 competent pairs (IQR 7–9 %) | **B** | CIs per family; include the CRNN family |
| V5 | **Spaces inserted at PAW gaps** | <ul><li>60 % of 10,655 wrongly inserted word-internal spaces fall at a PAW boundary</li><li>Chance rate 35 % (31–38 % across datasets)</li></ul> | **B+** | <ul><li>Bootstrap CIs per model and dataset</li><li>Add the CRNN family</li><li>Image check: are these gaps physically wider?</li><li>P-L2 repair as the V5 "useful" test</li></ul> |
| V6 | VLMs make as many similar-letter confusions per character as a CRNN | <ul><li>1.7–3.6 vs 1.2–1.5 per 100 characters on the same Muharaf lines</li><li>The metric reproduces the paper's substitution rate exactly (4.17 %)</li></ul> | **B** | CIs; the definition depends on the groups, so report the data-derived groups too |
| V7 | 65 % of letter substitutions fall in no hand-written group | 185,902 substitutions | **B−** | Data-derived, position-aware groups (R2.6 follow-up) |
| V8 | **Maghrebi qāf read as fā** | <ul><li>Base AIN-7B: 26.6 % [20.9, 32.1] on RASAM-1, 22.6 % [16.7, 28.9] on RASAM-2, vs 2.3 / 3.4 % on KHATT / NAKBA</li><li>Fine-tunes: 4–10 %; Qwen fine-tunes 9.8–10.4 % on RASAM vs 4.9–6.4 % elsewhere</li><li>(`M/results/characterize/20261009T111328Z/substitutions.json`)</li></ul> | **B** | <ul><li>More competent base models</li><li>The CRNN, which trained with RASAM, as a contrast</li><li>The E6 intervention</li></ul> |
| V9 | Line difficulty is reliable; text explains little | <ul><li>Split-half reliability 0.54–0.81</li><li>R² 0.06–0.32 from text features, 0.15–0.63 with image features added</li></ul> | **B** | R7: leave-one-dataset-out and leave-one-family-out (gate V2) |
| V10 | Medial forms hardest; dotted letters +10–17 points; tall strokes easiest | 10 complete blocks | **B** | CIs; control for letter frequency and line difficulty; the Latin analogue (ascenders easy, minims hard) |
| V11 | Ensemble ceiling | 0.8–3.6 points | **C** | |
| V12 | Muharaf label noise does not change rankings | Dropping 126 flagged lines: −0.4 to +0.9 points, no rank change | **B** | R4.1 label-error rates |
| V13 | Muharaf is stored at 60 px | A fact; its effect on difficulty is untested | **D** | R9 probe |
| V14 | E1: unfreezing the ViT with a differential LR | In-domain −7.6 [−9.1, −6.2], one seed, 200 lines; the gain is in letter identity, not dots | **C+** | E1.1 held-out bench; second seed |
| V15 | **AIN-7B contamination signature on KITAB-KHATT** (new, *recomputed*) | <ul><li>Diacritic-insensitive CER 6.4 % vs ≥ 20.7 % for every non-AIN model, Claude Opus, GPT-5 and GPT-5.6 rows included (those API rows have only 10–30 lines; the best 200-line non-AIN run is the fine-tuned Fanar at 20.9 %)</li><li>Its raw output reproduces GT whitespace quirks (double spaces, a space before ".") on 67 of the 96 lines that have them; the fine-tuned AIN on 65; the AIN-NAKBA LoRAs on 10–14; every non-AIN run on ≤ 26 (PaddleOCR and base Qwen3-VL-8B 26, the rest ≤ 13)</li><li>Raw exact matches: 23 / 200 (fine-tuned AIN 22, AIN-NAKBA LoRAs 13–14) vs ≤ 4 for every non-AIN run</li><li>The AIN paper names no OCR training sets (checked 2026-10-13)</li></ul> | **B** (as a hygiene flag) | <ul><li>A KHATT sample outside KITAB</li><li>A test for copied GT typos</li><li>Exclude the AIN family from the "best VLM" column until then</li></ul> |
| V16 | Baseer-Nakba 1.0 % on NAKBA test | It was trained on NAKBA (in-domain) | — | Never compare it with zero-shot rows |

### 1.4 Method findings (they decide how every later claim is tested)

| # | Finding | Evidence *(recomputed)* | Grade |
|---|---|---|---|
| M1 | **Seed churn is large at the line level** | <ul><li>Reseeding the base recipe changes the edit count of 86–89 % of KHATT lines and 74–78 % of NAKBA lines</li><li>Total per-line churn Σ\|Δ\| is 3.8 CER points on KHATT and 2.9–3.0 on NAKBA, for a net change of −0.07 / +0.31</li><li>(`shape/base.json`, `night/rep2_*.json`, `inkcrop_base.json`)</li></ul> | **A** (method) |
| M2 | **CRNN readings depend on numerics and batch composition** | <ul><li>Same weights and images, fp16 at batch size 1 vs fp32 at batch size 32: 38 % of KHATT and 22 % of NAKBA lines change; churn 0.8 / 0.4 points; net 0</li><li>Batch composition alone changes 5–7 % of KHATT lines across evaluation packs, because the BiLSTM reads the right padding</li></ul> | **A** (method) |
| M3 | **Only the crop is a concentrated line-level effect** | Lines better / worse beyond the 90th-percentile replicate band (per length tercile): <ul><li>crop 18 / 1</li><li>targeted vs random 8 / 3 (a reseed gives 10 / 8)</li><li>combo vs base (cropped) 13 / 12 (a reseed gives 5 / 10)</li></ul> | **A** (method) |
| M4 | Geometry transfer with single-step ablations as a causal diagnostic | Separates slant from tone and stroke width | **B** (needs the VLM side) |
| M5 | CTC forced alignment as a letter locator | Aligned spaces fall in low-ink columns 51 % of the time vs 30 % for random shifts; alef and lam spans are 30 % taller | **C** (the VML-HD check is still outstanding) |

### 1.5 Problems found in the existing results (fix before anything is written up)

1. **Two different "NAKBA blind" samples.**
   - The packs disagree: `eval_data.pkl` (used for the shape, replicate, random and augmented scoring) and
     `inkcrop_eval.pkl` (used for the crop, combo and TTA scoring) share only 118 of their 600 line ids.
   - The base CRNN scores 13.68 % on one and 13.38 % on the other.
   - The ledger quotes both as "NAKBA blind". Example: the augmented CRNN is 14.6 % in the X-T3 table and 14.3 % in the
     ink-crop table.
   - Fix: score the full 2,671-line blind set, or at least the 1,082-line union, for every CRNN.
2. **The CRNN's KHATT set is 183 of the 200 lines, and the 17 dropped lines are the worst loose crops.**
   - The CTC-length filter (`width/4 < len+4` in `xai_prepare_eval_data.py` and `geom_inkcrop_export.py`) removes them;
     their ink extent is 0.06–0.18 *(recomputed)*.
   - So every CRNN KHATT number is on the easier subset, while VLM rows use all 200.
   - Fix: crop first, filter second, report all 200.
3. **X-T2's Muharaf lines are mostly CRNN training lines.**
   - `xai_prepare_aligner_data.py` gives role `train` to every Muharaf *test* row except its 300 held-out lines: the first
     300 eligible rows of 8–60 characters, test indices ≤ 422.
   - `transfer_eval.pkl` takes a random 300 test lines. About 239 of them are in the CRNN's training data *(recomputed;
     panel ids ignored)*.
   - "Muharaf 6.2 → 8.2 %" is therefore mostly an in-sample sensitivity test, and 6.2 % is not a test CER.
   - The CRNN also has no standard-split Muharaf number. For comparison, the Luleå CRNN scores 10.1 % raw on the full
     test set.
4. **The held-out 7.6 % Muharaf CER covers only 8–60-character lines** (300 lines).
5. **TTA per-line readings are stored reversed.**
   - `xai_geom_tta.py` stores hypotheses in visual order (it never reverses them; `geom_transfer_eval.py` does), so
     `night/tta_inkcrop_*.json` hold reversed strings. The edit counts are right.
   - Result: the **TTA gallery in ledger v19** (block `d-day4`, from `galleries/tta_frag.html`) displays reversed
     readings, and its highlighted characters are meaningless.
   - Fix: reverse the strings on ingestion, fix the script, rebuild the gallery.
6. **`vlm_ink_extent.py` groups predictions by folder, not by model.**
   - So each ledger row mixes the base and the fine-tuned model (400 rows = 200 + 200).
   - Per model *(recomputed)*, loose vs tight crops:

     | Model | Loose | Tight |
     |---|---|---|
     | Base AIN | 5.5 % | 7.1 % |
     | Fine-tuned Fanar | 25.5 % | 21.8 % |
     | Fine-tuned Qwen3-VL-8B | 27.2 % | 22.7 % |
     | Base Qwen3-VL-4B | 35.8 % | 57.3 % |

   - The conclusion (VLMs are not hurt the way PaddleOCR is) holds; the per-model numbers in the ledger do not.
7. **CRNN and VLM CERs use different normalisations.**
   - CRNN CER strips harakat but does not fold alef / hamza.
   - VLM CER-light keeps harakat but folds alef forms.
   - So NAKBA's hamza convention (about 2 points) penalises only the CRNN in CRNN-vs-VLM tables.
   - Fix: score both at the same ladder rungs (§3.2).
8. **CRNN readings are not deterministic across evaluation scripts** (M2).
9. **The targeted and random runs see about 8.7k more line copies per epoch** (2,904 lines × 3) than the base, so they
   take more optimisation steps.
   - Only targeted − random is a clean contrast.
   - Example: the shape model's lower Muharaf CER in `shape_transfer.json` (5.2 % vs 6.2 %) may be just the extra steps.
10. **KHATT halves A and B are not text-disjoint.**
    - They are a row-parity split of a 200-line sample whose fixed paragraphs were written by many writers.
    - 78 of 183 lines share ≥ 30 % of their word trigrams with another line, 35 lines come from the fixed pangram
      paragraphs, and 24 are letter-list drills.
    - 44 of 91 half-B lines share a text cluster with half A *(recomputed)*.
    - KITAB's KHATT copy has no writer ids (`source` = "khatt" for all 200 rows).
11. **The novelty "independent check" swaps only the CNN features.** Spans and right/wrong labels still come from the
    evaluated model's forced alignment and greedy reading (`xai_letter_shapes.py --feat-run`).
12. **Training is unseeded** (`xai_train_aligner.py` sets no `random` or `torch` seed), so a "second seed" is really an
    uncontrolled rerun.
13. TTA denominators drop out-of-charset characters (`len(tgt)`), while the batched evaluator keeps them (`len(label)`).
    The difference is negligible but inconsistent.
14. The E8 1.7B gallery contains readings with newlines (a repeated «العنوان:١…١٩»), which break the ledger HTML
    (lines 770–788 of `paper-ledger.html` are fragments). `change_gallery.marked` should escape newlines.
15. AIN-7B on KITAB-KHATT (V15).
16. **The queue item "PAW-inventory coverage curve" is already answered** by `W/results/geometry/paw_coverage.json`.
    Muharaf train has 20,589 distinct PAWs; 4,846 cover 95 % and 16,029 cover 99 %. The rasm-PAW figures are 12,395,
    1,863 and 7,835. It only needs adding to the Atlas.
17. **The aligner checkpoints once per epoch** (about 2 min). The standing rule allows losing only "a few samples", so
    add step-level checkpoints before the seed sweep.
18. Housekeeping: `M/docs/research/NEXT_STEPS.md` still lacks the XAI block (an open item in `HANDOFF-2026-10-12.md`).

---

## 2. What is really strong, and the papers it makes

### 2.1 Shortlist

| Rank | Claim | Grade | Why a reviewer would accept it |
|---|---|---|---|
| 1 | **Arabic HTR benchmarks carry measurable validity defects:** <ul><li>leakage (V1)</li><li>contamination (V15)</li><li>loose crops plus a silent length filter (T1, §1.5 #2)</li><li>output caps (V2)</li><li>diacritic policy (V3)</li><li>hamza convention (T11)</li><li>label noise that does not change rankings (V12)</li></ul> | A / B+ | Each claim can be verified directly from the data; effects are large (2–10 points) and appear in many models |
| 2 | **Spaces at PAW gaps** (V5) | B+ | Arabic-specific; n = 10,655 spaces; a clear chance baseline; many models |
| 3 | **Script style changes what a dot means** (V8) | B | Large and asymmetric (ق→ف but not ف→ق); clear mechanism |
| 4 | **Slant is a causal but small factor, with clean nulls for tone, noise and ruled lines** (T2, T4) | B− | Single-factor interventions in both directions, with single-step ablations |
| 5 | **Per-line churn under reseeding** (M1–M3) | A (method) | Changes how every before/after result should be reported; well quantified |

**Not on the shortlist yet:**
- letter novelty (T6) and targeted data (T7, T8): C;
- E1 (V14): C+;
- geometry distances (T5, T12): D as explanations;
- E8, attribution and LocateAnything: D.

### 2.2 How a demanding reviewer would read the CRNN results

- **One architecture, single unseeded runs.** Interventions under 1 point on 91–183 lines sit inside the replicate's own
  ±0.4–1.4 interval.
- **The evaluation sets are small and filtered.**
  - KHATT has 183 lines, and those exclude the 17 worst crops.
  - Half B has 91 lines and is not text-disjoint from half A.
  - There are no writer ids, so bootstrap CIs treat correlated lines as independent.
- **There is one setting: zero-shot transfer.** Training is Muharaf (plus RASAM ×4); the targets are KHATT and NAKBA.
  Effects may vanish in domain (an in-domain control needs NAKBA train, which is not cached locally).
- **The baseline is weak against the state of the art.**
  - KHATT: 26 % zero-shot vs about 8.5 % for an in-domain CRNN (Luleå) and 6.4 % for a possibly contaminated VLM.
  - NAKBA blind: 13.4–13.7 % vs 6.9 % for NAKBA-trained Baseer. Baseer's figure folds the hamza; the CRNN's comparable
    figure is about 11.5 %.
  - That is acceptable for a characterisation paper if it is framed as "a fixed recogniser under domain shift", and if a
    standard-split CRNN (E5) and the VLMs show the same direction.
- **The split is non-standard.** Training uses Muharaf test lines.
- **Most interventions were measured only on the CRNN.** Generality across families is the main gap, and the store makes
  part of it testable on CPU (§4).

### 2.3 Paper groupings

| Paper | Content | Missing before submission | Venue type | Effort |
|---|---|---|---|---|
| **P1 Benchmark audit**, e.g. "Auditing Arabic handwriting benchmarks: leakage, contamination, crops, conventions and metrics" | <ul><li>V1, V2, V3, V12, V15</li><li>T1 as an artefact, plus the 17-line filter</li><li>T11</li><li>The normalisation-ladder recommendation (R2.1 / R2.7)</li><li>Muharaf's 60 px storage</li></ul> | <ul><li>R4.1 label-error rates and κ with CER-HV</li><li>R2.7 rule</li><li>The contamination test (V15)</li><li>Crop re-score on all 200 lines and PaddleOCR cropped</li><li>Counted hamza audit</li><li>Crop looseness on all sets</li></ul> | Evaluation / benchmark track at ICDAR or DAS; an ArabicNLP workshop; LREC-COLING (resources); a short IJDAR / PRL paper | 2–3 weeks; mostly CPU plus about 4 h of human review |
| **P2 Characterisation** (the user's main direction), e.g. "Characterising Arabic handwriting for recognition: positional forms, dots, PAWs and style across CRNN and VLM families" | <ul><li>V4–V10 on common lines across families</li><li>T5 descriptors</li><li>T6 if it passes P1.2</li><li>Line × model matrix: hard core and family splits</li><li>Each characteristic paired with its Latin analogue</li></ul> | <ul><li>The §3 framework</li><li>V2 validity (R7, leave-one-dataset-out and leave-one-family-out)</li><li>CIs everywhere</li><li>One causal probe per headline characteristic (R9 or synthetic renders)</li><li>Writer or text clusters</li></ul> | Journal (IJDAR, Pattern Recognition) or ICDAR main track | 6–10 weeks |
| **P3 Methods**, e.g. "Geometry transfer and letter-novelty-targeted data for cross-dataset Arabic HTR" | M4, T2, T4, T1 (crop), T6, T7, T8; positioned against `garridomunoz2025` (correlational, Latin) | <ul><li>P2.1 seeds and factorial</li><li>Second architecture (E5)</li><li>Second target domain</li><li>VLM geometry check</li></ul> | ICDAR or ICFHR-type conference, or a workshop (historical documents or ML) | 4–6 weeks of mostly GPU; risk: the effects stay small |
| Negative-results note | E8 / E8b, LocateAnything zero-shot, attribution pointing game | Nothing; it is already negative | Appendix of P2 or P3, or a 4-page workshop paper | Small |

**Recommendation:** P1 first, because it is the cheapest and strongest and protects P2 and P3 from benchmark artefacts.
Build the §3 framework at the same time, because P2 *is* that framework's output. Start P3 only after the P2.1 decision.

---

## 3. The standard analysis for every experiment

### 3.1 Principles

- **One store, many views.** Every reading of every line by every system, at every rung of the R2.1 ladder.
- **The unit is the line.** Means are summaries of line distributions, never the only evidence.
- **Every pair comparison carries its noise yardstick.** Use the replicate pair (and later the seeds) for seed churn and
  the numerics pair for evaluation noise.
- **The same lines, the same rung, the same image pipeline** for any cross-family comparison.
- **Galleries are the project convention** (best / worst / random, images, GT, readings, wrong characters highlighted),
  plus four new kinds: hard core, family split, flips under reseeding, unexplained residual.

### 3.2 The per-line store

**Location.** `W/results/store/` (regenerable; commit only small JSON summaries; the Parquet files are git-ignored).

**Environment.**
- `.geomvenv` has numpy, scipy, scikit-image, pyarrow and torch but **no rapidfuzz**.
- `M/.venv` has rapidfuzz and pandas but no scipy or scikit-image.
- Install rapidfuzz into `.geomvenv`, with the uv cache on the SSD:
  `UV_CACHE_DIR=/Volumes/PortableSSD/uv-cache uv pip install --python .geomvenv/bin/python rapidfuzz`
  (small wheels; pandas is optional, since pyarrow is enough).
- The store imports `arabic_ocr_bench.characterize` (runs, errors, text, samples). That package lives on branch
  `research-characterization` (main checkout), not on this branch. **Merge it in first** (open question 6).

**Files and schema.**

```
runs.json            run_id -> {
    family: crnn | crnn+llm | vlm-base | vlm-ft | ocr-engine | api,
    model, base_model, params_m,
    train: {data, n_lines, intervention, repeat_file, geom_aug, seed (null = unseeded), epochs, run_dir},
    eval:  {pack: eval_data | inkcrop_eval | transfer_eval | bench, numerics: fp32-bs32 | fp16-bs1 | bf16-hf,
            decode: ctc-greedy | llm-greedy | vlm-greedy, prompt_version},
    sources: [{path, format, tags}], notes: ["hyp stored reversed", "batch-dependent", ...]}

lines.parquet        one row per (run_id, transform, sample_key)
    run_id, transform        # none | inkcrop | tta_conf | inkcrop+tta_conf | as_muharaf | as_muharaf_only_slant | ...
    sample_key               # canonical "{dataset}/{split}/{index:06d}"
    dataset, split, index, eval_pack
    hyp                      # logical order, as emitted (cleaned_text for VLMs)
    hyp_raw                  # VLM raw_text, else null
    error, latency_ms, out_tokens, at_token_cap
    conf, conf_kind          # ctc_mean_maxprob | vlm_mean_logprob | null
    stored_edits, stored_ref_len   # what the source file says (audit)
    edits[8], ref_len[8]     # rungs: raw, canonical, tatweel_digits, orthographic, no_diacritics, rasm, no_spaces,
                             # plus the side rung crnn_native = canonical + tatweel and harakat stripped, no folds
    in_train                 # no | yes | suspect | unknown

samples.parquet      one row per sample_key (model-independent)
    gt_raw, gt_crnn, n_chars, n_words, n_paws, letters_by_form (json), oov_word_share, unseen_paw_share, xent_5gram,
    img_h, img_w, ink_extent, slant_deg, slant_density_pct, stroke_px, pen_contrast, ruled_share, dot_count,
    ink_px_per_char, edge_ink,
    flags: loose_crop, slant_outlier, ruled, multi_line, low_res, short_line, digits_only, latin, letter_list,
           text_cluster_id, diacritised_gt, hamza_heavy, rare_shape_count, cerhv_flag, consensus_suspect,
           excluded_by_ctc_filter,
    in_train_sets: [crnn_aligner, suite_v1_ft, ...], writer_id, page_id   # null until R0.4

letters_<ds>.parquet one row per (run_id, transform, sample_key, gt_index); analysis sets only (about 8M rows if
                     everything, so build per dataset)
    ch, form, rasm, dotted, ambiguous, paw, word, rel_pos, outcome (ok | sub | del), hyp_ch,
    ins_before (count), novelty_cnn, novelty_pix, span0, span1

words_<ds>.parquet   run_id, transform, sample_key, word_index, gt_word, hyp_word, ok, gt_in_train_vocab,
                     hyp_in_vocab, n_paws
```

**Id mapping** (verified today: 0 ground-truth mismatches after cleaning on every join).

| Source id | Store key |
|---|---|
| `kitab_khatt/NNNNNN` | `kitab_khatt/train/NNNNNN` (KITAB stores its 200 lines as split "train") |
| `nakba_lines/NNNNNN` | `nakba_lines/test/NNNNNN` |
| `nakba_blind/NNNNNN` | `nakba_blind/blind_test/NNNNNN` |
| `muharaf_lines/NNNNNN` (transfer pack) | `muharaf_lines/test/NNNNNN`, with `in_train` from the reconstructed held-out list |
| `muharaf_lines/test/…`, `rasam1/test/…` (aligner and E8) | unchanged |

**Source adapters (one per format).**

1. **`crnn_per_line`** (`geom_transfer_eval.py` format).
   - Layout: `{cer, lines, paired, per_line: {tag: {id: [edits, label_len, hyp]}}}`; the tag is
     `dataset[__transform]`.
   - Files: `shape/{base,targeted_eval,random_eval}.json`, `night/{rep2_eval,rep2_inkcrop,combo_inkcrop,shape_inkcrop,randctl_inkcrop,shape_transfer}.json`,
     `inkcrop_{base,aug,trained}.json`, `{base,aug}_evaldata.json`, `transfer_cer.json`, `aug_transfer.json`.
   - The files carry no run metadata, so add a registry `W/configs/analysis/sources.yaml` mapping each file to its run
     (for example `night/combo_inkcrop.json → crnn/combo@s0, pack inkcrop_eval, fp32-bs32`).
2. **`crnn_tta`** (`xai_geom_tta.py`).
   - Tags: `ds`, `ds__tta`, `ds__inkcrop`, `ds__inkcrop__tta`.
   - **Reverse every hypothesis.**
   - Numerics fp16-bs1. Note that `ds` is the 0° variant, not the batched base reading.
3. **`e8`** (`e8*/per_line.json`).
   - Layout: `{ds: {id: {gt, crnn, e8}}}`. This gives two runs (the CRNN reference on `eval_data` plus Muharaf and
     RASAM held-out, and E8), with edits computed here.
4. **`vlm_jsonl`**.
   - Use `characterize/runs.py` (`iter_readings`, `canonical`) on `M/` with globs `results/**`, `exp3090-results/**` and
     `suite-v1-artifacts/**`: standard prompt only, latest success per sample, `smoke/` skipped, `MODEL_RENAMES`
     applied.
   - **Filter by `dataset`**, because some files hold several datasets.
   - **Split by the `model` field**, which avoids the `vlm_ink_extent.py` bug.
   - Keep `raw_text` for the truncation and commentary detectors.
5. **`letters`** (new dump from `xai_letter_shapes.py --dump-letters`, §3.6).
   - Fields: sample_key, logical GT index, ch, form, span, novelty_cnn, novelty_pix, ok.
6. **GT and images.**
   - GT: the HF Arrow files by row index (the canonical source), cleaned once.
   - Images: `inkcrop_eval.pkl` and `transfer_eval.pkl` (local). `eval_data.pkl` exists only on the box; its images can
     be rebuilt from HF by index.

**Reconciliation (the first output of `store_build.py`).**
- For every CRNN (file, tag), recompute edits at the `crnn_native` rung and assert equality with the stored edits.
- For every VLM pair, reproduce the leaderboard's CER (161 of 162 pairs already match, per R0.3).
- Re-derive the key-findings numbers and list every discrepancy.
- This turns §1.5 #1, #5, #6 and #8 into automatic checks.

### 3.3 Analysis A: sample level

**A1. Distribution per (run, dataset, rung).**
- Rungs: `crnn_native` (continuity with the ledger), `no_diacritics` (the headline across families) and `rasm`.
- Micro CER, macro (mean-of-lines) CER and median line CER.
- **Bucket shares:**
  - exact (0)
  - ≤ 2 %
  - ≤ 5 % (at least 95 % of characters right)
  - ≤ 10 %
  - ≤ 25 %
  - ≤ 50 %
  - ≤ 100 % (failing miserably)
  - **> 100 %** (hallucination or runaway output)
- Plus: empty output, length ratio hyp/GT < 0.5 and > 1.5, tail and head cuts, loops, script mismatch.
- *Example (recomputed), base CRNN on KHATT:* no line is exact, 1.1 % are ≤ 5 %, 10.9 % are at 50–100 %, 9 readings
  are empty and 13 are under half the GT length. With the crop: 2.2 %, 0 and 0. On Muharaf held-out, 16.3 % of lines
  are exact.

**A2. Pair comparisons (B against A on the same lines).**
- (a) An 8 × 8 bucket transition matrix.
- (b) The per-line ΔCER distribution, overlaid on the **noise distributions**: the replicate pair (seed churn) and the
  numerics pair (evaluation noise).
- (c) Micro ΔCER with three intervals:
  - a paired line bootstrap (as now);
  - a **cluster bootstrap** over writer / page or `text_cluster_id`. Single-linkage text clusters chain into one
    71-line component on KHATT, so use page or writer ids when R0.4 has them;
  - a seed-pooled interval when there are ≥ 2 seeds (hierarchical bootstrap over seeds, then lines).
- (d) Lines better / worse **beyond the noise band** (the 90th percentile of the replicate's |Δ| per length tercile; per
  line ± 2 SD once there are ≥ 3 seeds), plus a sign test.
- (e) **Where the gain comes from**: the share of net Δ from lines that started in each A bucket. For the crop, almost
  all of it comes from the 50–100 % bucket.

**A3. The line × run matrix per dataset** (rows = sample_key, columns = runs grouped by family, cell = line CER at the
`no_diacritics` rung), sorted by consensus difficulty. Line labels:

| Label | Definition |
|---|---|
| `solved_by_all` | CER ≤ 0.10 for every competent run of every family |
| `hard_core` | CER > 0.25 for every run (runs flagged `suspect` excluded) → candidates for label review (R4.1) or "unreadable" |
| `crnn_only_failure` / `vlm_only_failure` | Every CRNN > 0.25 while the median VLM ≤ 0.15, and the reverse |
| `noise_sensitive` | Δ beyond the noise band between replicate-equivalent runs, or a per-line SD across seeds above the 90th percentile |
| `fixed_only_by:X` | Improves beyond the band under intervention X in ≥ 2 of 3 seeds, and under no other run |
| `regressed_by:X` | The mirror image |

*Example (recomputed), KHATT 183 lines × the CRNN cropped × 13 non-AIN VLM runs:*
- the hard core is **20 lines**;
- **27 lines** are VLM-only failures and **2** are CRNN-only failures;
- Spearman correlation between CRNN line CER and the median VLM line CER is 0.36.

**A4. Other sample-level metrics.**
- Insertion and deletion run lengths (longest run).
- Truncation (the last ≥ 10 % of the GT deleted; reuse the logic of `M/scripts/line_end_check.py`).
- Repetition loops.
- Relative position of the first error and the error-position profile (the line-end effect, R2.8).
- Word error rate and word exact match.
- **Calibration** where `conf` exists: Spearman(conf, line CER), ECE over 10 bins, and the risk–coverage curve (AURC)
  for selective reading. This needs the CTC confidence stored per line (§3.6); VLM runs store no log-probabilities
  today.

**A5. Galleries** (all via `change_gallery.gallery`):
- best / worst / random for every pair;
- plus hard core, family split, flips under reseeding, and the unexplained residual (§3.5).

### 3.4 Analysis B: letter, PAW and word level

**B1. Letter outcomes.**
- For each GT letter (ch, positional form, rasm class, dotted, rasm-ambiguous, PAW and word index, relative position),
  record the outcome: right, substituted (with the read character) or deleted, plus insertions attached to positions.
- Reuse `characterize/samples.letter_outcomes` (rapidfuzz opcodes on the letters view), extended to return the
  substituted character.

**B2. Character × form error rates per run.**
- Wilson 95 % intervals plus a line-cluster bootstrap.
- A Δ heatmap for each pair, with marks where the cluster interval excludes 0.
- The top movers.

**B3. Confusions.**
- The top-30 GT→hyp pairs per run, tagged dot, orthographic, data-derived shape (R2.6) or style (Maghrebi ق→ف, ف→ب).
- Δ per 1,000 GT occurrences between runs.
- Reuse `characterize/errors.classify_ops`, `sid_profile` and `scripts/substitution_analysis.py`.

**B4. Dots.**
- Letters with the right skeleton and the wrong dots, per 100 dotted letters.
- The dot tax per line (the rasm-rung difference).

**B5. PAWs.**
- PAW accuracy (all letters right) by PAW length and by whether the PAW was seen in training (the `geom_paw_coverage.py`
  inventory).
- Spaces inserted or deleted at PAW gaps per 100 gaps (`errors.is_paw_gap_space`, `paw_gap_base_rate`).

**B6. Words.**
- Word error rate.
- Out-of-vocabulary versus in-vocabulary word error, against the Muharaf-train vocabulary built from the local Arrow
  files.
- **Fluent-but-wrong words**: a wrong word that is still a real training-vocabulary word (the E8 definition).
- Hallucinated inserted words.
- Real-word substitution share.

**B7. Novelty join.**
- Join X-G10 per-letter novelty (from the new dump) by (sample_key, logical index) to **every** run's letter outcome,
  VLMs included.
- Report error by novelty band per run and family, and Δ error by band for each intervention.

**B8. Letter galleries.**
- For the character × form cells that moved most, show crops from the base CRNN's GT alignment. Spans are image
  positions, so they serve every model's outcome.
- Colour the outcome per run, and show the nearest training letter (as in `build_letter_gallery.py`).

### 3.5 Pattern detectors

Each detector is a function `detect(sample_row, line_row) -> (flag, score, evidence)` in `scripts/detectors.py`. Every
detector documents its rule, threshold and source.

**Output.** For each detector:
- prevalence per dataset;
- for each run: % of lines flagged, share of the run's edits on flagged lines, mean CER flagged vs not, and **excess
  edits explained** = Σ over flagged lines of (edits − n · CER of unflagged lines), divided by total edits;
- a gallery of 6 flagged lines.

**Residual.** Lines with CER above the run's 80th percentile that no detector flags. This is the feed for new
detectors; review it by hand in every analysis.

| ID | Level | Detector | Rule (threshold) | Reuse |
|---|---|---|---|---|
| D1 | image | Loose crop | Ink extent < 0.35 | `geom_ink_extent.py` |
| D2 | image | Slant outlier | Slant where < 5 % of training lines lie | `geom_measure.py` |
| D3 | image | Ruled lines | Long thin horizontal components > 2 % of ink | `geom_measure.strip_rules` |
| D4 | image | Neighbour-line intrusion | Edge-ink rows or several baselines | `M/scripts/image_layer.py` |
| D5 | image | Low effective resolution | Ink px per character < training p10, or dot diameter < 2 px at the model's input height | `image_layer.py`, `geom_dots.py` |
| D6 | image | Stroke or pen outlier | Stroke width or pen contrast outside training p5–p95 | `geom_measure.py` |
| D7 | text | Fragment line | ≤ 20 characters; digits only; Latin content | `characterize/text.py` |
| D8 | text | Letter-list drill / fixed text | ≥ 4 single-letter tokens; `text_cluster_id` size > 1 | new |
| D9 | text | Rare shapes and unseen units | A letter × form with < 10 training examples (R1.7); unseen PAWs; 5-gram cross-entropy > p90 | `geom_paw_coverage.py`, `geom_text_shift.py` |
| D10 | label | Label suspect | A CER-HV flag; or ≥ 3 strong models agree with each other (pairwise CER < 0.1) and all miss the GT (CER > 0.3); or GT length is an outlier against image width | `M/scripts/rescore_muharaf_clean.py`, `build_label_review.py` |
| D11 | prediction | Truncation | Tail cut (the last ≥ 10 % deleted); head cut; output at the token cap | `line_end_check.py` |
| D12 | prediction | Repetition loop | A word n-gram (n ≤ 4) repeated ≥ 3 times in a row, a character 4-gram ≥ 5 times, or length ratio > 1.5 | new |
| D13 | prediction | Whole-line failure or meta-commentary | Line CER > 0.8, or a commentary regex | `errors.WHOLE_LINE_CER`, `M/src/.../analysis/refusal.py` |
| D14 | prediction | Script or digit mismatch | ≥ 20 % Latin or CJK where the GT has none; mixed digit systems | `analysis/align.py`, `text.hygiene` |
| D15 | prediction | Fluent rewrite | ≥ 50 % of wrong words are in-vocabulary and line CER > 0.3 | E8 definition |
| D16 | prediction | Diacritic insertion | Hyp marks per 100 letters exceed the GT's by more than 2 | `errors.diacritic_mismatch` |
| D17 | prediction | Orthographic only (hamza, ya, ta marbuta) | ≥ 50 % of the line's edits vanish at the orthographic rung | ladder |
| D18 | prediction | Dot only | ≥ 50 % of edits vanish at the rasm rung | ladder |
| D19 | prediction | Maghrebi dot convention | ق→ف or ف→ب substitutions | `substitution_analysis.py` |
| D20 | prediction | Space at a PAW gap | ≥ 1 space inserted at a PAW boundary | `errors.is_paw_gap_space` |
| D21 | prediction | Lam-alef ligature error | A ligature-class edit | `errors.classify_ops` |
| D22 | prediction | Order reversal | Edit distance of the reversed hyp to the GT < 0.5 × that of the hyp (would have caught §1.5 #5) | new |
| D23 | prediction | Memorisation / contamination | GT whitespace or typo quirks reproduced; exact-match rate ≥ 3× the median of models with similar CER (V15) | new |
| D24 | prediction | Numeric / batch instability (CRNN) | The reading changes between fp32-bs32 and fp16-bs1 | paired runs |

### 3.6 Scripts

**New scripts** (all in `W/scripts/`, CPU unless noted).

1. **`store_build.py`** — ingests every source into `results/store/` and writes `reconcile.json`.
   - Input: `configs/analysis/sources.yaml` plus `--vlm-root M/`.
   - Reuses `characterize/runs.py`, `errors.ladder_strings` / `ladder_edits`, `geom_data` and the HF Arrow files.
   - Options: `--add FILE` (append one new result), `--rebuild`.
   - Tests in `W/tests/test_store.py`: id mapping, reversal detection, equality of rungs with stored edits, base and
     fine-tuned rows kept apart.
2. **`store_features.py`** — writes the sample features and image / text flags into `samples.parquet`.
   - Reuses `geom_measure`, `geom_ink_extent`, `geom_text_shift`, `geom_paw_coverage`, `characterize/text.line_features`
     and `image_layer` functions.
   - Builds text clusters from shared trigrams and the letter-list flag.
   - About 10–30 min on the laptop for about 6k samples; images are streamed, not all held in RAM.
3. **`analyze_lines.py`** — Analysis A.
   - Arguments: `--runs <ids|globs> --pair A:B ... --noise A:B --numerics A:B --datasets ... --rung no_diacritics`.
   - Writes `results/analysis/<stamp>/A/{buckets,pairs,labels_<ds>}.json` and `matrix_<ds>.parquet`.
   - Figures go to `figs/*.svg`, with data JSON for the ledger charts (follow the `dataviz` conventions when building
     them).
4. **`analyze_letters.py`** — Analysis B.
   - Writes `results/analysis/<stamp>/B/{letters,confusions,paws,words,novelty_join}.json`, heatmaps and crop
     galleries.
   - Needs CTC spans for crops (`xai_align.py` output or the new letter dump).
5. **`detectors.py`** (library) and **`run_detectors.py`**.
   - Writes `results/analysis/<stamp>/D/detectors.json`, `residual_<ds>.json` and the galleries.
6. **`build_evidence_section.py`** — builds the ledger block from an analysis bundle (pattern: `build_day4_section.py`
   plus `change_gallery`).
   - Writes the galleries to a supporting data file, `docs/research/evidence-data.js` (as `aug-gallery-data.js` does),
     so the ledger HTML stays small. It is 4.4 MB now and the artifact limit is 16 MB.
7. **`evidence.py <exp-id>`** — the one-command driver for every later experiment:
   `store_build --add` → `analyze_lines` → `analyze_letters` → `run_detectors` → `build_evidence_section`.

**Changes to existing scripts** (small, on this branch).

| Script | Change |
|---|---|
| `xai_geom_tta.py` | Store hypotheses in logical order; store all 5 variants' edits, readings and confidence per line (enables offline gate and selector studies); add an `--fp32` option |
| `geom_transfer_eval.py` | Batch size 1 or packed sequences (removes the padding dependence); per-line CTC confidence; write `numerics` into the output |
| `xai_letter_shapes.py` | `--dump-letters` (per-letter records with logical index, span and both novelties); `--inkcrop`; `--align-run` (take spans and right/wrong labels from a second model) |
| `xai_prepare_eval_data.py` / `geom_inkcrop_export.py` | Keep all lines; apply the CTC-length check after the optional crop; write the list of filtered ids |
| `xai_train_aligner.py` | A `--seed` argument; step-level checkpoints; a `--standard-split` mode (Muharaf train only; held-out = the full test set) |
| `vlm_ink_extent.py` | Group by the `model` field |
| `change_gallery.py` | Escape newlines in readings; mark "reversed?" when D22 fires |

### 3.7 Ledger integration

- **New part-2 block `d-evidence`: "Per-line and per-letter evidence".** One sub-block per result, each with:
  1. a bucket-shift bar chart;
  2. the transition matrix;
  3. lines better / worse beyond seed noise, next to the replicate's own numbers;
  4. ΔCER with line, cluster and (when available) seed-pooled intervals;
  5. galleries (best / worst / random plus the four new kinds), loaded lazily from `evidence-data.js`;
  6. the letter Δ heatmap and confusion Δ;
  7. its row in the detector table.
- **Key-findings table:** add three columns:
  - **Grade** (A–D, §1.1);
  - **Line pattern** (for example, "concentrated: 18 better / 1 worse beyond noise");
  - **Replicated in** (models, seeds, families).
- **EXP rows:** "done when" gains "analysis bundle built". Each row links to its bundle.
- **Corrections block:** a short dated list of the §1.5 fixes, so earlier numbers can be traced.
- **Publishing:** republish to the same URL only after the user has reviewed the fixes. Nothing is published by this
  plan.

### 3.8 Implementation order

| Step | Work | Effort (laptop) | Done when |
|---|---|---|---|
| 0 | rapidfuzz into `.geomvenv`; merge `research-characterization`; `sources.yaml` | ½ day | `import arabic_ocr_bench.characterize` works in `.geomvenv` |
| 1 | `store_build.py` with the 5 adapters; reconciliation; tests | 1 day | Every stored CRNN edit reproduced; leaderboard CERs reproduced; §1.5 #1, #5, #6, #8 reported automatically |
| 2 | `analyze_lines.py` (A1–A5) on the CRNN family and the KHATT VLM block | 1 day | Bucket tables, transition matrices, beyond-noise counts, KHATT line × model matrix and labels |
| 3 | `store_features.py` + `detectors.py` (image and text first, prediction-side next) | 1 day | Detector table and residual gallery for KHATT and NAKBA |
| 4 | `analyze_letters.py` (B1–B6, B8); B7 after the GPU letter dump | 1½ days | Character × form tables with CIs for CRNNs and VLMs on KHATT; confusion Δ; PAW and word tables |
| 5 | `build_evidence_section.py`, key-findings columns, corrections block | ½ day | Ledger block built locally and reviewed |
| 6 | `evidence.py` driver; update `NEXT_STEPS.md` and `progress.json` | ½ day | One command per experiment |

### 3.9 The evaluation contract for every later experiment

1. **Score the standard panel** with per-line output in the store format (logical order, confidence, numerics):
   - KHATT, all 200 lines, as stored and ink-cropped;
   - NAKBA test, 600 lines;
   - NAKBA blind, full 2,671 (or the fixed 1,082 union);
   - Muharaf held-out, 300;
   - RASAM-1 held-out, 100;
   - RASAM-2 test.
2. **Always include the in-domain regression check** (Muharaf and RASAM held-out). The night queue never scored it for
   the shape, random or combo models.
3. **Carry a noise reference:** at least one replicate of the baseline, plus the numerics pair once.
4. **Run `evidence.py <exp-id>`.** The gallery and ledger block follow from it. Every result that moves a metric by
   ≥ 0.5 points gets the A, B and detector views before it enters the key-findings table.
5. **Pre-register the decision rule** in the EXP row before the run (§5.2 gives the rules for the queued runs).

---

## 4. Which models and runs get the analysis now

**Coverage map** *(recomputed; ground-truth join exact everywhere)*:

| Dataset | CRNN lines | VLM runs reading the same lines |
|---|---|---|
| KHATT | 183 | 18 runs: 15 read all 200 rows, 3 (Qwen3-VL-4B suite fine-tune, PaliGemma base and fine-tuned) read rows 20–199 |
| NAKBA blind | 1,082 (union of both samples) | 6 NAKBA-trained runs on all 2,671 (Baseer ×2, AIN-NAKBA checkpoints 1200 / 1600 / 1936, visft-400) |
| NAKBA test | 600 | 239 lines with Qwen3-VL-4B base and fine-tuned; 116 with PaliGemma; 64 with PaddleOCR and the exp3090 Qwen fine-tune; 58 with 12 other runs |
| Muharaf held-out | 300 | 300 with Qari-0.2, TrOCR and nakba-baseline-qwen3vl; 296 with Qwen3-VL-4B base and fine-tuned; 83 with the suite fine-tunes |
| RASAM-1 held-out | 100 | 99–100 with Qwen3-VL-4B base and fine-tuned and PaliGemma; 53 with others |

| Run(s) | Datasets now | Why now | Analyses |
|---|---|---|---|
| CRNN base (3 packs, numerics pair from TTA 0°) | KHATT, NAKBA test, NAKBA blind A and B, Muharaf and RASAM held-out (E8 file), transfer tags | Reference for everything; gives the numerics noise band | A, B, D |
| CRNN replicate (rep2) | Same, minus held-out | **The noise yardstick** for every retraining claim | A (noise), B (letter churn) |
| Random control, shape-targeted | KHATT (halves), NAKBA, cropped | Tests T7 at the line and letter level; are the gains confined to novelty bands and character × form cells? | A, B, B7 (after the dump) |
| Combo; combo + TTA; base TTA | Cropped KHATT, NAKBA | T8 and the TTA gate; rebuild the reversed TTA gallery | A, D (D2 gate study) |
| Augmented; crop-trained | All packs | Nulls: do they move any bucket or cell? Robustness profile | A, B |
| Transfer tags (base, augmented, shape) | KHATT, NAKBA; Muharaf flagged in-train | Which lines and letters slant and stroke changes move | A, B |
| E8 0.6B and 1.7B | KHATT, NAKBA, NAKBA blind A, Muharaf and RASAM held-out | Validation for D12, D13 and D15 (loops, fluent rewrites) | A, D |
| **18 VLM runs on KHATT 183** (AIN ×5, Fanar ×2, Qwen3-VL-4B ×3, Qwen3-VL-8B ×2, PaliGemma ×2, PaddleOCR, Gemini Flash and Flash-Lite, Pixtral) | KHATT | **Cross-family generality of every CRNN claim**: crop, slant, novelty, hard core; the AIN contamination test | A3 matrix, B (letters and novelty join), D (D1, D11–D16, D23) |
| 6 NAKBA-trained VLM runs | NAKBA blind (1,082 lines shared with the CRNN) | Is the hamza convention learned by NAKBA-trained models (T11)? Checkpoint behaviour | A, B (orthographic), D17 |
| Qari-0.2, TrOCR, Qwen3-VL-4B base and fine-tuned | Muharaf held-out 300; RASAM-1 100 | In-domain family comparison on lines the CRNN never trained on | A, B |
| Suite fine-tunes (AIN, Fanar, Qwen 4B and 8B) | RASAM-1 / 2 | The Maghrebi convention (V8) with letter galleries | B3, D19 |

**Leave out for now:**
- page-scale sets (`kitab_hindawi`, `kitab_historicalbooks`);
- API rows with fewer than 50 lines;
- `smoke/` runs;
- runs with > 5 % of outputs at the token cap (score only after X6);
- `kitab_muharaf` for fine-tunes (leaked).

**Making it permanent:** the §3.9 contract. Every new evaluation script writes the store format directly, and the EXP
row is not "done" until `evidence.py` has run.

---

## 5. Next experiments, prioritised

### 5.1 What the literature says about the open approaches

Sources are the ledger paper records and `references.bib`; web checks today are in "Sources" at the end. "IG" is the
expected information gain for the papers in §2.3.

| Approach (ID; where planned) | What others found | Expected for us | IG | Compute | Priority |
|---|---|---|---|---|---|
| **Normalisation ablation** (X-G3: height 32 / 48 / 64 / 96, pad vs resize, deslant, binarisation; EXP, ideas #11) | <ul><li>`retsinas2022best`: padding beats resizing (IAM line WER 15.89 vs 16.19 %); column max-pooling 4.93 vs 5.54 % CER; CTC shortcut 4.93 → 4.62 %</li><li>`liebl2020crnn`: line height and binarisation interact</li><li>`vinciarelli2001norm`: deslanting +10.8 % relative in an HMM-era recogniser</li><li>Our X-T3: augmentation gives robustness, not transfer</li></ul> | ≤ 1 point each, except very low heights (dots, X-G7 proxy) | M | 3090, about 12 trainings at about 1 h | P2.5 |
| **Dot-preserving resize, dot channel** (X-G7; pool #2) | No Arabic study found | Nothing at 64 px (dots survive); only below 48 px | L | 3090 | P3 |
| **Factorised rasm + dot head** (X-G9; pool #4) | <ul><li>`ahmad2019subcore` (HMM): modelling core shapes and diacritics separately beat character-shape models on two databases (abstract)</li><li>`alazzawi2026gap`: explicit positional-form labels gave no consistent gain</li><li>Dot pairs are 13 % of letter substitutions (R2.6)</li></ul> | Small (≤ 0.5 points) but tells us whether neural CTC benefits from the rasm × dots factorisation | M | 3090, 4 trainings | P3 |
| **Allograph library per writer and style** (pool #5) | Allographic features identify writers (`bulacu2007allographic`); no use in recognition found | An analysis tool, not a training method | M | CPU after the letter dump | P1.2 (part) |
| **Allograph-stitching synthesis** (X-G11; pool #6) | <ul><li>`dinges2016synthesis` (unverified): connection models for synthetic Arabic</li><li>Font renders helped printed Ottoman (`bilgintasdemir2023ottoman`)</li><li>Arabic GAN generators are weak (`fogel2020scrabblegan`, `kang2020ganwriting`)</li></ul> | Risky: unnatural joins; only if P2.1 shows targeted shapes help | M | 3090 + engineering (days) | P3 |
| **Geometry pretext pretraining** (X-G12; pool #8) | Pretext tasks usually give small gains; nothing Arabic found | Small | L | 3090 | P3 |
| **Geometry-matched augmentation; transfer used as augmentation** (pool #7, #14) | <ul><li>`garridomunoz2025`: visual divergence is a weak predictor of out-of-distribution error (r ≈ −0.3, against 0.62 for text)</li><li>`kohut2023choose`: both appearance and language matter when choosing pretraining data</li><li>Our slant augmentation had no effect</li></ul> | Small | L | 3090 | P3 |
| **Pen-contrast augmentation** (pool #12) | No literature found; RASAM is already in the CRNN's training data | Small for the CRNN; E6 is the direct test for VLMs | L | 3090 | P3 |
| **Slant-density data selection** (pool #13) | The same family as T7 | Wait for P2.1 | L | 3090 | after P2.1 |
| **Text-side test: CTC + character n-gram LM decoding** (the E8 follow-up "lattice-constrained rescoring"; T12) | <ul><li>n-gram LMs in CTC beam search are standard in HTR toolkits and usually give moderate gains</li><li>`garridomunoz2025`: text divergence is the strongest predictor of out-of-distribution error (Latin)</li><li>Our E8 shows that an unconstrained LLM decoder hurts</li></ul> | If text distance drives KHATT's residual gap, an LM of modern text should cut KHATT CER more than NAKBA's; an LM trained only on Muharaf text should not | M–H | <ul><li>GPU minutes to dump CTC posteriors, then CPU beam search</li><li>A modern-MSA text sample needs a download (to the SSD)</li></ul> | P2.8 |
| **Better TTA selector** (X-G8 follow-up) | <ul><li>IAM TTA (arXiv 2307.00664): test CER 4.80 → 4.37 %, oracle 3.02 %</li><li>Test-time adaptation (arXiv 2308.15037): augmentation-consistency confidence tracks CER; CTC confidence does not</li><li>Ours: confidence selection 25.4 % vs oracle 23.2 %</li></ul> | An agreement-based selector may close part of the 2.2-point gap; a slant gate removes the NAKBA cost | M | GPU minutes (store all variants), then CPU | P1.3 / P0 |
| **E5 CRNN reference** (Retsinas recipe; standard split) | The Luleå papers get 8.5 % on KHATT and 10.1 % on Muharaf with this recipe, trained in domain | Needed for comparability and as a second architecture for every diagnostic | **H** | 3090, 4 trainings | **P2.2** |
| **E6 Maghrebi adaptation** | <ul><li>`vidalgorene2024rasam2`: about 40 % lower CER with Maghrebi training data; documents the confusions</li><li>`farazi2026vlmocr`: VLM correction fails when the prior is script-mismatched (abstract)</li><li>`kohut2023finetuning`: 16 lines give −25 % relative CER (abstract)</li></ul> | RASAM training lines: a large gain; a prompt hint: small | M | Hint about 1 GPU h; training about 10 h | P2.7 |
| **E7 multilingual pretraining with a freeze sweep** | <ul><li>`aradillas2021boosting`: freezing at most CNN layers 1–3 is harmless; full unfreezing is best (up to −6 points)</li><li>`alazzawi2026crossling` / `seqmodel`: joint training helps at 100–1,000 lines; the transfer lives in the BiLSTM</li></ul> | Likely reproduces their results; overlaps with the Luleå group | L–M | 3090 + new data (IAM, NUST, PHTD) | P3 |
| **E3′ data scaling on their K grid** | `alazzawi2026gap`: KHATT needs about 150k characters for 10 % CER | A known shape | L | 3090, heavy | P3 |
| **A1 augmentation items** (13 kashida, 14 PAW-gap jitter ≤ ±20 %, 16 dot minimal pairs, 17 Maghrebi dot swap, 18 harakat, 19 coverage text, 20 fragments, 21 style mixing; the catalogue) | <ul><li>Generic augmentation helps (`wigington2017augment`, `luo2020learn`)</li><li>Rare-character oversampling on IFN/ENIT (`eltay2022adaptive`, unverified)</li><li>No study of Arabic-specific items found</li></ul> | The targeted error may move; total CER unlikely to | M | 3090; measured with framework B | P3 (P2 if a target error dominates in the detector table) |
| **E1.1 / E1.2 ViT unfreeze** | Baseer-Nakba ladder: 0.16 → 0.10 with the encoder unfrozen at a differential LR | E1.1 tells whether our −7.6 holds outside Muharaf | **H** | E1.1 about 1 h; E1.2 about 10 h | **P2.4** |
| **R9 resolution and noise probe** | <ul><li>`rice1995isri`: 300 → 200 dpi gives about +50 % errors</li><li>`hegghammer2021noisy`: Arabic degrades more than English under the same noise</li></ul> | Tests the Muharaf 60 px hypothesis (V13) | M | About 2 GPU h | P2.6 |
| **P-L1 dot-restoration ceiling and H(dots \| rasm, context)** | `alshaibani2025dotless`: dots carry about 0.4–0.55 bits per character; `shannon1951` | Gain capped at the dot tax (about 1.8 points) | M | CPU | P1.9 |
| **P-L2 PAW-space repair** | `alluhaybi2020connect`: ligation affects word recognition | A V5 "useful" test for the PAW characteristic | M–H | CPU | P1.5 |
| **R2.7 diacritic scoring rule** | `doctor2022graphemic`, `heakl2025kitab` | Decides the P1 metric recommendation | H (for P1) | CPU | P1.4 |
| **R3.2 IRT; R3.3 challenge subsets** | `rodriguez2021irt` | The subsets fall out of the A3 matrix | M | CPU | P1.1 (part) |
| **R4.1 label review** | `alazzawi2026labelerr`: 6–9 % label errors in Muharaf | Label-error rate per set; κ with CER-HV | H (for P1) | Human, 2–3 h | P1.4 |
| **R7 V2 validity** (mixed effects, leave-one-dataset-out, leave-one-family-out) | Gate V2 of the framework | Which characteristics survive | **H** | CPU | P1.7 |
| **R0.4 writer and page ids** | `mahmoud2014khatt`: the original KHATT has writer metadata; the KITAB copy does not | Cluster-robust intervals; writer variance | H | CPU + data access | P1 (blocked: open question 3) |
| **R1.6 lexical diversity; R1.7 shape tail** | `alazzawi2026crossling` Table 3; `alazzawi2026gap` Fig. 6 | Cheap table rows for P2 | M | CPU, minutes | P1.6 |
| **X-A3 image reliance** | `greekvlm2026`: reliance on the prior differs by model; fluent errors with little use of the image | Supports the fluent-but-wrong story (T10, D15) | M | GPU, about 2 h | P3 (P2 if P2 needs it) |
| X-A2b, X-A4, X-A5, X-A6, X-A7, X-D0, X-D1, X-C2 | Attribution-evaluation caveats (ledger) | Diagnostic; does not move the characterisation papers | L | GPU | P3 / parked |
| C-track curriculum | <ul><li>`louradour2014curriculum`: mostly faster convergence</li><li>`wu2021curricula`: curricula help with small budgets or noisy labels</li></ul> | Parked | L | CPU / GPU | parked |
| P-L5 checkpoint averaging | `izmailov2018swa`, `wortsman2022wiseft`; Baseer-Nakba 0.10 → 0.079 | Not characterisation | L | GPU, cheap | when idle |
| E2 tokenizer view | `alomari2026iconicity` | Low | L | CPU | P3 |

### 5.2 Ordered plan with decision rules

**P0. Prerequisites** (about 1.5 laptop days and 2–3 GPU hours; nothing else is trustworthy without it).

| Step | Work | Compute | Done when |
|---|---|---|---|
| P0.1 | Store and reconciliation (§3.8 steps 0–1) | CPU | Every key-findings number is re-derived; §1.5 issues are listed by the tool |
| P0.2 | Fix the TTA reversal; rebuild the TTA gallery; fix `vlm_ink_extent.py` grouping; escape newlines in galleries | CPU, 1 h | Ledger block corrected locally |
| P0.3 | **Harmonised CRNN evaluation:** base, rep2, random, shape, combo, augmented and crop-trained on the §3.9 panel; KHATT on all 200 lines; full NAKBA blind; fp32 at batch size 1 (or packed); per-line confidence | 3090, about 1.5 h total | One line set per dataset for every CRNN |
| P0.4 | **Letter dump:** `xai_letter_shapes.py --dump-letters` for base, rep2, random, shape and combo, on raw and cropped lines, plus a variant whose spans and labels come from a second model (`--align-run`) | 3090, about 1.5 h | Per-letter Parquet for B7 and P1.2 |
| P0.5 | TTA with all five variants stored (base, combo) | 3090, about 1 h | Gate and selector studies run offline |

**P1. CPU analyses on existing data** (each high information gain; about 1–2 days each).

- **P1.1 Cross-family line × model matrix.** KHATT 183 × (CRNN family + 18 VLM runs); NAKBA blind 1,082 × 6 NAKBA-trained
  runs.
  - Outputs: hard core, family-split lines, consensus difficulty, challenge subsets (R3.3), detector table.
  - Decision:
    - if the crop-fixed lines are not elevated for VLMs (expected), T1 goes to P1 as a pipeline artefact;
    - if the CRNN's hard lines are also hard for VLMs (ρ ≥ 0.5 after cropping), the shared difficulty is a property of
      the handwriting and goes to P2;
    - the hard core goes to R4.1 review.
- **P1.2 The X-G10 confound battery** (needs P0.4).
  - (a) Novelty on ink-cropped lines.
  - (b) Conditional logistic regression of letter error on novelty with line fixed effects, plus letter × form fixed
    effects.
  - (c) **Cross-family:** do VLM letter errors on KHATT rise with CRNN novelty?
  - (d) Spans and labels from a second model.
  - (e) Letter-list lines kept as a separate stratum.
  - **Pass** (keep T6 as a P2 / P3 centrepiece): after cropping and within lines, the misread odds of letters beyond the
    95th percentile against bands 1–3 are ≥ 2 (CI excludes 1) for the CRNN, **and** ≥ 1.5 for the median VLM.
  - **Fail:** the effect vanishes within lines. Then reframe novelty as a line-level (writer or scan) descriptor and
    drop the letter-level claim.
- **P1.3 TTA slant gate**, simulated offline (needs P0.5, or approximated now with the stored 0° and selected readings
  plus per-line slant from `geom_measure`).
  - Choose the gate on KHATT half A and NAKBA blind sample A; test it on the other halves.
  - Keep it if NAKBA loses no more than +0.1 and KHATT keeps ≥ 70 % of the TTA gain.
- **P1.4 Hygiene battery for P1.**
  - D23 contamination for every model × dataset.
  - Crop looseness and filter exclusions for every set.
  - Hamza emission rates (NAKBA-trained vs others) and the counted manual audit (100 sites, about 1 h).
  - R2.7 rank correlation.
  - **R4.1 label review** (about 2–3 h; the tool already exists).
  - Done when every P1 claim has a number with an interval.
- **P1.5 PAW gaps.**
  - Per-model and per-dataset CIs, with the CRNN family added.
  - Image check: gap width in pixels at the inserted spaces against other PAW gaps.
  - P-L2 repair (WER and CER before and after on every competent pair).
  - Keep V5 at B+ if the PAW share stays above chance in ≥ 80 % of competent pairs (CI excludes the per-dataset chance
    rate).
- **P1.6 Text tables:** R1.6 and R1.7; add P-L3 (already computed) to the Atlas.
- **P1.7 V2 validity (R7).**
  - Mixed model of line CER on the features, with random effects for model and dataset.
  - Leave-one-dataset-out and leave-one-family-out.
  - Text cross-entropy per line as a predictor, against the `garridomunoz2025` claim for Latin.
  - Done when there is a ranked list of characteristics that survive.
- **P1.8 KHATT fixed-text natural experiment.** The 35 lines from fixed paragraphs (same text, several writers) give the
  writer and visual variance with text held constant. Descriptive; small n.
- **P1.9 Dot-restoration ceiling (P-L1).** Only if P1.1–P1.7 leave time.

**P2. GPU runs that decide publication** (the 3090 queue in §5.3).

- **P2.1 Seeds and factorial for T7 / T8.**
  - Design: {base, random ×4, targeted ×4} × {raw training, crop training} × 3 seeds = 18 cells. 5 exist with one seed,
    so about 13–15 new trainings, about 15 GPU hours, with `--seed`.
  - Evaluate on a **text-cluster-disjoint** KHATT split. Run a **second target**: select on NAKBA blind sample A, test on
    sample B.
  - Add a **hard-example control**: repeat the 2,904 highest-loss training lines.
  - **Real:**
    - seed-pooled targeted − random on held-out lines ≤ −0.5 points, with a 95 % interval below 0 (hierarchical
      bootstrap over seeds and lines);
    - targeted beats the hard-example control;
    - bands 4–5 improve by ≥ 1.5 points while bands 0–3 stay within ±0.5 in ≥ 2 of 3 seeds;
    - Muharaf held-out stays within +0.3.
  - **Close X-G11 lite** (report it as negative) if the pooled effect is above −0.3 or its interval includes 0.
- **P2.2 E5.**
  - A standard-split CRNN (Muharaf train + RASAM train only; held-out = full Muharaf test) and a Retsinas-recipe CRNN
    (column max-pool, CTC shortcut, median padding), × 2 seeds each.
  - Repeat the crop, slant-transfer and novelty diagnostics on them.
  - Decision: any key finding that changes direction is not robust.
- **P2.3 VLM geometry sensitivity.**
  - Models: fine-tuned Qwen3-VL-8B, fine-tuned AIN, fine-tuned Fanar and base Qwen3-VL-8B.
  - Inputs: KHATT slant-only shear, KHATT cropped, Muharaf upright on its **61 held-out lines plus fresh held-out lines**.
  - About 4–6 GPU hours.
  - Decision: if VLM |Δ| < 0.5 points (CI) where the CRNN moves ≥ 1, slant sensitivity is a property of a fixed-height
    CNN; otherwise it holds across families.
- **P2.4 E1.1 held-out bench** (about 1 h). Run E1.2 (about 10 h) only if arm (c) gains ≥ 3 points on 2 of the 4 held-out
  sets.
- **P2.5 Reduced X-G3:** height {48, 64, 96} × {pad, resize}, with the crop on, × 2 seeds. Run after P2.1. Decision:
  adopt only a setting that wins on ≥ 2 targets without in-domain loss.
- **P2.6 R9 resolution probe:** KHATT downscaled to 60 px for the CRNN and 3 VLMs. Decision: if CER rises ≥ 3 points,
  Muharaf's 60 px storage becomes a P1 / P2 claim.
- **P2.7 E6 Maghrebi:** the prompt-hint arm first (cheap), the RASAM-training arm later. Primary outcome: ق→ف on
  RASAM-1 / 2.
- **P2.8 Text-side test.**
  - Dump the base and combo CRNN posteriors on the panel (GPU minutes).
  - Run CTC beam search with character n-gram LMs on CPU: (a) Muharaf-train text only, (b) Muharaf train plus a modern
    MSA sample.
  - Decision: if (b) − (a) gains ≥ 1 point on KHATT and ≤ 0.3 on NAKBA, text distance carries part of KHATT's gap and
    T12 moves from D to B. If no LM helps, the residual gap is visual or writer-related.

**P3. Later or gated:** X-G9, X-G7, A1 items, X-G11 stitching, X-G12, E7, E3′, X-A3 and the rest of the attribution
track, X-C2, the C-track and the S-track.

### 5.3 Suggested 3090 queue (about three nights)

| Night | Jobs | Approx. time |
|---|---|---|
| 1 | P0.3 harmonised evaluation for 7 checkpoints; P0.5 TTA all variants; P0.4 letter dumps; seeds batch 1 (targeted s2, random s2, base s3) | about 8 h |
| 2 | Seeds batch 2 (targeted s3, random s3, crop-train + random s1 / s2, combo s2 / s3, hard-example control s1 / s2) | about 9 h |
| 3 | P2.2 E5 (4 trainings); P2.4 E1.1; P2.3 VLM geometry (if VRAM allows, after the CRNN jobs) | about 9 h |

Notes for the queue:
- Add `--seed` and step-level checkpoints to `xai_train_aligner.py` before night 1 (rule: lose only a few samples).
- Launch with `Win32_Process Create`, and give the live-view command with every launch (log tail plus a status check).
- One shared venv and cache on the box.
- Update `xai-tasks-and-runbook.md` §2 when a step needs a fix (the reproducibility rule).

---

## 6. Open questions for you

1. **Which paper first?** The recommendation is P1 (audit), with the §3 framework built in parallel. Is there a target
   venue or deadline?
2. **May the CRNN be retrained on the standard split** (Muharaf train only)? This is needed for any comparison with
   published numbers, and it changes every CRNN number in the ledger.
3. **Writer and page ids.**
   - Can we get the original KHATT (with writer ids) or Muharaf page ids?
   - May we download them to the SSD?
   - Without them, KHATT intervals stay line-level, or rely on crude text clusters.
4. **3090 budget.** Is about 25–30 GPU hours over the next week acceptable (§5.3)?
5. **Human time.** About 3–4 h for R4.1, the hamza audit and the hard-core review. Who reviews, and should a second
   reviewer give a κ?
6. **Code location.** Merge `research-characterization` into this worktree branch (recommended, so the store can import
   `characterize`), or build the analysis in the main checkout?
7. **AIN contamination.** Report it as a suspected contamination with the quirk evidence, or contact the AIN or
   KITAB-Bench authors first? Contacting them is outward-facing; there is a related question about the Luleå contact
   (X9).
8. **Galleries in the ledger.** May the new galleries move to a supporting `evidence-data.js` file, to keep the ledger
   under the size limit?
9. **VLM log-probabilities.** May the runner save token log-probabilities for local HF models, for calibration (A4)?
   This changes the cache keys only for new runs.
10. **NAKBA train.** An in-domain control (CRNN trained with NAKBA train) answers the "zero-shot only" objection, but
    NAKBA train is not cached locally. Download it to the SSD?

---

## Sources (web checks, 2026-10-13)

- AIN paper: [arXiv 2502.00094](https://arxiv.org/abs/2502.00094v2) ([HTML](https://arxiv.org/html/2502.00094v2)) and
  the [model card](https://huggingface.co/MBZUAI/AIN). It names no OCR training sets; KHATT and KITAB are not mentioned.
- Garrido-Munoz and Calvo-Zaragoza, "On the Generalization of Handwritten Text Recognition Models", CVPR 2025:
  [CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Garrido-Munoz_On_the_Generalization_of_Handwritten_Text_Recognition_Models_CVPR_2025_paper.html),
  [arXiv HTML](https://arxiv.org/html/2411.17332v2). Seven Latin-script datasets, no Arabic; textual divergence
  correlates 0.62 with out-of-distribution error, visual divergence about −0.3.
- TTA on IAM: [arXiv 2307.00664](https://arxiv.org/pdf/2307.00664). Test-time adaptation of line recognisers ("Is it
  an i or an l"): [arXiv 2308.15037](https://arxiv.org/pdf/2308.15037).
- Choosing pretrained HTR models for single-writer fine-tuning: [arXiv 2305.02593](https://arxiv.org/pdf/2305.02593).
- Ahmad and Fink, multi-stage sub-core-shape HMMs (IJDAR 22(3), 2019):
  [KFUPM record](https://pure.kfupm.edu.sa/en/publications/handwritten-arabic-text-recognition-using-multi-stage-sub-core-sh/),
  [2015 paper](https://faculty.kfupm.edu.sa/ics/irfanics/docs/Ahmad2015-MSH.pdf).
- Not web-checked; cited from memory: Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors", ICML
  2022.

**To add to `references.bib`** (mark unverified where only the abstract was read): `garridomunoz2025generalization`,
`kohut2023choose`, `ahmad2019subcore`, `sun2022knnood`, and the two TTA / test-time-adaptation preprints.
