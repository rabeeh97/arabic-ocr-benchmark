# Research map (2026-10-15)

Built from `docs/research/research_map.json` by `scripts/build_research_map.py`; the ledger's top section shows the same content. Every item sits under each question it serves. Grades: A strong and replicated, B solid with one gap, C suggestive, D weak, N settled negative.

## At a glance

- 11 findings graded A
- 23 graded B
- 5 settled negatives
- 15 proposed next
- 39 planned or gated
- 45 not started
- 3 dropped
- 166 items in the map

## Strongest results

- **Ink-band crop before resizing**: KHATT 31.0 → 23.3 % on all 200 lines (−7.5 [−10.6, −4.9], 3 seeds); −7.3 on the standard-split CRNN; 35–36 lines better, none worse; NAKBA and Muharaf unchanged; VLMs not affected
- **NAKBA's alif-hamza is an annotation convention**: About 2 points of NAKBA CER (13.8 → 11.6 % folded); systems write hamza at their training labels' rate (NAKBA-trained 17–20 %, Muharaf-trained 2–5 %); NAKBA test and blind labelled at different rates (16.8 vs 10.6 %)
- **kitab_muharaf is 86.5 % inside Muharaf train**: Tagged in-train; fine-tune scores on it never quoted
- **End-of-line excess is a pipeline artefact (early stop, token caps)**: Gemini-flash truncated on 74 % of lines; fix X6
- **Normalisation ladder: error owed to encoding, orthography, diacritics, dots and spacing**: Every reading in the store is scored at all seven rungs
- **Word-internal spaces fall at PAW gaps**: Above chance with the interval clear of it in 87 of 89 system × set pairs; 1.2–2.9× chance; every family
- **Dot tax and rasm-CER**: Measured on every system and the full panel: dot tax 1–4 points, 6–11 % of CER (see no-dots CER)
- **Do systems confuse letters that sound alike or look alike? (makhārij question)**: Look-alike 4–15× chance in every family; sound-alike at chance, even for the LLM decoder; no historical support for articulatory letter design
- **Positional-form tail (train vs test)**: At most 0.2 % of test letters have a letter × form seen < 50 times; the tail is not the gap at full data
- **CRNN generalisation to unseen sets (aligner recipe)**: Full panel: NAKBA 13.6–13.7 %, KHATT 31.0 % (23.3 % cropped); earlier 26.4 % was the 183-line subset
- **No-dots CER: how much of the error is the dots?**: About 90 % of every family's error survives without dots; dot tax 1–4 points; lines right except dots 0.1–4 %; three-dot letters the most error-prone (10–20 per 100)

## Proposed next round

- `R4.1` Label review in the review tool (458 candidates) and the 20 KHATT hard-core lines — Needs about 2 h of human review: a label-error rate per set, κ with CER-HV, and a 'can a human read it?' field for the hard-core lines (a human ceiling) (T0, T10)
- `PAW-gap-width` Are the PAW gaps that get split as wide as real word gaps? (pixel gaps from the letter dumps) — CPU, from existing letter spans (T1, T5)
- `P-L2` PAW-space repair: PAW-aware decoding or lexicon merge of a space after a non-joining letter — Needs CTC posteriors for the decoding version (minutes on the GPU) (T1, T6)
- `P-L1` Dot-restoration ceiling and H(dots | rasm, context) — First evidence (§16): the most frequent word restores 95 % of known words; folded into the two-stage reader experiment (T2, T6)
- `novel-clusters` What are the novel shapes? Allograph clusters of novel KHATT/NAKBA letters against training — CPU, from the letter dumps (T3)
- `shape-nakba` Shape targeting for a second dataset: select on NAKBA blind half A, test on half B (2 seeds vs random) — About 4 GPU hours (T3, T8)
- `E6` Maghrebi adaptation: few-shot curve (16 / 100 / 300 lines), prompt hint, dot-swap augmentation — About 1.5 GPU hours for the CRNN curve (T4, T8)
- `lm-decoding` CTC decoding with character language models (Muharaf text, NAKBA text, modern Arabic sample) — 15 GPU minutes to dump posteriors, then CPU; tests whether text distance is fixable (T6, T1, T2)
- `nakba-indomain` In-domain control: CRNN trained on NAKBA train (data on the box, verified) — 1–2 seeds, 1–2.5 GPU hours; score with and without train/test text overlaps (T8, T0)
- `two-stage-reader` Two-stage reader: skeleton-only CRNN, then a dot restorer constrained to the read skeleton (n-gram first, then a small LLM) — Does a dotless recogniser read skeletons better, and does constrained restoration beat the model's own dots? Known vs unseen words separately; about 1 GPU hour + CPU (T2, T6, T8)
- `shape-library` Allograph library (shape atlas): letter glyphs with skeleton, separate dots and entry/exit join points; alignment + ink skeleton extraction; cycle-consistency quality gate — Foundation of synthesis, shape tokens and personalised adaptation; also a characterisation result (shapes per letter × form per dataset); CPU, a few days (T3, T10, T11)
- `writer-adapt-crnn` Writer adaptation for the CRNN: RASAM-2 manuscripts and NAKBA; K = 4, 8, 16 lines chosen at random, to cover the most letter forms (set cover), or richest in the writer's novel shapes; fine-tune, adapter, or a no-training prototype memory — Short GPU runs; tests 'show a few samples that capture most letters' (T7, T8, T4)
- `vlm-incontext-writer` Writer calibration for VLMs in context: one prompt with a labelled reference line from the same writer (image + label), then the test line; vs the test line alone — Qwen3-VL-8B base and fine-tuned; RASAM-2 manuscripts, NAKBA, Muharaf pages; reference chosen at random vs covering most letters; needs a VLM on the 3090 (T7, T8, T10)
- `khatt-full` Full KHATT v1.0 (Teklia copy): add its 1,038 test lines to the panel; an in-domain KHATT control from its 4,672 train lines — Downloaded to the SSD. Shared fixed paragraphs: 767 unique texts in 1,038 test lines, and texts recur across splits, so score fixed-text and unique-text lines separately; no writer IDs (T0, T7, T8)
- `decision-rules` Decision rule and dependencies for every open item; a claims register per write-up (claim, evidence, what is missing) — Done in PLAN-2026-10-15 for the proposed items (T11)

## T0 · Are the numbers right? Benchmark and measurement validity

*Which measured differences are pipeline, labelling or scoring artefacts rather than properties of the script or the model?*

The strongest results are here. Loose KHATT crops cost fixed-height models 7.5 CER points, and an old length filter hid the worst of them. Hamza and digit conventions differ between label sets, and even between NAKBA's two test splits. KHATT labels carry spacing quirks. 11 % of NAKBA blind lines repeat training text and are memorised by NAKBA-trained systems, though corpus scores barely move. A PaddleOCR flag scrambled its output; kitab_muharaf is mostly Muharaf train. The per-line store, page-cluster intervals and the corrections log now guard every number. Open: label-noise rate and human readability (needs review), near-duplicates, the diacritic scoring rule.

**What we found**

- [A] `X-S1` Ink-band crop before resizing: KHATT 31.0 → 23.3 % on all 200 lines (−7.5 [−10.6, −4.9], 3 seeds); −7.3 on the standard-split CRNN; 35–36 lines better, none worse; NAKBA and Muharaf unchanged; VLMs not affected
- [A] `T11-hamza` NAKBA's alif-hamza is an annotation convention: About 2 points of NAKBA CER (13.8 → 11.6 % folded); systems write hamza at their training labels' rate (NAKBA-trained 17–20 %, Muharaf-trained 2–5 %); NAKBA test and blind labelled at different rates (16.8 vs 10.6 %)
- [A] `X1` kitab_muharaf is 86.5 % inside Muharaf train: Tagged in-train; fine-tune scores on it never quoted
- [A] `R2.8` End-of-line excess is a pipeline artefact (early stop, token caps): Gemini-flash truncated on 74 % of lines; fix X6
- [A] `R2.1` Normalisation ladder: error owed to encoding, orthography, diacritics, dots and spacing: Every reading in the store is scored at all seven rungs
- [A] `R1.7` Positional-form tail (train vs test): At most 0.2 % of test letters have a letter × form seen < 50 times; the tail is not the gap at full data
- [A] `no-dots-cer` No-dots CER: how much of the error is the dots?: About 90 % of every family's error survives without dots; dot tax 1–4 points; lines right except dots 0.1–4 %; three-dot letters the most error-prone (10–20 per 100)
- [B] `X2-clean` Cleaned Muharaf (CER-HV flags): removing flagged test lines moves CER < 1 point, no rank change
- [B] `family-matrix` Line × system matrix: CRNN and VLM difficulty correlate weakly: KHATT ρ ≈ 0.35; 20 hard-core lines; most hard lines are hard for one family
- [B] `std-split` Standard-split CRNN (no test lines in training): Muharaf validation 11.8 %; KHATT 33.4–33.9 %, cropped 26.0–26.7 %; the comparable reference
- [B] `R1.6` Lexical diversity per set (unique lines, duplication, Jaccard, TTR): KHATT most repetitive (mean max Jaccard 0.39, fixed-text paragraphs); NAKBA train duplicates 17.5 % of lines
- [B] `foundations` Characterisation foundations: GT recovery (R0.1), canonical runs (R0.3), positional-form analyser and profiler (R1.1, R1.2), diacritic-mismatch tax (R2.3), error taxonomy (R2.4), consensus quality groups (R3.1), exact leakage (R4.0), image layer first pass (R6.0), Wave-0 literature (L0)
- [B] `nakba-leak` NAKBA's official split repeats training text: 11 % of blind lines (2.9 % of characters) repeat NAKBA-train text; Baseer reads them at 0.9–1.5 % CER against 7.1 % on clean lines, unseen-data CRNNs at ~50 % (short fragments); corpus CER moves 6.91 → 7.08 %
- [B] `conventions` Label convention fingerprint per set: NAKBA test vs blind labelled differently (hamza 16.8 vs 10.6 %, Arabic-Indic digits 84 vs 38 %); KHATT double spaces on 34 % of lines and a space before 51 % of punctuation
- [C] `dialect-spelling` Writers spell as they speak: dialect mergers in the labels: Muharaf train labels hold Levantine spellings (متل for مثل 52×, كتير for كثير 16×, ضهر for ظهر 15×); articulation shows in the writers' spelling, not the models' errors; automatic pairing is noisy
- [–] `detectors-done` Pattern detectors implemented: D1 loose crop, D2 slant, D3 ruled, D5 low resolution, D6 stroke/pen, D7 fragment, D8 letter list, D9 unseen words, D10 label suspect, D10b label tail, D11 truncation, D12 loop, D13 whole-line, D14 script mismatch, D15 fluent rewrite, D16 diacritic insertion, D17 orthographic only, D18 dot only, D19 Maghrebi dots, D20 PAW-gap space, D22 reversed reading

**Proposed next round**

- `R4.1` (experiment) Label review in the review tool (458 candidates) and the 20 KHATT hard-core lines: Needs about 2 h of human review: a label-error rate per set, κ with CER-HV, and a 'can a human read it?' field for the hard-core lines (a human ceiling) — source: alazzawi2026labelerr — also in T10
- `nakba-indomain` (experiment) In-domain control: CRNN trained on NAKBA train (data on the box, verified): 1–2 seeds, 1–2.5 GPU hours; score with and without train/test text overlaps — source: plan open question 10 — also in T8
- `khatt-full` (experiment) Full KHATT v1.0 (Teklia copy): add its 1,038 test lines to the panel; an in-domain KHATT control from its 4,672 train lines: Downloaded to the SSD. Shared fixed paragraphs: 767 unique texts in 1,038 test lines, and texts recur across splits, so score fixed-text and unique-text lines separately; no writer IDs — source: Teklia/KHATT_v1.0, mahmoud2014khatt — also in T7, T8

**Planned**

- `R2.7` (experiment) Diacritic scoring rule: rank correlation with and without marks: Base VLMs pay up to 10 CER points for correct marks — source: doctor2022graphemic, heakl2025kitab
- `R3.3` (experiment) Challenge subsets (hard core, dot-dense, OOV-heavy, fragments) as dataset filters: Hard core (20 KHATT lines), family-split lines and detector flags now exist — source: NEXT_STEPS — also in T11
- `detector-residual` (experiment) Review the residual: worst lines no detector explains, and write new detectors: 1–17 % of each run's worst fifth unexplained — source: plan §3.5 — also in T11
- `detectors-open` (experiment) Detectors still to write: D4 neighbour-line intrusion, D23 memorisation / contamination signature, D24 numeric instability (CRNN), D21 lam-alef ligature — source: plan §3.5 — also in T11
- `paper-groupings` (design) Write-up groupings: P1 benchmark audit, P2 characterisation across families (the main direction), P3 methods (geometry transfer, targeted data): User view 2026-10-13: nothing is main-conference grade yet; P2 becomes possible once the cross-family evidence is in — source: PLAN-2026-10-13 §2.3 — also in T11
- `seed-instability` (experiment) Lines that flip between identical trainings (3 seeds): characterise them and add a detector — source: ours (review 2026-10-14) — also in T8

**Parked**

- `C6` (check) AIN-7B suspiciously strong on KITAB-KHATT (6.4 % vs ≥ 20.7 %), copies GT whitespace quirks: Unverified (AIN's paper names no training sets); AIN left out of cross-model claims — source: planning agent; user decision 2026-10-13

**Not started (borrowed or ours)**

- `X6` (fix) Output-token caps for API and VLM runs: Needed before scoring the truncated runs — source: NEXT_STEPS X6
- `R4.2` (experiment) Near-duplicate leakage (MinHash text plus perceptual image hash) — source: baek2019wrong
- `R4.3` (experiment) Spelling-inconsistency report (same word spelled several ways in one set) — source: doctor2022graphemic — also in T4
- `R4.4` (experiment) Per-dataset conventions table from the dataset papers — source: NEXT_STEPS — also in T4
- `X5-len` (fix) Length-stratified reporting: fragments vs full lines on the leaderboard: Short lines fail disproportionately (also a cross-family predictor) — source: NEXT_STEPS X5 — also in T11
- `R3.2` (experiment) IRT 2PL difficulty and discrimination per line — source: rodriguez2021irt — also in T8
- `X3-retrain` (experiment) Retrain without the 1,424 CER-HV-flagged Muharaf train lines — source: NEXT_STEPS X3 — also in T8
- `X7` (check) Do full-resolution Muharaf scans exist (public copy stored at 60 px)? — source: NEXT_STEPS X7, rice1995isri — also in T5
- `aug-20` (idea) Augmentation 20: fragment crops for short lines — source: ours — also in T5
- `P-L6` (experiment) Per-sample loss logging (dataset cartography) — source: swayamdipta2020cartography — also in T8
- `R1.3-R1.4` (experiment) Profiler: diacritic-policy share per set (R1.3); datasheet cards on the Atlas (R1.4) — source: NEXT_STEPS — also in T11
- `aug-18` (idea) Augmentation 18: harakat on/off renders with matching label — source: ours — also in T6

**Done**

- `C3` (fix) The old evaluation silently dropped 17 KHATT lines (CTC length filter applied before the crop): Every earlier KHATT CRNN number was on an easier 183-line subset; the panel scores all lines — source: planning agent — also in T11
- `C4` (fix) The aligner CRNN trained on most Muharaf and RASAM test lines: In-train lines flagged and excluded everywhere; standard-split CRNN trained (2 seeds) — source: planning agent — also in T8
- `C5` (fix) PaddleOCR output scrambled by our reverse_output flag: Un-reversed: KHATT 85.3 → 63.0 % CER; patch for the main checkout written (not applied there yet); TrOCR not reversed — source: store reversal detector
- `C2` (fix) Two different NAKBA blind samples in earlier results: The panel scores all 2,671 lines — source: planning agent
- `C7` (fix) CRNN readings depended on batch padding and numerics: Panel evaluator scores one line at a time in fp32 — source: planning agent — also in T11
- `X4-names` (fix) Model-name hygiene (runs launched under another BASES entry): Handled by MODEL_RENAMES in characterize/runs.py — source: NEXT_STEPS X4 — also in T11


## T1 · Connectivity, PAWs and word spacing

*Does Arabic's split of words into parts (PAWs) make word boundaries ambiguous to recognisers, and is that visual?*

Every family inserts word-internal spaces at PAW gaps far more than chance (87 of 89 pairs). Spaces cost more than dots: the space tax is 2–11 CER points against 1–4. PAWs per word was word length. Open: are split PAW gaps as wide as word gaps (pixel measurement); PAW-aware decoding; PAW tokens for a shape vocabulary.

**What we found**

- [A] `V5-paw` Word-internal spaces fall at PAW gaps: Above chance with the interval clear of it in 87 of 89 system × set pairs; 1.2–2.9× chance; every family
- [B] `space-tax` Spaces cost more than dots: Space tax 2–11 CER points against a dot tax of 1–4; largest for base VLMs
- [N] `paw-per-word` PAWs per word as a difficulty predictor: It was word length: not significant for CRNNs once letters per word is controlled

**Proposed next round**

- `PAW-gap-width` (experiment) Are the PAW gaps that get split as wide as real word gaps? (pixel gaps from the letter dumps): CPU, from existing letter spans — source: chat 2026-10-14 — also in T5
- `P-L2` (experiment) PAW-space repair: PAW-aware decoding or lexicon merge of a space after a non-joining letter: Needs CTC posteriors for the decoding version (minutes on the GPU) — source: alluhaybi2020connect — also in T6
- `lm-decoding` (experiment) CTC decoding with character language models (Muharaf text, NAKBA text, modern Arabic sample): 15 GPU minutes to dump posteriors, then CPU; tests whether text distance is fixable — source: plan P2.8, garridomunoz2025 — also in T6, T2

**Planned**

- `A1` (experiment) Augmentation ablation (items 13, 14, 16, 17 against the 1–4 baseline) — source: NEXT_STEPS A1 — also in T5, T2, T4

**Not started (borrowed or ours)**

- `P-L3` (experiment) PAW-inventory coverage curve: distinct PAWs for 95 / 99 % coverage — source: lehal2013urdu — also in T3
- `aug-14` (idea) Augmentation 14: PAW-gap jitter (widen or narrow gaps between word parts) — source: ours
- `D21` (check) Lam-alef ligature errors (detector planned) — source: plan §3.5


## T2 · Dots: skeleton (rasm) against dots (i'jam)

*How much of the error is dots, at what resolution do dots get lost, and can context restore them?*

Without dots, about nine tenths of every family's error remains: models misread skeletons, not dots, and lines that are right except for their dots are rare (0.1–4 %). Three-dot letters are the most dot-error-prone. A quarter to a third of words share their dotless form with another word, yet the most frequent word restores 95 % of known words. The dots matter for the fifth of words that are unseen, which echoes why iʿjām was invented. Confusions are visual, not phonetic. Open: a two-stage reader (skeleton recogniser plus constrained dot restorer), dot-writing styles (dash, caret, displaced), and a dot-preserving resize below 48 px.

**What we found**

- [A] `R2.2` Dot tax and rasm-CER: Measured on every system and the full panel: dot tax 1–4 points, 6–11 % of CER (see no-dots CER)
- [A] `makharij` Do systems confuse letters that sound alike or look alike? (makhārij question): Look-alike 4–15× chance in every family; sound-alike at chance, even for the LLM decoder; no historical support for articulatory letter design
- [A] `no-dots-cer` No-dots CER: how much of the error is the dots?: About 90 % of every family's error survives without dots; dot tax 1–4 points; lines right except dots 0.1–4 %; three-dot letters the most error-prone (10–20 per 100)
- [B] `R2.6` VLMs make as many similar-letter errors per character as the CRNN; their share is lower
- [B] `R2.9` Maghrebi dot convention: base AIN reads ق as ف on 23–27 % of RASAM qāfs (2–3 % elsewhere): Fine-tuning on Muharaf cuts it to 4–10 %
- [B] `space-tax` Spaces cost more than dots: Space tax 2–11 CER points against a dot tax of 1–4; largest for base VLMs
- [B] `ijam-ambiguity` How ambiguous is Arabic without dots? (why iʿjām was invented): 25–30 % of tokens share their dotless form with another word; the most frequent word restores 95 % of known words, one word of context about +1; 19 % of held-out tokens unseen, where dots carry what context cannot

**Proposed next round**

- `P-L1` (experiment) Dot-restoration ceiling and H(dots | rasm, context): First evidence (§16): the most frequent word restores 95 % of known words; folded into the two-stage reader experiment — source: alshaibani2025dotless, shannon1951 — also in T6
- `lm-decoding` (experiment) CTC decoding with character language models (Muharaf text, NAKBA text, modern Arabic sample): 15 GPU minutes to dump posteriors, then CPU; tests whether text distance is fixable — source: plan P2.8, garridomunoz2025 — also in T6, T1
- `two-stage-reader` (experiment) Two-stage reader: skeleton-only CRNN, then a dot restorer constrained to the read skeleton (n-gram first, then a small LLM): Does a dotless recogniser read skeletons better, and does constrained restoration beat the model's own dots? Known vs unseen words separately; about 1 GPU hour + CPU — source: user idea 2026-10-14; ahmad2019subcore, alshaibani2025dotless — also in T6, T8

**Planned**

- `X-G7` (experiment) Dot-preserving resize and a dot input channel at 32–48 px: CPU proxy: dots survive at 64 px, lost at 32 px; model test open — source: ours (pool #2) — also in T5
- `P-L4` (experiment) Similarity-choice model of confusions (Müller-style); human vs model confusability: Same-definition part done 2026-10-09 — source: mueller2012letter — also in T3
- `R9` (experiment) Resolution and noise probe: KHATT at 60 px and three noise levels, CRNN and 3 VLMs — source: rice1995isri, hegghammer2021noisy, rethinking2021 — also in T5
- `A1` (experiment) Augmentation ablation (items 13, 14, 16, 17 against the 1–4 baseline) — source: NEXT_STEPS A1 — also in T5, T1, T4
- `dot-styles` (experiment) Dot-writing styles: two dots as a dash, three as a caret, displaced (delayed) dots, dots touching the body; prevalence per set and effect on dot errors — source: ours (review 2026-10-14) — also in T5
- `loops-teeth` (experiment) Skeleton features that fail: loops (ف ق و م ه ع), teeth (ب ت ث ن ي, س ش), tails; tooth-count errors against run length (like minims in Latin): Top confusions are loop and tooth pairs (ح→ع, ف→م, و→م, ن→ل) — source: ours (review 2026-10-14) — also in T3
- `shape-tokens` (experiment) Shape-token recognition: the reader emits shapes (dotless class × form, discovered clusters, or a shape BPE of PAW-sized tokens); a decoder picks letters under Arabic's joining grammar (lexicon / n-gram, then transducer, then a constrained small LLM; top-k beams, speculative decoding): Hypothesis: separating 'what the ink looks like' from 'which letter' generalises better to new writers; readout: KHATT novelty bands — source: user idea 2026-10-14; ahmad2019subcore, alazzawi2026gap (positional labels did not help) — also in T3, T6, T8

**Parked**

- `zhang2020ran` (literature) Separate dots head (radical analysis): R2.6 says no case except style-specific dots (E6) — source: zhang2020ran

**Not started (borrowed or ours)**

- `X-G9` (experiment) Factorised output: rasm CTC plus a dot-pattern head (also C3) — source: ours (pool #4), ahmad2019subcore — also in T3
- `X-A6` (experiment) Do attribution maps land on dots? — source: ours — also in T9
- `E2` (experiment) Tokenizer view: do dot-pair swaps get unrelated token sequences; fertility vs CER — source: alomari2026iconicity — also in T6
- `aug-16` (idea) Augmentation 16: dot minimal-pair renders with new label — source: ours
- `aug-17` (idea) Augmentation 17: Maghrebi dot-convention swap — source: ours — also in T4
- `geometry-lit` (literature) Untouched geometry follow-ups: second baseline estimator (Abu-Ain), AHTID/MW baseline ground truth (Mezghani), polygon-curvature descriptor (Abdi), contour codebooks (Ahmed), FATF curve features (Parvez), dots validated on CTC spans — source: paper ledger, geometry group — also in T5

**Dropped**

- `aug-15` (idea) Augmentation 15: dot dropout: Breaks image–label agreement; small displacement only — source: ours


## T3 · Letter shapes, positional forms and allographs

*Which letter shapes are hard, why, and does adding the missing shapes help?*

Letter-shape novelty passes the confound battery (within-line OR ≈ 16 for CRNNs, 4.4 for fine-tuned VLMs); pixel novelty is weak. Shape-targeted repeats beat random on KHATT, including held-out lines, but nowhere else. The positional-form tail is covered. Since most error is skeleton misreading, the next questions are which skeleton features fail (loops, teeth, tails, vertical stacking) and the shape programme: an allograph library with joins and dots kept separate, synthesis from it, and shape-token recognition with a constrained decoder. Feasibility: CTC spans are anchors, not cut lines, and handwritten PAWs often touch, so extraction needs alignment and ink together.

**What we found**

- [A] `R1.7` Positional-form tail (train vs test): At most 0.2 % of test letters have a letter × form seen < 50 times; the tail is not the gap at full data
- [A] `no-dots-cer` No-dots CER: how much of the error is the dots?: About 90 % of every family's error survives without dots; dot tax 1–4 points; lines right except dots 0.1–4 %; three-dot letters the most error-prone (10–20 per 100)
- [B] `X-G10` Letter-shape novelty predicts letter errors: Within-line OR ≈ 16 (CRNN), 4.4 (fine-tuned VLMs); holds within letter × form, on cropped lines, without letter lists, from a model that never saw the lines; pixel novelty weak
- [B] `X-G11L` Shape-targeted training repeats (no target labels): KHATT −0.83 [−1.31, −0.33] vs random repeats (3 seeds, every targeted seed better); held-out half B −0.93; no effect on NAKBA, Muharaf, RASAM
- [B] `X-B1` CTC character spans from the CRNN aligner: Basis of the letter dumps; VML-HD validation open
- [B] `shape-feasibility` Can real letter shapes be cut out? (feasibility of the shape programme): CTC spans are narrow anchors (median 12 px at 64 px; 9 % one column) that often miss parts of a letter; large ink components match label PAWs exactly on only 6–9 % of lines (PAWs touch)
- [N] `X-S2` Combined recipe (cropped training + shape repeats): Equals shape repeats + test-time crop; training on cropped lines adds nothing; its NAKBA gain is a repetition effect

**Proposed next round**

- `novel-clusters` (experiment) What are the novel shapes? Allograph clusters of novel KHATT/NAKBA letters against training: CPU, from the letter dumps — source: chat 2026-10-14
- `shape-nakba` (experiment) Shape targeting for a second dataset: select on NAKBA blind half A, test on half B (2 seeds vs random): About 4 GPU hours — source: plan P2.1 — also in T8
- `shape-library` (experiment) Allograph library (shape atlas): letter glyphs with skeleton, separate dots and entry/exit join points; alignment + ink skeleton extraction; cycle-consistency quality gate: Foundation of synthesis, shape tokens and personalised adaptation; also a characterisation result (shapes per letter × form per dataset); CPU, a few days — source: user idea 2026-10-14; elarian2015 (KFUPM), dinges2016synthesis, bulacu2007allographic — also in T10, T11

**Planned**

- `hard-example-control` (experiment) Hard-example control: repeat the highest-loss training lines instead of shape-matched ones — source: plan P2.1 — also in T8
- `P-L4` (experiment) Similarity-choice model of confusions (Müller-style); human vs model confusability: Same-definition part done 2026-10-09 — source: mueller2012letter — also in T2
- `P1.8` (experiment) KHATT fixed-text natural experiment (same text, several writers) — source: characteristics-framework V3 — also in T7
- `R8` (experiment) Decision experiments (gate V5): coverage-balanced selection vs random; transformation choice: Shape targeting is a first instance of (a) — source: characteristics-framework §7 — also in T10, T11
- `loops-teeth` (experiment) Skeleton features that fail: loops (ف ق و م ه ع), teeth (ب ت ث ن ي, س ش), tails; tooth-count errors against run length (like minims in Latin): Top confusions are loop and tooth pairs (ح→ع, ف→م, و→م, ن→ل) — source: ours (review 2026-10-14) — also in T2
- `vertical-stacking` (experiment) Vertical stacking of letters in handwriting (as in محمد, لمح) and reading-order errors; detector from overlapping spans — source: ours (review 2026-10-14) — also in T5
- `synthesis-v1` (experiment) Synthesis from the library: join glyphs at their connection points, re-dot, measured PAW and word gaps, geometry and background variation; text from any source; sampling random / novelty-targeted / one coherent style: Realism checks (real-vs-synthetic classifier, geometry match, no seam shortcut), then CRNN training against real repeats, a StackMix-style naive cut and font renders; readouts: novelty bands and unseen words — source: user idea 2026-10-14; elarian2015, stackmix2021, dinges2016synthesis — also in T6, T5, T10
- `shape-tokens` (experiment) Shape-token recognition: the reader emits shapes (dotless class × form, discovered clusters, or a shape BPE of PAW-sized tokens); a decoder picks letters under Arabic's joining grammar (lexicon / n-gram, then transducer, then a constrained small LLM; top-k beams, speculative decoding): Hypothesis: separating 'what the ink looks like' from 'which letter' generalises better to new writers; readout: KHATT novelty bands — source: user idea 2026-10-14; ahmad2019subcore, alazzawi2026gap (positional labels did not help) — also in T6, T8, T2
- `lexicon` (idea) A comprehensive Arabic lexicon with proper names (اسم علم): words, person names, cities, countries, places: Sources: Ayaspell (~300k forms, GPL/LGPL/MPL), aspell-ar (~900k, Buckwalter-based), Pecina et al. LREC 2012 (9M forms; location and licence to check), Alotaibi & Lee fine-grained NE corpora, ANETAC (79,924 names), Wikidata / GeoNames filtered by Arabic label, Arabic Wikipedia. Uses: synthesis text, lexicon-constrained decoding, dot restoration, OOV analysis — source: user idea 2026-10-15 — also in T6, T10

**Gated on something else**

- `X-G11` (experiment) Allograph-stitching synthesis for rare shapes and style swaps: Now part of the shape programme: gated on the allograph library (synthesis-v1) — source: ours (pool #6), dinges2016synthesis — also in T4
- `X-D1` (experiment) Freeze-by-letter: protect weights that detect each letter group (wild) — source: ours (ideas #3) — also in T9
- `render-refine` (experiment) Render, then refine: a small image model trained on our pairs turns a stitched skeleton into realistic ink (ControlNet-like), label kept exact: Only if seams in synthesis-v1 are visible or exploited — source: ours — also in T5
- `personalised-synthesis` (experiment) Personalised synthesis: generate many lines in one writer's own allographs from their K samples, then fine-tune: Joins all three shape ideas; gated on the library and synthesis-v1 — source: user idea 2026-10-14 — also in T7

**Parked**

- `C-track` (experiment) Curriculum: letters first, then words (C0 data, C1 vs controls, C2 letter-incremental, C3 factorised head, C4 averaging) — source: NEXT_STEPS C-track — also in T8

**Not started (borrowed or ours)**

- `P-L3` (experiment) PAW-inventory coverage curve: distinct PAWs for 95 / 99 % coverage — source: lehal2013urdu — also in T1
- `X-G9` (experiment) Factorised output: rasm CTC plus a dot-pattern head (also C3) — source: ours (pool #4), ahmad2019subcore — also in T2
- `X-A2b` (experiment) Larger letter-family sample (the خ lead) — source: attribution spike — also in T9
- `X-D0` (experiment) Per-module gradient probe by letter family (no training) — source: ours (ideas #2) — also in T9
- `aug-10` (idea) Augmentation 10: rare-character oversampling — source: eltay2022adaptive (unverified)
- `aug-19` (idea) Augmentation 19: coverage-balanced text for synthetic lines — source: ours — also in T10
- `slant-density` (idea) Slant-density score for data selection — source: ours (pool #13) — also in T5
- `aug-5-7-8-9-12-13` (idea) Augmentations 5 learned warping, 7 GAN/diffusion, 8 font renders, 9 glyph concatenation, 12 downscale to 60 px, 13 kashida stretching — source: luo2020learn, fogel2020scrabblegan, bilgintasdemir2023ottoman, dinges2016synthesis, ours — also in T5
- `E3` (experiment) Data scaling on the K grid, in characters per letter × form (E3, E3′) — source: alazzawi2026gap — also in T8, T10

**Done**

- `pos-labels` (literature) Explicit positional-form labels did not help (gap paper) — source: alazzawi2026gap


## T4 · Script styles and transcription conventions

*How do style conventions (Maghrebi dots, hamza writing) change what models must learn, and how fast can they learn them?*

Maghrebi qāf/fāʾ convention (base AIN 23–27 % ق→ف). Without Maghrebi training a CRNN reads RASAM at ~47 % CER instead of ~12 %. Hamza use follows the training labels. Writers spell as they speak (Levantine متل for مثل in Muharaf labels). Open: a few-shot Maghrebi curve, dot-swap augmentation, the style inventory.

**What we found**

- [A] `T11-hamza` NAKBA's alif-hamza is an annotation convention: About 2 points of NAKBA CER (13.8 → 11.6 % folded); systems write hamza at their training labels' rate (NAKBA-trained 17–20 %, Muharaf-trained 2–5 %); NAKBA test and blind labelled at different rates (16.8 vs 10.6 %)
- [B] `R2.9` Maghrebi dot convention: base AIN reads ق as ف on 23–27 % of RASAM qāfs (2–3 % elsewhere): Fine-tuning on Muharaf cuts it to 4–10 %
- [B] `maghrebi-gap` Without Maghrebi training lines a CRNN reads RASAM at ~47 % CER (vs ~12 % with them): Standard-split CRNN, 2 seeds
- [B] `conventions` Label convention fingerprint per set: NAKBA test vs blind labelled differently (hamza 16.8 vs 10.6 %, Arabic-Indic digits 84 vs 38 %); KHATT double spaces on 34 % of lines and a space before 51 % of punctuation
- [C] `dialect-spelling` Writers spell as they speak: dialect mergers in the labels: Muharaf train labels hold Levantine spellings (متل for مثل 52×, كتير for كثير 16×, ضهر for ظهر 15×); articulation shows in the writers' spelling, not the models' errors; automatic pairing is noisy

**Proposed next round**

- `E6` (experiment) Maghrebi adaptation: few-shot curve (16 / 100 / 300 lines), prompt hint, dot-swap augmentation: About 1.5 GPU hours for the CRNN curve — source: vidalgorene2024rasam2, kohut2023finetuning, farazi2026vlmocr — also in T8
- `writer-adapt-crnn` (experiment) Writer adaptation for the CRNN: RASAM-2 manuscripts and NAKBA; K = 4, 8, 16 lines chosen at random, to cover the most letter forms (set cover), or richest in the writer's novel shapes; fine-tune, adapter, or a no-training prototype memory: Short GPU runs; tests 'show a few samples that capture most letters' — source: user idea 2026-10-14; MetaHTR, kohut2023finetuning — also in T7, T8

**Planned**

- `A1` (experiment) Augmentation ablation (items 13, 14, 16, 17 against the 1–4 baseline) — source: NEXT_STEPS A1 — also in T5, T1, T2

**Gated on something else**

- `X-G11` (experiment) Allograph-stitching synthesis for rare shapes and style swaps: Now part of the shape programme: gated on the allograph library (synthesis-v1) — source: ours (pool #6), dinges2016synthesis — also in T3

**Parked**

- `S1-styles` (design) Script-styles inventory (traits, datasets, samples, licences) — source: ideas #7 — also in T10

**Not started (borrowed or ours)**

- `R4.3` (experiment) Spelling-inconsistency report (same word spelled several ways in one set) — source: doctor2022graphemic — also in T0
- `R4.4` (experiment) Per-dataset conventions table from the dataset papers — source: NEXT_STEPS — also in T0
- `aug-17` (idea) Augmentation 17: Maghrebi dot-convention swap — source: ours — also in T2
- `aug-21` (idea) Augmentation 21: style mixing across Maghrebi, Naskh, Ajami — source: ours
- `pen-contrast-aug` (idea) Pen-contrast augmentation (imitate a cut nib): RASAM pen contrast is an outlier for 50–68 % of lines — source: ours (pool #12) — also in T5


## T5 · Writing geometry and image conditions

*Which geometric and image properties (crop, slant, stroke, resolution, rules, tone) change recognition, and for which model families?*

Ink crop: −7.5 points on KHATT for every CRNN. Slant is causal but small, and slant TTA gated on measured slant is nearly free. Slant/stroke augmentation hurts on unseen sets. Tone, noise and ruled lines: no effect. Jointly, pixels per character and line length predict errors in every family; loose crops and slant only for CRNNs. Open: normalisation ablation (height, padding, deslant), resolution probe, a VLM check on crop and slant.

**What we found**

- [A] `X-S1` Ink-band crop before resizing: KHATT 31.0 → 23.3 % on all 200 lines (−7.5 [−10.6, −4.9], 3 seeds); −7.3 on the standard-split CRNN; 35–36 lines better, none worse; NAKBA and Muharaf unchanged; VLMs not affected
- [B] `X-T2` Geometry transfer: slant is causal but small; stroke thinning helps NAKBA; tone, noise and rules no effect: KHATT as Muharaf (slant only) −1.1; Muharaf upright +2.0
- [B] `X-G1-6` Geometry measures per line (loci, slant, baseline, hinge, stroke, dots) and per-dataset profiles: KHATT off the training distribution in slant (34 %) and stroke (46 %); RASAM in pen contrast (50–68 %); NAKBA ruled lines (16–24 %); contour histograms identify the dataset but not the gap
- [B] `validity` Which characteristics predict errors jointly (gate V2): Every family: few pixels per character, short lines, unseen words; long words help. Loose crop and slant only for CRNNs; pen contrast only for fine-tuned VLMs. Low explained variance, poor leave-one-set-out transfer
- [C] `X-G8` Slant TTA with confidence selection, gated on measured slant: KHATT −0.85; ungated costs NAKBA +0.4; gated: +0.01 cost, keeps 61–69 % of the gain (rule needs 70 %)
- [N] `X-S2` Combined recipe (cropped training + shape repeats): Equals shape repeats + test-time crop; training on cropped lines adds nothing; its NAKBA gain is a repetition effect
- [N] `X-T3` Slant and stroke augmentation in training: Hurts on unseen sets (NAKBA +0.8 [+0.7, +1.0]); slant-robust in-domain

**Proposed next round**

- `PAW-gap-width` (experiment) Are the PAW gaps that get split as wide as real word gaps? (pixel gaps from the letter dumps): CPU, from existing letter spans — source: chat 2026-10-14 — also in T1

**Planned**

- `X-G7` (experiment) Dot-preserving resize and a dot input channel at 32–48 px: CPU proxy: dots survive at 64 px, lost at 32 px; model test open — source: ours (pool #2) — also in T2
- `X-G3` (experiment) Normalisation ablation: height 48/64/96 × pad vs resize, deslant, binarisation (2 seeds) — source: retsinas2022best, liebl2020crnn, vinciarelli2001norm
- `R9` (experiment) Resolution and noise probe: KHATT at 60 px and three noise levels, CRNN and 3 VLMs — source: rice1995isri, hegghammer2021noisy, rethinking2021 — also in T2
- `vlm-geometry` (experiment) VLM check: fine-tuned Qwen3-VL-8B on cropped and slant-sheared KHATT: 4–6 GPU hours, more setup; decides whether crop and slant effects are CRNN-only — source: plan P2.3 — also in T8
- `A1` (experiment) Augmentation ablation (items 13, 14, 16, 17 against the 1–4 baseline) — source: NEXT_STEPS A1 — also in T1, T2, T4
- `E5-retsinas` (experiment) Retsinas-recipe CRNN (column max-pool, CTC shortcut, median padding) — source: retsinas2022best — also in T8
- `R6-R11` (experiment) Image layer second pass (R6), causal synthetic probes (R9, gate V3), release (R11, gate V6) — source: NEXT_STEPS, characterization-plan — also in T11
- `dot-styles` (experiment) Dot-writing styles: two dots as a dash, three as a caret, displaced (delayed) dots, dots touching the body; prevalence per set and effect on dot errors — source: ours (review 2026-10-14) — also in T2
- `vertical-stacking` (experiment) Vertical stacking of letters in handwriting (as in محمد, لمح) and reading-order errors; detector from overlapping spans — source: ours (review 2026-10-14) — also in T3
- `synthesis-v1` (experiment) Synthesis from the library: join glyphs at their connection points, re-dot, measured PAW and word gaps, geometry and background variation; text from any source; sampling random / novelty-targeted / one coherent style: Realism checks (real-vs-synthetic classifier, geometry match, no seam shortcut), then CRNN training against real repeats, a StackMix-style naive cut and font renders; readouts: novelty bands and unseen words — source: user idea 2026-10-14; elarian2015, stackmix2021, dinges2016synthesis — also in T3, T6, T10

**Gated on something else**

- `pix2pix` (literature) Learned image normalisation (pix2pix): Only after X-G3 shows normalisation helps — source: pix2pix-normalisation2024
- `render-refine` (experiment) Render, then refine: a small image model trained on our pairs turns a stitched skeleton into realistic ink (ControlNet-like), label kept exact: Only if seams in synthesis-v1 are visible or exploited — source: ours — also in T3

**Parked**

- `pen-trajectory` (idea) Pen-trajectory recovery with sigma-lognormal primitives (long shot) — source: ours (pool #11), plamondon-kinematic

**Not started (borrowed or ours)**

- `X7` (check) Do full-resolution Muharaf scans exist (public copy stored at 60 px)? — source: NEXT_STEPS X7, rice1995isri — also in T0
- `pen-contrast-aug` (idea) Pen-contrast augmentation (imitate a cut nib): RASAM pen contrast is an outlier for 50–68 % of lines — source: ours (pool #12) — also in T4
- `X-G12` (experiment) Geometry-prediction pretext pretraining on unlabelled scans — source: ours (pool #8) — also in T8
- `geom-matched-aug` (idea) Geometry-matched augmentation from unlabelled target images: Slant augmentation already hurt; low priority — source: ours (pool #7)
- `slant-density` (idea) Slant-density score for data selection — source: ours (pool #13) — also in T3
- `aug-5-7-8-9-12-13` (idea) Augmentations 5 learned warping, 7 GAN/diffusion, 8 font renders, 9 glyph concatenation, 12 downscale to 60 px, 13 kashida stretching — source: luo2020learn, fogel2020scrabblegan, bilgintasdemir2023ottoman, dinges2016synthesis, ours — also in T3
- `aug-20` (idea) Augmentation 20: fragment crops for short lines — source: ours — also in T0
- `geometry-lit` (literature) Untouched geometry follow-ups: second baseline estimator (Abu-Ain), AHTID/MW baseline ground truth (Mezghani), polygon-curvature descriptor (Abdi), contour codebooks (Ahmed), FATF curve features (Parvez), dots validated on CTC spans — source: paper ledger, geometry group — also in T2

**Dropped**

- `aug-6` (idea) Augmentation 6: pixel span masking — source: li2025htrvt

**Done**

- `R7-DIQA` (experiment) Image-only difficulty predictor, leave-one-dataset-out: Part of the validity model: image features alone transfer poorly across sets — source: ye2013diqa — also in T11
- `aug-1-4` (idea) Augmentations 1–4: affine, elastic, morphology, photometric (baseline) — source: wigington2017augment, retsinas2022best


## T6 · Language and text: what the decoder knows

*How much of the gap is text distance (unseen words, register), and can a language prior help without hallucinating?*

Unseen words hurt every family. KHATT text is the farthest from training. An unconstrained LLM decoder hurts (E8). Errors are visual. Without dots, context restores 95 % of known words but not unseen ones, so a language prior helps only under the image's constraint. Open: constrained decoding with character language models and a comprehensive lexicon (words, names, places), and synthesis from any text.

**What we found**

- [A] `makharij` Do systems confuse letters that sound alike or look alike? (makhārij question): Look-alike 4–15× chance in every family; sound-alike at chance, even for the LLM decoder; no historical support for articulatory letter design
- [B] `validity` Which characteristics predict errors jointly (gate V2): Every family: few pixels per character, short lines, unseen words; long words help. Loose crop and slant only for CRNNs; pen contrast only for fine-tuned VLMs. Low explained variance, poor leave-one-set-out transfer
- [B] `text-distance` Unseen words hurt every family; KHATT text is the farthest from training: Unseen words 33–35 % vs Muharaf train; KHATT 4.0 bits per character; significant in every family's joint model
- [B] `R1.6` Lexical diversity per set (unique lines, duplication, Jaccard, TTR): KHATT most repetitive (mean max Jaccard 0.39, fixed-text paragraphs); NAKBA train duplicates 17.5 % of lines
- [B] `ijam-ambiguity` How ambiguous is Arabic without dots? (why iʿjām was invented): 25–30 % of tokens share their dotless form with another word; the most frequent word restores 95 % of known words, one word of context about +1; 19 % of held-out tokens unseen, where dots carry what context cannot
- [C] `dialect-spelling` Writers spell as they speak: dialect mergers in the labels: Muharaf train labels hold Levantine spellings (متل for مثل 52×, كتير for كثير 16×, ضهر for ظهر 15×); articulation shows in the writers' spelling, not the models' errors; automatic pairing is noisy
- [N] `paw-per-word` PAWs per word as a difficulty predictor: It was word length: not significant for CRNNs once letters per word is controlled
- [N] `E8` LLM decoder on CRNN features (E8, E8b): No-go at 0.6B and 1.7B: KHATT 40 %, more fluent-but-wrong words

**Proposed next round**

- `P-L2` (experiment) PAW-space repair: PAW-aware decoding or lexicon merge of a space after a non-joining letter: Needs CTC posteriors for the decoding version (minutes on the GPU) — source: alluhaybi2020connect — also in T1
- `P-L1` (experiment) Dot-restoration ceiling and H(dots | rasm, context): First evidence (§16): the most frequent word restores 95 % of known words; folded into the two-stage reader experiment — source: alshaibani2025dotless, shannon1951 — also in T2
- `lm-decoding` (experiment) CTC decoding with character language models (Muharaf text, NAKBA text, modern Arabic sample): 15 GPU minutes to dump posteriors, then CPU; tests whether text distance is fixable — source: plan P2.8, garridomunoz2025 — also in T1, T2
- `two-stage-reader` (experiment) Two-stage reader: skeleton-only CRNN, then a dot restorer constrained to the read skeleton (n-gram first, then a small LLM): Does a dotless recogniser read skeletons better, and does constrained restoration beat the model's own dots? Known vs unseen words separately; about 1 GPU hour + CPU — source: user idea 2026-10-14; ahmad2019subcore, alshaibani2025dotless — also in T2, T8

**Planned**

- `synthesis-v1` (experiment) Synthesis from the library: join glyphs at their connection points, re-dot, measured PAW and word gaps, geometry and background variation; text from any source; sampling random / novelty-targeted / one coherent style: Realism checks (real-vs-synthetic classifier, geometry match, no seam shortcut), then CRNN training against real repeats, a StackMix-style naive cut and font renders; readouts: novelty bands and unseen words — source: user idea 2026-10-14; elarian2015, stackmix2021, dinges2016synthesis — also in T3, T5, T10
- `shape-tokens` (experiment) Shape-token recognition: the reader emits shapes (dotless class × form, discovered clusters, or a shape BPE of PAW-sized tokens); a decoder picks letters under Arabic's joining grammar (lexicon / n-gram, then transducer, then a constrained small LLM; top-k beams, speculative decoding): Hypothesis: separating 'what the ink looks like' from 'which letter' generalises better to new writers; readout: KHATT novelty bands — source: user idea 2026-10-14; ahmad2019subcore, alazzawi2026gap (positional labels did not help) — also in T3, T8, T2
- `cross-family-ensemble` (experiment) Cross-family combination: pick per line by confidence, a VLM correcting the CRNN under the skeleton constraint, or ROVER across families: Reopens the dropped ensemble idea in cross-family form; needs VLM log-probabilities — source: ours; NEXT_STEPS (ensembles dropped among VLMs) — also in T8
- `lexicon` (idea) A comprehensive Arabic lexicon with proper names (اسم علم): words, person names, cities, countries, places: Sources: Ayaspell (~300k forms, GPL/LGPL/MPL), aspell-ar (~900k, Buckwalter-based), Pecina et al. LREC 2012 (9M forms; location and licence to check), Alotaibi & Lee fine-grained NE corpora, ANETAC (79,924 names), Wikidata / GeoNames filtered by Arabic label, Arabic Wikipedia. Uses: synthesis text, lexicon-constrained decoding, dot restoration, OOV analysis — source: user idea 2026-10-15 — also in T3, T10

**Parked**

- `R5` (experiment) Language layer: register, dialect, entropy, numbers and names: OOV share already measured and strongest — source: NEXT_STEPS R5
- `zeghli` (literature) Bigram study (no test planned) — source: zeghli2025bigram

**Not started (borrowed or ours)**

- `E2` (experiment) Tokenizer view: do dot-pair swaps get unrelated token sequences; fertility vs CER — source: alomari2026iconicity — also in T2
- `perea-root` (experiment) Do real-word substitutions share the root more often than chance? (CAMeL Tools) — source: perea2010tl
- `X-A3` (experiment) Image-reliance check: log-prob with and without the image — source: greekvlm2026 — also in T9
- `aug-18` (idea) Augmentation 18: harakat on/off renders with matching label — source: ours — also in T0

**Done**

- `R2.5` (experiment) Fluency-hallucination rate (frequent MSA words with no aligned counterpart): Covered by detector D15 and the fluent-but-wrong share in analysis B — source: heakl2025kitab


## T7 · Writers and hands

*How much of line difficulty is the writer, and do the effects hold within writers?*

Writers can now be studied. RASAM-2 has 15 manuscripts of 24–31 lines, RASAM-1 three large ones, and NAKBA is one writer. Page-cluster intervals were added; they match line intervals because seed noise dominates. The full KHATT v1.0 (1,038 test lines) is downloaded, but has no writer IDs. Next: writer adaptation, both for the CRNN (few lines chosen to cover the letters) and for VLMs in context (a labelled reference line in the same prompt), and a calibration sentence covering every letter form.

**Proposed next round**

- `writer-adapt-crnn` (experiment) Writer adaptation for the CRNN: RASAM-2 manuscripts and NAKBA; K = 4, 8, 16 lines chosen at random, to cover the most letter forms (set cover), or richest in the writer's novel shapes; fine-tune, adapter, or a no-training prototype memory: Short GPU runs; tests 'show a few samples that capture most letters' — source: user idea 2026-10-14; MetaHTR, kohut2023finetuning — also in T8, T4
- `vlm-incontext-writer` (experiment) Writer calibration for VLMs in context: one prompt with a labelled reference line from the same writer (image + label), then the test line; vs the test line alone: Qwen3-VL-8B base and fine-tuned; RASAM-2 manuscripts, NAKBA, Muharaf pages; reference chosen at random vs covering most letters; needs a VLM on the 3090 — source: user idea 2026-10-15; in-context writer adaptation (arXiv 2603.29450), MetaWriter — also in T8, T10
- `khatt-full` (experiment) Full KHATT v1.0 (Teklia copy): add its 1,038 test lines to the panel; an in-domain KHATT control from its 4,672 train lines: Downloaded to the SSD. Shared fixed paragraphs: 767 unique texts in 1,038 test lines, and texts recur across splits, so score fixed-text and unique-text lines separately; no writer IDs — source: Teklia/KHATT_v1.0, mahmoud2014khatt — also in T0, T8

**Planned**

- `R0.4` (experiment) Writer, page and document IDs; between-writer variance: NAKBA file names and RASAM manuscript names are now in the panel — source: mahmoud2014khatt
- `P1.8` (experiment) KHATT fixed-text natural experiment (same text, several writers) — source: characteristics-framework V3 — also in T3
- `calibration-sentence` (design) A calibration sentence: the shortest natural Arabic text covering every letter and positional form (for a user to write once): Set cover over letter × form on real text; feeds writer adaptation and the own-dataset copy-texts — source: user idea 2026-10-15; altwaijry2021hijja — also in T10

**Gated on something else**

- `personalised-synthesis` (experiment) Personalised synthesis: generate many lines in one writer's own allographs from their K samples, then fine-tune: Joins all three shape ideas; gated on the library and synthesis-v1 — source: user idea 2026-10-14 — also in T3

**Not started (borrowed or ours)**

- `srihari` (literature) Writer individuality and children's handwriting (feeds S2) — source: srihari2002individuality, srihari2016children — also in T10

**Done**

- `cluster-ci` (experiment) Cluster bootstrap over NAKBA pages and RASAM manuscripts for every comparison: Page clusters for NAKBA (845 / 323 pages) and RASAM; with seed noise included the page intervals match the line intervals, so no conclusion changes — source: plan §3.3 — also in T11


## T8 · Models, training data and transfer

*What transfers across datasets and model families, and what is a fair reference?*

CRNN and VLM difficulty correlate weakly, and the per-line better of the two is far below either: KHATT 14.3 % against 17.4 / 18.1 %, NAKBA test 4.1 % against 10.2 / 18.7 %. So a cross-family combination is a strong candidate. The standard-split CRNN is the comparable reference. Repetition helps NAKBA slightly. Open: in-domain NAKBA and KHATT controls, the combination, E1, the multilingual freeze sweep, data scaling, the cross-script check.

**What we found**

- [A] `X-E1` CRNN generalisation to unseen sets (aligner recipe): Full panel: NAKBA 13.6–13.7 %, KHATT 31.0 % (23.3 % cropped); earlier 26.4 % was the 183-line subset
- [B] `R2.6` VLMs make as many similar-letter errors per character as the CRNN; their share is lower
- [B] `X-G11L` Shape-targeted training repeats (no target labels): KHATT −0.83 [−1.31, −0.33] vs random repeats (3 seeds, every targeted seed better); held-out half B −0.93; no effect on NAKBA, Muharaf, RASAM
- [B] `maghrebi-gap` Without Maghrebi training lines a CRNN reads RASAM at ~47 % CER (vs ~12 % with them): Standard-split CRNN, 2 seeds
- [B] `family-matrix` Line × system matrix: CRNN and VLM difficulty correlate weakly: KHATT ρ ≈ 0.35; 20 hard-core lines; most hard lines are hard for one family
- [B] `std-split` Standard-split CRNN (no test lines in training): Muharaf validation 11.8 %; KHATT 33.4–33.9 %, cropped 26.0–26.7 %; the comparable reference
- [B] `nakba-leak` NAKBA's official split repeats training text: 11 % of blind lines (2.9 % of characters) repeat NAKBA-train text; Baseer reads them at 0.9–1.5 % CER against 7.1 % on clean lines, unseen-data CRNNs at ~50 % (short fragments); corpus CER moves 6.91 → 7.08 %
- [B] `cross-family-oracle` CRNN and VLM fail on different lines: Line-wise better of best CRNN and best fine-tuned VLM: KHATT 14.3 % (17.4 / 18.1), NAKBA test 4.1 % (10.2 / 18.7), blind 4.8 % (11.1 / 6.4), RASAM-1 20.0 % (36.7 / 22.4)
- [C] `repetition` Repeating training lines (any) helps NAKBA slightly: Random repeats vs base: NAKBA test −0.21 [−0.34, −0.07], blind −0.19
- [N] `X-S2` Combined recipe (cropped training + shape repeats): Equals shape repeats + test-time crop; training on cropped lines adds nothing; its NAKBA gain is a repetition effect
- [N] `E8` LLM decoder on CRNN features (E8, E8b): No-go at 0.6B and 1.7B: KHATT 40 %, more fluent-but-wrong words

**Proposed next round**

- `shape-nakba` (experiment) Shape targeting for a second dataset: select on NAKBA blind half A, test on half B (2 seeds vs random): About 4 GPU hours — source: plan P2.1 — also in T3
- `E6` (experiment) Maghrebi adaptation: few-shot curve (16 / 100 / 300 lines), prompt hint, dot-swap augmentation: About 1.5 GPU hours for the CRNN curve — source: vidalgorene2024rasam2, kohut2023finetuning, farazi2026vlmocr — also in T4
- `nakba-indomain` (experiment) In-domain control: CRNN trained on NAKBA train (data on the box, verified): 1–2 seeds, 1–2.5 GPU hours; score with and without train/test text overlaps — source: plan open question 10 — also in T0
- `two-stage-reader` (experiment) Two-stage reader: skeleton-only CRNN, then a dot restorer constrained to the read skeleton (n-gram first, then a small LLM): Does a dotless recogniser read skeletons better, and does constrained restoration beat the model's own dots? Known vs unseen words separately; about 1 GPU hour + CPU — source: user idea 2026-10-14; ahmad2019subcore, alshaibani2025dotless — also in T2, T6
- `writer-adapt-crnn` (experiment) Writer adaptation for the CRNN: RASAM-2 manuscripts and NAKBA; K = 4, 8, 16 lines chosen at random, to cover the most letter forms (set cover), or richest in the writer's novel shapes; fine-tune, adapter, or a no-training prototype memory: Short GPU runs; tests 'show a few samples that capture most letters' — source: user idea 2026-10-14; MetaHTR, kohut2023finetuning — also in T7, T4
- `vlm-incontext-writer` (experiment) Writer calibration for VLMs in context: one prompt with a labelled reference line from the same writer (image + label), then the test line; vs the test line alone: Qwen3-VL-8B base and fine-tuned; RASAM-2 manuscripts, NAKBA, Muharaf pages; reference chosen at random vs covering most letters; needs a VLM on the 3090 — source: user idea 2026-10-15; in-context writer adaptation (arXiv 2603.29450), MetaWriter — also in T7, T10
- `khatt-full` (experiment) Full KHATT v1.0 (Teklia copy): add its 1,038 test lines to the panel; an in-domain KHATT control from its 4,672 train lines: Downloaded to the SSD. Shared fixed paragraphs: 767 unique texts in 1,038 test lines, and texts recur across splits, so score fixed-text and unique-text lines separately; no writer IDs — source: Teklia/KHATT_v1.0, mahmoud2014khatt — also in T0, T7

**Planned**

- `hard-example-control` (experiment) Hard-example control: repeat the highest-loss training lines instead of shape-matched ones — source: plan P2.1 — also in T3
- `vlm-geometry` (experiment) VLM check: fine-tuned Qwen3-VL-8B on cropped and slant-sheared KHATT: 4–6 GPU hours, more setup; decides whether crop and slant effects are CRNN-only — source: plan P2.3 — also in T5
- `E5-retsinas` (experiment) Retsinas-recipe CRNN (column max-pool, CTC shortcut, median padding) — source: retsinas2022best — also in T5
- `E1` (experiment) E1 frozen vs unfrozen ViT; E1.1 bench of the diff-LR arm; E1.2 train arm (b); E1.3 compare: Preliminary in-domain −7.6 points (one seed) — source: NEXT_STEPS E1, Baseer-Nakba
- `shape-tokens` (experiment) Shape-token recognition: the reader emits shapes (dotless class × form, discovered clusters, or a shape BPE of PAW-sized tokens); a decoder picks letters under Arabic's joining grammar (lexicon / n-gram, then transducer, then a constrained small LLM; top-k beams, speculative decoding): Hypothesis: separating 'what the ink looks like' from 'which letter' generalises better to new writers; readout: KHATT novelty bands — source: user idea 2026-10-14; ahmad2019subcore, alazzawi2026gap (positional labels did not help) — also in T3, T6, T2
- `cross-family-ensemble` (experiment) Cross-family combination: pick per line by confidence, a VLM correcting the CRNN under the skeleton constraint, or ROVER across families: Reopens the dropped ensemble idea in cross-family form; needs VLM log-probabilities — source: ours; NEXT_STEPS (ensembles dropped among VLMs) — also in T6
- `seed-instability` (experiment) Lines that flip between identical trainings (3 seeds): characterise them and add a detector — source: ours (review 2026-10-14) — also in T0

**Gated on something else**

- `ketaba-dora` (literature) DoRA / rsLoRA arm: After E1 says whether encoder or decoder is the bottleneck — source: Ketaba-OCR

**Parked**

- `C-track` (experiment) Curriculum: letters first, then words (C0 data, C1 vs controls, C2 letter-incremental, C3 factorised head, C4 averaging) — source: NEXT_STEPS C-track — also in T3

**Not started (borrowed or ours)**

- `R3.2` (experiment) IRT 2PL difficulty and discrimination per line — source: rodriguez2021irt — also in T0
- `X3-retrain` (experiment) Retrain without the 1,424 CER-HV-flagged Muharaf train lines — source: NEXT_STEPS X3 — also in T0
- `X-G12` (experiment) Geometry-prediction pretext pretraining on unlabelled scans — source: ours (pool #8) — also in T5
- `E7` (experiment) Multilingual pre-training with a freeze sweep — source: aradillas2021boosting, alazzawi2026seqmodel, alazzawi2026crossling
- `E3` (experiment) Data scaling on the K grid, in characters per letter × form (E3, E3′) — source: alazzawi2026gap — also in T10, T3
- `R10` (experiment) Cross-script check (gate V4): Persian PHTD, Urdu NUST-UHWR, Pashto, IAM — source: marti2002iam, characteristics-framework — also in T11
- `P-L5` (experiment) Checkpoint weight averaging (AIN-7B NAKBA adapters) — source: Baseer-Nakba, izmailov2018swa
- `P-L6` (experiment) Per-sample loss logging (dataset cartography) — source: swayamdipta2020cartography — also in T0
- `E4` (experiment) Baseer replay (lower rank, fewer steps, NAKBA-domain lines) — source: NEXT_STEPS E4
- `aug-11` (idea) Augmentation 11: cross-lingual / cross-dataset mixing — source: alazzawi2026crossling
- `X5-print` (experiment) Add a line-level printed Arabic set and a page track — source: NEXT_STEPS X5, Baseer — also in T10
- `vlm-logprobs` (experiment) Save token log-probabilities in the VLM runner for calibration and selective reading: Plan open question 9; changes cache keys only for new runs — source: PLAN-2026-10-13 §3.3 A4 — also in T11

**Done**

- `C4` (fix) The aligner CRNN trained on most Muharaf and RASAM test lines: In-train lines flagged and excluded everywhere; standard-split CRNN trained (2 seeds) — source: planning agent — also in T0


## T9 · Attribution and localisation: where models look

*Do models look at the right letters, and can attribution guide training?*

Attribution maps are valid (9× on ink) and point at the token's own letters about 3× chance. Wrong tokens look in the right place as often as correct ones, so errors are misreadings, not misdirected looking. LocateAnything zero-shot failed. Open, and parked behind the characterisation work: image reliance, dots, per-module gradients, freeze-by-letter.

**What we found**

- [B] `X-A1` Attribution maps validated (attention, relative attention, gradient × input): 9× chance on ink; all beat random patches on deletion
- [B] `X-B2` Pointing game against CTC spans: About 3× chance; wrong tokens look in the right place as often as correct ones
- [B] `X-B1` CTC character spans from the CRNN aligner: Basis of the letter dumps; VML-HD validation open
- [N] `X-C1` LocateAnything-3B zero-shot: No-go on 42 lines

**Gated on something else**

- `X-D1` (experiment) Freeze-by-letter: protect weights that detect each letter group (wild) — source: ours (ideas #3) — also in T3
- `X-C2` (experiment) Fine-tune LocateAnything on word / PAW pseudo-labels — source: wang2026locateanything

**Not started (borrowed or ours)**

- `X-A6` (experiment) Do attribution maps land on dots? — source: ours — also in T2
- `X-A2b` (experiment) Larger letter-family sample (the خ lead) — source: attribution spike — also in T3
- `X-D0` (experiment) Per-module gradient probe by letter family (no training) — source: ours (ideas #2) — also in T3
- `X-A3` (experiment) Image-reliance check: log-prob with and without the image — source: greekvlm2026 — also in T6
- `X-A4-A5-A7` (experiment) Chefer relevance maps (X-A4), map sensitivity (X-A5), attribution on other models incl. the CRNN (X-A7) — source: chefer2021, attribution-eval
- `grounding-lit` (literature) Q-Mask, PositionOCR, InstructOCR, Chinese segmentation-free spotting, VML-HD, character segmentation (niss2023) — source: qmask2026, positionocr2026, instructocr2025, chinesesegfree2022, vmlhd, niss2023charseg

**Dropped**

- `pointing-loss` (idea) Pointing game as a training signal: Wrong tokens already look at the right place — source: ours (pool #10)


## T10 · Our own dataset (design)

*What would a dataset that answers these questions look like?*

A living design guide exists (dataset-design.md). Findings feed it: size in characters per letter × form, writer metadata, a tight crop policy, a documented hamza and diacritic convention, shared copy-texts across writers.

**Proposed next round**

- `R4.1` (experiment) Label review in the review tool (458 candidates) and the 20 KHATT hard-core lines: Needs about 2 h of human review: a label-error rate per set, κ with CER-HV, and a 'can a human read it?' field for the hard-core lines (a human ceiling) — source: alazzawi2026labelerr — also in T0
- `shape-library` (experiment) Allograph library (shape atlas): letter glyphs with skeleton, separate dots and entry/exit join points; alignment + ink skeleton extraction; cycle-consistency quality gate: Foundation of synthesis, shape tokens and personalised adaptation; also a characterisation result (shapes per letter × form per dataset); CPU, a few days — source: user idea 2026-10-14; elarian2015 (KFUPM), dinges2016synthesis, bulacu2007allographic — also in T3, T11
- `vlm-incontext-writer` (experiment) Writer calibration for VLMs in context: one prompt with a labelled reference line from the same writer (image + label), then the test line; vs the test line alone: Qwen3-VL-8B base and fine-tuned; RASAM-2 manuscripts, NAKBA, Muharaf pages; reference chosen at random vs covering most letters; needs a VLM on the 3090 — source: user idea 2026-10-15; in-context writer adaptation (arXiv 2603.29450), MetaWriter — also in T7, T8

**Planned**

- `R8` (experiment) Decision experiments (gate V5): coverage-balanced selection vs random; transformation choice: Shape targeting is a first instance of (a) — source: characteristics-framework §7 — also in T3, T11
- `synthesis-v1` (experiment) Synthesis from the library: join glyphs at their connection points, re-dot, measured PAW and word gaps, geometry and background variation; text from any source; sampling random / novelty-targeted / one coherent style: Realism checks (real-vs-synthetic classifier, geometry match, no seam shortcut), then CRNN training against real repeats, a StackMix-style naive cut and font renders; readouts: novelty bands and unseen words — source: user idea 2026-10-14; elarian2015, stackmix2021, dinges2016synthesis — also in T3, T6, T5
- `calibration-sentence` (design) A calibration sentence: the shortest natural Arabic text covering every letter and positional form (for a user to write once): Set cover over letter × form on real text; feeds writer adaptation and the own-dataset copy-texts — source: user idea 2026-10-15; altwaijry2021hijja — also in T7
- `lexicon` (idea) A comprehensive Arabic lexicon with proper names (اسم علم): words, person names, cities, countries, places: Sources: Ayaspell (~300k forms, GPL/LGPL/MPL), aspell-ar (~900k, Buckwalter-based), Pecina et al. LREC 2012 (9M forms; location and licence to check), Alotaibi & Lee fine-grained NE corpora, ANETAC (79,924 names), Wikidata / GeoNames filtered by Arabic label, Arabic Wikipedia. Uses: synthesis text, lexicon-constrained decoding, dot restoration, OOV analysis — source: user idea 2026-10-15 — also in T6, T3

**Parked**

- `S1-styles` (design) Script-styles inventory (traits, datasets, samples, licences) — source: ideas #7 — also in T4
- `S-track` (design) Own dataset: S0 partner and ethics, S1 content design, S2 sampling, S3 annotation protocol, S4 pilot — source: dataset-design.md, altwaijry2021hijja, srihari2016children

**Not started (borrowed or ours)**

- `aug-19` (idea) Augmentation 19: coverage-balanced text for synthetic lines — source: ours — also in T3
- `srihari` (literature) Writer individuality and children's handwriting (feeds S2) — source: srihari2002individuality, srihari2016children — also in T7
- `E3` (experiment) Data scaling on the K grid, in characters per letter × form (E3, E3′) — source: alazzawi2026gap — also in T8, T3
- `X5-print` (experiment) Add a line-level printed Arabic set and a page track — source: NEXT_STEPS X5, Baseer — also in T8


## T11 · Infrastructure, standards and literature

*How do we make every result traceable and every characteristic reusable?*

Built: store, panel, seeds, analyses A and B, 18 detectors, page-cluster intervals, corrections log, research map with coverage check, SSD backups. Open: correction for multiple comparisons; reporting in WER and lines fixed; decision rules, dependencies and a claims register per item; gates V3–V6; items that need the main checkout (PaddleOCR patch, NEXT_STEPS, Atlas tracker).

**What we found**

- [A] `R2.1` Normalisation ladder: error owed to encoding, orthography, diacritics, dots and spacing: Every reading in the store is scored at all seven rungs
- [B] `validity` Which characteristics predict errors jointly (gate V2): Every family: few pixels per character, short lines, unseen words; long words help. Loose crop and slant only for CRNNs; pen contrast only for fine-tuned VLMs. Low explained variance, poor leave-one-set-out transfer
- [B] `X-B1` CTC character spans from the CRNN aligner: Basis of the letter dumps; VML-HD validation open
- [B] `foundations` Characterisation foundations: GT recovery (R0.1), canonical runs (R0.3), positional-form analyser and profiler (R1.1, R1.2), diacritic-mismatch tax (R2.3), error taxonomy (R2.4), consensus quality groups (R3.1), exact leakage (R4.0), image layer first pass (R6.0), Wave-0 literature (L0)
- [B] `shape-feasibility` Can real letter shapes be cut out? (feasibility of the shape programme): CTC spans are narrow anchors (median 12 px at 64 px; 9 % one column) that often miss parts of a letter; large ink components match label PAWs exactly on only 6–9 % of lines (PAWs touch)
- [–] `store` Per-line store, standard panel, seeded trainer, analyses A/B, 18 detectors, corrections log: Every stored CRNN edit reproduced exactly
- [–] `detectors-done` Pattern detectors implemented: D1 loose crop, D2 slant, D3 ruled, D5 low resolution, D6 stroke/pen, D7 fragment, D8 letter list, D9 unseen words, D10 label suspect, D10b label tail, D11 truncation, D12 loop, D13 whole-line, D14 script mismatch, D15 fluent rewrite, D16 diacritic insertion, D17 orthographic only, D18 dot only, D19 Maghrebi dots, D20 PAW-gap space, D22 reversed reading

**Proposed next round**

- `shape-library` (experiment) Allograph library (shape atlas): letter glyphs with skeleton, separate dots and entry/exit join points; alignment + ink skeleton extraction; cycle-consistency quality gate: Foundation of synthesis, shape tokens and personalised adaptation; also a characterisation result (shapes per letter × form per dataset); CPU, a few days — source: user idea 2026-10-14; elarian2015 (KFUPM), dinges2016synthesis, bulacu2007allographic — also in T3, T10
- `decision-rules` (design) Decision rule and dependencies for every open item; a claims register per write-up (claim, evidence, what is missing): Done in PLAN-2026-10-15 for the proposed items — source: review 2026-10-14

**Planned**

- `R3.3` (experiment) Challenge subsets (hard core, dot-dense, OOV-heavy, fragments) as dataset filters: Hard core (20 KHATT lines), family-split lines and detector flags now exist — source: NEXT_STEPS — also in T0
- `detector-residual` (experiment) Review the residual: worst lines no detector explains, and write new detectors: 1–17 % of each run's worst fifth unexplained — source: plan §3.5 — also in T0
- `R8` (experiment) Decision experiments (gate V5): coverage-balanced selection vs random; transformation choice: Shape targeting is a first instance of (a) — source: characteristics-framework §7 — also in T10, T3
- `gates` (design) Validation gates V1 reliable, V2 predictive, V3 causal, V4 generalises, V5 useful, V6 adoptable: V1–V2 partly met for several characteristics; V3–V6 open — source: characteristics-framework §7
- `R6-R11` (experiment) Image layer second pass (R6), causal synthetic probes (R9, gate V3), release (R11, gate V6) — source: NEXT_STEPS, characterization-plan — also in T5
- `box-cleanup` (fix) Trim the 3090 footprint (8.3 GB of pickles and runs) once the next queue is done: All weights backed up to the SSD — source: user instruction 2026-10-13
- `detectors-open` (experiment) Detectors still to write: D4 neighbour-line intrusion, D23 memorisation / contamination signature, D24 numeric instability (CRNN), D21 lam-alef ligature — source: plan §3.5 — also in T0
- `paper-groupings` (design) Write-up groupings: P1 benchmark audit, P2 characterisation across families (the main direction), P3 methods (geometry transfer, targeted data): User view 2026-10-13: nothing is main-conference grade yet; P2 becomes possible once the cross-family evidence is in — source: PLAN-2026-10-13 §2.3 — also in T0
- `fdr` (design) Correct for multiple comparisons (false-discovery rate) wherever many cells are scanned (letter × form moves, detector tables) — source: review 2026-10-14
- `wer-lines` (design) Report WER and 'lines fixed' alongside CER for every comparison — source: review 2026-10-14
- `main-checkout-sync` (fix) Items that need the main checkout: apply the PaddleOCR patch, update NEXT_STEPS.md from the map, update the Atlas tracker (progress.json) and republish: This session cannot write there; ask the user — source: memory: research-atlas-live-page; first handoff

**Parked**

- `X9` (check) Contact the Luleå authors (licence, code, our VLM numbers): Outward-facing: ask the user first — source: NEXT_STEPS X9

**Not started (borrowed or ours)**

- `X5-len` (fix) Length-stratified reporting: fragments vs full lines on the leaderboard: Short lines fail disproportionately (also a cross-family predictor) — source: NEXT_STEPS X5 — also in T0
- `R10` (experiment) Cross-script check (gate V4): Persian PHTD, Urdu NUST-UHWR, Pashto, IAM — source: marti2002iam, characteristics-framework — also in T8
- `R1.3-R1.4` (experiment) Profiler: diacritic-policy share per set (R1.3); datasheet cards on the Atlas (R1.4) — source: NEXT_STEPS — also in T0
- `L-track` (literature) Literature tasks L1 diacritics, L2 annotation schema, L3 script types, L4 architectures, L5 dataset taxonomy, L6 other languages, L7 verify unverified references — source: NEXT_STEPS L-track
- `vlm-logprobs` (experiment) Save token log-probabilities in the VLM runner for calibration and selective reading: Plan open question 9; changes cache keys only for new runs — source: PLAN-2026-10-13 §3.3 A4 — also in T8

**Done**

- `C3` (fix) The old evaluation silently dropped 17 KHATT lines (CTC length filter applied before the crop): Every earlier KHATT CRNN number was on an easier 183-line subset; the panel scores all lines — source: planning agent — also in T0
- `C7` (fix) CRNN readings depended on batch padding and numerics: Panel evaluator scores one line at a time in fp32 — source: planning agent — also in T0
- `R7-DIQA` (experiment) Image-only difficulty predictor, leave-one-dataset-out: Part of the validity model: image features alone transfer poorly across sets — source: ye2013diqa — also in T5
- `cluster-ci` (experiment) Cluster bootstrap over NAKBA pages and RASAM manuscripts for every comparison: Page clusters for NAKBA (845 / 323 pages) and RASAM; with seed noise included the page intervals match the line intervals, so no conclusion changes — source: plan §3.3 — also in T7
- `X4-names` (fix) Model-name hygiene (runs launched under another BASES entry): Handled by MODEL_RENAMES in characterize/runs.py — source: NEXT_STEPS X4 — also in T0

