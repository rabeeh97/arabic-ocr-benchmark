# Experiment plan 2026-10-15

This plan picks up from the research map (ledger top section; [`RESEARCH-MAP.md`](RESEARCH-MAP.md); source
[`research_map.json`](research_map.json)). Every row is a map item, written as a question, a decision rule fixed before the
run, its dependencies, the compute it needs, and where its evidence will go. Items are grouped into waves by what they need,
not by importance; the recommendation at the end says what to do first.

**Conventions for every experiment** (from PLAN-2026-10-13 §3.9, now standard):
- Score on the panel: full test sets, as stored and ink-cropped, one line at a time in fp32. The full KHATT v1.0 test split
  is added once A6 is done.
- Run the store and the standard analyses: line-level A (buckets, transitions, beyond-noise lines), letter-level B, the
  detectors, and the dots and convention views.
- Use at least 2 seeds for any training claim; 3 seeds where the effect is under 1 point.
- Report line, page-cluster and seed-pooled intervals. Correct for multiple comparisons when scanning letter or detector cells.
- Report WER and lines fixed as well as CER.
- Add a gallery to the ledger evidence block.

## Wave A: laptop, now (no new data; about 1 week of CPU work in total)

| ID | Question | Decision rule | Depends on | Output |
|---|---|---|---|---|
| A1 `PAW-gap-width` | Are the PAW gaps that get split as wide as real word gaps? | If split PAW gaps overlap the word-gap distribution, spacing is visually ambiguous, which is a characterisation finding. If they are narrower, the models over-split and the cause is in the model | letter dumps (have) | gap-width distributions per set; T1 panel |
| A2 `novel-clusters` | What are the novel shapes? | Clusters that exist in KHATT or NAKBA but not in Muharaf train → targets for synthesis and selection | letter dumps | allograph atlas v0 (cluster medoids, counts per set) |
| A3 `loops-teeth` | Which skeleton features fail: loops, teeth, tails? | A feature class whose error is above the letter-level baseline in every family becomes a T3 finding | store | confusion by feature class; tooth-count detector; error vs run length |
| A4 `dot-styles` | Do dash and caret dots, displaced dots or touching dots raise dot errors? | Dot errors on a style at least 2× the separate-dot rate (interval clear of 1) → a dot-style finding and a synthesis requirement | components + spans | prevalence per set; dot-error ratio |
| A5 `shape-library` | Can a clean glyph library be extracted? | Proceed to synthesis if at least 30 % of letters pass the cycle-consistency gate and every letter × form has at least 20 glyphs | A2 | the library (glyph, skeleton, dots, join points, source, novelty) |
| A6 `khatt-full` | Build the full KHATT panel (1,038 test lines) | — | download (done) | KHATT as stored and cropped in the panel; fixed-text and unique-text flags |
| A7 `calibration-sentence` | What is the shortest natural text covering every letter × form? | — | text corpora | a few candidate sentences and their coverage |
| A8 `lexicon` | Assemble words, names and places | — | downloads to the SSD | lexicon (Ayaspell, aspell-ar, Wikidata/GeoNames names, Wikipedia vocabulary) with licences recorded |
| A9 `fdr`, `wer-lines` | Hygiene in the analyses | — | — | FDR-adjusted letter and detector tables; WER and lines fixed in pairs |
| A10 `seed-instability` | Which lines flip between identical trainings? | If unstable lines share a detector profile (short, loose, unseen), that profile is a noise source to report | 3-seed runs (have) | an instability detector |
| A11 `R4.1` | Label-error rate and human readability (human, about 2 h) | Hard-core lines judged unreadable or mislabelled leave the headline metric, with a note | review tool | label-error rate per set; human ceiling |

## Wave B: the first GPU night (CRNN only; about 8 h)

| ID | Question | Decision rule | Cost | Output |
|---|---|---|---|---|
| B1 `lm-decoding`, `P-L2` | Dump CTC posteriors (base, combo, standard split; panel + full KHATT) for every decoding study on the laptop: character LMs (Muharaf text, NAKBA text, modern Arabic), lexicon-constrained beams, PAW-aware spacing | If a modern-text LM gains ≥ 1 point on KHATT and ≤ 0.3 on NAKBA, text distance carries part of the gap. If PAW-aware spacing lowers WER without raising CER, adopt it as post-processing | 15–30 min GPU, then CPU | decoding tables; T6 and T1 panels |
| B2 `two-stage-reader` | Does a skeleton-only CRNN read skeletons better, and does a constrained dot restorer beat the model's own dots? | Adopt the two-stage reader if its full CER ≤ the normal CRNN's on two sets, or if it wins on unseen words without losing on known ones | 1 training (about 1 h) + CPU restorer | skeleton CER of both models; restored-dot CER, known vs unseen |
| B3 `writer-adapt-crnn` | Do a few lines from a writer help, and do letter-covering lines help more? | Coverage-chosen lines beat random lines by ≥ 1 point at K = 8 on most RASAM-2 manuscripts → the coverage rule is useful (gate V5) | 15 manuscripts × 3 K × 3 selections, short fine-tunes (about 2 h) | adaptation curves per writer |
| B4 `E6` | How many Maghrebi lines fix the dot convention? | Report the K at which ق→ف falls under 5 % and RASAM CER halves | about 1.5 h | few-shot curve (CER and ق→ف) |
| B5 `nakba-indomain` | In-domain control | Do the crop, novelty and PAW-space findings hold in-domain? A finding that reverses in-domain is restricted to "under domain shift" | about 1 h (1 seed) | in-domain panel scores; diagnostics repeated |
| B6 `khatt-full` | All existing CRNNs on the full KHATT test | Re-check every KHATT claim at five times the sample (crop, shape targeting, novelty) | about 30 min | updated KHATT numbers with tighter intervals |

## Wave C: VLM night (needs a VLM on the 3090; about 6–8 h)

| ID | Question | Decision rule | Cost | Output |
|---|---|---|---|---|
| C1 `vlm-incontext-writer` | Does a labelled reference line from the same writer, in the same prompt, help a VLM read the next line? | Keep if the gain is ≥ 2 points on RASAM-2 manuscripts or NAKBA for base or fine-tuned Qwen3-VL-8B. Compare a random reference with a letter-covering one, and a calibration sentence (A7) | about 3–4 h | per-writer gain; prompt designs |
| C2 `vlm-geometry` | Are the crop and slant effects CRNN-only? | If the VLM |Δ| < 0.5 points (interval) where the CRNN moves ≥ 1, the effect belongs to fixed-height CNNs | about 2 h | VLM on cropped and sheared KHATT |
| C3 `vlm-logprobs` | Save token log-probabilities | — (enables C4) | with C1 | calibrated VLM confidences |
| C4 `cross-family-ensemble` | Can we get near the per-line oracle (KHATT 14.3 %, NAKBA test 4.1 %)? | Adopt if a confidence-based selector closes ≥ 30 % of the gap between the best single family and the oracle on two sets | CPU after C3 | combination results |

## Wave D: the shape programme (after A5; weeks)

| ID | Question | Decision rule | Depends on |
|---|---|---|---|
| D1 `synthesis-v1` | Do synthetic lines built from real glyphs help on novel shapes and unseen words? | Real-vs-synthetic classifier ≤ 75 % and geometry within real ranges before training. Then keep if novelty-targeted synthesis beats real repeats and the StackMix-style control on KHATT bands 4–5 by ≥ 1.5 points (3 seeds) without in-domain loss | A5, A8 |
| D2 `shape-tokens` | Does reading shapes, then letters, generalise better to new writers? | Compare letters vs dotless class × form vs discovered clusters vs shape BPE. Keep the shape vocabulary that lowers KHATT high-novelty error with no overall loss | A2, A5, B1 decoders |
| D3 `personalised-synthesis` | Do lines synthesised in one writer's own shapes beat fine-tuning on their K lines alone? | ≥ 1 point over B3 at the same K | D1, B3 |
| D4 `render-refine` | Only if D1's seams are visible or exploited | — | D1 |
| D5 `shape-nakba` | Does shape targeting work for a second dataset? | Pooled targeted − random ≤ −0.5 on held-out NAKBA (interval below 0) → a general method | A2 |

## Wave E: gated or later

X-G3 normalisation ablation, R9 resolution probe, X-G7 dot-preserving resize, X-G9 factorised head, E1 / E1.1, E7 freeze
sweep, E3 data scaling, R10 cross-script (gate V4), the attribution track (X-A*, X-D*), the curriculum track (C0–C4), the
own-dataset track (S0–S4), R4.2–R4.4, IRT, the L-track literature tasks. Status and reasons are on the map.

## Claims register (which future write-up each finding feeds)

| Write-up | Findings already at A or B | Missing before writing |
|---|---|---|
| Benchmark and evaluation audit | crop and length filter, hamza and digit conventions, NAKBA split differences, NAKBA text leak, kitab_muharaf leakage, end-of-line artefact, PaddleOCR flag, no-dots and space taxes | R4.1 label errors and human ceiling, R2.7 diacritic rule, B6 full KHATT, near-duplicates (R4.2) |
| Characterisation across families | PAW-gap splits, novelty passes confound battery, look-alike vs sound-alike, skeleton vs dots, undotted ambiguity, cross-family oracle, validity model | A1 gap widths, A3 skeleton features, B5 in-domain control, C2 VLM geometry, R10 cross-script |
| Methods (shape programme) | shape targeting on KHATT, feasibility | A5 library, D1 synthesis, D2 shape tokens, D5 second dataset, B3 / C1 writer adaptation |

## Needs the main checkout (this session cannot write there)

1. Apply `docs/research/patches/0001-paddleocr-reverse-output.diff`.
2. Refresh `docs/research/NEXT_STEPS.md` from the map. The map now supersedes it as the backlog.
3. Update the Atlas tracker (`progress.json`) and republish the Atlas.
