# Characterising our data and the Arabic text it contains: full plan

> **Status:** proposed · **Last updated:** 2026-10-06 · **Wave:** 0 (the empirical half)
> Companion to [`script-characteristics.md`](script-characteristics.md), which
> is the literature half. That note says *what should matter*. This plan
> *measures it on our data* and checks *whether it actually predicts model
> errors*.

## Why, and what "useful" means

We have 17 dataset configs and **489 prediction files** (`results/**/predictions.jsonl`,
each with GT and the model output), plus the exp3090 and suite_v1 runs. So far we
have read them only as CER averages.

A characterisation is only useful if it changes a decision. Each output below is
tied to one:

| output | the decision it changes |
|---|---|
| **Per-dataset datasheet cards** (text, image and annotation profile) | which datasets to trust, train on, or report separately |
| **Label-noise estimate per dataset**, with flagged samples | whether to clean or swap Muharaf (6–9 % label errors reported in `alazzawi2026labelerr`); which test rows to exclude |
| **Error taxonomy per (model, dataset)**: dot / positional form / PAW-space / diacritic / ligature / other | what E1 (ViT) and E2 (tokenizer) should target and measure |
| **Difficulty model**: which sample features predict CER, and by how much | the *ranked* list of what makes Arabic OCR hard, measured rather than taken from papers; the core academic finding |
| **Coverage and distance map**: how far each test set sits from our training data | explains the transfer results (why RASAM lags) and guides data selection |
| **Arabic vs. English reference statistics** on the same measures | the "studied like English" comparison the project asks for |
| **Stratified challenge subsets** (e.g. dot-dense lines, diacritised lines, long lines) | sharper leaderboard columns than one CER per dataset |
| **Numbers for the dataset spec**: shape coverage targets, size, strata | [`dataset-design.md`](dataset-design.md) |

## Research questions

- **RQ1 (inventory):** what units does each dataset actually contain:
  characters, (letter × positional form) shapes, dot groups, PAWs, ligatures,
  diacritics? How heavy-tailed is each distribution?
- **RQ2 (language):** what variety and register is each dataset (Classical,
  MSA, dialect, archaic spelling)? How do vocabulary, entropy, morphology and
  ambiguity compare across datasets and against English?
- **RQ3 (image):** what visual conditions does each dataset have (resolution,
  stroke width, degradation, skew, overlap, script style)?
- **RQ4 (annotation):** what transcription conventions does each dataset
  follow? How noisy are its labels?
- **RQ5 (relations):** how similar are the datasets to each other and to our
  training data? Is there leakage between them?
- **RQ6 (difficulty):** which of the RQ1–RQ4 features explain per-sample CER,
  across models and architectures? Do the explanations differ by model
  family, for example VLM vs. CRNN vs. API?
- **RQ7 (errors):** what kinds of errors do models make, and how does the mix
  change with fine-tuning and with model family?

## Scope

- **Datasets:** all line-track sets first: `muharaf_lines`, `kitab_khatt`,
  `nakba_lines`, `rasam1/2`, `arams_htr`, the KITAB line sets and `synthetic`.
  The page sets (`kitab_hindawi`, `kitab_historicalbooks`, …) get the text and
  annotation layers only, flagged as page-scale.
- **Splits:** for each dataset, the test split we score on and the train split
  if one exists. Muharaf `train` is mandatory because it is our training data.
- **Reference corpora** (text only, for the language layer):
  - *Arabic, undiacritised MSA:* Arabic Wikipedia sample
  - *Arabic, Classical, diacritised:* Tashkeela
  - *English, handwriting GT:* IAM lines
  - *English, general:* English Wikipedia sample

  The four samples are comparable in size, e.g. 1M characters each.
- **Size:** compute text features on every line. Compute image features on
  every line where it's cheap, and on a seeded sample of ≤ 2,000 per dataset
  otherwise.

## The layers (what we measure)

### L1: Script and text layer (GT only; CPU, cheap)
| feature | how |
|---|---|
| Unicode inventory and encoding hygiene | code-point histogram; presentation forms, bidi/zero-width characters, Persian look-alikes (ک/ی/ہ), unnormalised combining-mark order, stray tatweel; reuse the `normalize.py` building blocks to *count*, not fold |
| Letter × positional-form shapes | joining-type state machine (Unicode `ArabicShaping.txt` classes) → isolated/initial/medial/final per letter; shape count, Gini coefficient and tail mass |
| Dot groups | map each letter to its rasm class (19 classes); dotted-letter density; how often dot-confusable pairs co-occur in a line |
| PAWs | split words at non-joining letters; PAWs per word, distribution of single-letter PAWs |
| Ligatures and decoration | lam-alef rate, tatweel rate |
| Diacritics | density per letter; type mix (fatha, kasra, damma, shadda, sukun, tanwin, dagger alef); **full vs. partial** diacritisation per line; *invalid placements* (a mark on a non-letter, doubled marks), which are a label-noise signal: e.g. Muharaf `test/000000` has `اسَعد` |
| Non-Arabic content | digits (which system), Latin, punctuation, brackets and editorial marks |
| Length | characters, words and PAWs per line; tokens per line for each model tokenizer (feeds E2) |

### L2: Language layer (GT and reference corpora; CPU)
| feature | how |
|---|---|
| Vocabulary | type/token ratio, hapax ratio, Zipf exponent, Heaps curve |
| Entropy | character unigram/bigram/trigram entropy; conditional entropy H(dots \| rasm) = information carried by the dots |
| Cross-set lexicon | OOV rate of each test set against Muharaf train; vocabulary Jaccard overlap between datasets |
| Morphology | clitic rate, lemma/root diversity, average analyses per undiacritised word = ambiguity (needs **CAMeL Tools**; new optional dependency) |
| Variety and register | dialect-ID score per line (CAMeL dialect ID) and an archaic-orthography flag; a manual check on 50 lines per dataset |
| Arabic vs. English | the same metrics on IAM and English Wikipedia, so every Arabic number has a reference point |

### L3: Image layer (CPU; seeded sample)
| feature | how |
|---|---|
| Geometry | width, height, aspect ratio; estimated x-height / text height in pixels (the effective resolution the model sees); for VLMs, image tokens per line at each model's resize policy |
| Ink | stroke width (distance transform on an Otsu mask), ink density, ink/background contrast |
| Degradation | blur (Laplacian variance), noise estimate, JPEG blockiness, background non-uniformity and bleed-through proxy |
| Layout within the line | skew and slant angle, baseline drift, ascender/descender overlap from neighbouring lines (ink at the top and bottom borders) |
| Script style | from metadata where available. Otherwise annotate a 50-line sample per dataset by hand (Naskh / Ruq'ah / Maghrebi / Diwani / mixed) to get a style mix per dataset |

### L4: Annotation and quality layer
| check | how |
|---|---|
| Conventions | per dataset: diacritics policy, how unreadable text is marked, abbreviations, numbers, line-break handling, normalisation already applied. Read the dataset papers and confirm against the data |
| **Label noise by model consensus** | we already have many strong models' predictions per sample. Flag a sample when *all* good models disagree with the GT in the same way (high CER for every model, agreement among the predictions). Then review the top-k by hand, as the CER-HV recipe does (`alazzawi2026labelerr`). Output: an estimated label-error rate per dataset and a flag list |
| GT vs. image consistency | length ratio GT characters / image width in x-heights; outliers mean truncated or misaligned labels |
| Duplicates and leakage | exact and near-duplicate GT lines (normalised text hash, MinHash) within and across datasets and splits; perceptual image hash for duplicate images; Muharaf train vs. every test set especially |
| Writer and document metadata | recover writer, page and document IDs from the source datasets into `Sample.meta` (the adapters currently keep only source/split/row). This is needed for writer-level variance and for checking that splits are writer-disjoint |

### L5: Relations between datasets
- Distances between the character, shape and dot-group distributions (Jensen–Shannon) → a dataset similarity map.
- Image-feature distribution distances → a "visual distance from Muharaf train" score for each test set.
- Correlate these with the transfer gains already measured (exp3090, suite_v1). Test whether distance from the training data predicts post-fine-tune CER.

### L6: Error taxonomy and difficulty model (the payoff)
- **Error taxonomy** (`scripts/error_taxonomy.py`): align GT with each prediction by character edit operations, using `char_edit_counts` / Levenshtein from `metrics.py`, with backtrace. Classify each operation:
  - **dot:** same rasm class, different letter;
  - **positional / shape-similar:** e.g. ت↔ن in initial form, ر↔د;
  - **PAW-space:** a space inserted or deleted at a PAW boundary;
  - **diacritic:** a harakat insertion, deletion or substitution;
  - **ligature:** a lam-alef split or merge;
  - **hamza/alef/ya/ta-marbuta orthographic:** a variant that `diac_insensitive` would fold;
  - **other substitution / deletion / insertion;**
  - **hallucination:** a whole-line failure, e.g. the meta-commentary we saw from base VLMs. Tag this separately so it doesn't swamp the rest.

  Run it over all 489 prediction files. Also compute **rasm-CER**: both sides mapped to the 19 rasm classes. The difference CER − rasm-CER is the "dot tax".
- **Difficulty model:** one row per (sample, model), with per-sample CER as the outcome and the L1/L3/L4 features as predictors.
  - Fit a mixed-effects model with model and dataset as random effects.
  - Cross-check it with gradient boosting plus SHAP.
  - Read off effect sizes: e.g. "+1 SD dot density → +x CER points". Check whether this holds within a model family and after fine-tuning.
  - Drop hallucinated outputs first, and handle them separately as a failure-rate model.
- **Item-response view (optional):** treat models as "test takers" and samples as "items" to estimate per-sample difficulty and discrimination. This shows which samples separate good from bad models (useful for a small, sharp challenge set) and finds unanswerable items (often label errors, which links back to L4).

### L7: Controlled probes (later, optional, needs GPU)
Natural data mixes factors together. Synthetic rendering isolates them. Render
the *same* text in a few fonts, then vary exactly one factor:
- dots removed;
- diacritics on vs. off;
- an isolated-form rendering (no joining);
- resolution (x-height 8 → 40 px).

Run these through 3–4 models. This turns L6's correlations into causal
statements. The infrastructure exists: `datasets/synthetic.py`.

## Phases and deliverables

| phase | work | deliverable | compute |
|---|---|---|---|
| **P0 Inventory** | which datasets load locally or from cache; licences; which source metadata (writer/page/style) exists; which prediction files are usable (no duplicate runs, prompt version known) | inventory table at the top of `data-profile.md` | none |
| **P1 Text profiler** | `src/arabic_ocr_bench/characterize/text.py` (pure functions: joining forms, rasm map, PAWs, diacritics, hygiene) + `scripts/profile_datasets.py` writing `results/characterize/features_text.parquet` (one row per sample) + tests | L1 tables | CPU, minutes |
| **P2 Error taxonomy** | `characterize/errors.py` + `scripts/error_taxonomy.py` over all predictions; rasm-CER | taxonomy table per (model, dataset); `error-taxonomy.md` | CPU |
| **P3 Annotation audit** | conventions; consensus label-noise flags with manual review of the top 50 per dataset; duplicate/leakage scan; metadata recovery in adapters | label-noise rates, flag lists, leakage report | CPU + ~2–3 h manual review |
| **P4 Language layer** | reference corpora; entropy, Zipf, OOV; CAMeL morphology/dialect ID | Arabic-vs-English and cross-dataset language tables | CPU |
| **P5 Image layer** | `characterize/image.py`; sampled features; manual style labels | L3 tables | CPU, ≤ 1 h |
| **P6 Relations + difficulty model** | JS distance map; mixed-effects + SHAP; IRT optional | ranked difficulty drivers; similarity map | CPU |
| **P7 Synthesis** | one **datasheet card per dataset**; `data-profile.md`; an "Atlas" artifact page with charts; challenge subsets registered as dataset filters; implications pushed into `dataset-design.md`, E1/E2 hypotheses and `findings-and-notes.md` | the shareable result | — |
| P8 (optional) | controlled synthetic probes | causal check of the top 3 drivers | GPU |

P1 and P2 are first because they are cheap, need no new dependencies, and
answer the dot/loci questions directly. P3 comes next because label noise can
invalidate everything after it.

## Engineering notes

- New package `src/arabic_ocr_bench/characterize/` (`text.py`, `errors.py`,
  `image.py`, `lang.py`). Keep the functions pure, with no I/O, and put the
  thin CLIs in `scripts/`.
- Output is parquet/JSON under `results/characterize/<run-stamp>/`, keyed by
  `sample_id` so it joins with `predictions.jsonl` and `scores.jsonl`.
- CAMeL Tools and opencv/scikit-image go in an optional `[characterize]`
  extra in `pyproject.toml`. P1 and P2 need neither.
- Tests cover hand-built cases:
  - joining forms of `بيت` (init/med/fin) and `دار` (all isolated, 3 PAWs);
  - a ب→ت dot error, a dropped fatha and a lam-alef split, each classified correctly;
  - the rasm map collapses ب ت ث ن ي (initial forms) to one class.
- Reproducible: seeded samples, and versioned rasm and shape tables in the
  repo.

## Risks
- **Prediction files are heterogeneous** (prompt versions, truncated runs).
  P0 must pick one canonical run per (model, dataset).
- **The joining logic is approximate in handwriting.** Writers sometimes break
  or add joins. GT-derived positional forms are the *intended* forms, so say
  so in the write-up.
- **Correlation is not causation in L6,** which is why P8 exists.
- **Manual labels** (script style, label-noise review) take real time. Budget
  for them and keep the sample sizes small.

## Revised phases (2026-10-07)

After [`characteristics-framework.md`](characteristics-framework.md), every
characteristic has to inform a decision (D1 data, D2 model, D3
transformations, D4 evaluation) and pass the validation gates V1–V6. This
reorders the work: error decomposition comes first, because it turns every
existing prediction file into evidence at no compute cost.

| phase | work | gate it serves | compute |
|---|---|---|---|
| **P0** | inventory (done); one canonical run per (model, dataset) | — | none |
| **P1** | text profiler: shapes, dots, PAWs, diacritics, hygiene, exact leakage (done) | V1 | CPU |
| **P2** | **error decomposition**: normalisation ladder, rasm-CER / dot tax, diacritic-mismatch tax, PAW-space errors, error taxonomy, hallucination rate | V2 inputs; D3/D4 | CPU |
| **P3** | **consensus difficulty**: ISRI-style quality groups, IRT difficulty and discrimination, challenge subsets | D4; feeds P4 | CPU |
| **P4** | **annotation audit**: label-noise flags (consensus + manual top-50), near-duplicate/MinHash and image-hash leakage, convention and inconsistency report (same word spelled several ways) | D4, D1 | CPU + manual |
| **P5** | language layer (register, OOV, entropy, numbers and names) and tokenizer view (fertility, dot-pair tokens) | D2, D3 | CPU |
| **P6** | image layer (resolution in px per x-height, dot size, degradation, style labels) | D2, D3 | CPU |
| **P7** | **validity analysis (V2)**: mixed-effects + leave-one-dataset-out + leave-one-model-family-out; which characteristics survive | V2 | CPU |
| **P8** | **decision experiments (V5)**: (a) normalisation and post-processing choices re-scored on existing predictions; (b) coverage-balanced vs random fine-tuning data at equal size; (c) pre-processing (binarise/deskew) per model family | V5 | (a) CPU, (b)(c) GPU |
| **P9** | **causal probes (V3)**: synthetic single-factor renders (dots, diacritics, resolution, isolated forms) | V3 | GPU |
| **P10** | **cross-script check (V4)**: run the toolkit on Persian/Urdu/Pashto sets and on IAM | V4 | CPU (+ GPU for models) |
| **P11** | synthesis and release (V6): datasheet cards, Atlas, library, write-up | V6 | — |
