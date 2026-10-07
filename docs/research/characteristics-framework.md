# Which characteristics, and why: a decision-driven framework

> **Status:** draft · **Last updated:** 2026-10-07 · **Wave:** 0
> This note decides *what* we characterise. [`characterization-plan.md`](characterization-plan.md)
> says *how* and *in which order*. [`script-characteristics.md`](script-characteristics.md)
> covers the script-level literature. Citations are BibTeX keys in [`references.bib`](references.bib).

## 1. The rule: a characteristic earns its place by changing a decision

Profiling a dataset can produce hundreds of numbers. We keep a characteristic
only if it can inform at least one of four decisions:

| | decision | example question |
|---|---|---|
| **D1** | **Data:** which datasets to select, mix, augment, or create | Which shapes or styles are under-covered in our training data? What must the school dataset contain? |
| **D2** | **Model:** which base to use, which output units, which input resolution, what to fine-tune | Is the vision encoder losing dots? Should the output unit be a character, a shape, or rasm + dots? |
| **D3** | **Transformations:** normalisation and pre-/post-processing | Is folding X lossless? Does binarisation help a VLM? Can a language model put dots back? |
| **D4** | **Evaluation:** metrics, splits, what to report, what to exclude | Is this test set leaked? Is a CER gap recognition or annotation policy? |

A characteristic is **validated** when it also clears a bar for *mattering*
(§7). It has to explain model errors, or change a conclusion when we act on
it. Until then it is a hypothesis.

## 2. What the field has measured before (and what we copy)

**Printed OCR, 1990s (ISRI/UNLV annual tests).** Rice, Nartker and Nagy ran
OCR engines on thousands of pages.
- **Page quality groups.** They ranked page difficulty by the *median accuracy
  across all engines* and split pages into five quality groups
  (`rice1995isri`).
- **Measured effects.** Resolution: 300 → 200 dpi gave about 50 % more errors.
  They also measured skew, font features, and a taxonomy of error *causes*
  (imaging defects, similar symbols, typography) (`nagy2000frontier`).
- **We copy:** cross-model consensus difficulty (we have 489 prediction files),
  error causes rather than just error counts, and resolution as a first-class
  variable.

**Document image quality assessment (DIQA).** This field predicts OCR accuracy
from image features alone, without ground truth (`ye2013diqa`; ACM survey
2023). **We copy:** layer L3 is designed as a no-ground-truth difficulty
predictor, so it also works on unlabelled data such as the school scans.

**HTR datasets (IAM, KHATT, IFN/ENIT, AHTID).** Each reports writer count,
demographics, lexicon size and writer-disjoint splits (`marti2002iam`,
`mahmoud2014khatt`). **We copy:** these as the minimum datasheet fields.

**Benchmark hygiene.**
- Scene-text recognition benchmarks: inconsistent evaluation subsets alone
  caused accuracy gaps of more than 15 % (`baek2019wrong`).
- Arabic-script HTR: test sets carry 6–9 % label errors
  (`alazzawi2026labelerr`).
- **We copy:** leakage, duplicate and label-noise audits before *any*
  comparison. We already found `kitab_muharaf` 86.5 % in-train.

**Instance difficulty.**
- *Dataset cartography* (`swayamdipta2020cartography`): items that stay hard
  during training are often mislabelled.
- *Item response theory for leaderboards* (`rodriguez2021irt`): treat models as
  test takers to find informative items and annotation errors.
- **We copy:** a per-sample difficulty and discrimination score, plus
  challenge subsets.

**Script-structure decomposition in other scripts.**
- **Chinese.** Thousands of characters with a long tail are decomposed into
  about 500 *radicals*. That lets models recognise rare or unseen characters
  (`zhang2020ran` and successors).
- **Urdu Nastaliq.** Recognition works on the *ligature*, a unit between
  character and word. Each ligature splits into a **primary component** (the
  body) and **secondary components** (dots and diacritics). About 2,300 units
  cover 99 % of text (`lehal2013urdu`).
- **Online Arabic.** Dots are "delayed strokes", and dedicated detection and
  re-attachment steps improve recognition.
- **We copy:** the decomposition of Arabic into **rasm (body) + i'jam (dots) +
  harakat**, as a measurement, as a metric (rasm-CER), and as a possible model
  output design (D2). The PAW is Arabic's equivalent of the Urdu ligature unit.

**Script normalisation.**
- Graphemic normalisation of Perso-Arabic script folds visually identical but
  canonically different letters. It gave significant gains for language
  modelling and machine translation in 8 languages (`doctor2022graphemic`).
- **We copy:** an *explicit*, documented normalisation ladder, where each rung
  is justified as lossless or as a deliberate fold.

**Arabic OCR evaluation.**
- KITAB-Bench (`heakl2025kitab`) reports results by font, diacritics,
  elongation and tilt. Diacritics and kashida trigger hallucinations and
  broken words.
- A noise study of Tesseract, Textract and Document AI shows Arabic degrades
  far more than English under the *same* synthetic noise (`hegghammer2021noisy`).
- The Latin–Arabic gap study (`alazzawi2026gap`) is the **closest existing
  cross-dataset Arabic characterisation**: shape distributions, confusion
  shares and label noise across six Arabic-script sets.

**Has anyone characterised Arabic datasets this way? Partly.** Dataset papers
report their own statistics, and surveys tabulate size and writers (ACM CSUR
2025 Arabic OCR survey; IJCV 2026 Arabic HTR review). `alazzawi2026gap` goes
furthest, but it uses one CRNN, no VLMs, and none of the following:
diacritic policy, normalisation, register/language, PAW spacing, leakage, or
image layer. **No common, reusable characterisation standard for Arabic OCR
data exists that we could find.** That is the gap we can fill and publish.

## 3. The catalogue: candidate characteristics

**Priority key:** ★★★ = measure now (cheap and decision-critical); ★★ = next;
★ = later or conditional.

| # | characteristic | mechanism (why it could matter) | evidence it mattered | decisions | priority |
|---|---|---|---|---|---|
| A | **Shape inventory and tail**: (letter × position) counts, coverage per shape, shapes needed for 95 % | Each shape needs its own training examples; a heavy tail means rare shapes are never learned | Arabic 82 shapes vs Latin 50, heavy tail → gap (`alazzawi2026gap`); Chinese long tail → radical models | D1 (coverage-balanced selection and synthesis), D2 (output units), D4 (tail-shape error) | ★★★ *(have)* |
| B | **Dot structure**: dot-ambiguous share, dot bits, dot-group mix | Dots are small; the vision encoder or resolution can lose them | ~30 % of Arabic substitutions vs 15 % Latin; Urdu primary/secondary split | D2 (resolution, unfreeze ViT, rasm+dots head), D3 (dot restoration by a language model), D4 (rasm-CER) | ★★★ *(have; needs error side)* |
| C | **Connectivity and spacing**: PAWs/word, PAW gap vs word gap | Gaps inside words look like gaps between words, so models insert or drop spaces | Reading studies; WER is very sensitive to this | D3 (whitespace-insensitive scoring, re-segmentation), D4 (separate space errors), D2 (PAW units) | ★★★ |
| D | **Diacritic policy**: none / sparse / partial / full per dataset, mark types, invalid marks | Ground truth that is mostly *undiacritised* but has stray marks penalises a model that is right | Muharaf: a third of lines sparse; others none. KITAB-Bench: diacritics cause hallucination | D3 (strip or keep by policy), D4 (profile per dataset), D1 (paired condition in our dataset) | ★★★ *(have)* |
| E | **Orthographic variation**: hamza/alef forms, ta-marbuta/ha, ya/alef-maqsura, Persian look-alikes, digit systems | The same word is spelled two ways, both "right"; scoring counts the difference as an error | Normalisation gains in `doctor2022graphemic`; our `light` vs `diac_insensitive` gap | D3 (which folds are lossless), D4 | ★★★ |
| F | **Encoding hygiene**: bidi/zero-width, presentation forms, non-NFC, tatweel | Invisible or rendering artefacts in the ground truth | `kitab_isippt` 6 Persian forms per 1,000 chars; KITAB kashida failures | D3 (mandatory canonicalisation), D4 | ★★★ *(have)* |
| G | **Leakage and duplicates**: exact/near-duplicate lines and images within and across sets | Inflated scores | `kitab_muharaf` 86.5 %; scene-text inconsistencies >15 % | D4 above all, D1 | ★★★ *(exact done; near-dup next)* |
| H | **Annotation quality**: label-noise rate, transcription conventions | A wrong label sets a hard ceiling and adds noise to fine-tuning | 6–9 % label errors in Muharaf | D4, D1 (clean or replace training data) | ★★★ |
| I | **Instance difficulty**: cross-model consensus CER, IRT difficulty and discrimination | Separates hard-but-valid items from broken items; builds sharp subsets | ISRI quality groups; IRT leaderboards | D4 (challenge subsets), H (flags label errors) | ★★★ (needs no new data) |
| J | **Granularity and length**: chars/line, page vs line, text height in px | The prompt, token budget and resize policy assume one granularity | Our KITAB page-scale finding; ISRI resolution effect | D2 (input resolution, `max_new_tokens`), D4 (granularity flag) | ★★ |
| K | **Language and register**: Classical / MSA / dialect / archaic spelling, OOV vs training data, entropy, named-entity and number density | The VLM decoder's language prior "corrects" towards fluent MSA, so it hallucinates on dialect, archaic text and names | Our base-VLM confabulation; Arabic word entropy 12–13 vs 10–11 bits for English | D2 (strength of the language prior), D3 (feasibility of lexicon/LM post-correction), D1 (domain mix) | ★★ |
| L | **Image quality**: blur, noise, contrast, bleed-through, stroke width, skew/slant, line overlap | Degradation removes small features first, and dots are small features | DIQA; ISRI; `hegghammer2021noisy` (Arabic hit harder) | D3 (binarise, denoise, deskew: do they help VLMs?), D1 (augmentation) | ★★ |
| M | **Script style and hand**: Naskh, Ruq'ah, Maghrebi, Diwani, Nastaliq; print vs handwriting | Different shape inventories and dot conventions (Maghrebi ف/ق use different dots) | Our RASAM gap (~50 % CER after fine-tuning) | D1, D2 (unfreeze ViT for style shift) | ★★ |
| N | **Writer and demographics**: writer ID, age/grade, handedness | Variance between writers; individuality grows with grade | `srihari2016children`; no study links demographics to CER | D1 (school dataset), D4 (writer-disjoint splits) | ★ now, ★★★ for our dataset |
| O | **Tokenizer view**: fertility per line, whether dot-pairs share tokens | Decoder units may not match visual units | Our tokenizer discussion; fertility audit in the char-CTC experiment | D2 (tokenizer and output head) | ★★ |

## 4. New layers designed specifically for Arabic

Six measurements we define ourselves, because no existing standard covers them.
Each one is cheap and computed from (GT, prediction) pairs we already have.

1. **The normalisation ladder: an error decomposition.** Score each prediction
   at successive rungs: *raw* → *canonical encoding* (NFC, bidi, presentation
   forms) → *+tatweel and digits* → *+orthographic folds* (alef/hamza,
   ya/maqsura, ta-marbuta, look-alikes) → *+diacritics stripped* → *rasm
   only* → *+whitespace removed*. The CER drop at each rung is the share of
   error **owed to** encoding, orthography, diacritic policy, dots and
   spacing. What is left at the bottom is "shape recognition" proper.
   - Our three profiles are already rungs of this ladder; we add the missing
     ones.
   - This makes D3 measurable: a fold is worth applying exactly when its rung
     removes error that is policy, not recognition.
2. **Rasm-CER and the dot tax.** CER after mapping both sides to the 19
   position-aware skeletons. Dot tax = CER<sub>diac-insens.</sub> − rasm-CER.
   This is the Arabic counterpart of Chinese radical-level error.
3. **Shape-coverage curve.** For a training set: the share of test-set shape
   occurrences whose shape has at least *k* training examples. The prediction:
   per-shape error falls with training count, and a curve tells us what *k*
   is enough. This sizes data (D1) in shapes, not lines.
4. **Diacritic-policy fingerprint and mismatch tax.** Per dataset: the policy
   class plus the mark-type mix. Per (model, dataset): the CER that comes only
   from *inserting* marks where the ground truth has none, or *dropping* stray
   marks. This answers "is the model wrong, or does it just diacritise
   differently?"
5. **PAW-boundary space error rate.** Space insertions and deletions that fall
   at PAW boundaries, as a share of all space errors.
6. **Fluency hallucination rate.** Predicted words with no aligned counterpart
   that are frequent MSA words. This is a signature of the language prior
   overriding the image, which matters for base VLMs.

## 5. Transformations the characteristics should decide (D3)

| transformation | stage | lossless? | decided by | how we validate |
|---|---|---|---|---|
| Canonical encoding (bidi/ZW strip, presentation-form fold, NFC) | eval + post | yes | F | Always on; it already is |
| Tatweel strip, digit unification | eval + post | yes for content | F, E | Ladder rung; no ranking change expected |
| Orthographic folds (alef/hamza, ya/maqsura, ta-marbuta, look-alikes) | eval; optionally post | **no**: they erase real spelling distinctions | E, D | Ladder rung. Report both, and fold only where the dataset's GT is itself inconsistent (measure that inconsistency) |
| Diacritic strip | eval/post *per dataset policy* | no | D | Mismatch tax; apply only when the dataset policy is "none" or "sparse/inconsistent" |
| Whitespace re-segmentation at PAW gaps | post | no | C | Space-error rate before and after; WER change |
| LM dot restoration (predict i'jam given the rasm) | post | no | B, K | Run on rasm-correct but dot-wrong predictions: share fixed vs share broken; check rare words and names separately |
| Meta-commentary / hallucination filter | post | n/a | I, §4.6 | Already have `cleaned_text`; measure the false-positive rate |
| Binarisation, denoise, deskew | pre | no (image) | L | Per model family: CNN/CTC usually benefit; VLMs may not. Test on a stratified subset |
| Resize to a target x-height / max image tokens | pre | no | J, B | Dot-error share vs pixels per dot; the sweep in P8 |

## 6. Questions this should answer

- **Gap.** How much of the Arabic error is *policy* (normalisation, diacritics)
  vs *recognition*? Is it the same for CRNN, VLM and API models?
- **Errors by architecture.** Do VLMs fail differently from CTC models? The
  hypothesis: fewer shape errors, more fluent hallucinations.
- **Resolution.** Below what dot-size in pixels does dot-error explode? Does it
  differ by vision encoder (fixed 448 px SigLIP vs Qwen's dynamic
  resolution)?
- **Tail.** How many training examples does a shape need? Do rare shapes
  improve with synthetic rendering?
- **Diacritics.** Does a model trained on sparsely diacritised Muharaf learn
  to *add* marks, and is that why its raw-profile CER is worse elsewhere?
- **Annotation.** Which datasets are internally inconsistent, for example
  spelling the same word with and without hamza?
- **Transfer.** Does distance between datasets (shape, register, style,
  image) predict transfer after fine-tuning?
- **Cross-script.** Do Persian, Urdu and Pashto share the same difficulty
  drivers? They use identical joining rules, so the framework should port
  unchanged.
- **Humans vs models.** Does the model confusion matrix match human
  letter-confusability (as `mueller2012letter` did for Latin)? If not, models
  fail for non-perceptual reasons, such as the language prior.

## 7. Validating a new characteristic until others can rely on it

A characteristic or metric graduates from "idea" to "standard" through six
gates:

| gate | test | done when |
|---|---|---|
| **V1 Reliable** | deterministic, unit-tested, computed after canonicalisation; bootstrap CI across resamples | CI width is small relative to between-dataset differences |
| **V2 Predictive** | per-sample mixed-effects model of CER, with model and dataset as random effects. Must add explained variance *beyond* trivial baselines (line length, image height). Must hold **leave-one-dataset-out** and **leave-one-model-family-out** | a significant, stable effect in held-out folds |
| **V3 Causal** | a controlled intervention: synthetic rendering that changes only that factor (P8), or a natural experiment (KHATT fixed-text paragraphs written by many writers; paired diacritised/undiacritised lines in our own dataset) | the effect reproduces under intervention |
| **V4 Generalises** | the same code on other Arabic-script languages (Persian PHTD, Urdu NUST-UHWR, Pashto) and the analogous metric on Latin (IAM) | reproduces known results, e.g. the 30 % vs 15 % confusion gap |
| **V5 Useful** | acting on it changes an outcome: excluding leaked rows changes a conclusion; a ladder-chosen fold changes a ranking; coverage-balanced data selection beats random selection at equal size | a decision-level effect |
| **V6 Adoptable** | released as a library + datasheet template + the Atlas; documented; compared with human perception data where relevant | others can run it in one command |

## 8. What changes in the plan

See the revised phases in [`characterization-plan.md`](characterization-plan.md)
§ "Revised phases (2026-10-07)". In short:
- **error decomposition** (the ladder, rasm-CER, taxonomy) moves first;
- **consensus difficulty** and **annotation audit** follow;
- every characteristic passes the V2 predictive test before we build anything
  on it;
- two **decision experiments** are added (data selection by coverage, and
  transformation choice), which make up gate V5;
- a **cross-script** check is added, which makes up gate V4.
