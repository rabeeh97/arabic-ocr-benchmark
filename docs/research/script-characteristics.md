# Arabic script & language as a recognition problem

> **Status:** draft · **Last updated:** 2026-10-06 · **Wave:** 0
> Citations are BibTeX keys in [`references.bib`](references.bib). A claim marked
> **(unverified)** was not checked against the primary source.

## Question

What *properties of the Arabic script and language* make recognising it hard,
how does each one compare with Latin/English (where the same questions were
studied for decades), and which of them should we measure on our own data,
target in experiments, and control for in our own dataset?

Each property below is laid out as **what it is → the English-side analogue →
evidence → how we measure it here**.

## Key papers

| key | venue, year | data | method | key result |
|---|---|---|---|---|
| `alazzawi2026gap` | ICPR-W (TIPS) 2026 | 6 Arabic-script (KHATT, **Muharaf**, PHTD, NUST-UHWR, PHTI, Ajami) + 3 Latin (IAM, READ-2016, NorHand) | one CRNN, training size K=100…full; HTR-VT control | Arabic−Latin gap **+26.6 CER at K=100, +5.8 at full scale**; Arabic has **82 visual shapes vs 50 (IAM)** and a heavy-tailed shape distribution; **~30 % of Arabic substitutions are visually-similar confusions vs 15 % IAM**; KHATT needs ~150k training chars to hit 10 % CER vs ~34k for IAM; gap persists in a Transformer |
| `alazzawi2026labelerr` | arXiv 2601.16713, 2026 | KHATT, Muharaf, PHTI, Ajami, PHTD | CER-based detector + human verification | label errors common; **Muharaf 6–9 % label errors per split** (as cited by `alazzawi2026gap`); cleaning → up to −1.8 CER |
| `alazzawi2026crossling` | arXiv 2605.02089, 2026 | KHATT, NUST-UHWR, PHTD | joint training across Arabic-script languages | joint training helps at 100–1,000 lines; CRNN transfers more reliably than ViT |
| `alshaibani2025dotless` | Computational Linguistics 51(2), 2025 | Quran, poetry, news, Wikipedia, UN parallel EN–AR | dotless (rasm-only) re-encoding; entropy, Zipf/Heaps, 7 NLP tasks | 31 symbols → **19 rasm classes**; char entropy **4.27 → 3.72–3.87 bits**; English 4.12–4.17; word entropy AR 12.2–13.2 vs EN 9.7–11.0 bits; vocab AR ≈ 1.8–2.3× EN (Wikipedia / UN); tasks comparable dotless |
| `alomari2026iconicity` | arXiv 2608.02935, 2026 | Arabic NLP tasks | 2,000 random remappings onto the 19 undotted shapes | NLP models rely on distribution, not glyph iconicity — the *visual* distinctions matter for OCR, not for downstream LM |
| `lorigo2006survey` | IEEE TPAMI 28(5), 2006 | — | survey | canonical list of Arabic HTR difficulties: positional forms, PAWs, dots, ligatures, overlap, baseline |
| `mahmoud2014khatt` | Pattern Recognition 47(3), 2014 | 1,000 writers | dataset | demographic metadata per writer: age group, gender (677 M / 323 F), handedness (928 R / 72 L), country, qualification |
| `altwaijry2021hijja` | Neural Comput. & Appl., 2021 **(venue unverified)** | 591 children, 7–12 y, Riyadh | isolated letters | 47,434 samples, **108 classes = letters × positional forms**; first children's Arabic set |
| `srihari2016children` | J. Forensic Sci., 2016 | grades 2–6, same school district, 3-year longitudinal | FDE features + automatic comparison | children's writing starts close to the taught model; **individuality emerges markedly between grades 3 and 4** and inter-writer distance grows with grade |
| `srihari2002individuality` | J. Forensic Sci. 47(4), 2002 | 1,500 US writers, stratified by gender/age/ethnicity | writer features | handwriting is individual; the canonical "representative writer sample" design |
| `marti2002iam` | IJDAR 5, 2002 **(unverified details)** | 657 writers, English | dataset | the Latin reference for line-level HTR and writer-disjoint splits |
| `mueller2012letter` | Acta Psychologica 139, 2012 | review of 70+ confusion studies | letter-confusion modelling | confusability = perceivability × similarity × bias; clusters such as E–F–P, O–Q–C–G |
| `perea2010tl` | Psychon. Bull. Rev. 17(3), 2010 | Arabic readers | masked priming | transposed-letter priming only when **root letter order** is preserved — morphology shapes letter-position coding |
| `alluhaybi2020connect` | Q. J. Exp. Psychol., 2020 | Arabic readers | lexical decision | letter connectedness (ligation) affects word recognition **(details unverified)** |
| `zeghli2025bigram` | Atten. Percept. Psychophys. 87(4), 2025 | Arabic readers | flanking-letters lexical decision | open-bigram effects like Latin — some orthographic coding is universal |
| `shannon1951` | Bell Syst. Tech. J. 30, 1951 | English | prediction experiments | English ≈ 0.6–1.3 bits/char *in context* — the reference point for contextual redundancy |

## Properties

### 1. Positional forms — the letter's "locus" in the word
- **What:** 28 letters, each with 2–4 contextual glyphs (isolated / initial / medial
  / final). The shape is chosen by the neighbours' joining class, not by the
  writer. A recogniser therefore has to learn **~80–110 visual classes for ~30
  output symbols**: Hijja uses 108 position-classes, and `alazzawi2026gap`
  counts 82 shapes in KHATT/Muharaf.
- **English analogue:** upper vs. lower case (≈ 50 shapes in IAM). The
  difference is that case is rare and predictable, while Arabic positional
  forms occur in every word.
- **Evidence:** this is the main structural explanation `alazzawi2026gap` gives
  for the Arabic−Latin gap. More shapes means fewer training examples per shape,
  and the shape distribution has a heavy tail.
- **We measure:** the positional-form distribution per dataset, derived from
  joining classes with Unicode `ArabicShaping.txt` logic, and the coverage of
  rare (letter, position) pairs in our 8k Muharaf training lines.

### 2. Connectivity, PAWs and word segmentation
- **What:** 6 letters never join to the left (ا د ذ ر ز و, plus ء and the
  ة/ى finals). They split a word into **pieces of Arabic words (PAWs)**. The
  space between PAWs inside a word can be as wide as the space between words.
- **English analogue:** inter-word spacing in cursive handwriting, and
  word-shape ("Bouma") theories of reading.
- **Evidence:** eye-tracking studies find Arabic readers barely need
  inter-word spaces, because ligatures are position-informative (unverified,
  JEMR / J. Psycholinguist. Res. 2026). Ligation affects lexical access
  (`alluhaybi2020connect`). For OCR, PAW gaps cause **space insertion and
  deletion errors**, which WER punishes heavily.
- **We measure:** PAWs per word and the PAW-gap vs. word-gap ratio (GT-side
  only). In errors, we count space insertions/deletions that fall at a PAW
  boundary.

### 3. Rasm vs. i'jam — the dots
- **What:** many letters share one skeleton (*rasm*) and are told apart only by
  the number and position of dots (ب ت ث ن ي in initial/medial position, ج ح خ,
  د ذ, ر ز, س ش, ص ض, ط ظ, ع غ, ف ق). 15 of the 28 base letters are dotted.
  Remove the dots and 31 symbols collapse to **19 rasm classes**
  (`alshaibani2025dotless`).
- **English analogue:** the letter-confusion matrices of perception research
  (`mueller2012letter`: E–F–P, O–Q–C–G), and the OCR confusion sets rn↔m,
  cl↔d, 0↔O.
- **Evidence:** dot or shape confusions are **~30 % of Arabic substitutions vs.
  15 % in IAM**. The top pairs are ف↔ق, ب↔ي and ت↔ن (`alazzawi2026gap`). In
  handwriting, dots are small, get displaced, merge, or vanish under
  binarisation and downscaling (`lorigo2006survey`). From the NLP side, dots
  carry only **~0.4–0.5 bits/char**. That suggests a language prior could
  restore many of them, which is a testable post-OCR idea.
- **We measure:** the share of each model's character substitutions that stay
  inside the same rasm class (the *dot-error rate*), plus a rasm-only CER.
  The gap between CER and rasm-CER is the "dot tax".

### 4. Ligatures, stacking, kashida, baseline
- **What:**
  - **Ligatures:** the mandatory lam-alef (لا), and optional calligraphic
    ligatures in Naskh and Ruq'ah.
  - **Vertical stacking:** letters can stack vertically (e.g. لمح).
  - **Kashida/tatweel:** an elongation that carries no meaning.
  - **Baseline:** letters sit on a baseline, but descenders and ascenders
    overlap neighbouring PAWs and lines.
- **English analogue:** ascenders and descenders, and the IAM-era
  slant/height normalisation.
- **We measure:**
  - lam-alef and tatweel frequency in GT;
  - in errors, whether lam-alef is split or merged;
  - inter-line overlap, later, from images.

### 5. Diacritics (harakat): see A1, `diacritics.md`
They are optional, mostly absent in modern text, and present in religious and
children's text. They are a separate topic because they are both a visual and an
annotation-policy problem.

### 6. Language-level statistics
- **Entropy and redundancy:**
  - Arabic unigram char entropy is 4.19–4.27 bits, vs. 4.12–4.17 for English on
    comparable corpora.
  - Word entropy is much higher in Arabic: **12.2–13.2 vs. 9.7–11.0 bits**.
  - Arabic vocabulary is **≈ 1.8–2.3× English** on comparable corpora.
  - Source for all three: `alshaibani2025dotless`.
  - Root-and-pattern morphology plus attached clitics make the vocabulary
    open-ended, so lexicon-constrained decoding (a classic for English OCR)
    gives less help.
- **Ambiguity without diacritics:** analysers report about **12 analyses per
  undiacritised MSA word** out of context (Habash-group figures, unverified
  exact source). This is why GT diacritisation policy matters.
- **Letter position and morphology in reading:** Arabic readers code letter
  position relative to the **root** (`perea2010tl`), but also show
  Latin-like open-bigram effects (`zeghli2025bigram`). The orthographic
  "units" are partly universal and partly morphological.
- **Variety:** Classical, MSA and dialect, Arabic-Indic vs. Western digits,
  embedded Latin. Each one shifts the character distribution.
- **We measure:**
  - char and word entropy per dataset;
  - out-of-vocabulary rate of the test sets against the training lexicon;
  - the share of digits and Latin characters.

### 7. Writer variability and demographics
- **What:** how much handwriting varies between and within writers, and which
  writer attributes explain that variation.
- **English analogue:** CEDAR individuality (`srihari2002individuality`),
  children's development (`srihari2016children`), and the IAM writer-disjoint
  protocol.
- **Evidence:**
  - Individuality grows with schooling and **diverges sharply around grades
    3–4**. Younger children write close to the taught model.
  - KHATT is the main Arabic precedent for recording age group, gender,
    handedness, country and education.
  - Hijja is the only children's Arabic set, and it covers **isolated letters
    only**, not lines.
  - There is a research line predicting age, gender and handedness *from*
    handwriting (an ESWA 2023 survey). It shows these attributes leave visible
    traces, but not that they change OCR difficulty. **We found no study that
    measures HTR CER as a function of writer demographics. That is an open
    gap.**
- **We measure:** per-writer CER variance where writer IDs exist (KHATT
  metadata), as an upper bound on how much demographics *could* matter.

## Synthesis

1. **The Arabic−Latin gap is real, measured and only partly a data problem.**
   Label noise explains some of it, and Muharaf, our training set, is among
   the noisiest. Positional-form inflation and heavy-tailed shape frequencies
   explain the remainder (`alazzawi2026gap`).
2. **Dots are the single most identifiable source of error.** They are
   visually small, script-specific, and cheaply detectable in alignments.
   That makes them the ideal target for an error taxonomy and for both
   experiments: a frozen ViT may lose fine dot detail, and a tokenizer may
   merge dotted/undotted pairs.
3. **Language statistics cut both ways.** Arabic has high word entropy and an
   open vocabulary, so lexicons help less. But dots carry only ~0.4 bits/char
   given the rasm, so a strong language prior can restore them.
4. **Children's handwriting is under-studied for Arabic text lines.** Only
   letter-level data exists (Hijja), and grade is the demographic with the
   best-supported effect on writing variability.

## Implications for us

- **[data] Muharaf label noise.** `alazzawi2026labelerr` reports 6–9 % label
  errors in Muharaf. Find out whether their cleaned version is public. If it
  is, re-score the existing fine-tunes on cleaned test lines and consider
  retraining on cleaned training lines. This may move every number in
  `findings-and-notes.md`. → new item in `findings-and-notes.md`.
- **[metric]** Add **rasm-CER** (CER after mapping both sides to the 19 rasm
  classes) and a **dot-error share** next to CER-light. The gap between
  them is the "dot tax". → `metrics.py`, via `scripts/error_taxonomy.py`.
- **[metric]** Report **space errors at PAW boundaries** separately from
  ordinary WER.
- **[hypothesis] E1 (ViT unfreeze).** If the frozen encoder limits fine
  detail, unfreezing should reduce the *dot-error share* more than the
  overall CER. Use this as the pre-registered primary outcome of E1, and
  Maghrebi as the secondary.
- **[hypothesis] E2 (tokenizer).** Check whether the VLM tokenizers map
  dot-pairs (ب/ت, ف/ق) to shared or unrelated tokens. Test whether
  dot-errors concentrate where the tokenization differs.
- **[hypothesis] Data scaling.** Measure training data in **characters per
  (letter, position) class**, not lines. `alazzawi2026gap` puts the 10 % CER
  point at ~150k characters for KHATT. Our 8k Muharaf lines are about this
  order of magnitude; compute it exactly in `data-profile.md`.
- **[dataset]** The copy-text must balance **(letter × positional form)
  coverage** and dot-group minimal pairs, not just letter frequency.
- **[dataset]** Stratify writers by **grade, crossing the grade 3–4
  boundary**. Record handedness and gender in the KHATT style. Because the
  link between demographics and CER is unmeasured in the literature, treat it
  as a research question our dataset can answer.
- **[dataset]** Use writer-disjoint splits, and **label quality control from day
  one**: double annotation on a subset, plus a CER-HV-style error check
  (`alazzawi2026labelerr`).

## Open questions
- Is the cleaned Muharaf release public, and under which licence?
- How do *VLMs* (not CRNNs) distribute their errors across these categories? We
  can answer this ourselves from existing predictions.
- Does a language prior fix dot errors after OCR, and would it then
  hallucinate on rare words, proper nouns and dialect?
- Is there any Arabic study that links the reader's (or writer's) age to
  letter-confusion patterns? This would mirror the developmental confusion
  matrices that exist for Latin script.
