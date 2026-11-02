# 01 · Is this a valid idea? (assessment, 2026-11-02)

**Short answer: yes, as both a paper and a product.** There are caveats on novelty, scope and claims of completeness, set out below.

## What the idea is

**Tadqiq (تدقيق, "auditing / proofreading"; working name)** is a toolkit that finds likely errors in Arabic
datasets automatically. It covers the ground truth (GT) and, where there is one, the paired signal: the image for OCR/HTR, and later audio for ASR.

- **It fixes an error only when the fix is provably safe.** Everything else is routed to people.
- **It is built for growth.** People confirm flags, report new kinds of error, and those reports become new detectors.

It has three layers:

1. **Detectors.** These are automatic checks in tiers by what they need: text only, image only, image + text, a reader model, the whole
   dataset. Seventeen checks are built and validated (`docs/research/dataqa.md`), covering 18 of the 71 types; the other 53 are specified in
   [02-error-taxonomy.md](02-error-taxonomy.md).
2. **Evidence and adjudication.**
   - **Several independent signals per line.** These include checks, two or more readers, a character LM and a lexicon.
   - **Model agreement tells the two cases apart.** When the models agree with each other but not with the GT, the GT is suspect. When the models disagree with each other, the line is hard.
   - **People decide in an arena-style vote** when the evidence is not conclusive.
3. **A taxonomy that grows.** Every confirmed error has a type. New types reported by people are grouped and named.
   Once a type has enough confirmed examples, it becomes a public detection task with its own test set.

## Why it is worth doing

- **The problem is real and measured, not assumed.** In our own work, three public Maghrebi benchmarks had:
  - broken test crops;
  - unproofread pages;
  - references pasted twice or shifted to the wrong line.

  Fixing the test set moved the best system's CER on RASAM-2 from 7.71 to 5.47 %. The scans we ran found two more things:
  - KHATT shares 67 % of its test text with its training text;
  - AMIDDA's test split is exactly the union of three other test sets.

  None of these is documented on the dataset cards.
- **Arabic needs its own tools.** Generic data-quality tools (cleanlab-style label-noise detection, schema validators,
  annotation platforms) do not know about:
  - rasm vs dots;
  - clitics;
  - hamza and alif variants;
  - right-to-left storage;
  - Maghrebi dotting;
  - pieces of connected letters (PAWs);
  - the Unicode traps of Arabic text (presentation forms, ی vs ي, invisible marks).

  These are where most Arabic GT errors live.
- **It fits current demand.** Arabic LLMs and OCR systems are being built at national scale across the Arab world. Their quality depends on data
  that nobody audits systematically. Digitisation programmes also need a way to accept or reject vendor output.

## Honest caveats

- **"We covered almost every error type" is too strong.** What exists covers most errors of **line-level OCR/HTR**
  datasets. Still missing:
  - page and layout data (region and reading-order errors, wrong polygons);
  - word and character-level annotation (boxes);
  - transcription *policy* errors: the text is consistent but follows a different convention than the dataset claims, e.g. normalised
    vs diplomatic transcription, or expanded abbreviations;
  - semantic and label errors in NLP datasets (sentiment, NER, dialect labels);
  - speech transcripts (audio–text alignment).

  The taxonomy is open-ended by design, and the product's job is to keep finding new types. The paper should claim
  *a taxonomy and a toolkit that grow, with measured coverage*, never completeness.
- **Novelty has to be argued.** Related work to read and position against (not yet checked in this project):
  - label-noise detection in general (confident learning / cleanlab, dataset cartography);
  - HTR label-error papers, e.g. the cleaned-Muharaf study already in `references.bib` (`alazzawi2026labelerr`);
  - Arabic NLP tool suites (CAMeL Tools, Farasa) for morphology and the lexicon;
  - data cards and documentation (Datasheets for Datasets, HF dataset cards);
  - annotation and arena platforms (Label Studio, Argilla, Chatbot Arena).

  Our contribution is the combination: an Arabic-specific taxonomy, image–text and reader-based detectors, a
  benchmark that measures detectors (synthetic and verified real errors), and audit findings on public datasets.
- **Auto-correction must be conservative.** "100 % sure" is achievable only for a small class of fixes:
  - lossless encoding fixes;
  - re-cuts from page geometry that a reader confirms;
  - removing a pasted copy that two independent readers confirm.

  Everything else is a *suggested patch* that a person accepts. Fixes are never applied in place; the toolkit writes a patch file with provenance.
- **Precision varies by dataset.** The thresholds come from RASAM. Measured precision on other sets ranges from about 50 % to 100 %, depending on the check. The product must show
  per-check precision estimates and let people calibrate them with votes.
- **Licensing limits hosting.** Many Arabic datasets are non-commercial (Muharaf CC-BY-NC-SA) or research-only (KHATT).
  The platform can always host *audit results*, i.e. flags and patches keyed by line id. It can host the data itself only where the licence allows.

## Verdict

Go. Start with the toolkit and paper, which reuse what is built and validated. Design the platform so the toolkit is its
engine and the paper's benchmark is its first public "leaderboard of detectors". The order and the agent-ready tasks are in
[05-roadmap.md](05-roadmap.md).
