# 04 · Paper plan (v0.1, 2026-11-02)

## Working titles

1. *Tadqiq: Finding and Fixing Ground-Truth Errors in Arabic Text-Recognition Datasets*
2. *Is Your Arabic Benchmark Clean? A Taxonomy, Toolkit and Benchmark for Dataset Errors in Arabic OCR/HTR*
3. *Agree to Disagree: Reader Ensembles, Script-Aware Checks and the Hidden Errors of Arabic HTR Datasets*

Recommendation: title 2 for a dataset/benchmark track; title 1 for a tools/demo track.

## Claim (one sentence)

Public Arabic OCR/HTR datasets contain systematic, measurable ground-truth and crop errors. This matters because the errors change the CER figures systems report and the ranking of those systems. Most of the errors can be found automatically by script-aware checks and reader ensembles, and a small, provably safe subset can be fixed automatically.

## Contributions

1. **Taxonomy.** 71 error types in 10 families (v0.1: 18 built, 49 planned, 4 future) ([02-error-taxonomy.md](02-error-taxonomy.md)). Each type has a definition, an example, the evidence it needs and its fix class. Arabic-specific types (rasm/dots, clitics, PAWs, Unicode look-alikes, Maghrebi dotting) are covered explicitly.
2. **Toolkit.** An open-source Python package with tiered detectors: text, image, joint, reader and dataset. It has a patch format with provenance and a conservative fix policy (A0/A1/S/H). The package is `tadqiq`, derived from `dataqa.py`.
3. **Detection benchmark.**
   - **TadqiqBench-syn:** injected errors on clean lines from about 4 datasets, which gives exact recall.
   - **TadqiqBench-real:** human-verified real errors, which give precision and recall. Starting sources:
     - RASAM by-eye and clean lists, about 120 lines;
     - the cleaned-Muharaf flags (if the licence allows);
     - new verification of about 1k flags across datasets.
4. **Audit of public datasets.** Findings so far:
   - **RASAM-1/2 and TariMa:** broken crops, unproofread pages, doubled and shifted references.
   - **KHATT:** 67 % of the test GTs also appear in train.
   - **AMIDDA:** the test split is exactly the union of three other test sets.
   - **Muharaf and NAKBA:** off-script lines.

   Extend the audit to about 10 datasets.
5. **Downstream effect.** Cleaning changes CER and the ranking of systems. Our RASAM-2 result moves the best system from 7.71 to 5.47 % CER (page-bootstrap CIs). Add 1–2 more datasets, and show that training on cleaned train data helps, or say plainly that it does not.

## Experiments

| # | question | design | status |
|---|---|---|---|
| E1 | How common is each error type? | Scan about 10 datasets; report flag rates. Estimate precision from a stratified sample of 50 flags per check and dataset, with Wilson CIs | 9 sets scanned; precision by eye on small samples |
| E2 | Detector recall and precision | TadqiqBench-syn at 1/5/10 % injection rates, per type; TadqiqBench-real | not started |
| E3 | Do reader ensembles separate "GT wrong" from "hard"? | k = 2–4 readers (CRNN folds, Kraken, HATFormer, a VLM). Plot d_rr vs d_rg and compute AUROC for GT-error vs hard on verified lines. Ablate the number and diversity of readers | readings exist for 2 cross-fitted CRNN folds |
| E4 | Are the A1 fixes safe? | Precision of applied A1 patches on verified lines (target ≥ 99 %, with CIs) | not started |
| E5 | Does cleaning change the conclusions? | Score systems on the official vs cleaned test sets. Test whether rankings flip (Kendall τ with a bootstrap). Train on the official vs cleaned train sets | RASAM done |
| E6 | Cost | CPU seconds per 1k lines per tier; reader cost separately | quick to measure |
| E7 (optional) | Human-in-the-loop efficiency | Lines verified per hour: random order vs flag-ranked order (a pilot with 2–3 annotators) | needs people |

## Datasets (licence check required for each before publishing examples)

- **Already done:**
  - RASAM-1/2 (Apache-2.0);
  - TariMa;
  - AMIDDA;
  - Muharaf (CC-BY-NC-SA);
  - KHATT (research licence; the Teklia HF copy);
  - NAKBA / Omar al-Saleh;
  - KITAB KHATT.
- **Candidates:**
  - an MSA printed-OCR set (e.g. a public synthetic or printed Arabic line set);
  - a Persian/Urdu set, to test whether the taxonomy holds for other Arabic-script languages;
  - an ASR transcript set, as a pilot for the S family.

## Figures

1. **Taxonomy overview.** A tree of families and types, with the tier icons.
2. **Gallery.** One line per major error type: image, GT, readings and flag. These are real examples (RASAM-1 000475 / 000387 / 000709, RASAM-2 000325, KHATT 000143, Muharaf 000141 and 000979, AMIDDA 000622).
3. **Firing-rate heat map:** dataset × check.
4. **Reader-ensemble scatter plot** (d_rr vs d_rg), coloured by verified label.
5. **Recall by error type at injection rate** (TadqiqBench-syn).
6. **CER before and after cleaning**, with CIs, and the ranking change.

## Venues (verify every date before planning around it)

- **ICDAR 2027** main or dataset track, and the IJDAR journal extension. This is the best fit for the community.
- **ArabicNLP 2027** (co-located with ACL/EMNLP). This is the best fit for the Arabic audience and for a toolkit or demo.
- **LREC-COLING 2028.** Strong for resource and toolkit papers.
- **NeurIPS Datasets & Benchmarks.** Ambitious; needs the broad audit and the human study.
- **Workshops:** WML (ICDAR workshop on machine learning), and HIP (historical document imaging).

Suggested path: an ICDAR or ArabicNLP paper first, then a journal version with the platform data (votes, new types).

## Related work to read and cite (to do: none checked yet in this project)

- **Confident learning / cleanlab.** Northcutt et al., and the "pervasive label errors in test sets" study.
- **Dataset cartography** (Swayamdipta et al.).
- **Label errors in HTR.** The cleaned-Muharaf study (`alazzawi2026labelerr`), and any IAM/READ ground-truth audits.
- **Arabic NLP tools:** CAMeL Tools, Farasa, MADAMIRA (lexicon and morphology).
- **Documentation:** Datasheets for Datasets, Data Statements, HF dataset cards.
- **Annotation and arenas:** Label Studio, Argilla, Chatbot Arena, Dawid–Skene.
- **Arabic OCR/HTR benchmarks:** KHATT, Muharaf, RASAM, the Calfa work.

## Threats to validity

- **Thresholds were tuned on RASAM, which risks overfitting.** Mitigation: keep held-out datasets for E1/E2, and report the default thresholds frozen before the scan.
- **Synthetic errors may be easier than real ones.** Mitigation: always report the syn and real results side by side.
- **Readers trained on the same data share its biases.** Mitigation: cross-fitting, architecture diversity, and VLM readers that never saw the data.
- **Verification by one person introduces bias.** Mitigation: two annotators on a subset; report kappa.
- **Licensing limits the examples we can show.** Mitigation: publish flags by line ID; show images only where allowed.

## Writing plan

**Format.**
- 8 pages plus references.
- Sections: intro → taxonomy → toolkit → benchmark → audit → downstream → discussion (limits, product outlook in a paragraph).
- Draft in `docs/tadqiq/paper/` (LaTeX, `references.bib` shared with `docs/research/`).
