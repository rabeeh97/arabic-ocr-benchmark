# The dataset we would build: living design guide

> **Status:** collecting lessons; nothing collected yet · **Last updated:** 2026-10-09
>
> This is **the** place for everything we learn about what an ideal Arabic
> handwriting dataset needs: what text it contains, how it is labelled, how
> images are captured, how it is split and released. It decides nothing about
> *when* we collect (that is the S track in `NEXT_STEPS.md`).
>
> **How to keep it current.** When a finding, paper or experiment changes what
> the dataset should contain, edit the relevant rule here in the same session.
> Each rule gives its **why** and a source: a bib key, a task ID or a findings
> entry. A rule without a why gets cut. Add a line to the change log at the
> bottom. Questions we can't settle yet go to §9 (open decisions) with the
> evidence that would settle them.

## 0. What "the perfect dataset" means here

There is no universally perfect dataset. Ours should do four things the public
sets don't:

1. **Make the hard parts of Arabic measurable.** Positional forms, dots, PAW
   segmentation, diacritics and script styles should each be covered on
   purpose, so per-phenomenon error rates have enough samples.
2. **Make metrics trustworthy.** Public sets carry 6–9 % label errors
   (`alazzawi2026labelerr`), and our consensus check finds the same.
   A 1-point CER difference means nothing on such labels.
3. **Generalise honestly.** Splits must be writer-disjoint *and* text-disjoint,
   and the test text must be unseen by large models.
4. **Be releasable.** The licence and consent are settled before collection.

The first concrete instance planned is a **school set** (§8). The rules in
§1–§7 apply to any set we build, including a later adult or multi-style one.

Research questions it should answer that no existing set can:
- Does recognition difficulty depend on **writer age/grade**? No HTR study
  measures this (`script-characteristics.md` §7). The only children's set,
  Hijja, has isolated letters only.
- How do the **diacritised and undiacritised** versions of the *same* text
  compare?
- Which errors dominate on children's lines, and how far do models rely on
  the **language prior** rather than on the strokes?

## 1. Text: what goes on the page

Build the copy-text from blocks. Each block has a job, and the mix is
verified with the R1 text analyzer before printing.

| block | what | why (evidence) |
|---|---|---|
| **T1 Letter sheets** | All 28 letters in every positional form (isolated, initial, medial, final), hamza seats (ء أ إ آ ؤ ئ), lām-alif, tāʾ marbūṭa, alif maqṣūra. Written isolated, Hijja-style | The inventory and its long tail drive the Arabic gap (`alazzawi2026gap`: 82 shapes, iso/fin merged; Muharaf train has 88 counted the same way). It is also stage-1 data for the curriculum track (C0, C1) and is comparable with Hijja's 108 classes |
| **T2 Dot minimal pairs** | Real words differing only in dots in the same position, e.g. بيت/بنت/نبت, حرّ/جرّ/خرّ, سرّ/شرّ, صيف/ضيف, طلّ/ظلّ, ذلّ/دلّ, حرب/حزب, عرب/غرب, فطر/قطر | Dot-based confusions are 13 % of letter confusions and 29–41 % of a CRNN's substitutions are "similar letters" (R2.6). Minimal pairs make dot reading measurable without language-model help |
| **T3 Measured confusion pairs** | Words that put ا/ل, ع/م and the other top non-dot pairs from the R2.6 confusion list in the same slot | 65 % of real letter confusions fall *outside* the hand-written shape groups (R2.6), so the pair list comes from measured confusions, not intuition. Refresh it from `substitutions.json` |
| **T4 Orthographic pairs** | ة/ه, ى/ي and the hamza seats in natural words (مدرسة/مدرسه, على/علي) | 16 % of confusions are orthographic (R2.6). We need to tell "can't see it" from "spelling convention" |
| **T5 PAW and spacing stress** | Words full of non-connectors (د ذ ر ز و ا), e.g. وزراء, دار, ورد; long fully joined words; tight and wide word gaps | Deletions and insertions are 28 % of edits (R2.6); that is segmentation, not letter identity |
| **T6 Frequency backbone** | Natural MSA sentences covering the most frequent words | Realistic language; enables the lexical-diversity measure (R1.6) |
| **T7 Coverage-selected sentences** | Sentences picked greedily from a licence-clean corpus to fill every (letter × form) class to a floor (§6) | Balancing by shape, not frequency, fixes the tail (R1.7) |
| **T8 Pseudo-words** | Orthographically valid non-words (e.g. letter-swapped real words) | **Our idea:** a VLM's language prior can "read" a word it can barely see. Pseudo-words measure pure visual reading. Literacy tests use the same trick on children |
| **T9 Paired diacritics** | The same sentences written once with full ḥarakāt and once without | Isolates the diacritic effect (A1); a third of Muharaf lines carry sparse marks, which models then add or drop |
| **T10 Numbers and forms** | Arabic-Indic and Western digits, dates, phone-style numbers, names, addresses, form fields, punctuation, a few Latin words | The product-shaped axis (`findings-and-notes.md`); mixed-direction text breaks many models |
| **T11 Free writing** | A short prompt ("describe your school day") | Natural spelling, spacing and dialect, which copied text can't give |

**Text rules**
- **New text, not famous text.** No Qurʾān, well-known poetry or Wikipedia
  leads in the test split: large models have memorised them, which inflates
  scores. Write new sentences or use licence-clean, low-exposure sources.
  Check them with the contamination tooling (`training/contamination.py`).
- **Every copy-text needs a clean licence** (own writing or public domain).
- **Shared vs unique text.** Some text is copied by many writers, to compare
  writers on the same content; the rest is unique per writer, for a
  text-disjoint test (§5).

## 2. Labelling: what the transcription records

| rule | why |
|---|---|
| **Two text layers for copied text:** the *prompt* (what the writer was asked to copy) and a *diplomatic* transcription (what is actually on the page, misspellings included) | A child's spelling slip is not a label error. Keeping both separates writer error from annotator error, and the difference is itself data. CER-HV mixes the two |
| **Transcribe as written.** Never add diacritics, hamzas or dots the writer didn't write, and never fix spelling in the diplomatic layer | Normalisation belongs at scoring time (the normalisation ladder, R2), not in the labels. If it is baked into labels it can't be undone |
| **One Unicode policy:** NFC, Arabic-block code points (ي U+064A, not Persian ی U+06CC; ك U+0643), no presentation forms, no tatweel unless written as an elongation | Encoding hygiene is catalogue item F: `kitab_isippt` has 6 Persian forms per 1,000 chars, and KITAB has kashida failures (`characteristics-framework.md`) |
| **Mark uncertainty explicitly:** reserved markup for illegible characters, struck-out text and the writer's own corrections; never guess silently | A silent guess becomes a label error |
| **Line polygon and baseline** for every line, plus reading order | Standard (PAGE-XML-style). The schema comes from A2 (`annotation-schema.md`, pending) |
| **A subset with word boxes, and a smaller one with letter/PAW boxes** | Enables dot-size and px-per-x-height measures (P6) and per-letter analysis without forced alignment |
| **QA verdict codes = the label-review tool's:** correct, transcription, segmentation, orientation, script, non-text, unsure | Our QA numbers stay comparable with CER-HV (`alazzawi2026labelerr`) and with R4.1 |
| **Double-annotate ~10 % and report inter-annotator CER** | This is the noise floor; no model claim below it is meaningful |
| **Before release, run model-consensus label checking and human review** (AUC 0.94 against CER-HV's flags) | Catches the 6–9 % class of errors public sets ship with. Clean labels also matter for reward-based fine-tuning: RL on noisy ground truth amplifies the noise |
| **Corrections never overwrite silently:** versioned labels with an errata file | Benchmarks stay reproducible. Same principle as the review tool, which stores decisions apart from the data |

## 3. Images: capture

| rule | why |
|---|---|
| **Keep full-resolution originals**: ≥ 300 dpi colour scans, archived; release derived greyscale crops alongside, never instead | The public Muharaf copy is 60 px tall, which may be why it is hardest (X7, R9). It can't be undone after release |
| **Record px per x-height and dot size per line** | Image-layer characteristics (P6) need them; resolution and dot size interact |
| **A phone-capture subset** of the same pages | Realism, and a paired scan-vs-phone comparison for free |
| **Record pen type, paper, ruled vs unruled** | Cheap metadata that explains image variation |
| **Fixed form templates with page IDs**, but some unruled free pages | Easy line segmentation, plus a natural layout subset |

## 4. Writers, styles and metadata

| rule | why |
|---|---|
| **Anonymous writer ID on every line**; demographics stored apart from images | Ethics, and writer-disjoint splits |
| **Grade/age** stratified around grades 3–4 | Individuality emerges between grades 3 and 4 (`srihari2016children`) |
| **Gender and handedness** | Comparable with KHATT (`mahmoud2014khatt`) |
| **Regional script tradition** (Mashriqi vs Maghribi, and the writer's schooling country) | Maghribi writes qāf with one dot above and fā with one dot below. Base AIN-7B reads ق as ف on 23–27 % of RASAM qāfs vs 2–3 % elsewhere (R2.9). The style *must* be a label, not a hidden variable |
| **Script style per line** (naskh-like, ruqʿa-like, mixed) | Style is a likely driver of the cross-dataset gap. In adult writing ruqʿa is common |
| **Enough lines per writer (≥ 20)** | Writer-level analysis and writer adaptation need it |
| Region/dialect, first language, school type: **only if an analysis will use them** | Collect the minimum (data protection) |

## 5. Splits and leakage

| rule | why |
|---|---|
| **Writer-disjoint** train / val / test | Standard since IAM (`marti2002iam`) |
| **Text-disjoint test:** no test line's text appears in train, by any writer | About 6 % of Muharaf test lines occur in train (X3), and `kitab_muharaf` overlaps Muharaf train 86.5 % (X1). Shared copy-text is the classic trap |
| **A separate "shared-text" diagnostic split** | Writer comparisons on identical content, kept out of the headline test |
| **Deduplicate against all public sets** (exact text and near-duplicate images) | Prevents cross-set leakage when people combine sets (R4) |
| **A hidden test set** scored on request | Resists contamination of future model pretraining |
| **Fixed, published split files** | Reproducibility |

## 6. Size

- **Size in characters per (letter × positional form) class, not in lines.**
  About 150k training characters reach 10 % CER on KHATT with a CRNN
  (`alazzawi2026gap`). Set a **floor per class** from the train/test
  shape-tail measure (R1.7) and our own scaling curve (E3).
- T1 and T7 exist to raise the rare classes to the floor. Check coverage with
  the R1 analyzer *before* printing forms.

## 7. Release

- Licence and access tier decided **before** collection and written into the
  consent form.
- A datasheet (Gebru et al.), the labelling guide, the QA log (verdict codes
  and inter-annotator CER) and baseline numbers from our harness.
- Release the cleaning log **with an explicit licence**. CER-HV released
  logs and predictions without one, which blocks reuse (X3).
- Model predictions for every line, so others can rerun our error analyses.

## 8. The first instance: a school set (plan)

| item | plan | why |
|---|---|---|
| Writers | Pupils on both sides of grades 3–4; pilot 20–30 writers | §4, `srihari2016children` |
| Sheets per pupil | T1 letter sheet → T2–T5 word sheet → T6/T7/T9 line sheets → T11 free writing | The same writer at every granularity is the natural data for the curriculum track (C0, C1) |
| Ethics | Ethics approval, school or ministry permission, parental consent plus child assent, no names, faces or school names on sheets | Minors; blocking (S0) |
| Limits | Children write one regional style and little ruqʿa | Style coverage (§4) needs a later adult set; see §9 |

## 9. Open decisions

| decision | options | what settles it |
|---|---|---|
| Scope | school only · school + adult multi-style (Maghribi, ruqʿa) | Pilot cost; whether the style results (E6) hold up |
| Diacritic policy for training targets | keep as written · strip · separate layer | A1, X4 |
| Letter/PAW box subset size | none · 1 % · 5 % | Annotation cost from the pilot |
| Per-class floor (§6) | from E3 and R1.7 | Those two results |
| Pseudo-word share (T8) | 0 · 5 · 10 % | A pilot reading test: do VLMs' errors jump on non-words? |
| Hidden test hosting | own server · a benchmark platform | Release plan |

## Change log

- **2026-10-09:** rewritten from the 2026-10-06 school spec into a general
  living guide. Added text blocks T1–T11 (R2.6 confusion pairs, orthographic
  pairs, PAW stress, pseudo-words, coverage selection), two-layer labelling,
  verdict codes, full-resolution capture (Muharaf 60 px), the Maghribi style
  label (R2.9), the text-disjoint split (X1/X3), licensed cleaning logs
  (CER-HV lesson), and links to the curriculum track (C0/C1).
- **2026-10-06:** first school-set spec.
