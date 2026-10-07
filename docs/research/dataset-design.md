# Our own dataset: living design spec

> **Status:** started · **Last updated:** 2026-10-06
> Every requirement links back to the research note or experiment that motivates
> it. A requirement without a "why" gets cut.

## Goal (draft)

A line-level Arabic handwriting dataset from school students. It is designed so
it can answer questions existing sets cannot:
- Does recognition difficulty depend on **writer demographics**, especially
  grade/age? No study measures this for HTR. See
  [`script-characteristics.md`](script-characteristics.md) §7.
- How do **diacritised vs. undiacritised** versions of the *same* text compare?
- Which errors dominate on **children's** text lines? The only children's set,
  Hijja, has isolated letters only.

## 0. Ethics & legal: blocking, start first
- [ ] The writers are minors. We need school or ministry permission and
      ethics/IRB approval from an institution.
- [ ] Parental consent plus the child's assent. Use plain-language information
      sheets in Arabic.
- [ ] No names, faces or school names on the scanned sheets. Assign each writer
      an anonymous ID. Store demographics separately from the images.
- [ ] Decide the release licence and the access tier (open vs. gated) *before*
      collection, and write it into the consent form.
- [ ] Check the data-protection law of the collection country.

## 1. Writers & sampling
| requirement | why |
|---|---|
| Stratify by **grade**, with grades below and above the grade 3–4 boundary | Individuality diverges at grades 3–4 (`srihari2016children`) |
| Record gender and handedness, KHATT-style | Precedent for comparability (`mahmoud2014khatt`) |
| *(open)* region/dialect, school type, first language, instrument and paper | Collect only what an analysis will use. Decide after Wave B |
| Writer-disjoint train/val/test | Standard since IAM (`marti2002iam`) |
| Pilot with ~20–30 writers before scaling | Validates the forms, consent and capture flow |

## 2. Content (what students write)
| requirement | why |
|---|---|
| Copy-text balanced over **(letter × positional form)**, not letter frequency | The shape inventory and its heavy tail drive the Arabic gap (`alazzawi2026gap`) |
| **Dot-group minimal pairs** (ب/ت/ث/ن/ي, ف/ق, ج/ح/خ…) | Dots cause ~30 % of substitutions |
| PAW-rich words and lam-alef | Segmentation and ligature errors |
| The **same text, diacritised and undiacritised** | Isolates the effect of diacritics (A1) |
| Free writing (a short prompt) | Natural spelling and spacing |
| Arabic-Indic and Western digits, a little Latin, form-style fields | Product-shaped axis (`findings-and-notes.md`) |

## 3. Capture
- Fixed scanner DPI (≥ 300). Add a phone-capture subset for realism.
- Printed forms with guide lines, plus some unruled pages.

## 4. Annotation
- Unicode and diacritics policy: from A1 (`diacritics.md`, pending).
- Schema (line polygons, reading order, uncertain-character markup): from A2
  (`annotation-schema.md`, pending).
- Double-annotate a subset and report inter-annotator CER.
- Run an automated label-error check before release (`alazzawi2026labelerr`).
  Public sets carry 6–9 % label errors.

## 5. Size
- Size it in **characters per (letter, position) class**, not lines. About 150k
  training characters reach 10 % CER on KHATT with a CRNN
  (`alazzawi2026gap`). Refine this with our own data-scaling curve
  (`findings-and-notes.md`, the 1k/2k/4k/8k open question).

## 6. Release
- Publish a datasheet (Gebru et al.) with the data, plus baseline numbers from
  our leaderboard harness.
