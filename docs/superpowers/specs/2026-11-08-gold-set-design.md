# Gold set: design (2026-11-08)

## Intent

One fixed, verified set of lines that every model we train or test is scored on, and every future comparison is
reported on, with tags so results can be sliced (diacritised vs not, fragments, script style, difficulty...).
It replaces today's situation where paid models were scored on tiny, different subsets and some benchmark samples
were simply the first 200 lines of a split.

## Goals

- A uniform random sample of lines per source dataset, reproducible from a seed, extendable in steps of 50 (50, 100, 150 ...).
- Labels verified by hand under written guidelines, so that different labelers land on the same text; measured by an agreement report.
- Original and verified label both kept, so a result can be reported against either, and so dataset label noise is measured as a by-product.
- Tags for slicing, including things the research has shown to matter: diacritics, short fragments, Latin/digits, script style, medium, resolution, difficulty.
- A per-model "dataset was in this model's training mix" flag in results, so generalisation of a single-dataset model can be read off the same set.

## Non-goals

- Pages or word crops (lines only). No page or writer IDs for now.
- Proving the "not enough data" thesis (that needs the training experiments) or certifying third-party models (their training data is unknown).
- Releasing the data publicly (dataset licences; kept local).

## Anti-goals (would make it a failure even if it runs)

- Labels that depend on the model being evaluated (no OCR or AI help while labeling; difficulty computed from a panel that is not among the compared models).
- A "random" set that is quietly curated. Hard and recorded lines are separate tiers, never pooled into the Arabic headline number.
- Labelers priming each other (labels are per labeler and blind; toolkit notes appear only after a verdict).

## Decisions (the project owner's, 2026-11-08)

- Lines only; test splits only; where a dataset has no test split, decide later (the four KITAB-Bench samples are single-split and used as they are).
- 50 lines per dataset first, then more in steps of 50, as nested prefixes of one seeded order. Never replace lines dropped as unusable.
- Exact image duplicates are removed; the same text on different images is fine.
- Difficulty is assigned after labels are final, from a model panel separate from the compared models.
- Thin slices (for example diacritised lines mostly coming from Muharaf) are reported as such and become collection targets, not oversampled.
- Relabeling follows written guidelines (diacritics, scratched text, words cut at line ends and so on) and records a labeler ID.
- Extra tiers wanted: a few lines in other Arabic-script languages and one English set (VLM sanity check), and a recorded hard-case set that nobody has published. Kept local.

## My calls (change any of them)

- Image duplicates also include re-encoded copies (difference hash plus pixel correlation), because KHATT appears re-encoded inside AMIDDA.
- Datasets rank by priority so originals keep a shared line and AMIDDA keeps only what no other dataset has.
- Transcription is image-grounded and diplomatic at the spelling level but standard at the letter level (missing dots do not create new letters); diacritics only when clearly visible.
- The first five lines of every dataset are calibration lines every labeler does, blind.
- English (IAM) is a separate tier with its own tag, never in the Arabic number.
- Persian and Urdu are left as an open slot: no directly downloadable, real, line-level set was found.
- NAKBA uses its official `test` split; the page-disjoint `blind_test` is one config change away.
- Toolkit notes ship with each line and are shown only after the labeler's verdict; the tool never changes a verdict because of them.

## Left to the builder / later

- Difficulty band script and the model panel (after labels are frozen).
- Persian / Urdu source and the recorded hard-case protocol (section 12 of the guidelines is a draft).
- A scoring entry point that reads the frozen gold file and carries the "in training mix" flag.
- Whether to double-label more than the calibration lines.

## Implemented

`src/arabic_ocr_bench/gold/` (readers, image hashing, pool, tags, schema, tranche export, labeling app, agreement),
`scripts/gold_*.py`, `gold/` (config, guidelines, README, pool, tranche manifest), `tests/test_gold.py`.
