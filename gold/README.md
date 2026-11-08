# Gold set

A seeded, de-duplicated, hand-verified sample of **lines** from the datasets we benchmark on. From now on every
trained model is scored on it and every comparison is reported on it. Design and the decisions behind it:
[`docs/superpowers/specs/2026-11-08-gold-set-design.md`](../docs/superpowers/specs/2026-11-08-gold-set-design.md).
Labeling rules: [`GUIDELINES.md`](GUIDELINES.md).

## What is in here

| path | what | git |
|---|---|---|
| `config.yaml` | seed, tranche size, dedup thresholds, dataset registry (source, split, tier, language, medium, era, script style) | tracked |
| `pool/<dataset>.jsonl` | every eligible test line, ranked by the seed; duplicates listed with `dup_of` | tracked |
| `pool/_meta.json` | per dataset: source snapshot, rows, duplicates, eligible lines | tracked |
| `tranches/tNN/manifest.json` | the line ids of tranche NN | tracked |
| `tranches/tNN/lines.jsonl`, `images/` | the lines to label: original text, image, auto tags, toolkit notes | **local only** (dataset licences) |
| `labels/<labeler>/tNN.jsonl` | one labeler's verdicts, append-only | **local only** |

## Commands (run from the repo root)

```bash
export PY=/Users/rabeehjouni/.venvs/kraken-rasam/bin/python     # any env with numpy, Pillow, pyarrow, PyYAML
PYTHONPATH=src $PY scripts/gold_build_pool.py                    # rank + de-duplicate (first run about 3 min, then cached)
PYTHONPATH=src $PY scripts/gold_export_tranche.py --tranche 1    # ranks 1-50 of every dataset (750 lines)
PYTHONPATH=src $PY scripts/gold_label_app.py --tranche gold/tranches/t01   # http://127.0.0.1:8770/
PYTHONPATH=src $PY scripts/gold_agreement.py --tranche t01       # after two labelers did the calibration lines
```

The labeling tool asks for a **labeler ID** (stored in the browser, written into every record, and used as the folder
name). Labelers never see each other's labels, and the toolkit's notes for a line appear only after that labeler
saved a verdict for it.

## How the sample is drawn

1. For each dataset, every row of its test split gets `sha256(seed | dataset | line_id)`.
2. Lines whose **image** is a duplicate (identical pixels, or the same picture re-encoded: 256-bit difference hash within 10 bits and pixel correlation >= 0.97) of an earlier kept line are set aside. Datasets are processed in priority order, originals first and AMIDDA last. The same text on a different image is **not** a duplicate.
3. The rest are ranked by that hash. Tranche *k* is ranks `50(k-1)+1 .. 50k`. Any prefix is a uniform random sample; adding rows to a dataset later never reorders the lines already ranked.
4. Nothing is replaced when a line is later judged unusable: it is logged, and the unusable rate per dataset is itself a result.

Changing `seed` creates a different gold set. Never edit it once labeling has started.

## Datasets used (tranche 1 = 50 lines each)

| key | source (split) | medium / style | eligible lines | tranches of 50 |
|---|---|---|---:|---:|
| rasam1 | calfa-ai/RASAM-1 (test) | handwritten, Maghrebi | 734 | 14 |
| rasam2 | calfa-ai/RASAM-2 (test) | handwritten, Maghrebi | 372 | 7 |
| tarima | calfa-ai/tarima (test) | handwritten, Maghrebi | 229 | 4 |
| iskandar | calfa-ai/iskandar (test) | handwritten manuscript | 523 | 10 |
| baybars | calfa-ai/baybars (test) | handwritten manuscript | 1,528 | 30 |
| nakba | U4RASD/omar-al-saleh-manuscripts-segments (test) | handwritten, 20th century | 2,095 | 41 |
| muharaf | aamijar/muharaf-public (test) | handwritten manuscript | 1,334 | 26 |
| khatt | Teklia/KHATT_v1.0 (test) | handwritten, modern | 1,038 | 20 |
| arams | AraMS-28k-HTR, local zip (test) | handwritten manuscript | 6,744 | 134 |
| onlinekhatt | ahmedheakl/arocrbench_onlinekhatt | handwritten, pen | 200 | 4 |
| historyar | ahmedheakl/arocrbench_historyar | handwritten, historical | 200 | 4 |
| patsocr | ahmedheakl/arocrbench_patsocr | printed, several fonts | 500 | 10 |
| isippt | ahmedheakl/arocrbench_isippt | printed, screenshot style | 499 | 9 |
| amidda | calfa-ai/AMIDDA (test) | mixed sources | 4,416 | 88 |
| iam (English, separate tier) | Teklia/IAM-line (test) | handwritten, English | 2,915 | 58 |

`onlinekhatt`, `historyar`, `patsocr` and `isippt` are single-split benchmark samples (the KITAB-Bench copies), so there is
no test split to choose from. AMIDDA's test split is exactly the union of other datasets' test splits plus new material:
2,268 of its 6,684 lines were set aside as duplicates (all of RASAM-1, RASAM-2, TARIMA and ISKANDAR test, 157 baybars,
113 KHATT, 139 Muharaf, 1 historyar), which leaves 4,416 distinct lines. NAKBA's official test split shares pages with
train; a page-disjoint `blind_test` (2,671 lines) exists and can replace it in `config.yaml` before the pool is built.

**Not used and why:** `arocrbench_hindawi`, `arocrbench_historicalbooks` (pages), `arocrbench_evarest`, `arocrbench_adab`
(word crops), `arocrbench_muharaf` (86.5 % in Muharaf train; the Muharaf test split replaces it), `arocrbench_khatt`
(a sample of KHATT; the KHATT test split replaces it).

**Open slots:** other-language Arabic-script lines (Persian, Urdu: no directly downloadable real line-level set found on
2026-11-08; the Persian handwriting set on the Hub is page-level and UTRSet-Real needs a manual download) and the
recorded hard-case set (`GUIDELINES.md` section 12). Neither has lines yet.

## Tags

Automatic, written into `lines.jsonl` at export (`tags_version` 1):

- **source:** dataset, tier, language, medium (handwritten / printed), era, script style, split
- **text:** chars, words, length band, `is_fragment` (<= 20 characters), script, Arabic-letter share, Latin, digits (and which kind), punctuation, tatweel, diacritics present and level (none / sparse / partial / full), dot-ambiguous share, medial-letter share, lam-alef count
- **image:** width, height, aspect, height tier, ink density, contrast, background (light / dark / coloured)

Added by labelers: verdict, correction types, image issues, line kind, legibility, diacritics visible, script style seen.

Added after labels are final (not built yet, see below): difficulty band.

## Workflow

1. **Calibration.** Two labelers independently label the five calibration lines (marked in the tool) of every dataset: 75 lines. Run `gold_agreement.py`, discuss the disagreements, bump the guidelines version.
2. **Label tranche 1** dataset by dataset.
3. **Freeze** tranche 1: copy the verified lines to a versioned release file and never edit it again.
4. **Score** every model with the existing harness on the verified lines, always reporting against both `text_original` and `text_verified`, with the "dataset was in this model's training mix" flag.
5. **Tranches 2+** the same way when more lines are needed.
6. **Difficulty band (after step 3).** Average line CER over a model panel that is *not* among the models being compared, rank all lines, and tag the hardest 10 / 25 / 50 %. Both within each dataset and across all datasets, because pooled ranks let the hardest datasets dominate the top band. Slices under about 30 lines are descriptive only.

## Known thin slices (tranche 1, from the auto tags)

Of 700 Arabic-tier lines: 95 diacritised (21 Muharaf, 32 historyar, 13 AMIDDA, the rest scattered), 70 short fragments
(NAKBA alone is 46 %), 41 with digits, 10 with Latin letters, 21 with no Arabic letters at all. Diacritised lines come
mostly from two datasets: report that, and treat it as a data-collection target rather than oversampling.
