# Automatic data-QA checks for line datasets

*Added 2026-11-01. Code: `src/arabic_ocr_bench/dataqa.py` (definitions in `CHECKS`), scanner
`scripts/dataqa_scan.py`, cross-dataset overlap `scripts/dataqa_overlap.py`, summary
`scripts/dataqa_summary.py`, RASAM validation `scripts/dataqa_validate_rasam.py`, tests `tests/test_dataqa.py`.
Outputs: `results/dataqa/`.*

The RASAM clean-up (RS4i–RS4l) found the same kinds of defect again and again: crops that show more than one
line, vertical margin strips, 1-px crops, references pasted twice, references cut short, references that belong to
the neighbouring line. This note turns the defects that can be found **without a human** into general checks.
They run on any line dataset that yields `(id, image, ground truth)`. Each check gives a line a flag and a reason.
A flag means "look at this line"; it does not prove the line is wrong.

## How to run

```bash
# any dataset in configs/datasets.yaml (needs `datasets`), or any cached HF line dataset offline:
PYTHONPATH=src:scripts python scripts/dataqa_scan.py --hf calfa-ai/RASAM-1 --split test \
    [--reads readings.json ...] [--limit N]
# -> results/dataqa/<name>_<split>.json (flags per line) and .html (flagged lines per check: image, GT, reading)

PYTHONPATH=src python scripts/dataqa_overlap.py calfa-ai/RASAM-1:test calfa-ai/AMIDDA:train  # leakage by GT text
PYTHONPATH=src python scripts/dataqa_summary.py                                              # firing-rate table
```

`--reads` takes `{id: {"hyp": reading, "ms": manuscript, "page": page}}` (or `{id: reading}`) from **any**
recogniser. With readings, the reader tier runs as well. With page fields, `shifted` runs too. The scanner reads the
HF parquet cache directly, so it needs neither `datasets` nor the network.

## The checks

| check | tier | wrong side | rule (thresholds set on RASAM/TariMa) |
|---|---|---|---|
| `empty_gt` | text | GT | no letters and no digits (digit-only page or verse numbers are content) |
| `repeated_gt` | text | GT | spaces removed, a span of ≥ 15 characters recurs within 12 characters of its end, or the GT is X+X |
| `char_run` | text | GT | the same character (not space or tatweel) ≥ 6 times in a row |
| `off_script` | text | GT | < 50 % of the letters are Arabic script (GT ≥ 5 letters) |
| `degenerate` | image | image | height or width < 8 px |
| `vertical` | image | image | height ≥ 1.8 × width, GT > 2 letters (vertical or marginal strip) |
| `blank` | image | image | a flat image; or < 0.3 % of pixels darker than 0.8 × the local paper brightness, with GT ≥ 3 letters |
| `multi_line` | image | image | after deskewing (±8°), the row profile has ≥ 2 line peaks. Each peak is ≥ 45 % of the highest and holds ≥ 40 % of the main line's ink mass. Peaks are ≥ max(15 % H, 0.8 × core height) apart, the valley between them is ≤ 50 % of the lower peak, and the extra peak lies ≥ 12 % H from the edges |
| `neighbour_clip` | image | image | as `multi_line`, but the extra peak lies at the edge: a neighbour line cut in half. Informational only |
| `length_mismatch` | joint | either | robust z (median/MAD over the dataset) of log(GT letters) − log(core width / core height) is beyond ±3.5. + means the GT is too long for the ink; − means it is too short |
| `gt_truncated` | reader | GT | sub + del ≤ 15 % of GT and insertions ≥ 15 % + 5: the reader reads the whole GT and more |
| `image_truncated` | reader | image | sub + ins ≤ 15 % and deletions ≥ 30 % + 5: the GT holds text the image does not show |
| `doubled_gt` | reader | GT | the longest repeat (≥ 5 characters, gap ≤ 12) is a paste: deleting its second copy cuts the edit distance to the reading by ≥ 60 % of its length. Natural repeats ("الاثنين في الاثنين") are read twice, so deleting one gains nothing |
| `shifted` | reader | GT | own CER ≥ 30 %, GT ≥ 15 characters, and the reading of another line on the same page matches the GT at CER < min(30, own/2) |
| `unreadable` | reader | either | CER ≥ 60 %, GT ≥ 10 characters, and no more specific flag |
| `duplicate_image` | dataset | either | same quantised 64×16 thumbnail. The flag says whether the GTs differ and whether the copies cross splits |
| `duplicate_gt` | dataset | either | the same GT of ≥ 25 characters on ≥ 2 lines |

**Image measurements.**
- **Padding:** dark regions that touch the border, solid enough to be the filler around rotated or polygon crops. Padding is never counted as ink.
- **Ink:** found by Otsu over the non-padding pixels; for `blank`, against the local paper brightness instead.
- **Core:** the main line's band of rows on the deskewed ink, so ink clipped from neighbour lines does not count.

**Not automated here:** garbled references (uncorrected HTR). `scripts/rasam_clean_lists.py` finds them with a character LM (bits per character of the reference minus that of the reading). That needs an LM for the script and period, so it is left out of the generic set.

## Validation on RASAM (test splits, readings from the two cross-fitted CRNN folds)

- **By-eye examples** (`data_issues_curated.json`, 25 lines): 13 of the 17 non-garbled examples are flagged.
  - All 3 degenerate, 2 of 4 wrong-place, 1 of 3 neighbours, both shifted, the vertical one, 1 of 2 doubled and both truncated.
  - **Misses:**
    - Two faint wrong-place crops. Their ink amount equals faint real text; see the limits below.
    - One strongly skewed neighbours crop, where loops fill the gap between the lines.
    - One doubled reference whose repeat the reader garbles, so dropping the copy saves only 2 of 11 edits.
- **Earlier clean lists** (46 test lines; `results/dataqa/rasam_validation.json`):
  - Every earlier `vertical`, `truncated` and `shifted` line is flagged (2/2 each).
  - 15 of the 25 re-cut crops are flagged (degenerate 7, multi-line 8).
  - The 2 earlier `blank` lines are missed: they are faint, and the strict blank rule lets them through.
  - A looser rule (< 1 % ink) caught them, but it also flagged 186 faint real lines in Muharaf (see the limits below).
- **Garbled lines:** the 7 garbled and 9 externally-flagged lines are mostly not flagged. This is expected: they need the LM or a person.
- **New finds:** `doubled_gt` adds RASAM-1/test/000387. Its GT has "لانها لانا" where the image has "لانا" once. This was not in any earlier list.

## Firing rates (% of lines)

| scan | lines | read | multi_line | neighbour_clip | length_mismatch | other flags |
|---|---:|---:|---:|---:|---:|---|
| RASAM-1 test | 734 | 734 | 3.1 | 16.2 | 1.5 | vertical 2, repeated 1, doubled 2, gt_truncated 2, image_truncated 1, shifted 1, unreadable 2 |
| RASAM-2 test | 372 | 372 | 4.0 | 9.1 | 5.1 | degenerate 7 (1.9 %) |
| TariMa test | 229 | 229 | 2.6 | 8.7 | – | – |
| AMIDDA test (first 1,500) | 1500 | – | 3.0 | 13.3 | 2.3 | degenerate 7, vertical 2, repeated 1 |
| Muharaf test | 1334 | – | 2.0 | 3.4 | 1.5 | off_script 25 (Latin catalogue lines), empty_gt 6, duplicate_gt 8, blank 1 |
| KHATT v1.0 test (Teklia) | 1038 | – | – | 7.3 | 2.1 | duplicate_gt 319 (30.7 %, by design), blank 4, vertical 3 |
| NAKBA / Omar al-Saleh test | 2095 | – | 1.5 | 3.2 | 2.1 | off_script 68 (English day and month names), duplicate_gt 10 |
| KITAB KHATT | 200 | – | – | 8.5 | 0.5 | duplicate_gt 11, vertical 1 |

**Precision checked by eye**, on random samples from the galleries:
- `multi_line`: about 9/12 on RASAM-1, 6/10 on AMIDDA, 4/8 on Muharaf. Errors are vocalised lines (a row of harakat), tall single words and digits.
- `blank`: 4/4 on KHATT (tiny text in an almost empty canvas: real bad crops).
- `length_mismatch`: about 6/8 on Muharaf (seals, signatures, Latin shelf marks). It is a lead to look at, not a verdict.
- Reader-tier flags on RASAM: all checked flags were real, apart from `unreadable`, which is by definition a "look" flag.

## Train/test overlap by GT (`dataqa_overlap.py`, GTs of ≥ 20 letters)

| probe | against | shared |
|---|---|---:|
| RASAM-1 / RASAM-2 / TariMa test | own train + validation | 0 / 1 / 0 |
| RASAM-1 / RASAM-2 / TariMa test | AMIDDA train | 0 / 1 / 0 |
| RASAM-1 / RASAM-2 / TariMa test | AMIDDA test | 100 % each: AMIDDA's test split contains all three test sets |
| Muharaf test | own train + validation | 21 (2.0 %) |
| NAKBA test | own train | 51 (3.3 %) |
| KHATT v1.0 test | own train + validation | 641 (67 %): KHATT's fixed paragraphs are copied by many writers, so test text is mostly seen in training. Note this for models with a language prior |

## Limits

- **Faint text and empty texture overlap.** A crop that shows only background texture (RASAM-2 "wrong-place" crops) has
  about as many dark pixels as a faint pencil or blue-ink line on lined paper (Muharaf). No global or local-contrast
  ink measure separated the two. `blank` is therefore strict, and faint crops are left to the reader tier: a reader
  run on the original crop will flag them as `unreadable` or `image_truncated`.
- **Multi-line counting is a projection heuristic.** It fails on strong skew with bridging loops, and it can fire on
  vocalised lines. The gallery shows the peak rows so the call can be checked quickly.
- **Thresholds come from RASAM.** For a new dataset, read the gallery for each check before acting on a flag.
