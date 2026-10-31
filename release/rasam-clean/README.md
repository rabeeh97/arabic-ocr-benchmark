# RASAM-1 / RASAM-2 / TariMa: corrected crops and a clean test view

Corrections to the line crops of the Hugging Face release of
[calfa-co/rasam-dataset](https://github.com/calfa-co/rasam-dataset) (configs `RASAM-1`, `RASAM-2`, `tarima`), made **automatically**
from the page images, the PAGE XML and two cross-fitted line readers. The only human labels used are someone else's: the manual check of
RASAM-1 in [PR #7](https://github.com/calfa-co/rasam-dataset/pull/7). We did not hand-correct any line.

Nothing here changes the official splits or references. Use it to:

1. train on fixed crops and without lines whose reference does not match the image, and
2. report test CER on the **clean view** next to the official number.

## What is in it

| file | content |
|---|---|
| `manifest.csv` | the 447 lines we act on: `config, split, index` (Hugging Face row index), manuscript, page, `action` (`recut`, `drop`, `recut+drop`), `reasons`, the PAGE XML line id of the re-cut, the reference |
| `recut/<config>/<split>/<index>.png` | 250 crops re-cut from the page images (196 train, 29 validation, 25 test) |
| `drop.json` | the 202 lines removed from the clean view (161 train, 20 validation, 21 test), as `<config>/<split>/<index>` |
| `score.py` | scorer, standard library only: official, drop-only and clean CER from a predictions file keyed by test index |
| `counts.json` | the counts below, and the rule descriptions |

### Re-cut crops (250)

These crops are broken in the published files but intact on the page. Most are MS 1936 crops of RASAM-2 that are 1–7 px tall. We cut
them again from the page image along the line's PAGE XML polygon, and kept only re-cuts whose reading matches the reference. The test
split has 25 of them: 23 in RASAM-2 and 2 in RASAM-1.

### Dropped lines (202)

A line is dropped when its reference does not describe its image and the crop cannot be fixed. The rules are in
`scripts/rasam_clean_lists.py` of the [benchmark repo](https://github.com/rabeeh97/arabic-ocr-benchmark).

| rule | lines | meaning |
|---|---:|---|
| `external` | 71 | marked invalid in the manual check of PR #7 (RASAM-1) |
| `garbled` | 66 | the reference is far less probable under a character LM than what two cross-fitted readers read |
| `garbled_page` | 45 | the line is on a page whose references are, as a whole, much less probable than the readings: MS 1977 pages 0017, 0071, 0158 (the same pages were called "uncorrected HTR" in [issue #2](https://github.com/calfa-co/rasam-dataset/issues/2)) |
| `blank` | 18 | the crop contains no ink, or the reference is empty |
| `shifted` | 12 | the reference belongs to a neighbouring line |
| `vertical` | 7 | the crop is a vertical or marginal strip |
| `truncated` | 4 | the crop is cut short, and the page does not hold the full reference either |

Test lines dropped: 17 in RASAM-1, 2 in RASAM-2 and 2 in TariMa. The test view therefore has 717 / 370 / 227 lines.

**Caveat.** The automatic rules come from CRNN readings. They could favour readers that make CRNN-like errors. To check this, we also
score a view that drops only the lines found by people (PR #7 plus our own by-eye list). Both views give the same numbers within
about 0.2 points, and the ranking of the 11 systems we score is unchanged. The only swap is between two of our own models that
are within noise of each other (see `results/rasam/clean/bench.json` in the benchmark repo).

Agreement with PR #7 on RASAM-1 is low: precision 0.24, recall 0.30. The two methods catch largely different defects, so neither
list is complete.

## Scoring

```bash
python score.py --config RASAM-2 --pred preds.json [--recut-pred recut_preds.json]
```

- `preds.json` holds `{"<test index>": "<reading>"}`, or a list in test order, for the official crops.
- `recut_preds.json` holds the same model's readings of `recut/<config>/test/*.png`. Without it, you get the official and the drop-only
  CER.
- References are loaded from the Hugging Face dataset (this needs `datasets`), or pass `--refs refs.json`.
- The CER is micro CER: total edits divided by total reference characters. Before comparing, invisible format characters are removed,
  Arabic presentation forms are folded and the text is NFC-normalised.

## Licence

The re-cut images are derived from the RASAM and TariMa page images (Apache-2.0, © the RASAM / Calfa authors). They are redistributed
under the same licence. The lists and the scorer are released under the benchmark repository's licence.
