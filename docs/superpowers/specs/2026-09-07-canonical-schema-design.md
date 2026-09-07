# Canonical line schema + offline conversion layer (with charset & normalization profiles)

Date: 2026-09-07
Status: approved design, pre-implementation
Path class: architectural

## Problem

Every corpus in this benchmark is consumed through a bespoke `DatasetAdapter`
that yields a minimal `Sample(id, image, ground_truth, track, meta)`. That is
enough to score a hosted model, but not enough for the CTC training programme
(Tier 0) the project is about to start: no page context, no per-line difficulty
signal, no separation of the transcription *as written* from a
diacritic-normalised counterpart, no single place where per-line metadata
(script, era, licence, split unit, validation level) lives, and no character
vocabulary.

We want one canonical record that every corpus converts into **before** any
training/analysis pipeline runs, plus the character-set and normalization
machinery Tier 0 depends on.

## Scope (decided with the user)

- **Offline JSONL export, additive.** A new `scripts/build_canonical.py`
  iterates each corpus's *existing* adapter and writes
  `data/canonical/<corpus>.jsonl` plus a lazy crop cache. The benchmark runner,
  scorer and leaderboard are **not** touched.
- **Coverage:** all 15 datasets currently wired in `configs/datasets.yaml`, all
  pre-segmented line-crop shape, plus a **new AraMS-28k-HTR adapter** — the one
  corpus with real `book_id` / `page_id`, per-line `alignment_confidence`,
  per-book `script`, a `validation_level` (PV/LV), and a paired
  diacritic-normalised transcription.
- **AraMS-28k-HTR via a one-time HF mirror.** Download the Zenodo archive once,
  push a cleaned dataset to a HF repo under the user's account, then stream it
  like the other HF corpora. The mirror script also emits a local
  **Kraken-format tree** (`<id>.png` + `<id>.gt.txt`) for `ketos train` in
  experiment 0.1.
- **Charset + normalization profiles are merged into this plan** (not a
  follow-up) because experiment 0.1 needs a character vocabulary and 0.2 needs a
  union charset:
  - `normalize.py` gains `standard` and `diac_full` profiles (details below);
    `raw` / `light` / `diac_insensitive` are unchanged.
  - `scripts/build_charset.py` produces `data/canonical/_charset.json`.

Out of scope: page-track corpora (Muharaf-full, AraMS *pages*, OpenITI MAKHZAN),
SARD, the `training/crops.py` model-input preprocessing module (belongs to the
Tier-0 spec), and any change to how hosted models are run or scored.

## The canonical record

`src/arabic_ocr_bench/canonical.py` defines a frozen dataclass `CanonicalLine`
and a pure `to_canonical()` builder. Serialised JSON shape:

```jsonc
{
  "line_id": "arams28k/book_03/p0172/l007",     // hierarchical when book/page known, else "<corpus>/<split>/<nnnnnn>"
  "corpus": "arams28k",
  "track": "line",                              // line | page

  "book_id": "book_03", "page_id": "p0172",     // null for pre-segmented HF sets
  "page_path": null,                            // repo-relative page-image path, when held
  "polygon": null, "bbox": null,                // null unless the source is page-level
  "zone": "main",                               // main|margin|catchword|header|colophon
  "angle_deg": 0.0,                             // 0.0 for pre-cropped; computed at page-crop
  "reading_order": 7,                           // source row index when nothing better
  "insertion_anchor": null,

  "text_diplomatic": "…",                       // source transcription, verbatim, harakat intact
  "text_diac_normalised": null,                 // paired diacritic-normalised transcription — non-null only where the source ships one (AraMS-28k). NOT the same as text_l3 (which is our normalize() output); this is the corpus's own counterpart, used as supervision for 0.4 arms B/C.
  "text_edited": null,                          // editorial reconstruction; null for every current corpus
  "text_l1": "…", "text_l2": "…", "text_l3": "…",  // = normalize(text_diplomatic, raw|light|diac_insensitive)

  "script": "maghribi",                         // per-line; from corpora.yaml (corpus constant) or the per-book table for AraMS
  "era": "19c", "genre": "manuscript", "writer_id": null,
  "diacritic_density": 0.031,                   // harakat chars / non-space chars of text_diplomatic
  "scan_quality": null,                         // heuristic pass later; null now
  "alignment_confidence": 100,                  // per-line where the source gives it (AraMS confidence score); 100 for gold pre-aligned line sets
  "validation_level": null,                     // "PV" | "LV" | null — AraMS page-validated vs line-validated; null elsewhere

  "split": "test", "split_unit": "book",        // provider|book|page|writer
  "license": "CC-BY-NC-SA-4.0",
  "provenance": {
    "source_id": "arams28k/test/006745",       // original Sample.id, always
    "hf_id": "<user>/arams28k-htr-lines", "hf_revision": "<commit-or-null>",
    "split": "test", "row": 6745, "converter_version": 1
  }
}
```

### Deviations from the originally suggested schema

- **`text_raw_diacritized` dropped**; the diplomatic form is the reference
  edition. A *separate* `text_diac_normalised` is added for the one corpus
  (AraMS) that genuinely ships a paired stripped transcription — it is source
  data, not our `normalize()` output.
- **`text_l1/l2/l3` are defined as `normalize(text_diplomatic, PROFILES[N-1])`**
  with `PROFILES[:3] = ("raw", "light", "diac_insensitive")`. One source of truth.
- **`text_diplomatic` vs `text_edited` separate from day one** — the field that
  cannot be retrofitted.
- **`script` is per-line.** Corpus-constant for 14 corpora (`corpora.yaml`);
  per-book for AraMS (its books are one script each).
- **Added `track`, `validation_level`, `text_diac_normalised`.**
- **`split_unit`** ∈ `provider | book | page | writer` (no `line`).

### Field population tiers

- **Always populated:** `line_id`, `corpus`, `track`, `text_diplomatic`,
  `text_l1/l2/l3`, `script`, `era`, `license`, `split`, `split_unit`,
  `provenance`, `diacritic_density`, `zone` (`"main"`), `angle_deg` (`0.0`),
  `alignment_confidence`.
- **Only when the source carries it:** `book_id`, `page_id`, `page_path`,
  `polygon`, `bbox`, `reading_order`, `writer_id`, `genre`,
  `text_diac_normalised`, `validation_level`. Non-null today only for
  AraMS-28k-HTR.
- **Deferred / null now:** `text_edited`, `scan_quality`, `insertion_anchor`.

## `line_id`

- Hierarchical when the source has book/page structure:
  `<corpus>/<book_id>/<page_id>/l<line_no:03d>` (AraMS; any future page corpus).
- Flat otherwise: `<corpus>/<split>/<row:06d>` — identical to today's
  `Sample.id`, so existing result files stay joinable.
- `provenance.source_id` always holds the original `Sample.id` verbatim.

## AraMS-28k-HTR specifics

Zenodo `10.5281/zenodo.21499649` (~2.4 GB). Ids encode `book/page/line`.
**Assumptions to verify at mirror time — if any is false, flag it, it affects
Tier-0 arms:**

- Each line ships a raw diacritised transcription **and** a diacritic-normalised
  counterpart → `text_diplomatic` + `text_diac_normalised`.
- Each line has a numeric alignment/confidence value → `alignment_confidence`.
- Books are labelled page-validated (confidence-100 only, released) vs
  line-validated → `validation_level`. Per-book confidence-100 rate ranges
  ~8.4 % (book_03, Maghribi, degraded) to ~99.9 % (book_06, Naskh, clean).
- Book-level split: train 9 books / 19,739 lines, val 2 / 1,486, test 3 / 6,746.
  **Use the corpus's own split** so numbers compare to RefLAM (Kraken 23.31 %
  weighted CER; Maghribi 32.71 / Naskh 22.62 / Ruq'ah 11.65).
- Per-book script is a fixed table in `corpora.yaml` under `arams28k.books`.

The mirror pushes columns: `image`, `text`, `text_diac_normalised`, `book_id`,
`page_id`, `line_no`, `alignment_confidence`, `validation_level`, `script`,
`split`. `scripts/mirror_arams_to_hf.py --kraken-out <dir>` also writes
`<dir>/<split>/<book>/<page>_<line>.png` + `.gt.txt`.

## Module layout & data flow

| Unit | Responsibility | Tested by |
| --- | --- | --- |
| `src/arabic_ocr_bench/canonical.py` | `CanonicalLine`, `to_canonical(sample, corpus_meta, *, hf_revision=None)`, `diacritic_density(text)`. Pure, no I/O. `corpus_meta` = merged dict: `script`(/`books`)/`era`/`genre`/`split_unit` from `corpora.yaml` + `license`/`split`/`hf_id` from `datasets.yaml`; per-line overrides (`script`, `alignment_confidence`, `validation_level`, `text_diac_normalised`, `book_id`, `page_id`) come from `Sample.meta`. | `tests/test_canonical.py` |
| `src/arabic_ocr_bench/normalize.py` | Add `standard` and `diac_full` to `PROFILES`; `strip_harakat(s, marks=WIDE)` gains a mark-set param; add `TASHKEEL_WIDE` / `TASHKEEL_FULL` constants. No change to `raw`/`light`/`diac_insensitive` behaviour. | extend `tests/test_normalize.py` |
| `configs/corpora.yaml` | Per-corpus `script`/`era`/`genre`/`split_unit`; `arams28k.books: {book_01: {script: …}, …}`. `license` stays in `datasets.yaml`. | schema-load test |
| `src/arabic_ocr_bench/datasets/arams.py` | Streams the HF mirror; `Sample.meta` carries every AraMS field above. | adapter smoke |
| `scripts/mirror_arams_to_hf.py` | One-time: Zenodo → parse → HF `push_to_hub`; `--kraken-out <dir>` also writes the PNG/`.gt.txt` tree. | manual run |
| `scripts/build_canonical.py` | Per corpus in `datasets.yaml` (`--only` filter): adapter → `to_canonical` → `data/canonical/<corpus>.jsonl` sorted by `line_id`; `--no-crops` toggles `crops/<corpus>/<line_id>.png`; `--kraken-out <dir>` emits the Kraken tree for a corpus. Idempotent (skip when JSONL newer than adapter + `corpora.yaml` + `canonical.py`, `--force` overrides). Writes `_manifest.json`. | `--only synthetic` CI smoke |
| `scripts/build_charset.py` | Reads `data/canonical/*.jsonl`; `--level {l1,l2,l3,standard}`, `--min-count 20`, `--corpora a,b,…` (default all). Emits `data/canonical/_charset.json`: global + per-corpus + per-split charset, size, OOV mass, frequency tail, train→test character gap, and a `union` charset over the requested corpora. **Never drops a line.** | `tests/test_charset.py` on synthetic |

No change to `runner.py`, `report.py` (that's the eval-readiness spec),
`cli.py`, `make_leaderboard.py`.

## `normalize.py` additions

- `standard` = `light` + alef-maqsura→yaa (ى→ي) + taa-marbuta→haa (ة→ه). For the
  RefLAM/CAMeL cross-paper table only; `light` stays strict so the headline
  still measures the distinction. Existing leaderboard numbers unaffected.
- `diac_full` = `diac_insensitive` with the stripped set = `TASHKEEL_WIDE`
  (`U+064B–065F, U+0670`) **plus** `U+0610–061A` and `U+06D6–06ED` (Qur'anic
  marks). `diac_insensitive` (WIDE) stays the headline diacritic profile;
  `diac_full` is scored alongside and the WIDE↔FULL delta is reported.
- `text_l1/l2/l3` in the canonical record stay on `raw`/`light`/`diac_insensitive`;
  `standard` / `diac_full` are computed at analysis time, not stored.

## `configs/corpora.yaml` — proposed values

| corpus | script | era | genre | split_unit |
| --- | --- | --- | --- | --- |
| muharaf_lines | ruqah | 19c-21c | correspondence | provider |
| kitab_khatt / kitab_onlinekhatt | modern_hw | 21c | prompts | provider |
| kitab_muharaf | ruqah | 19c-21c | historical_hw | provider |
| kitab_historyar | naskh | historical | manuscript | provider |
| kitab_historicalbooks | print | historical | book_print | provider |
| kitab_hindawi / kitab_adab | print | modern | book_print | provider |
| kitab_patsocr | print | classical | religious_print | provider |
| kitab_isippt | print | modern | slides | provider |
| kitab_evarest | scene_text | modern | in_the_wild | provider |
| kitab_arabicocr | print | modern | print | provider |
| nakba_lines | modern_hw | 20c | memoir | provider |
| rasam1 / rasam2 | maghribi | historical | manuscript | provider |
| arams28k | per-book (`books:` table) | 19c | manuscript | book |
| synthetic | print | n/a | synthetic | provider |

## Crop cache & lazy semantics

- Line-crop corpora (all 15 + AraMS mirror): no polygon; the crop *is* the
  source image, written once to `crops/<corpus>/<line_id>.png`.
- Page corpora (future): store `page_path` + `polygon`; crops regenerated on
  demand into `crops/<corpus>/<line_id>@<sha1(crop_params)>.png`. `crop_params`
  are passed at build time, **not** stored in the JSONL — they stay an ablation
  knob for the Tier-0 `training/crops.py` module.
- `data/canonical/` is git-ignored. JSONL is reproducible from adapters +
  `corpora.yaml`.

## Outputs

```
data/canonical/
  <corpus>.jsonl        # one CanonicalLine per row, sorted by line_id
  crops/<corpus>/…png
  _charset.json         # global + per-corpus + per-split + union charset, OOV, tail, train→test gap
  _manifest.json        # corpus -> {n_lines, converter_version, built_at, adapter_sha, normalize_sha}
```

## Consumed by Tier 0

| experiment | needs |
| --- | --- |
| 0.1 CTC baseline | AraMS canonical + `--kraken-out` tree; `_charset.json` (AraMS-scoped); Muharaf Kraken rec warm-start (Zenodo `10.5281/zenodo.14295489`, fetch separately) |
| 0.2 cross-corpus matrix | canonical for **arams28k, muharaf_lines, rasam1, rasam2, nakba_lines, kitab_khatt**; `_charset.json` `union` over those six; per-script + bootstrap-CI readout (eval-readiness spec) |
| 0.3 script-balanced sampling | per-line `script`; `_charset.json` union |
| 0.4 diacritics decoupling | `text_diplomatic` + `text_diac_normalised` (AraMS); `diac_full` profile; diacritic F1 (already in `metrics.py`) |
| 0.5 PV-vs-LV bias | `validation_level` + per-line `alignment_confidence` + per-book confidence-100 rate |
| 0.6 contamination probe | public vs unpublished line split (canonical `split` + provenance) |

## Muharaf split note

`aamijar/muharaf-public` parquet columns are **only `image` + `text`** — no
document/page/writer id. A collection-level split therefore **cannot** be built
from the HF mirror. The Muharaf paper states its splits are document-disjoint;
the eval-readiness spec decides between (a) citing that and setting
`split_unit: provider` with a footnote, or (b) reconstructing document
membership from the full GitHub repo. This spec assumes (a).

## Error handling

- A corpus whose adapter raises is logged and skipped; others still build.
  `_manifest.json` records only completed corpora. Non-zero exit if any
  requested corpus failed.
- `to_canonical` raises `ValueError` on empty `text_diplomatic` or a `line_id`
  collision within a corpus.

## Testing

- `tests/test_canonical.py`: both `line_id` forms; `text_l1/l2/l3` == `normalize()`;
  `diacritic_density` on a known string; null-field defaults; `text_diac_normalised`
  / `validation_level` passthrough from `Sample.meta`; collision raises.
- `tests/test_normalize.py`: `standard` folds ى/ة and `light` does not;
  `diac_full` strips a Qur'anic mark that `diac_insensitive` keeps; `raw`/`light`
  outputs byte-identical to before.
- `tests/test_charset.py`: known charset + OOV math on synthetic; train→test gap.
- `configs/corpora.yaml` load test: every wired corpus has an entry.
- CI smoke: `build_canonical.py --only synthetic --no-crops` → 10 `CanonicalLine`s;
  `build_charset.py --corpora synthetic`.

## Open inputs (assumed; confirm at implementation)

1. **AraMS HF mirror repo name** — assumed `rabeeh97/arams28k-htr-lines`.
2. **`corpora.yaml` `genre`/`era`** for KITAB print subsets — low-confidence.
3. **AraMS field assumptions** in the "AraMS-28k-HTR specifics" section — verified
   at mirror time.

## Deferred to later specs

- `training/crops.py` — polygon mask + page-median fill + pad-factor sweep +
  64px height-norm + width-bucket + pad-in-batch, **no deslant** (Tier-0 spec).
- Eval-readiness spec: contamination NxN matrix (pHash ≤6 + 5-gram Jaccard ≥0.9,
  pairwise, published), Muharaf split decision, `report.py` per-row additions
  (length ratio, truncation/repetition/empty flags, per-script breakdown,
  bootstrap CI), `docs/metrics.md` paragraph (codepoint edit distance, identical
  normalization pre-denominator, WIDE↔FULL delta, Maghribi qāf/fāʾ dotting).
- `scan_quality` heuristic; `text_edited` population.

## Known operational note (not part of this design)

Gemini + free-tier sweeps were dying on a HF Xet CAS `File reconstruction error`
fetching `arocrbench_historyar` / `arocrbench_onlinekhatt`. Fixed by
`HF_HUB_DISABLE_XET=1`, now permanent in `env.sh`.
