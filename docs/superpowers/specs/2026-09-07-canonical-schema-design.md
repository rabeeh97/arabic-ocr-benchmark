# Canonical line schema + offline conversion layer

Date: 2026-09-07
Status: approved design, pre-implementation
Path class: architectural

## Problem

Every corpus in this benchmark is consumed through a bespoke `DatasetAdapter`
that yields a minimal `Sample(id, image, ground_truth, track, meta)`. That is
enough to score a model, but not enough for the training and analysis work the
project is heading into: no page context, no difficulty signal, no separation of
the transcription *as written* from an editorial reconstruction, no single place
where per-line metadata (script, era, licence, split unit) lives.

We want one canonical record that every corpus converts into **before** any
downstream training/analysis pipeline runs, following the "get this right first"
advice captured in the project notes.

## Scope (decided with the user)

- **Offline JSONL export, additive.** A new `scripts/build_canonical.py`
  iterates each corpus's *existing* adapter and writes
  `data/canonical/<corpus>.jsonl` plus a lazy crop cache. The benchmark runner,
  scorer and leaderboard are **not** touched. Canonical files are a parallel
  artifact.
- **First-cut coverage:** all 15 datasets currently wired in
  `configs/datasets.yaml` (7 with leaderboard results + 8 wired-but-unrun), all
  of which are the pre-segmented line-crop shape, plus a **new AraMS-28k-HTR
  adapter** so the converter is exercised against a corpus that genuinely has
  `book_id` / `page_id` and needs a real `alignment_confidence`.
- AraMS-28k-HTR is brought in via a **one-time HF mirror**: download the Zenodo
  archive once, push a cleaned line-crop dataset to a HF repo under the user's
  account, then stream it like the other HF corpora (no 2.4 GB local dependency
  for Colab).

Out of scope for this pass: page-track corpora (Muharaf-full, AraMS pages,
OpenITI MAKHZAN), SARD, and any change to how models are run or scored.

## The canonical record

`src/arabic_ocr_bench/canonical.py` defines a frozen dataclass `CanonicalLine`
and a pure `to_canonical()` builder. Serialised JSON shape (one object per line
in the corpus JSONL):

```jsonc
{
  "line_id": "arams28k/book_03/p0172/l007",
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
  "text_edited": null,                          // editorial reconstruction; null for every current corpus
  "text_l1": "…", "text_l2": "…", "text_l3": "…",  // = normalize(text_diplomatic, raw|light|diac_insensitive)

  "script": "ruqah",                            // corpus-level constant (configs/corpora.yaml)
  "era": "19c-21c", "genre": "correspondence", "writer_id": null,
  "diacritic_density": 0.031,                   // harakat chars / non-space chars of text_diplomatic
  "scan_quality": null,                         // heuristic pass later; null now
  "alignment_confidence": 100,                  // 100 for gold pre-aligned sets; real score where we align

  "split": "test", "split_unit": "provider",    // provider|book|page|writer
  "license": "CC-BY-NC-SA-4.0",
  "provenance": {
    "source_id": "muharaf_lines/test/000123",  // original Sample.id, always
    "hf_id": "aamijar/muharaf-public", "hf_revision": "<commit-or-null>",
    "split": "test", "row": 123, "converter_version": 1
  }
}
```

### Deviations from the originally suggested schema

- **`text_raw_diacritized` dropped.** For every corpus we hold, the ground truth
  *is* the reference edition; it lives in `text_diplomatic`. Re-introduce only if
  a corpus ever ships two distinct editions.
- **`text_l1/l2/l3` are defined as `normalize.py` output**, not a fourth
  hand-maintained transcription: `text_lN = normalize(text_diplomatic, PROFILES[N-1])`
  with `PROFILES = ("raw", "light", "diac_insensitive")`. One source of truth.
- **`text_diplomatic` vs `text_edited` are separate columns from day one** even
  though `text_edited` is `null` everywhere today — this is the one field that
  cannot be retrofitted later.
- **`script` / `era` / `genre` are corpus-level constants** from
  `configs/corpora.yaml`, stored per line for convenience.
- **Added `track`** (`line` | `page`) — the leaderboard already depends on it.
- **`split_unit` has no `"line"` value.** Allowed: `provider` (pre-split by the
  dataset publisher — all 15 current corpora), `book`, `page`, `writer`.

### Field population tiers

- **Always populated:** `line_id`, `corpus`, `track`, `text_diplomatic`,
  `text_l1/l2/l3`, `script`, `era`, `license`, `split`, `split_unit`,
  `provenance`, `diacritic_density`, `zone` (defaults `"main"`),
  `angle_deg` (defaults `0.0`), `alignment_confidence`.
- **Populated only when the source carries it:** `book_id`, `page_id`,
  `page_path`, `polygon`, `bbox`, `reading_order` (falls back to source row
  index), `writer_id`, `genre`. Non-null today only for AraMS-28k-HTR.
- **Deferred / null now:** `text_edited`, `scan_quality`, `insertion_anchor`.

## `line_id`

- Hierarchical when the source has book/page structure:
  `<corpus>/<book_id>/<page_id>/l<line_no:03d>` (AraMS; any future page corpus).
- Flat otherwise: `<corpus>/<split>/<row:06d>` — identical to today's
  `Sample.id`, so existing result files stay joinable.
- `provenance.source_id` always holds the original `Sample.id` verbatim.

## Module layout & data flow

| Unit | Responsibility | Depends on | Tested by |
| --- | --- | --- | --- |
| `src/arabic_ocr_bench/canonical.py` | `CanonicalLine`, `to_canonical(sample, corpus_meta, *, hf_revision=None, align_conf=100)`, `diacritic_density(text)`. Pure, no I/O. `corpus_meta` is the merged dict the build script assembles per corpus: `script`/`era`/`genre`/`split_unit`/`alignment_confidence_default` from `corpora.yaml` + `license`/`split`/`hf_id` from `datasets.yaml`. | `normalize.py` | `tests/test_canonical.py` |
| `configs/corpora.yaml` | Per-corpus constants: `script`, `era`, `genre`, `split_unit`, `alignment_confidence_default`. `license` is read from `datasets.yaml`, not duplicated. | — | schema-load test |
| `src/arabic_ocr_bench/datasets/arams.py` | New adapter: streams the HF mirror of AraMS-28k-HTR, yields `Sample` with `meta` carrying `book_id`, `page_id`, `line_no`. | HF mirror repo | adapter smoke |
| `scripts/mirror_arams_to_hf.py` | One-time: Zenodo `10.5281/zenodo.21499649` -> parse `<id>.png` + `<id>.gt.txt` -> HF dataset (`image`, `text`, `book_id`, `page_id`, `line_no`, book-level `split`) -> `push_to_hub`. | Zenodo, HF auth | manual run |
| `scripts/build_canonical.py` | For each corpus in `datasets.yaml` (`--only` filter): instantiate its adapter, iterate `Sample`s, `to_canonical`, write `data/canonical/<corpus>.jsonl` sorted by `line_id`; unless `--no-crops`, write the line image to `data/canonical/crops/<corpus>/<line_id>.png`. Idempotent (skip when JSONL newer than adapter module + `corpora.yaml`, override `--force`). Writes `data/canonical/_manifest.json`. | adapters, `canonical.py`, `corpora.yaml` | `--only synthetic` CI smoke |

No change to `runner.py`, `report.py`, `cli.py`, `make_leaderboard.py`.

## `configs/corpora.yaml` — proposed values

| corpus | script | era | genre | split_unit | align_conf |
| --- | --- | --- | --- | --- | --- |
| muharaf_lines | ruqah | 19c-21c | correspondence | provider | 100 |
| kitab_khatt | modern_hw | 21c | prompts | provider | 100 |
| kitab_onlinekhatt | modern_hw | 21c | prompts | provider | 100 |
| kitab_muharaf | ruqah | 19c-21c | historical_hw | provider | 100 |
| kitab_historyar | naskh | historical | manuscript | provider | 100 |
| kitab_historicalbooks | print | historical | book_print | provider | 100 |
| kitab_hindawi | print | modern | book_print | provider | 100 |
| kitab_adab | print | modern | book_print | provider | 100 |
| kitab_patsocr | print | classical | religious_print | provider | 100 |
| kitab_isippt | print | modern | slides | provider | 100 |
| kitab_evarest | scene_text | modern | in_the_wild | provider | 100 |
| kitab_arabicocr | print | modern | print | provider | 100 |
| nakba_lines | modern_hw | 20c | memoir | provider | 100 |
| rasam1 | maghribi | historical | manuscript | provider | 100 |
| rasam2 | maghribi | historical | manuscript | provider | 100 |
| arams28k | naskh | historical | manuscript | book | per-line from source, else 90 |
| synthetic | print | n/a | synthetic | provider | 100 |

## Crop cache & lazy semantics

- Line-crop corpora (all 15 wired + AraMS mirror): no polygon; the crop *is* the
  source image, written once to `crops/<corpus>/<line_id>.png`. No
  `crop_params_hash`.
- Page corpora (future): store `page_path` + `polygon`; crops regenerated on
  demand into `crops/<corpus>/<line_id>@<sha1(crop_params)>.png`. `crop_params`
  (padding, deskew, min-height) are passed to the converter, **not** written into
  the JSONL, so they remain an ablation knob.
- `data/canonical/` is git-ignored. The JSONL is fully reproducible from the
  adapters + `corpora.yaml`.

## Outputs

```
data/canonical/
  <corpus>.jsonl        # one CanonicalLine per row, sorted by line_id
  crops/<corpus>/…png
  _manifest.json        # corpus -> {n_lines, converter_version, built_at, adapter_sha}
```

## Error handling

- A corpus whose adapter raises (download failure, gated without token) is logged
  and skipped; other corpora still build. `_manifest.json` records only corpora
  that completed.
- `to_canonical` raises `ValueError` on an empty `text_diplomatic` or a
  `line_id` collision within a corpus — these are data bugs, not skippable.
- `build_canonical.py` exit code is non-zero if any requested corpus failed.

## Testing

- `tests/test_canonical.py`: `to_canonical` on a synthetic `Sample` — both
  `line_id` forms, `text_l1/l2/l3` equal `normalize()` output, `diacritic_density`
  on a known string, null-field defaults, `provenance.source_id` preserved,
  collision raises.
- `configs/corpora.yaml` load test: every name in `datasets.yaml` (minus
  planned/commented) has a `corpora.yaml` entry with the required keys.
- CI smoke: `python scripts/build_canonical.py --only synthetic --no-crops`
  (offline, 10 lines) and assert the JSONL parses to 10 `CanonicalLine`s.

## Open inputs (assumed values, confirm before/at implementation)

1. **AraMS HF mirror repo name** — assumed `rabeeh97/arams28k-htr-lines`
   (matches the GitHub org `rabeeh97/arabic-ocr-benchmark`). Adjust if the HF
   username differs.
2. **`corpora.yaml` values** — the table above is a first proposal; `genre` and
   `era` for the KITAB print subsets are low-confidence.

## Follow-up work — separate spec, after this lands

Agreed in the design discussion, to be specced once `data/canonical/*.jsonl`
exists to build on:

- **`normalize.py` gains two reported profiles, no change to `raw` / `light` /
  `diac_insensitive`:**
  - `standard` = `light` + alef-maqsura→yaa (ى→ي) + taa-marbuta→haa (ة→ه). Used
    only for the cross-paper comparison table (RefLAM / CAMeL fold these at their
    headline; we keep `light` strict so the headline still measures the
    distinction). Existing leaderboard numbers are unaffected.
  - `diac_full` = `diac_insensitive` but the stripped mark set is WIDE
    (`U+064B–065F, U+0670`) **plus** the Qur'anic marks `U+0610–061A` and
    `U+06D6–06ED`. `diac_insensitive` (WIDE) stays the headline diacritic
    profile; `diac_full` is scored alongside it and the WIDE↔FULL delta is
    reported as a number (RefLAM's and Farazi's CERs are not otherwise
    comparable). The tashkeel set becomes a parameter of `strip_harakat`.
- **`text_l1/l2/l3` in the canonical record stay = `normalize(text_diplomatic,
  {raw,light,diac_insensitive})`.** `standard` / `diac_full` forms are computed
  at analysis/report time, not stored per line.
- **`scripts/build_charset.py`** — reads `data/canonical/*.jsonl`, params
  `--level {l1,l2,l3,standard}` and `--min-count` (default 20); emits
  `data/canonical/_charset.json` with global + per-corpus + per-split charset,
  size, OOV mass, frequency tail, and the train→test character gap. **No line is
  ever dropped here.** The `<unk>`-vs-filter choice is an `oov_policy: unk |
  error` field in `training/config.py` — `unk` for training loaders, `error`
  (never silently drop) for any eval loader.
- **`docs/metrics.md`** gains a paragraph: codepoint-level edit distance,
  reference and hypothesis normalised identically before the denominator,
  WIDE↔FULL tashkeel delta, and the Maghribi qāf/fāʾ dotting note (same Unicode
  codepoint as Mashriqi, so a visual→symbol transfer problem that only
  script-diverse training data fixes; slice CER by the `script` field to show
  it).

## Known operational note (not part of this design)

The Gemini full crawl and the free-tier sweep were failing to download
`arocrbench_historyar` / `arocrbench_onlinekhatt` with a HF Xet CAS
`File reconstruction error`. Mitigation: `HF_HUB_DISABLE_XET=1` for those runs.
Tracked separately from this schema work.
