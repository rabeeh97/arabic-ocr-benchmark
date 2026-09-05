# Datasets — How to Obtain and Prepare Each One

Companion to `benchmark-matrix.md` and the Phase 1 plan. For every dataset we
plan to benchmark on (primary + stretch): where it lives, license, how to fetch
it (online or local), what the ground truth looks like, and how it maps onto our
`Sample` interface and the line / page tracks.

**Common contract.** Every dataset adapter yields `Sample(id, image, ground_truth,
track, meta)`:
- `id` — stable, globally unique (e.g. `muharaf_lines/test/000123`).
- `image` — a `PIL.Image` (line crop or full page).
- `ground_truth` — the reference transcription string, kept **raw** (no
  normalization; harakat preserved as in the source).
- `track` — `"line"` or `"page"`.
- `meta` — free dict: writer, era, script style, document type, source page id,
  polygon, dpi — whatever the source provides (used for the optional
  metadata-in-prompt experiment and for error slicing).

**Hold-out discipline.** Muharaf's published `test` split is the benchmark set
and must never be used for fine-tuning. Any dataset without an official split:
we define one deterministically (hash of `id`) and record it.

---

## Tier 1 — Primary: Muharaf

Paper: *Muharaf: Manuscripts of Handwritten Arabic Dataset for Cursive Text
Recognition*, NeurIPS 2024 Datasets & Benchmarks. arXiv `2406.09630`.
~1,600+ expert-transcribed historical page images (early 19th–early 21st c.,
predominantly Ruq'ah), with polygonal line coordinates and page-element tags.

### 1a. Muharaf-public — lines (`ML`, primary, start here)

| Field | Value |
| --- | --- |
| Source | Hugging Face: `aamijar/muharaf-public` |
| Level | line-level crops |
| Size | `train` 22,092 · `validation` 1,069 · `test` 1,334 (≈24,495 lines, ~1.32 GB) |
| Columns | `image` (Image), `text` (string, 1–181 chars) |
| License | CC-BY-NC-SA-4.0 (non-commercial; fine for research benchmarking — cite the paper) |
| Access | public, no gating |

Fetch (online, no local storage needed):
```python
from datasets import load_dataset
ds = load_dataset("aamijar/muharaf-public")          # streams from HF cache
test = ds["test"]                                     # the benchmark set
sample = test[0]; sample["image"], sample["text"]
```
Or pin a local snapshot:
```bash
hf download aamijar/muharaf-public --repo-type dataset --local-dir data/muharaf-public-lines
```

Adapter → `datasets/muharaf_lines.py`:
- `track="line"`, `ground_truth = row["text"]`,
  `id = f"muharaf_lines/{split}/{idx:06d}"`.
- Benchmark uses `split="test"` only. `train`/`validation` reserved for Phase 5.
- Smoke subset: `test.select(range(15))`.

### 1b. Muharaf-public — pages (`MP`, primary, Phase 3)

| Field | Value |
| --- | --- |
| Source | GitHub `MehreenMehreen/muharaf` (data + docs) and/or Zenodo record `11492215` |
| Related | `MehreenMehreen/ScribeArabic` (annotation tool), `MehreenMehreen/start_follow_read_arabic` (page-recognition pipeline) |
| Level | full page images + per-page JSON |
| Size | 1,216 public page images |
| License | CC-BY-NC-SA-4.0 |
| GT format | JSON per image: text-line polygons + transcription per line; page elements (page number, heading, graphic) tagged. JSON → PAGE-XML converter in the repo's `page-xml/` folder |

Fetch (local):
```bash
git clone https://github.com/MehreenMehreen/muharaf data/muharaf-repo
# follow its README to pull the public page images + JSON (Zenodo mirror: record 11492215)
```

Adapter → `datasets/muharaf_pages.py`:
- One `Sample` per page. `track="page"`.
- `ground_truth` = line transcriptions from the JSON concatenated in reading
  order: sort lines top→bottom by polygon y-centroid; within a band, right→left
  by x-centroid (RTL). One newline between lines. Skip non-text elements.
- `meta` carries the raw polygons + element tags + source filename.
- **Restrict to pages whose lines appear in the HF `test` split** so `MP` and
  `ML` describe the same held-out material. Build the page→split map by matching
  line text / image hashes against `aamijar/muharaf-public` `test`; record the
  resulting page id list in the repo (`data/muharaf_pages_test_ids.txt`).

### 1c. Muharaf-restricted — pages (`MR`, stretch)

| Field | Value |
| --- | --- |
| Size | 428 additional page images |
| License | proprietary |
| Access | email **Carlos Younes**, Phoenix Center for Lebanese Studies, USEK (contact details on the `MehreenMehreen/muharaf` GitHub) and request the restricted set |
| Use | optional extra benchmark slice, kept separate; never merged into training |

Same adapter as 1b once obtained; `id` prefix `muharaf_restricted/`.

---

## Tier 2 — Secondary Arabic HTR datasets (Phase 4)

### KHATT — KFUPM Handwritten Arabic Text (`KH`)

| Field | Value |
| --- | --- |
| Content | ~1,000 writers; 2,000 fixed-text + 2,000 unique-text paragraph images; extracted line images; GT at page / paragraph / line level; scans at 200/300/600 dpi |
| Official | KFUPM request form (Prof. Sabri Mahmoud's group); also LDC catalog `LDC2015T23` |
| HF mirrors | `benhachem/KHATT`, `johnlockejrr/KHATT_v1.0_dataset` (convenient, verify against official GT before trusting) |
| License | research use; check the KFUPM agreement / LDC terms |
| Tracks | `line` (extracted line images) and `page` (paragraph images) |

Fetch (online, mirror):
```python
from datasets import load_dataset
ds = load_dataset("benhachem/KHATT")
```
Fetch (official): submit the KFUPM form, download the v1.0 archive, unzip locally
to `data/khatt/`.

Adapter notes: use the "unique-text" paragraphs for the page track (fixed-text
paragraphs repeat the same content across writers → leakage-like redundancy if
mixed). Define a deterministic test split if the mirror lacks one.

### KITAB-Bench (`KB`) — **implemented, priority datasets**

| Field | Value |
| --- | --- |
| Content | 8,809 samples, 9 domains / 36 sub-domains. We use only the **text-recognition** subsets |
| Source | HF subsets under `ahmedheakl/arocrbench_*` — **ungated, no access form**, one `train` split each, columns `image` + (`text` or `answer`) + `source` |
| Paper | arXiv `2502.14949` (ACL 2025 Findings) |
| License | per each source dataset + the KITAB-Bench paper |
| Use for us | the priority benchmark datasets — small (200–500 lines), quick, cover handwritten + historical + printed Arabic in one consistent format |

One shared adapter, `datasets/kitab_bench.py::KitabBench`, selected by
`params.subset` in `configs/datasets.yaml`. Ground truth is whichever of
`text` / `answer` the subset uses. Registered datasets:

| name | HF subset | kind |
| --- | --- | --- |
| `kitab_khatt` | `arocrbench_khatt` | modern handwritten (KHATT lines) — **main handwriting target** |
| `kitab_muharaf` | `arocrbench_muharaf` | historical handwritten (Muharaf slice) — KITAB-scored comparison to `muharaf_lines` |
| `kitab_historicalbooks` | `arocrbench_historicalbooks` | degraded historical printed |
| `kitab_hindawi` | `arocrbench_hindawi` | clean digitised printed — easy anchor |

More subsets (`arocrbench_adab`, `_arabicocr`, `_patsocr`, `_isippt`, …) drop in
with the same adapter — just add an entry with `params.subset`.

```python
from datasets import load_dataset
ds = load_dataset("ahmedheakl/arocrbench_khatt", split="train")   # image, text, source
```

The full KITAB-Bench repo (`mbzuai/KITAB-Bench`) IS gated and its chart / table /
PDF-to-markdown tasks are a different track — out of scope here.

### NAKBA NLP 2026 — Omar Al-Saleh manuscripts (`NK`) — **implemented**

| Field | Value |
| --- | --- |
| Content | Line segments from 16 handwritten Arabic documents (Omar Al-Saleh memoirs, 1951–1965, ~1.6M words) — real degraded 20th-century handwriting with expert line-level transcription |
| Source | HF `U4RASD/omar-al-saleh-manuscripts-segments` — **gated, auto-approval**: open the page logged in, click *Agree and access repository* once, then `HF_TOKEN` streams it. Splits `train` / `test` / `blind_test` (blind_test has no GT) |
| Shared task | <https://acrps.ai/nakba-nlp-manu-understanding-2026> (Systems Track = OCR/HTR) |
| License | CC-BY-4.0 |
| Baseline | `U4RASD/ar-ms-baseline` — LoRA on Qwen3-VL-8B-Instruct, organisers report 0.23 CER / 0.50 WER on their test set. Wired as model `nakba-baseline-qwen3vl`; the untuned base is `qwen3-vl-8b` |

Served by the same `KitabBench` adapter — `params.subset:
U4RASD/omar-al-saleh-manuscripts-segments`, `split: test`. The page-level repo
`U4RASD/omar-al-saleh-manuscripts-full` is **manual**-approval and not wired.

### AHAWP — Arabic Handwritten Alphabets, Words, Paragraphs
- Mendeley Data. Includes paragraph-level handwritten Arabic with transcriptions.
- Fetch: download the Mendeley archive to `data/ahawp/`. Use the paragraph subset
  only; alphabets/isolated words are out of scope.
- Stretch.

### Historical Arabic Handwritten Text Recognition Dataset
- Mendeley Data. 40 meticulously transcribed historical Arabic pages from
  distinct books / regions / centuries.
- Fetch: Mendeley download → `data/hist-arabic-40/`. Small; useful as a
  hard, diverse page-track slice. Stretch.

### IFN/ENIT (`IE`, low priority)
- Handwritten Tunisian town/village names; word-level; ~26k word images, 5 sets
  (a–e).
- Access: request from TU Braunschweig (Institute for Communications Technology)
  via their IFN/ENIT agreement form. No open mirror we should rely on.
- Adapter: `track="word"` (treated as a degenerate line). Keep for literature
  comparability only.

---

## Tier 3 — Anchors / contrast (Phase 4+)

### Printed-Arabic anchor (`PR`)
Pick one, kept deliberately easy to give a low-CER reference point:
- **KITAB-Bench printed subset** (simplest — already in the `KB` pipeline), or
- **AMFDS** (Arabic Multi-Fonts Dataset — printed lines across many fonts,
  some handwritten-style), or
- **APTI** (Arabic Printed Text Image database — synthetic printed lines,
  request from the authors).

### Synthetic / structured contrast
- `NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset` (HF) — synthetic full-page
  Arabic with layout. Useful to see how much of Qari's strength is
  in-distribution vs transfers to real manuscripts. Stretch.

### OnlineKHATT — explicitly out of scope
Online (pen-trajectory) handwriting. Our benchmark is offline image→text only.
Noted so it isn't accidentally added.

---

## Storage & caching conventions

```
data/
  muharaf-public-lines/     # HF snapshot (or rely on ~/.cache/huggingface)
  muharaf-repo/             # git clone: public pages + JSON
  muharaf_pages_test_ids.txt
  muharaf-restricted/       # if access granted
  khatt/                    # official archive
  kitab-bench/              # HF subsets snapshot
  ...
```
- Prefer streaming from the HF cache for HF-hosted sets; only pin local
  snapshots for sets we filter/re-split (Muharaf pages) or that need an
  agreement (KHATT, IFN/ENIT, Muharaf-restricted).
- Never commit dataset images to the repo. Commit only: split id lists, the
  page→split map, and small `tests/fixtures/` golden samples we transcribe
  ourselves.

## Adding a new dataset — checklist

1. Add a row to `configs/datasets.yaml`: `name`, adapter module, `track(s)`,
   `split`, `license`, `source`.
2. Implement `datasets/<name>.py` yielding `Sample`s with a stable `id` scheme.
3. If no official split, add a deterministic `id`-hash split and record it.
4. Add the dataset column/rows to `benchmark-matrix.md`.
5. Verify: load 5 samples, render image + GT side by side, check encoding
   (UTF-8, RTL display), check the GT is raw (harakat intact).
