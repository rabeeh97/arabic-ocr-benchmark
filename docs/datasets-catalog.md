# Dataset catalog — what's in, what's next, what's blocked

Acquisition-and-status view of every Arabic OCR/HTR corpus this benchmark uses or
wants. For the adapter design and the `Sample(id, image, ground_truth, …)`
contract, see [`datasets.md`](datasets.md); this file is the shopping list.

All benchmark datasets are scored **line-level** (one image = one text line) on
their **test/held-out** split where one exists, or `train` for the small
KITAB-Bench subsets (which have no separate test split — they *are* the
eval set).

---

## 1. Live — wired in `configs/datasets.yaml`, results on the leaderboard

| name | source | lines (split) | script / type | license | notes |
| --- | --- | --- | --- | --- | --- |
| `muharaf_lines` | HF `aamijar/muharaf-public` | 1,334 (test) | handwritten, mostly Ruq'ah | CC-BY-NC-SA-4.0 | NeurIPS 2024; the headline handwriting target |
| `nakba_lines` | HF `U4RASD/omar-al-saleh-manuscripts-segments` | 372 (test) | handwritten, 1951–65 memoirs | CC-BY-4.0 | NAKBA NLP 2026 shared task; **gated (auto-approve)** |
| `kitab_khatt` | HF `ahmedheakl/arocrbench_khatt` | 200 (train) | modern handwritten (KHATT lines) | KHATT terms, research | KITAB-Bench slice |
| `kitab_onlinekhatt` | `arocrbench_onlinekhatt` | 200 | stylus-captured KHATT | KHATT terms | cleaner strokes |
| `kitab_muharaf` | `arocrbench_muharaf` | 200 | historical handwritten | CC-BY-NC-SA-4.0 | 200-line Muharaf slice |
| `kitab_historyar` | `arocrbench_historyar` | 200 | historical manuscripts | KITAB-Bench | degraded scans |
| `kitab_historicalbooks` | `arocrbench_historicalbooks` | 10 | degraded historical **print** | KITAB-Bench | tiny |
| `kitab_hindawi` | `arocrbench_hindawi` | 200 | clean digitised **print** | KITAB-Bench | easy anchor (≈0 CER for strong models) |
| `synthetic` | in-repo | 10 | rendered print | n/a | offline pipeline check, not a benchmark |

---

## 2. Just added (2026-09-06) — config-only, `KitabBench` generic adapter

All ungated on HF; the shared `KitabBench` adapter reads `image` + one of
`text` / `answer` / `transcription`. **No results yet — run notebook 04 (or
`bench run`) to populate them.** Added to notebook 04 / 04b `DATASETS` and the
`_HF_SPLIT` sample-image map.

| name | source | lines (split) | script / type | license |
| --- | --- | --- | --- | --- |
| `rasam2` | HF `calfa-ai/RASAM-2` | 372 (test) | **Maghrebi**-script manuscripts (BULAC) | Apache-2.0 |
| `rasam1` | HF `calfa-ai/RASAM-1` | 734 (test) | Maghrebi-script manuscripts, earlier set | Apache-2.0 |
| `kitab_adab` | `arocrbench_adab` | 200 (train) | short printed literary segments | KITAB-Bench |
| `kitab_patsocr` | `arocrbench_patsocr` | 500 | printed pages, classical text (`answer` col) | KITAB-Bench |
| `kitab_isippt` | `arocrbench_isippt` | 500 | printed Arabic from slides | KITAB-Bench |
| `kitab_evarest` | `arocrbench_evarest` | 800 | **scene text** / in-the-wild words | KITAB-Bench |
| `kitab_arabicocr` | `arocrbench_arabicocr` | 50 | small printed set | KITAB-Bench |

Why: RASAM adds the **Maghrebi script family** (absent everywhere else in the
suite); the extra KITAB subsets extend the **printed** axis (slides, scene text,
classical print) so the board isn't only handwriting.

---

## 3. Recommended next — need real work, not just a config line

| corpus | where | size | why it's worth it | what it takes | difficulty |
| --- | --- | --- | --- | --- | --- |
| **AraMS-28k-HTR** | Zenodo `10.5281/zenodo.21499649` (~2.42 GB) | line crops + `.gt.txt` | 27,971 real historical-manuscript lines — the single biggest new signal; pre-segmented (Kraken-style) so no polygon work | one-time download, host the crops (Drive or a HF mirror), write `datasets/arams.py` reading `<id>.png` + `<id>.gt.txt` pairs, pick a held-out split | **medium** — the format is trivial; the friction is the 2.4 GB download + where to keep it |
| **AraMS-28k** (pages) | Zenodo `10.5281/zenodo.22095333` (~2.05 GB) | 3,043 pages / 27,971+629 lines + JSONL | same corpus, page-level with line polygons — needed for a future page track | download + a page adapter that crops lines from the JSONL polygons | **medium-high** — only do this once a page track exists; use `-HTR` for line-level |
| **SARD** | HF `riotu-lab/SARD` | ~100k+ samples, 6 fonts | clean synthetic **print**, contamination-free, per-font splits — a controlled baseline | custom `datasets/sard.py`: splits are per font (`Amiri`, `Sakkal_Majalla`, …), GT is the `chunk` column, image is **base64 in `image_base64`** (not an HF `Image` feature) — decode + subsample (100k is too big to score whole) | **medium** — non-standard schema; sample ~200/font |
| **OpenITI MAKHZAN** | GitHub `OpenITI/MAKHZAN` (JOHD 2026) | 1,497 pages (822 Arabic) | diverse historical print/manuscript; also Persian/Ottoman/Urdu pages | page images + TEI-XML transcriptions; needs an XML parser + line segmentation, and a language filter to keep Arabic | **high** — page track + XML; lower priority |
| **RASAM 2 val/train** | HF `calfa-ai/RASAM-*` | +2,976 / +5,864 lines | more Maghrebi data if the 372/734-line test sets prove too small to rank on | change `split:` in the config | trivial (already wired) |

---

## 4. Blocked / gated / out-of-scope

| corpus | status | detail |
| --- | --- | --- |
| **KHATT** (full, 1,000 writers) | **request required** | KFUPM release form, approval takes **weeks**. Start the request now if wanted. We already have the KITAB-Bench KHATT slices (`kitab_khatt`, `kitab_onlinekhatt`, 200 lines each) as a stand-in. |
| **MADCAT** | **paid** | LDC, ~42,000 pages, licensed. Skip unless a grant covers it (user's call). |
| **RASAM 1/2 via Calfa/BULAC directly** | not needed | the HF mirrors (`calfa-ai/RASAM-1`, `calfa-ai/RASAM-2`) are Apache-2.0 and ungated — use those, no BULAC registration. |
| **`mbzuai/KITAB-Bench`** (full 13-subset repo) | **gated** | the chart / table / PDF-to-markdown tasks are a different track. The line-recognition subsets are all separately available as ungated `ahmedheakl/arocrbench_*` repos — that's what we use. |
| **Muharaf** (full GitHub, 36,311 lines / polygons) | partially in | `muharaf_lines` is the public HF line split (24,495 public lines, we score the 1,334-line `test`). The full polygon set is for a **page track** (`muharaf_pages`, planned, adapter not written). |
| **RASAM 2 "Maghribi"** whole-page PAGE-XML | superseded | the HF mirror is already line-crops + text; no PAGE-XML parsing needed. |

### Not datasets (referenced for a training pipeline, not scored here)

| artifact | where | purpose |
| --- | --- | --- |
| Muharaf Kraken **segmentation** model | Zenodo `10.5281/zenodo.14295555` | RefLAM's bootstrap line segmenter — only relevant if we build a Kraken page pipeline |
| Muharaf Kraken **recognition** model | Zenodo `10.5281/zenodo.14295489` | a warm-start for fine-tuning a Kraken recogniser |
| NAKBA eval scripts + 7 system outputs | shared-task repo | the competing systems are wired as **models** (`nakba-*` in `models.yaml`), not as data |

---

## 5. How to add a dataset

1. **If it's an HF `image` + text-column dataset** (any of `text` / `answer` /
   `transcription` / `label` / `gt` / `target`): just add a `configs/datasets.yaml`
   entry pointing `KitabBench` at it — `params: { subset: <org/name> }`, set
   `split`. Done. (This is how §2 was added.)
2. **Otherwise**: write `src/arabic_ocr_bench/datasets/<name>.py` yielding
   `Sample(id=f"<name>/<split>/<idx:06d>", image=<PIL>, ground_truth=<str>,
   track="line", meta={...})`, with a stable id scheme. Register it in
   `datasets.yaml`, add it to the notebook `_HF_SPLIT` map (for leaderboard
   sample images) and to notebook 04's `DATASETS` list.
3. Run `bench run --config <cfg>` or the Colab sweep; the leaderboard picks it
   up automatically from `results/*/summary.csv`.

Contamination check before trusting a new set: `06_colab_prepare_data.ipynb`
compares candidate text against the benchmark test splits.
