# Plan: Arabic Cursive Script Recognition Benchmark (Phase 1)

## Context

We are starting a new project from scratch. The goal of Phase 1 (per
`Rabeeh | Eyad (LLM Benchmarking).md`) is to build a **reproducible benchmark** that
measures how well current LLMs / vision-language models transcribe **cursive
handwritten Arabic**, and to compare them on the *same* data with the *same*
metrics. The hypothesis driving the project: general multimodal LLMs are
overtaking traditional HTR pipelines, and cheap fine-tuning (OpenAI, Vertex AI)
could close the remaining gap if data is available.

The background survey (`OCR Research.md`) establishes the domain challenges that
the benchmark must respect: Arabic is inherently cursive with context-dependent
letter forms and ligatures; letters are disambiguated by dots (i'jam); optional
harakat diacritics change meaning; text is right-to-left; historical manuscripts
add degradation and layout noise. Evaluation practice in the field is CER/WER
first, but naive CER/WER over-penalizes trivial orthographic variants, so
Arabic-aware normalization and a separate diacritic score are needed.

This document is **plan/spec only** — no code is written this round. It is the
blueprint the implementation phases will follow.

### Decisions locked in with the project owner

| Question | Decision |
| --- | --- |
| First milestone | Plan/spec only for now |
| Recognition granularity | **Both** line-level and page-level tracks |
| Model comparison set | Gemini, OpenAI, Claude, open / Arabic-specific models, plus Arabic-first API providers where reachable |
| Diacritics / normalization | Report **both** raw and Arabic-normalized CER/WER, plus a separate diacritic-accuracy number |

## Goals

- A config-driven harness: add a model or a dataset by editing config, not code.
- One unified model interface so every model is called the same way.
- Two evaluation tracks from day one:
  - **Line-level** — isolates recognition from layout; matches classical HTR benchmarks.
  - **Page-level** — whole-page image to full transcript; matches how LLMs are used in practice; reading order is part of the task.
- Metrics: raw CER/WER, Arabic-normalized CER/WER, diacritic accuracy, both micro (corpus-weighted) and macro (per-sample averaged); plus latency and cost per request.
- Every run fully logged (raw model output kept verbatim) and resumable.
- Clean train/benchmark separation so a future fine-tuning round never contaminates the test set.

## Non-goals (Phase 1)

- No fine-tuning yet (that is a later phase once the harness is trusted).
- No training of a custom HTR model.
- No text-detection / layout-analysis model of our own — for the page track we rely on dataset-provided ground truth; for line crops we use dataset-provided polygons.
- No web UI / leaderboard site — results are tables + a notebook.

## Architecture

### Repository layout

```
arabic-ocr-benchmark/
  pyproject.toml            # uv-managed; deps: litellm, datasets, jiwer, pandas,
                            #   pyarabic/camel-tools, pillow, pydantic, tenacity, tqdm
  configs/
    models.yaml             # model registry: id, provider, litellm name, pricing, params
    datasets.yaml           # dataset registry: loader, split, track(s), license
    run.yaml                # which models x datasets x tracks for a given run; concurrency, caching
  src/arabic_ocr_bench/
    datasets/
      base.py               # Sample dataclass: id, image, ground_truth, meta, track
      muharaf_lines.py       # HF aamijar/muharaf-public (line-level)
      muharaf_pages.py       # raw Muharaf pages + JSON polygons (page-level)
      registry.py            # name -> loader
    models/
      base.py                # ModelClient protocol: transcribe(image, prompt) -> Prediction
      litellm_client.py      # Gemini / OpenAI / Claude / any LiteLLM-routed API
      hf_local_client.py     # open + Arabic-specific weights (Qwen2.5-VL, Qari-OCR, paligemma-arabic, ...)
      arabic_api_client.py   # Arabic-first providers not in LiteLLM (thin REST adapters)
      registry.py
    prompting.py             # prompt templates per track; optional metadata injection
    normalize.py             # Arabic normalization variants (see below)
    metrics.py               # CER/WER (raw + normalized), diacritic accuracy, micro/macro, aggregation
    runner.py                # async orchestration, caching, checkpointing, cost/latency capture
    report.py                # results JSONL -> pandas tables (per-model, per-dataset, weighted overall)
    cli.py                   # `bench run --config configs/run.yaml`, `bench report <run_dir>`
  results/
    <run_id>/
      predictions.jsonl      # one row per (sample, model): raw output, cleaned output, latency_ms, tokens, cost_usd, error
      scores.jsonl           # one row per (sample, model): all metric values
      summary.csv / summary.md
      run_config.snapshot.yaml
  notebooks/
    analysis.ipynb           # load results/, charts, error inspection
  tests/
```

### Components and responsibilities

- **Dataset adapters** — each yields a uniform `Sample` (stable `id`, PIL image, ground-truth string, `track`, free-form `meta`). The harness never sees dataset-specific formats.
- **Model clients** — one `transcribe()` method. Three implementations:
  1. `litellm_client` for hosted APIs (Gemini, OpenAI, Claude, and anything LiteLLM can route, incl. OpenAI-compatible endpoints from Arabic providers).
  2. `hf_local_client` for open / Arabic-specific weights run locally (Metal on this Mac, or a GPU box) — Qwen2.5-VL, Qari-OCR (Qwen2-VL based), `Nada2125/paligemma-3b-arabic-handwritten-ocr`, Qalam if weights are available.
  3. `arabic_api_client` for Arabic-first providers with bespoke REST APIs (Tarjama/Pronoia, Arabic.AI OCR, Fanar, ALLaM/HUMAIN) — added only where an API key and vision endpoint are actually reachable; each is a ~30-line adapter.
- **Runner** — builds the (sample x model) work list, executes with bounded async concurrency, retries transient failures (tenacity), writes an on-disk cache keyed by `hash(image_bytes + model_id + prompt_version)` so re-runs are cheap and partial runs resume. Captures wall-clock latency, token counts, and computes cost from `models.yaml` pricing.
- **Metrics** — pure functions over `(ground_truth, prediction)`; no I/O. Computes the full metric family (below) per sample; `report.py` aggregates.
- **Report** — produces per-model x per-dataset tables and a **size-weighted overall** (each dataset contributes proportional to its sample/char count — never a plain average of averages, per the brief).

## Datasets

### Track A — line-level (start here, lowest friction)

- **Muharaf lines**: `aamijar/muharaf-public` on HF. Confirmed structure: `train` 22,100 / `validation` 1,070 / `test` 1,330 line examples, columns `image` + `text`. License **CC-BY-NC-SA-4.0** (non-commercial — fine for research benchmarking; note it).
- Benchmark set = the dataset's own **`test` split** (1,330 lines). The predefined split already gives us the clean hold-out the brief asks for; `train`/`validation` are reserved for any future fine-tuning.
- Smoke-test subset: first 10-20 test lines, wired through `run.yaml`.

### Track B — page-level

- The HF dataset is line-only, so the page track needs **raw Muharaf**: `github.com/MehreenMehreen/muharaf` (page images + per-page JSON with line polygons and transcriptions) or the Zenodo record `11492215`.
- Page ground truth = line transcriptions concatenated in reading order (top-to-bottom, right-to-left) from the JSON.
- Restrict Track B to the pages whose lines fall in the HF `test` split, so the two tracks describe the same held-out material and cannot leak into future training.
- The author also mentions a *restricted* Muharaf subset — evaluate requesting access; treat as an optional extra benchmark slice, kept separate.

### Later datasets (add via `datasets.yaml`, same `Sample` interface)

- **KHATT** — modern handwritten Arabic paragraphs (line + page).
- **KITAB-Bench** — existing Arabic OCR / document-understanding benchmark for VLMs; useful as an external cross-check and for non-cursive contrast.
- **IFN/ENIT** — constrained (Tunisian town names); low priority, mainly for comparability with older literature.
- A **less-cursive / printed** set (e.g. from KITAB-Bench or a fonts dataset) as an easy-difficulty anchor point.

## Prompting

- Versioned prompt templates (`prompt_version` string logged with every prediction).
- Line track: minimal instruction — "Transcribe the Arabic text in this image. Output only the transcription, no commentary." Specify whether to preserve harakat (we ask the model to preserve what it sees).
- Page track: add reading-order and layout guidance — one line per text line, top-to-bottom, ignore page furniture (page numbers, marginalia, graphics).
- Output cleaning in `litellm_client` / clients: strip code fences, leading "Here is..." fluff, surrounding quotes — but the **raw** response is always stored untouched in `predictions.jsonl`.
- Optional experiment (later, toggle in `run.yaml`): inject dataset `meta` (document type, era, script style) into the prompt and measure whether it moves CER.

## Metrics

Computed per sample, then aggregated:

- **CER / WER — raw**: exact strings, `jiwer`. Primary headline numbers.
- **CER / WER — normalized**: after Arabic normalization (below). Secondary.
- **Diacritic accuracy**: restrict to harakat marks — precision/recall/F1 of diacritic characters between prediction and ground truth (needs raw, un-stripped text).
- **Exact-line-match rate** (line track): fraction of lines transcribed perfectly (raw and normalized).
- **Aggregation**: report both
  - **micro** — total edits / total reference length across the corpus ("how good overall"),
  - **macro** — mean of per-sample rates ("how consistent across samples").
- **Overall across datasets**: size-weighted by reference character count, not a mean of dataset means.
- **Operational**: latency (ms, wall-clock per request), input/output tokens, cost (USD, from `models.yaml` pricing; note Gemini's image-token accounting differs from OpenAI's), failure/refusal rate.

### Arabic normalization (`normalize.py`)

Provide these as composable steps and record which were applied:

1. Strip tatweel / kashida (U+0640).
2. Unify alef forms (أ إ آ ٱ -> ا).
3. Unify ya / alef-maqsura (ى -> ي) and final ta-marbuta / ha (ة -> ه) — configurable, since these can be meaningful.
4. Unify hamza carriers (ؤ ئ -> normalized) — configurable.
5. Remove harakat / tanwin / shadda / sukun for the "diacritics-insensitive" variant.
6. Normalize Arabic-Indic vs ASCII digits.
7. Collapse whitespace, Unicode NFC, trim.

Use `pyarabic` and/or `camel-tools` rather than hand-rolling; `jiwer` handles the edit-distance math. Two canonical normalized profiles: **light** (steps 1-2, 6-7) and **diacritics-insensitive** (light + 3-5). Raw = no steps.

## Execution flow

1. `bench run --config configs/run.yaml` -> resolves model x dataset x track matrix.
2. Runner loads samples, checks cache, dispatches uncached work with bounded concurrency (per-provider limits in config), retries transient errors, writes `predictions.jsonl` incrementally (crash-safe).
3. `bench report <run_dir>` -> reads predictions, runs `metrics.py`, writes `scores.jsonl`, `summary.csv`, `summary.md`.
4. `notebooks/analysis.ipynb` -> tables, charts, and a worst-N error browser (image + GT + each model's output side by side).
5. Results directory is the checkpoint artifact (optionally synced to Drive/HF as the brief suggests).

## Implementation phases (for the round after this plan is approved)

- **Phase 1 — skeleton + line track, one model.** Repo, configs, `Sample`, `muharaf_lines`, `litellm_client` (Gemini *or* GPT), `normalize.py`, `metrics.py`, `runner` with cache, `report`. Run on 10-20 test lines. Deliverable: a `summary.md` with raw + normalized CER/WER + latency + cost for one model. TDD: `metrics.py` and `normalize.py` get unit tests with hand-checked Arabic fixtures first.
- **Phase 2 — full line track, multi-model.** Add OpenAI + Claude via LiteLLM; add `hf_local_client` with one open model (Qwen2.5-VL or Qari-OCR) and one Arabic-specific model. Run full `test` split (1,330 lines). Weighted aggregation, macro/micro, diacritic accuracy.
- **Phase 3 — page track.** Ingest raw Muharaf pages + JSON, `muharaf_pages` adapter, reading-order GT reconstruction, page-track prompts. Re-run all models.
- **Phase 4 — breadth.** Add `arabic_api_client` adapters for whichever Arabic-first providers we can get vision API access to; add KHATT and a KITAB-Bench slice; metadata-in-prompt experiment.
- **Phase 5 — decision point.** With a trusted harness and a clean hold-out, evaluate fine-tuning (OpenAI fine-tuner vs Vertex AI) on Muharaf `train`, benchmarked on the untouched `test` split.

## Verification

- **Unit**: `metrics.py` against known CER/WER cases (incl. RTL strings, diacritics, empty prediction); `normalize.py` each step against fixtures; cost math against a hand computation.
- **Golden sample**: 3 hand-transcribed line images checked into `tests/fixtures/`; assert the harness reproduces expected GT join and that a mocked model response scores as expected.
- **Cache/resume**: run smoke set, kill mid-run, re-run, confirm no duplicate API calls and identical final `scores.jsonl`.
- **End-to-end smoke**: `bench run` + `bench report` on 10 Muharaf test lines with one real API model; eyeball `summary.md` and 3 worst errors in the notebook.
- **Sanity ceiling**: a trivial "return empty string" baseline model should yield CER ~1.0 and WER ~1.0 — confirms the pipeline isn't silently rewarding nothing.

## Open questions to resolve during implementation

- Access path for raw Muharaf pages (GitHub clone vs Zenodo) and the *restricted* subset request.
- Which Arabic-first providers actually expose a vision/OCR API we can key into (Fanar 2.0 multimodal, Tarjama/Pronoia, Arabic.AI, ALLaM/HUMAIN) — spike each.
- Where heavy local models run (this Mac via Metal for small ones; a rented GPU for Qwen2.5-VL-7B+).
- Budget ceiling per full run, so `run.yaml` can cap sample counts per paid model.
