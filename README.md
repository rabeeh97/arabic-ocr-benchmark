# arabic-ocr-bench

A reproducible benchmark for **Arabic cursive script recognition** (handwritten
text recognition / HTR). It measures how well modern LLMs / vision-language
models — and classical OCR engines — transcribe cursive handwritten Arabic, on
the *same* data with the *same* metrics.

**Hypothesis under test:** general multimodal LLMs are overtaking traditional
HTR pipelines for cursive Arabic, and cheap fine-tuning could close the rest of
the gap.

## Status

Phase 1 — thin vertical slice. Line-level track on the Muharaf dataset, one
model, raw + Arabic-normalized CER/WER, latency and cost.

## Documentation

| Doc | What |
| --- | --- |
| [`docs/plan.md`](docs/plan.md) | Full phased plan and architecture |
| [`docs/benchmark-matrix.md`](docs/benchmark-matrix.md) | Every model × dataset track we target (core + stretch) |
| [`docs/models.md`](docs/models.md) | How to access and call each model |
| [`docs/datasets.md`](docs/datasets.md) | How to obtain and prepare each dataset |

Background research: `OCR Research.md`, `Rabeeh _ Eyad (LLM Benchmarking).md`.

## Quickstart

```bash
uv sync --extra dev                 # core + test deps
uv run pytest                       # unit tests (metrics, normalization)

# smoke run: 15 Muharaf test lines against the zero-cost mock model
uv run bench run --config configs/run.yaml
uv run bench report results/<run_id>
```

To run real models:

```bash
uv sync --extra dev --extra local   # adds torch / transformers for local models
cp .env.example .env                 # add API keys for hosted models
```

## Metrics

- **CER / WER** — raw (exact) and Arabic-normalized (light + diacritics-insensitive profiles)
- **Diacritic accuracy** — precision / recall / F1 over harakat
- **Aggregation** — micro (corpus-weighted) and macro (per-sample), size-weighted across datasets
- **Operational** — latency, token counts, USD cost, refusal rate

## License

MIT (harness code). Datasets retain their own licenses — see `docs/datasets.md`
(Muharaf is CC-BY-NC-SA-4.0, research use).
