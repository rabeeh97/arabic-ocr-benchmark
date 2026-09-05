# Paid Claude + GPT APIs on the N=10 set

**Run:** `20260903T130126Z` (2026-09-03). 12 paid models × `kitab_muharaf` + `kitab_khatt`,
rows `000000`–`000009`, prompt `line-v2`. `muharaf_lines` rows 0–9 are a **follow-up**
(its adapter streams over the network and stalled; the two KITAB-Bench subsets load from
local cache). **Measured cost of this run: $2.92.** Whole-session spend incl. calibration
probes: ~$3.3.

## Why

Two paid keys (`OPENAI_API_KEY_PAID`, `ANTHROPIC_API_KEY_PAID`) were added to benchmark
the frontier hosted models deliberately, without risking the free/trial keys. Goals: (1)
score Claude + GPT models on the same samples the free-tier models were scored on; (2) a
dedicated, verified **cost table** incl. a projection over all datasets; (3) fold results
into the leaderboard.

## Method

- **12 models, pinned to the `*_PAID` keys** via new `*-paid` rows in
  `configs/models.yaml`. `registry.build_model` now passes the `env_key` value to litellm
  explicitly, so litellm's provider-default lookup (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`)
  cannot shadow the paid key.

  | | models |
  |---|---|
  | Anthropic | `claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`, `claude-fable-5`, `claude-fable-5-1` |
  | OpenAI | `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` |

- **Samples:** `offset: 0`, `limit: 10`, no accumulate → a fixed deterministic slice of
  rows 0–9. Prompt `line-v2`. Config: `configs/run.paid-apis.yaml` (full, incl.
  `muharaf_lines`) and `configs/run.paid-apis.fast.yaml` (the two cached subsets, used for
  this run).
- **Caching:** `cache: true`. Each successful `(sample, model)` is written to
  `results/_cache/<sha1(sample_id|model|prompt_version|rCACHE_REV)>.json`; re-running the
  config re-spends nothing for rows already done. Errored/refused rows are never cached and
  retry. The `*-paid` names are new, so no cache collision with earlier runs.
- **Reasoning models:** `gpt-5` and `gpt-5.6-terra` run at `reasoning_effort=minimal`
  (transcription needs no deliberation). `gpt-5.6-sol` / `gpt-5.6-luna` **ignore that flag**
  (litellm drops it for those ids) and run at default effort with `max_tokens: 8000` — they
  still spend 2,700–3,700 hidden reasoning tokens per line. `claude-opus-5` / `-sonnet-5` /
  `-fable-5` / `-fable-5-1` have adaptive thinking on by default; those thinking tokens are
  billed as output and are included in every cost number below.
- **Cost mechanism:** `runner._cost_usd` = provider-reported `prompt_tokens` /
  `completion_tokens` × the `pricing` block. Image tokens are already inside `prompt_tokens`
  for both providers (no per-image constant needed). Pricing verified against the OpenAI and
  Anthropic pricing pages, 2026-09 (USD per 1M tokens):

  | model | input | output | | model | input | output |
  |---|---|---|---|---|---|---|
  | claude-haiku-4-5 | 1.00 | 5.00 | | gpt-4o-mini | 0.15 | 0.60 |
  | claude-sonnet-5 | 2.00 | 10.00 | | gpt-4o | 2.50 | 10.00 |
  | claude-opus-5 | 5.00 | 25.00 | | gpt-4.1 | 2.00 | 8.00 |
  | claude-fable-5 | 10.00 | 50.00 | | gpt-5 | 1.25 | 10.00 |
  | claude-fable-5-1 | 10.00 | 50.00 | | gpt-5.6-sol | 4.00 | 20.00 |
  | | | | | gpt-5.6-terra | 2.00 | 12.00 |
  | | | | | gpt-5.6-luna | 0.20 | 1.20 |

## Headline findings

- **Claude Fable 5 is the most accurate model in the set** on both handwriting subsets —
  CER·light **30.3 %** on `kitab_muharaf`, **23.1 %** on `kitab_khatt` — ahead of
  `claude-opus-5` (39.0 / 28.5) and `claude-fable-5-1` (39.2 / 30.6). It is also the
  priciest (~$0.015/line).
- **The GPT-5.6 reasoning trio is a bad trade for OCR.** `gpt-5.6-sol` is the single most
  expensive row by far — $1.48 for 20 lines (3,658 output tokens/line at $20/1M) at ~26–72 s
  per line — and scores *worse* than `claude-fable-5` (CER 67.8 / 47.8). `terra` and `luna`
  are similar. Their hidden reasoning tokens dominate cost; `reasoning_effort=minimal` is
  honoured only by `gpt-5` and (partly) `gpt-5.6-terra`.
- **`gpt-5` at `reasoning_effort=minimal` is cheap** ($0.019 for 20 lines) but middling on
  accuracy (CER 54.2 / 48.3).
- **Best value:** `claude-sonnet-5` (CER 46.5 / 32.8 at ~$0.001/line) and the free-tier
  `gemini-flash-lite` (CER 41.3 on `kitab_muharaf` at ~$0.0001/line) still look strong.
- `gpt-4o` / `gpt-4o-mini` / `gpt-4.1` each **refused 1** of the 10 `kitab_muharaf` lines
  (the same line the free-tier GPTs refuse). `gpt-4o`-class tiles the line image to
  ~14k–20k input tokens; `gpt-5`-class uses ~250–600.

## Table 1 — N=10 confirmation set (rows 0–9, prompt line-v2)

Paid models: exactly rows 0–9 (n=10). Free-tier / local rows: n shown, accumulated across
earlier runs — rows 0–9 are a subset. CER/WER = light-normalised micro %, lower is better.
Full detail and a per-line best/worst browser: `results/leaderboard.html`.

### kitab_muharaf

| model | n | err | CER·light | WER·light | exact% | diac F1 | s/line | cost $ |
|---|---|---|---|---|---|---|---|---|
| **claude-fable-paid** | 10 | 0 | **30.3** | 61.3 | 20.0 | 0.83 | 7.5 | 0.1320 |
| claude-opus-paid | 10 | 0 | 39.0 | 79.6 | 10.0 | 0.80 | 7.3 | 0.0840 |
| claude-fable-5-1-paid | 10 | 0 | 39.2 | 77.4 | 20.0 | 0.83 | 7.7 | 0.0670 |
| gemini-flash-lite (free) | 20 | 0 | 41.3 | 77.6 | 5.0 | 0.76 | 2.8 | 0.0025 |
| claude-sonnet-paid | 10 | 0 | 46.5 | 86.0 | 10.0 | 0.70 | 3.0 | 0.0100 |
| gpt-5-paid | 10 | 0 | 54.2 | 89.2 | 10.0 | 0.71 | 2.7 | 0.0078 |
| openrouter-qwen3-vl-235b (free) | 20 | 0 | 56.5 | 87.5 | 5.0 | 0.59 | 2.8 | 0.0019 |
| gpt-4.1-paid | 9 | 1 | 56.9 | 91.2 | 11.1 | 0.74 | 2.5 | 0.0111 |
| gpt-4o-paid | 9 | 1 | 57.8 | 98.9 | 0.0 | 0.56 | 2.1 | 0.0137 |
| openrouter-qwen2.5-vl-72b (free) | 20 | 0 | 62.1 | 93.8 | 5.0 | 0.56 | 1.9 | 0.0024 |
| claude-haiku-paid | 10 | 0 | 63.5 | 98.9 | 0.0 | 0.70 | 1.8 | 0.0032 |
| gpt-5.6-sol-paid | 10 | 0 | 67.8 | 94.6 | 10.0 | 0.72 | 72.2 | **0.8297** |
| gpt-5.6-terra-paid | 10 | 0 | 68.7 | 97.9 | 10.0 | 0.80 | 35.2 | 0.2165 |
| qari-ocr-0.3 (local) | 10 | 0 | 69.3 | 103.2 | 0.0 | 0.42 | 105.0 | 0 |
| gpt-4o-mini-paid | 9 | 1 | 76.9 | 96.7 | 11.1 | 0.15 | 2.9 | 0.0199 |
| gpt-5.6-luna-paid | 10 | 0 | 79.9 | 104.3 | 10.0 | 0.50 | 30.5 | 0.0286 |

(free-tier / local rows below CER 80 % omitted here for brevity — see the leaderboard.)

### kitab_khatt

| model | n | err | CER·light | WER·light | exact% | diac F1 | s/line | cost $ |
|---|---|---|---|---|---|---|---|---|
| **claude-fable-paid** | 10 | 0 | **23.1** | 62.3 | 10.0 | 0.80 | 9.0 | 0.1968 |
| claude-opus-paid | 10 | 0 | 28.5 | 61.5 | 10.0 | 0.44 | 8.3 | 0.1185 |
| claude-fable-5-1-paid | 10 | 0 | 30.6 | 68.8 | 10.0 | 0.65 | 8.5 | 0.0943 |
| openrouter-qwen3-vl-235b (free) | 22 | 0 | 31.1 | 72.3 | 0.0 | 0.39 | 7.5 | 0.0027 |
| claude-sonnet-paid | 10 | 0 | 32.8 | 82.8 | 0.0 | 0.90 | 3.3 | 0.0128 |
| openrouter-qwen2.5-vl-72b (free) | 22 | 0 | 35.3 | 75.3 | 0.0 | 0.59 | 11.2 | 0.0057 |
| gpt-4.1-paid | 10 | 0 | 37.5 | 75.4 | 0.0 | 0.72 | 2.7 | 0.0157 |
| gemini-flash-lite (free) | 22 | 0 | 42.5 | 82.0 | 0.0 | 0.77 | 11.9 | 0.0028 |
| gpt-4o-paid | 10 | 0 | 44.9 | 84.4 | 0.0 | 0.40 | 1.8 | 0.0194 |
| gpt-5.6-sol-paid | 10 | 0 | 47.8 | 90.2 | 0.0 | 0.88 | 60.0 | **0.6537** |
| gpt-5-paid | 10 | 0 | 48.3 | 93.4 | 0.0 | 0.20 | 3.2 | 0.0109 |
| gpt-5.6-luna-paid | 10 | 0 | 49.2 | 86.9 | 0.0 | 0.70 | 39.5 | 0.0383 |
| gpt-5.6-terra-paid | 10 | 0 | 51.2 | 86.1 | 0.0 | 0.90 | 47.9 | 0.2960 |
| claude-haiku-paid | 10 | 0 | 55.9 | 95.9 | 0.0 | 0.80 | 3.1 | 0.0061 |
| gpt-4o-mini-paid | 10 | 0 | 77.7 | 99.2 | 0.0 | 0.10 | 1.6 | 0.0310 |

## Table 2 — Cost table (paid models)

Generated by `scripts/cost_report.py`. `$/line` = mean per-line token means from this run ×
pricing. Projection = `$/line × dataset rows` (linear estimate; real cost varies with image
size and how much a model rambles). Dataset rows: `kitab_muharaf` **200**, `kitab_khatt`
**200**, `muharaf_lines` **1,334**, `kitab_historicalbooks` / `kitab_hindawi` ~200 each
(exact counts pending a download — update `results/_dataset_sizes.json`).

| model | in $/1M | out $/1M | mean in tok | mean out tok | $/line | run cost (n=20) | proj `muharaf_lines` (1,334) | proj one `kitab_*` (200) | **proj all 5 datasets** |
|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-paid | 1.00 | 5.00 | 290 | 35 | $0.00047 | $0.0093 | $0.62 | $0.09 | **$0.99** |
| claude-sonnet-paid | 2.00 | 10.00 | 328 | 49 | $0.00114 | $0.0229 | $1.52 | $0.23 | **$2.44** |
| claude-opus-paid | 5.00 | 25.00 | 328 | 339 | $0.01012 | $0.2025 | $13.51 | $2.02 | **$21.60** |
| claude-fable-paid | 10.00 | 50.00 | 328 | 263 | $0.01644 | $0.3288 | $21.93 | $3.29 | **$35.08** |
| claude-fable-5-1-paid | 10.00 | 50.00 | 330 | 95 | $0.00806 | $0.1613 | $10.76 | $1.61 | **$17.21** |
| gpt-4o-mini-paid | 0.15 | 0.60 | 17,237 | 36 | $0.00261 | $0.0508 | $3.48 | $0.52 | **$5.56** |
| gpt-4o-paid | 2.50 | 10.00 | 601 | 18 | $0.00169 | $0.0330 | $2.25 | $0.34 | **$3.60** |
| gpt-4.1-paid | 2.00 | 8.00 | 601 | 20 | $0.00137 | $0.0269 | $1.82 | $0.27 | **$2.91** |
| gpt-5-paid | 1.25 | 10.00 | 499 | 31 | $0.00093 | $0.0187 | $1.25 | $0.19 | **$1.99** |
| gpt-5.6-luna-paid | 0.20 | 1.20 | 253 | 2,745 | $0.00334 | $0.0669 | $4.46 | $0.67 | **$7.14** |
| gpt-5.6-terra-paid | 2.00 | 12.00 | 253 | 2,093 | $0.02562 | $0.5124 | $34.18 | $5.12 | **$54.68** |
| gpt-5.6-sol-paid | 4.00 | 20.00 | 253 | 3,658 | $0.07417 | $1.4835 | $98.95 | $14.83 | **$158.29** |

"proj all 5 datasets" = `muharaf_lines` (1,334) + 4 × `kitab_*` (~200 each) = ~2,134 lines.
Regenerate with real sizes: `uv run python scripts/cost_report.py results/20260903T130126Z`.

## Table 3 — Full leaderboard

`results/leaderboard.html` (interactive — includes the **Review — best & worst by model**
browser) and `results/leaderboard.json` (machine-readable, 64 model×dataset rows across 25
runs). Best CER·light per dataset:

| dataset | best model | CER·light |
|---|---|---|
| kitab_khatt | claude-fable-5 (paid) | 23.1 % |
| kitab_muharaf | claude-fable-5 (paid) | 30.3 % |
| muharaf_lines | gemini-flash-lite (free) | 48.0 % (paid models pending) |
| kitab_historicalbooks | openrouter-qwen3-vl-235b (free) | 31.2 % |
| kitab_hindawi | openrouter-qwen3-vl-235b (free) | 55.4 % |
| kitab_historyar | gemini-flash-lite (free) | 32.9 % |
| kitab_onlinekhatt | gemini-flash-lite (free) | 9.2 % |

## Update — 2026-09-03: four more KITAB-Bench subsets + incremental local sweep

**New datasets** (`ahmedheakl/arocrbench_*`, reuse the `KitabBench` adapter — entries in
`configs/datasets.yaml`, sizes in `results/_dataset_sizes.json`):

| dataset | rows | kind |
|---|---|---|
| `kitab_historicalbooks` | 10 | degraded historical **print** |
| `kitab_hindawi` | 200 | clean digitised **print** |
| `kitab_historyar` | 200 | historical **manuscripts** |
| `kitab_onlinekhatt` | 200 | **online** (stylus) handwriting |

**Free-tier run** (`configs/run.free-newdatasets.yaml`, `line-v2`, rows 0–19; historicalbooks
= all 10). ~10 free models; the Mistral free tier and the OpenRouter `:free` Gemmas hit
rate-limit / daily-cap errors (n < 20, some 0). NVIDIA NIM rows dropped — `nvidia-gemma-4-31b`
hangs on the larger page images. Findings:

- **`kitab_onlinekhatt`**: `gemini-flash-lite` reads it at **9.2 %** CER·light — cleaner
  strokes than scanned handwriting make it the easiest set in the whole benchmark.
- **`kitab_historicalbooks` / `kitab_hindawi`**: even "clean" digitised print is hard for the
  free models (best 31 % / 55 %); `openrouter-qwen3-vl-235b` leads both.
- **`kitab_historyar`**: ~33 % for the best free models (`gemini-flash-lite`,
  `qwen3-vl-235b`).

**Incremental local sweep (`04_colab_sweep.ipynb`, regenerated):** now defaults to
`ACCUMULATE = True`, `BATCH = 10`, `DATASETS` = all 7. Each Run-all scores 10 new lines per
(model, dataset) that still has un-scored lines, retries past errors, recomputes nothing
cached. Two harness fixes back it:

- `run_sweep` skips a (model, dataset) whose split is fully scored (`_accumulate_has_work`) —
  no empty run, no empty Colab commit.
- `runner.run` abandons a model after `_RL_ABORT` (8) consecutive rate-limit errors —
  treats it as out of daily quota and skips its remaining datasets for that run instead of
  retrying every line. Rows stay un-scored/uncached for a later run.

## Update — 2026-09-03: paid models on the four new subsets + full Gemini crawl

**Paid models on the new subsets** (`configs/run.paid-newdatasets.yaml`, rows 0–9,
`line-v2`). 9 models — Claude haiku/sonnet/opus/fable/fable-5.1 + gpt-4o-mini/4o/4.1/5;
the GPT-5.6 reasoning trio is excluded (2–4k hidden reasoning tokens/line). **$3.39,
0 errors.** Best CER·light per new subset:

| subset | best | CER·light | notes |
|---|---|---|---|
| `kitab_onlinekhatt` | `claude-fable-5` | **7.2 %** | online strokes — easiest set overall |
| `kitab_historyar` | `claude-fable-5` | **20.0 %** | Claude tier clearly ahead of every free model |
| `kitab_hindawi` | `claude-fable-5-1` | **1.1 %** | clean print, near-solved (fable-5 itself is erratic here at 72 %) |
| `kitab_historicalbooks` | see leaderboard | — | degraded print, still hard |

**Full Gemini crawl** (`scripts/run_gemini_full.py` + `configs/run.gemini-full.yaml`):
`gemini-flash-lite` / `gemini-flash` / `gemini-pro` over *every* sample of the six
KITAB-Bench subsets (`muharaf_lines` excluded — streaming adapter hangs). Throttled to
≤14 req/min via the new `_RateLimiter` (`rpm:` run-config key). Loop wrapper re-checks
coverage each pass, sleeps ~1 h on a daily quota, and exits when every (model, dataset)
is 100 % done. Fully resumable off the per-line cache. Progress:
`results/gemini_full_progress.json`.

## Update — 2026-09-05: GLM-5.3-Flash (Z.AI, paid) — vision confirmed, N=10

A third paid key (`ZAI_API_KEY_PAID`, Z.AI/Zhipu) was added. Added as `glm-5.3-flash-paid`
in `configs/models.yaml` (`openai/glm-5.3-flash` via LiteLLM's OpenAI-compatible path,
custom `api_base: https://api.z.ai/api/paas/v4`). Plain `glm-5.3` (no `-Flash`) is
text-only per Z.AI's own pricing docs and was not tried.

- **Vision confirmed empirically**, settling an open question from the model's own docs
  page: a 2-line smoke test correctly read a handwritten line image (`MK2A7 21` →
  predicted `MK2A-7 21`, a near-exact match) once `max_tokens` was raised from the
  default 256 to **4000** — at 256 the model spent its entire budget on hidden
  chain-of-thought and returned empty `cleaned_text` with no error, the same failure
  pattern already seen on GPT-5/GPT-5.6 and `nemotron-omni` in this doc.
- **N=10 run** (`configs/run.glm-paid.yaml`): `kitab_muharaf` + `kitab_khatt`, rows
  0–9, `line-v2`. `muharaf_lines` excluded (still unreliable to stream on this network
  — HF parquet fetch timeouts, independent of the adapter fix already on `main`).
  **0 errors, 20/20 lines scored.**

  | dataset | n | cer_micro_light | wer_micro_raw | exact | cost (n=10) |
  |---|---|---|---|---|---|
  | kitab_muharaf | 10 | 68.0 % | 95.7 % | 10 % | $0.0078 |
  | kitab_khatt | 10 | 78.2 % | 95.9 % | 0 % | $0.0139 |

  Accuracy is well behind the Claude/GPT paid tier (contrast `claude-fable-5` at 23–30 %
  CER·light on these same two sets) — GLM-5.3-Flash reads *something* off the image but
  the transcriptions are frequently unrelated to the ground truth.

- **Tried lowering the hidden reasoning spend, no win.** Z.AI's docs are explicit:
  *"GLM-5.3 and GLM-5.3-FLASH use forced thinking and cannot be disabled"* — the
  `thinking: {type: "disabled"}` field is silently coerced back to low effort. The
  documented workaround is `reasoning_effort: low/high/max`, so it was tried on the same
  2-line smoke sample with `cache: false` to force fresh calls. Result: no reliable
  reduction — the harder of the two lines burned its **entire 4000-token budget** and
  returned empty output on repeated attempts, with `reasoning_effort: low` set exactly
  as often as with it left at the default. Since the setting cost nothing but also
  bought nothing, it was **dropped from the config** rather than kept for no measured
  benefit. Net: mean **285 input / 2,081 output tokens per line** — most of a line's
  cost is hidden reasoning, not the image or the answer.
- **Cost**: list pricing $0.15 / $0.50 per 1M in/out (an on-and-off ~50% Z.AI promo may
  currently be running, unverified at run time — treat as an upper bound). Actual N=20
  spend **$0.0217**; **$0.00108/line** blended.

  | dataset | rows | projected cost |
  |---|---|---|
  | `muharaf_lines` | 1,334 | $1.44 |
  | `kitab_khatt` | 200 | $0.22 |
  | `kitab_muharaf` | 200 | $0.22 |
  | `kitab_historicalbooks` | 10 | $0.01 |
  | `kitab_hindawi` | 200 | $0.22 |
  | `kitab_historyar` | 200 | $0.22 |
  | `kitab_onlinekhatt` | 200 | $0.22 |
  | **all 7 datasets** | 2,344 | **$2.54** |

  For context this sits below `gpt-4o-mini-paid` ($5.56) and well below the reasoning-
  heavy GPT-5.6 trio, but the accuracy doesn't justify a full run at this price point —
  cheaper than most paid models, but not competitive on quality.
- Leaderboard regenerated (`results/leaderboard.html`, 143 rows / 36 runs) to include
  these rows.

## Follow-ups

- Fix the `muharaf_lines` adapter (it streams over the network and hangs) so paid, local,
  and Gemini runs can cover it.
- Run the local models on all subsets (Colab GPU, notebook 04 — now covers them all).
- Grow the free-tier coverage on the new subsets past n=20 (`accumulate` / re-run the config).

## Reproduce

```bash
uv run python scripts/list_paid_models.py                              # $0 — confirm ids
uv run bench run --config configs/run.paid-apis.smoke.yaml --report    # ~$0.01 pre-flight
HF_HUB_OFFLINE=1 uv run bench run --config configs/run.paid-apis.fast.yaml --report
uv run python scripts/make_leaderboard.py results -o results/leaderboard.html
uv run python scripts/cost_report.py results/20260903T130126Z --md /tmp/cost.md
```
