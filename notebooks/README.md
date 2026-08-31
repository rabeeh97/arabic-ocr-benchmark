# Running the benchmark on Google Colab

`01`–`03` are generated from `scripts/build_notebooks.py` — edit that and re-run
it, don't hand-edit the `.ipynb`.

| Notebook | Runtime | What it does |
| --- | --- | --- |
| `00_reference_gemini_pipeline.ipynb` | CPU | **Original prototype** (kept as-is, not generated). Gemini-2.5-flash only, raw CER/WER, CSV output. Superseded by `02` + `03`; see "About `00`" below. |
| `01_colab_local_models.ipynb` | **T4 GPU** | Interactive single/few local models on Muharaf — for exploring |
| **`04_colab_sweep.ipynb`** | **T4 GPU** (CPU fallback) | **Run-all-and-leave-it.** Several local models, partial then full pass. A phase is marked done in `results/sweep_manifest.json` **only after its results are pushed to the repo** — if the push fails (e.g. read-only `GH_TOKEN`) the entry is rolled back and it re-runs next session. Resumes on restart. With no GPU it still runs — drops the 7B 4-bit models, tiny partial-only fp32 CPU pass (proves the pipeline, not a real benchmark). |
| `02_colab_hosted_apis.ipynb` | CPU | Gemini / OpenAI / Claude on Muharaf via their APIs |
| `03_analysis.ipynb` | CPU | Worst-error browser (image + GT + prediction) |

## About `00_reference_gemini_pipeline.ipynb`

The team's first prototype (the Colab that was shared into the repo). It targets
the **same dataset and model** as the harness — `aamijar/muharaf-public` `test`
split, line-level, `gemini-2.5-flash`, pricing `0.30 / 2.50` per 1M tokens — and
tracks per-image + global CER/WER, latency, tokens and cost with resume support.
The `02` notebook does everything it does and more: LiteLLM (multi-model), the
`raw` / `light` / `diac_insensitive` normalization profiles, a diacritic F1
score, macro as well as micro aggregation, and structured `predictions.jsonl` /
`scores.jsonl` output instead of ad-hoc CSVs.

Its saved outputs are from an **interrupted run** — only ~9 images completed
before a storm of Gemini `503 "overloaded"` errors and a manual stop — so it
contains **no global CER/WER numbers to reconcile against**. When `02` finishes a
`gemini-2.5-flash` run, its `cer_micro_raw` is the value that should line up with
this notebook's `Global_CER` once someone completes a full run of it.

Each notebook: clone repo → install → run → `summary.md` → save results to Google Drive.

---

## One-time setup

### 1. GitHub token (needed — the repo is private)

Colab has to authenticate to clone `rabeeh97/arabic-ocr-benchmark`.

1. https://github.com/settings/personal-access-tokens/new → **Fine-grained token**
2. **Resource owner:** `rabeeh97` · **Repository access:** Only select repositories → `arabic-ocr-benchmark`
3. **Permissions → Repository → Contents: Read-only** (nothing else)
4. Generate, copy the token.
5. In Colab, open the **🔑 Secrets** panel (left sidebar) → **Add new secret**
   - Name: `GH_TOKEN` · Value: the token · **Notebook access: ON**

### 2. API keys (only for `02_colab_hosted_apis.ipynb`)

Add as Colab secrets (Notebook access ON), any subset:

| Secret name | Where to get it | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | Free tier, rate-limited — enough for a first run |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys | Paid |
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys | Paid |

The notebook runs whichever keys are present and skips the rest.

### 3. HF token — `HF_TOKEN` (recommended)

Create a **read** token at https://huggingface.co/settings/tokens and add it as a
Colab secret named `HF_TOKEN` (Notebook access ON). It:

- lifts anonymous HF rate limits and **speeds up the Muharaf download** (anonymous
  pulls throttle hard after ~25 MB), and
- is **required** for gated models — e.g. `paligemma-arabic-hw`, whose base
  `google/paligemma-3b-pt-448` needs the license accepted on its HF page first.

Not strictly required for Qari-OCR or Qwen2.5-VL, but recommended for every run.

---

## Opening a notebook in Colab

**Option A — from GitHub (private repos):**
1. https://colab.research.google.com → **File → Open notebook → GitHub**
2. Tick **Include private repos**, authorize if asked
3. Repo: `rabeeh97/arabic-ocr-benchmark`, Branch: `main`, pick the notebook

**Option B — upload:** download the `.ipynb` from GitHub, then Colab **File → Upload notebook**.

---

## Running `01_colab_local_models.ipynb`

1. **Runtime → Change runtime type → T4 GPU**.
2. Add the `GH_TOKEN` secret (above).
3. In the "Pick a model" cell set `MODEL` and `LIMIT`:
   - First run: `MODEL = 'qari-ocr-0.2'`, `LIMIT = 50` (~3-5 min after weights download).
   - Model names: `qari-ocr-0.2`, `qari-ocr-0.3`, `qwen2.5-vl-3b`, `qwen2.5-vl-7b`,
     `paligemma-arabic-hw` (see `configs/models.yaml`).
4. **Runtime → Run all**.
5. The summary table prints inline; full results land in `results/<run_id>/` and are
   copied to `MyDrive/arabic-ocr-bench/results/<run_id>/`.
6. Change `MODEL`, run the last few cells again — runs accumulate.
7. Set `LIMIT = None` for the full 1,330-line test split once a small run looks right.

## Running `02_colab_hosted_apis.ipynb`

1. CPU runtime is fine.
2. Add at least one API-key secret.
3. Run all. `limit` starts at 25 to keep paid-API spend near zero — raise it when happy.
4. `cost_usd_total` in the summary is estimated from `configs/models.yaml` pricing.

## Running `03_analysis.ipynb`

1. Run after you have at least one run saved to Drive.
2. It reads every `results/*/scores.jsonl` under `RESULTS`, builds a leaderboard,
   plots the CER distribution, and shows the 8 worst lines with the image, the
   ground truth, and each model's output.

---

## What "results" look like

```
results/<run_id>/
  predictions.jsonl   # one row per (line, model): raw + cleaned output, latency, tokens, cost, error
  scores.jsonl        # one row per (line, model): CER/WER (raw, light, diac_insensitive), diacritic P/R/F1
  summary.csv         # per model x dataset: micro/macro CER/WER, exact-match rate, refusal rate, cost
  summary.md          # the headline table
  run_config.snapshot.yaml
```

Metric meaning: **CER/WER** lower is better; **raw** = exact strings, **light** =
alef/digit/tatweel/whitespace normalized, **diac_insensitive** = also harakat-stripped;
**micro** = corpus-weighted, **macro** = per-line mean; **diac_f1** = harakat F1 on raw text.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No GPU` assertion | Runtime → Change runtime type → T4 GPU, then Run all |
| Clone fails / asks for password | `GH_TOKEN` secret missing, wrong name, Notebook-access off, or token lacks Contents:Read |
| CUDA out of memory | Lower `LIMIT`, or use `qari-ocr-0.2` / `qwen2.5-vl-3b` instead of 7B |
| Muharaf download slow | First time only; it's cached for the rest of the session |
| `paligemma-arabic-hw` 401/403 | Accept the base-model license on HF, add `HF_TOKEN` secret |
| Gemini `429` | Free-tier rate limit; the runner backs off and retries. Lower `concurrency` to 1-2 |
