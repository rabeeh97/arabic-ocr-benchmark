# Models — How to Call and Benchmark Each One

Companion to `benchmark-matrix.md` and the Phase 1 plan. For every model we plan
to test (core + stretch), this file records: how it is accessed, the exact
identifier, credentials needed, the call path in our harness, and any quirks that
affect a fair OCR comparison.

**Common contract.** In the harness every model is a `ModelClient` with one
method: `transcribe(image: PIL.Image, prompt: str) -> Prediction`. `Prediction`
carries `raw_text`, `cleaned_text`, `latency_ms`, `input_tokens`,
`output_tokens`, `cost_usd`, `error`. Three client implementations cover
everything below: `litellm_client`, `hf_local_client`, `arabic_api_client`.
The prompt comes from `prompting.py` and is track-specific (line vs page).

**Golden rule for all models:** store the raw response verbatim. Do cleanup
(strip ```` ``` ```` fences, "Here is the transcription:", surrounding quotes,
Markdown) only into `cleaned_text`, which is what scoring uses.

---

## Tier 1 — Hosted frontier VLMs (via LiteLLM)

All Tier 1 models go through `litellm_client`. One code path; switch models by
changing the identifier string. LiteLLM normalizes the request/response and
reports token usage; cost is computed by us from `configs/models.yaml` pricing
(do not trust LiteLLM's built-in cost for image requests — see Gemini note).

### Gemini 2.5 Flash / Pro

| Field | Value |
| --- | --- |
| LiteLLM id | `gemini/gemini-3.6-flash`, `gemini/gemini-3.5-flash-lite`, `gemini/gemini-3.1-pro-preview` (2.5-series is closed to new API keys) |
| Credential | `GEMINI_API_KEY` (Google AI Studio) **or** Vertex AI (`VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, ADC) |
| Image input | inline base64 in the message `content` list |
| Track fit | line + page (strong at full-page reading order) |

Quirks:
- Gemini bills images via a multimodal encoder, **not** pixel→token like OpenAI.
  Record `usageMetadata` from the response and price from our own table; a
  fixed per-image token count is a good enough approximation for budgeting.
- The free AI Studio tier rate-limits and quota-exhausts quickly (noted in the
  brief). For any full run use a billed key or Vertex.
- Can refuse or add safety commentary on manuscript images of people/letters —
  count as `error="refusal"`, do not let it pollute CER.

### OpenAI GPT-4o / GPT-4.1 / GPT-5 (vision)

| Field | Value |
| --- | --- |
| LiteLLM id | `openai/gpt-4o`, `openai/gpt-4.1`, `openai/gpt-5` |
| Credential | `OPENAI_API_KEY` |
| Image input | `image_url` with `data:` URI; set `detail: high` for dense script |
| Track fit | line + page |

Quirks:
- `detail: high` materially changes cost and accuracy on small line crops —
  fix it in config and log which value was used (`prompt_version` covers this).
- GPT-5 vision params differ (no `temperature` control on some tiers); keep a
  per-model `params` block in `models.yaml`.

### Claude Sonnet / Opus (latest vision)

| Field | Value |
| --- | --- |
| LiteLLM id | `anthropic/claude-haiku-4-5`, `anthropic/claude-sonnet-5`, `anthropic/claude-opus-5` |
| Credential | `ANTHROPIC_API_KEY` |
| Image input | base64 `image` block; max 8000×8000 px, ~5 MB/image |
| Track fit | line + page |

Quirks:
- Downscale very wide Muharaf line crops (up to ~1620 px) is fine; very tall
  page scans may need resizing to stay under the pixel cap — do it in the
  client and log the resize.
- Tends to preserve diacritics well; good candidate for the diacritic-accuracy
  metric comparison.

### NVIDIA NIM API catalog (Llama-3.2-Vision / Gemma-4)

| Field | Value |
| --- | --- |
| LiteLLM id | `openai/meta/llama-3.2-90b-vision-instruct`, `openai/meta/llama-3.2-11b-vision-instruct`, `openai/google/gemma-4-31b-it` |
| Credential | `NVIDIA_API_KEY` (free developer tier at build.nvidia.com — no card, rate-limited) |
| api_base | `https://integrate.api.nvidia.com/v1` (OpenAI-compatible; registry passes the key explicitly) |
| Image input | `image_url` data URI (JPG/PNG); same path as every other LiteLLM model |
| Track fit | page / document VQA — weak on bare handwritten line crops |

Quirks:
- **NVIDIA retired the hosted Nemotron-VL line.** As of 2026-09-02, `GET
  /v1/models` no longer lists `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` or
  `nvidia/nemotron-nano-12b-v2-vl` — both return `410 Gone` ("end of life
  2026-08-26"), and `google/gemma-3-27b-it` 410'd on 2026-05-12. The live chat
  VLMs are Meta Llama-3.2-Vision (11B/90B), Gemma 3 4B/12B, Gemma 4 31B,
  Phi-3-vision and `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.
- These are general VLMs, not OCR models. On the Sep 2026 pass `llama-3.2-90b`
  scored >100% CER on Muharaf lines (it describes the crop instead of
  transcribing) at ~135 s/line — keep expectations low for the line track.
- Free tier rate-limits and burns a small credit pool; keep `limit` modest and
  `concurrency` low. Expect `RateLimitError` rows on big passes, not silent data.
- `nvidia/nemotron-parse` (a dedicated document-parsing model) is *not* added: it
  uses a non-chat NIM response shape that `litellm_client` doesn't speak.

**Testing a Tier 1 model end to end:**
```
# configs/run.yaml: models: [gemini/gemini-2.5-flash], datasets: [muharaf_lines], limit: 15
bench run   --config configs/run.yaml
bench report results/<run_id>
```

---

## Tier 2 — Open-weight VLMs (local, via `hf_local_client`)

Run with `transformers` + `accelerate`, or serve with **vLLM** (OpenAI-compatible
endpoint) and then point `litellm_client` at `openai/<served-name>` with
`api_base=http://localhost:8000/v1`. vLLM is preferred for full runs (batching,
throughput); raw `transformers` is fine for a smoke test.

Hardware: this Mac (Metal) can run the 2B–7B models slowly; 32B needs 4-bit
quant and ≥24 GB; 72B needs a rented GPU. All large sizes are `○` in the matrix.

### Qwen2.5-VL

| Size | HF id |
| --- | --- |
| 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 32B | `Qwen/Qwen2.5-VL-32B-Instruct` |
| 72B | `Qwen/Qwen2.5-VL-72B-Instruct` |

- Needs `transformers>=4.49` (or the `qwen-vl-utils` helper) for the vision
  preprocessing; use `min_pixels`/`max_pixels` to cap tokens on large scans.
- vLLM serves it directly: `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --limit-mm-per-prompt image=1`.
- Strong general OCR; the natural open-model baseline.

### Llama-3.2-11B-Vision-Instruct — `meta-llama/Llama-3.2-11B-Vision-Instruct`
Gated on HF (accept license). Weaker at dense non-Latin script than Qwen-VL;
kept as a stretch cross-check.

### InternVL3-8B/14B — `OpenGVLab/InternVL3-8B` · Pixtral-12B — `mistralai/Pixtral-12B-2409`
Stretch. Same `hf_local_client` / vLLM path. Include only if Qwen-VL results
suggest open models are competitive.

**Testing a Tier 2 model:**
```
# option A: transformers, smoke test
python -m arabic_ocr_bench.models.hf_local_client --model Qwen/Qwen2.5-VL-7B-Instruct --self-test

# option B: serve then benchmark
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 &
# models.yaml entry: {id: qwen2.5-vl-7b, litellm: openai/Qwen2.5-VL-7B-Instruct, api_base: http://localhost:8000/v1}
bench run --config configs/run.yaml
```

---

## Tier 3 — Arabic-specific OCR / HTR models (local)

### Qari-OCR (NAMAA-Space)

| Version | HF id | Base | Best at |
| --- | --- | --- | --- |
| v0.2.2.1 | `NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct` | Qwen2-VL-2B-Instruct | raw text accuracy, diacritics (reported CER 0.061 / WER 0.160) |
| v0.3 | `NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` | Qwen2-VL-2B-Instruct | structured / multi-block layout, Markdown output |

- Loads as a `Qwen2VLForConditionalGeneration` model — same code as Qwen2.5-VL
  path with the Qwen2-VL processor. 2B params → runs on the Mac.
- v0.3 emits Markdown/structure tokens; the client must strip them into
  `cleaned_text` before scoring, or scoring will over-count errors.
- Trained largely on synthetic full-page Arabic; expect it to be strong on
  clean lines, less so on degraded historical manuscript pages — that contrast
  is exactly what the benchmark should surface.

### PaliGemma-3B Arabic Handwritten OCR

| Field | Value |
| --- | --- |
| HF id | `Nada2125/paligemma-3b-arabic-handwritten-ocr-1` (also `-arabic-handwritten-ocr`) |
| Form | **LoRA adapter (~23 MB)** — needs base `google/paligemma-3b-pt-448` (gated) |
| Load | `PaliGemmaForConditionalGeneration.from_pretrained(base)` then `PeftModel.from_pretrained(model, adapter)` |
| Track fit | **line only** — trained on single handwritten lines; do not run page track |
| Prompt | PaliGemma expects a task prefix, e.g. `"ocr\n"` or `"transcribe Arabic\n"` — check the adapter card / config for the exact training prefix |

### TR_OCR_LARGE — `David-Magdy/TR_OCR_LARGE` (implemented: `arch: trocr`)
- TrOCR-Large (0.6B, ViT encoder + RoBERTa-style decoder) finetuned on **KHATT +
  IAM** handwriting. A dedicated HTR model, not a VLM: no prompt, one line in →
  one string out. `hf_local_client` loads it via `TrOCRProcessor` +
  `VisionEncoderDecoderModel`; runs fine on CPU/MPS. Expect strong numbers on
  KHATT-like lines, weak on out-of-distribution scripts (Muharaf).

### MiniCPM-V 2.6 — `openbmb/MiniCPM-V-2_6` (implemented: `arch: minicpmv`)
- 8B multimodal model with multilingual document OCR (30+ languages incl.
  Arabic). Custom `model.chat(msgs, tokenizer)` API (`trust_remote_code`), so it
  has its own `hf_local_client` branch. 4-bit on a T4, dropped in CPU mode. Also
  the recommended **local Ollama** model (`ollama pull minicpm-v`).

### AIN, Qalam, other Arabic VLMs — stretch
- **Qalam** (SwinV2 encoder + RoBERTa decoder, purpose-built Arabic OCR/HWR):
  check whether weights are publicly released; if only via paper/API, treat as
  Tier 4. If weights exist, it needs its own `hf_local_client` branch (not a
  Qwen/PaliGemma architecture).
- **AIN** (MBZUAI Arabic VLM) — implemented as `ain-7b` (`arch: qwen2_vl`, 4-bit).

**Testing a Tier 3 model:** same as Tier 2. Add a `tests/fixtures/` golden line
with a hand transcription and assert CER is near the model's reported number as
a smoke check.

---

## Tier 4 — Arabic-first API providers (`arabic_api_client`, ~30-line adapters)

Each provider gets a thin adapter mapping our `transcribe()` to its REST call.
Add a row only once we have a working key **and** a vision/OCR endpoint.

### Fanar (QCRI, Qatar)

| Field | Value |
| --- | --- |
| Model | `Fanar-2-Oryx-IVU` (image-vision-understanding); platform "Fanar 2.0" |
| Access | Developer program at fanar.qa — request form + API key; some endpoints OpenAI-compatible |
| Weights | `QCRI/Fanar-2-Oryx-IVU` on HF also exists → could instead run via `hf_local_client` if license permits |
| Track fit | line + page |

If the API is OpenAI-compatible, route via `litellm_client` with a custom
`api_base` instead of writing an adapter.

### ALLaM / HUMAIN (SDAIA, Saudi Arabia)
- ALLaM is served via IBM watsonx.ai and the HUMAIN platform. Confirm a
  **multimodal** ALLaM endpoint exists (base ALLaM is text-only); if not, drop
  from the matrix. Credential + endpoint via HUMAIN/watsonx console.

### Tarjama Pronoia
- Enterprise Arabic AI engine advertising handwritten + bilingual OCR. No public
  self-serve API — contact sales for a key. Adapter: single POST with image,
  parse `text` field.

### Arabic.AI OCR
- `https://arabic.ai/ocr/` — commercial OCR API. Sign up for a key; adapter POSTs
  the image, reads the returned transcription. Good "productized Arabic OCR"
  reference point.

---

## Tier 5 — Classical / cloud OCR baselines (anchors)

These are not LLMs; they establish the "traditional OCR" floor the project's
hypothesis is measured against. Wrap each as a `ModelClient` that ignores the
prompt.

| Engine | How to run | Notes |
| --- | --- | --- |
| **Tesseract** | `pytesseract.image_to_string(img, lang="ara")`; install `tesseract-ocr-ara`; try `ara` vs the `tessdata_best` model | fast, free, weak on cursive manuscripts — expected high CER, that's the point |
| **Surya OCR** | `surya_ocr` Python API; supports Arabic; has its own line detector for page track | modern, strong classical baseline |
| **EasyOCR** | `easyocr.Reader(['ar'])` | stretch |
| **PaddleOCR** | Arabic recognition model + detector | stretch |
| **Google Cloud Vision** | `DOCUMENT_TEXT_DETECTION`; `GOOGLE_APPLICATION_CREDENTIALS` | best on page + printed tracks |
| **Azure AI Document Intelligence** | `prebuilt-read` model; endpoint + key | best on page + printed tracks |
| **Amazon Textract** | `detect_document_text`; AWS creds | Latin-biased; stretch |

---

## Tier 6 — Fine-tuned models (Phase 5)

Only after the harness is trusted and the Muharaf `test` split has been kept
untouched. Train on Muharaf `train` (22.1k lines) / pages, benchmark on `test`.

| Route | How |
| --- | --- |
| OpenAI fine-tuning | Upload JSONL of `{image, target_text}` pairs; fine-tune a vision-capable base; call the resulting `ft:` model id through `litellm_client` unchanged |
| Vertex AI fine-tuning | Supervised tuning of Gemini on GCS-hosted pairs; call the tuned endpoint via LiteLLM Vertex route |
| Qwen2.5-VL LoRA | `transformers` + `peft` + `trl` SFT on line pairs; serve the merged model via vLLM; benchmark as a Tier 2 model |
| Qari-OCR LoRA | Same as Qwen LoRA on the 2B base; cheapest to train |

Report fine-tuned numbers side by side with the same model's zero-shot row so
the fine-tuning delta is explicit.

---

## Adding a new model — checklist

1. Add a row to `configs/models.yaml`: `id`, client type, identifier,
   `api_base` (if served), `params`, pricing (`input_usd_per_1k`,
   `output_usd_per_1k`, `usd_per_image` if applicable).
2. If it needs a new call shape, add/extend a `ModelClient` in `src/.../models/`.
3. Add it to `benchmark-matrix.md` with intended phases.
4. Run the 15-line Muharaf smoke test; eyeball 3 outputs for encoding/RTL issues.
5. Confirm the empty-string sanity baseline still scores CER ≈ 1.0 (pipeline check).
