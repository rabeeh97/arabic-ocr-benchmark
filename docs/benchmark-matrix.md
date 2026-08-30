# Benchmark Matrix — Models × Datasets

Companion to `../we-are-starting-here-hashed-llama.md` (the Phase 1 plan) and to
`models.md` / `datasets.md` in this folder.

This is the full target surface: everything we currently plan to run, plus the
stretch models and datasets we may add later. It is a planning artifact, not a
commitment — cells say *when* we intend to run a pairing, not that it is done.

## Legend

**Cell values** = the plan phase in which that model × dataset track is first run:

| Symbol | Meaning |
| --- | --- |
| `1`–`5` | Planned in that phase (see plan's "Implementation phases") |
| `○` | Stretch — run only if time / budget / access allows |
| `–` | Not planned (track not meaningful for this model, or superseded) |

**Dataset track columns:**

| Col | Track | Level | Status |
| --- | --- | --- | --- |
| `ML` | Muharaf-public (lines) — `aamijar/muharaf-public` | line | primary |
| `MP` | Muharaf-public (pages) — GitHub/Zenodo | page | primary |
| `MR` | Muharaf-restricted (pages) | page | stretch (access request) |
| `KH` | KHATT | line + page | secondary |
| `KB` | KITAB-Bench (handwritten + printed subsets) | line/page | secondary |
| `IE` | IFN/ENIT | word | low priority |
| `PR` | Printed-Arabic anchor (APTI / AMFDS / KITAB printed) | line | contrast baseline |

## Matrix

### Tier 1 — Hosted frontier VLMs (LiteLLM, Phase 1–2)

| Model | ML | MP | MR | KH | KB | IE | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini 2.5 Flash | 1 | 3 | ○ | 4 | 4 | ○ | 4 |
| Gemini 2.5 Pro | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| GPT-4o | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| GPT-4.1 | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| GPT-5 (vision) | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| Claude Sonnet (latest) | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| Claude Opus (latest) | 2 | 3 | ○ | 4 | 4 | ○ | 4 |

### Tier 2 — Open-weight VLMs (local / vLLM / HF, Phase 2)

| Model | ML | MP | MR | KH | KB | IE | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| Qwen2.5-VL-32B-Instruct | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| Qwen2.5-VL-72B-Instruct | ○ | ○ | – | ○ | ○ | – | ○ |
| Llama-3.2-11B-Vision-Instruct | ○ | ○ | – | ○ | ○ | – | ○ |
| InternVL3-8B / -14B | ○ | ○ | – | ○ | ○ | – | ○ |
| Pixtral-12B | ○ | ○ | – | ○ | ○ | – | ○ |

### Tier 3 — Arabic-specific OCR / HTR models (local, Phase 2)

| Model | ML | MP | MR | KH | KB | IE | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qari-OCR v0.2.2.1 (Qwen2-VL-2B) | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| Qari-OCR v0.3 (Qwen2-VL-2B) | 2 | 3 | ○ | 4 | 4 | ○ | 4 |
| PaliGemma-3B Arabic-HW-OCR (LoRA) | 2 | ○ | – | 4 | ○ | ○ | ○ |
| AIN / other Arabic VLM | ○ | ○ | – | ○ | ○ | – | ○ |
| Qalam (SwinV2+RoBERTa) — weights TBD | ○ | ○ | – | ○ | ○ | – | ○ |

### Tier 4 — Arabic-first API providers (bespoke adapters, Phase 4)

| Model / Service | ML | MP | MR | KH | KB | IE | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fanar-2 Oryx IVU (QCRI) | 4 | 4 | ○ | 4 | 4 | ○ | 4 |
| ALLaM / HUMAIN (SDAIA) | ○ | ○ | – | ○ | ○ | – | ○ |
| Tarjama Pronoia | ○ | ○ | – | ○ | ○ | – | ○ |
| Arabic.AI OCR | ○ | ○ | – | ○ | ○ | – | ○ |

### Tier 5 — Classical / cloud OCR baselines (anchors, Phase 3–4)

| Engine | ML | MP | MR | KH | KB | IE | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tesseract (`ara`, `ara_best`) | 3 | 3 | ○ | 4 | 4 | ○ | 4 |
| Surya OCR | 3 | 3 | ○ | 4 | 4 | ○ | 4 |
| EasyOCR (Arabic) | ○ | ○ | – | ○ | ○ | – | ○ |
| PaddleOCR (Arabic) | ○ | ○ | – | ○ | ○ | – | ○ |
| Google Cloud Vision | ○ | 4 | – | ○ | 4 | – | 4 |
| Azure AI Document Intelligence | ○ | 4 | – | ○ | 4 | – | 4 |
| Amazon Textract | ○ | ○ | – | ○ | ○ | – | ○ |

### Tier 6 — Fine-tuned models (Phase 5, decision point)

| Model | ML | MP | MR | KH | KB | IE | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4o fine-tuned (OpenAI, Muharaf train) | 5 | 5 | ○ | ○ | – | – | – |
| Gemini fine-tuned (Vertex AI, Muharaf train) | 5 | 5 | ○ | ○ | – | – | – |
| Qwen2.5-VL LoRA (Muharaf train) | 5 | 5 | ○ | ○ | – | – | – |
| Qari-OCR LoRA (Muharaf train) | 5 | 5 | ○ | ○ | – | – | – |

## Priority summary

- **Phase 1** (thin slice): Gemini 2.5 Flash × Muharaf-lines only.
- **Phase 2** (full line track): Tiers 1–3 core rows × `ML`.
- **Phase 3** (page track): Tiers 1–3 core rows + Tesseract/Surya × `MP`.
- **Phase 4** (breadth): add Tier 4 + cloud baselines; add `KH`, `KB`, `PR`, `IE`.
- **Phase 5** (fine-tuning): Tier 6 × `ML` + `MP`, benchmarked against the untouched Muharaf `test` split.

## Notes on scope choices

- **Line-only Arabic models** (PaliGemma-3B Arabic-HW) are marked `–` / `○` for
  page tracks: they were trained on single lines and degrade on full pages; we
  keep them on line tracks where the comparison is fair.
- **72B / large open models** are all `○`: they need a rented GPU, so they run
  only if the smaller open models show enough promise to justify the cost.
- **IFN/ENIT** stays low priority everywhere — it is a constrained
  word-recognition set (Tunisian place names), useful only for comparability
  with older literature, not for cursive-manuscript performance.
- **Cloud OCR baselines** (Google/Azure/AWS) are most informative on page and
  printed tracks, where their layout engines matter; less so on pre-cropped lines.
