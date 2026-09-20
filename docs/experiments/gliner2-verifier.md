# 2026-09-20 — text-only confidence gate on OCR output (GLiNER2-Arabic)

- **Question:** can a small text-only model give a 0-1 confidence for an OCR transcription, so that
  refusing low-confidence outputs lowers CER on what we accept?
- **Model:** `Elafnawaf/gliner2-arabic-multitask` (0.3B, runs on CPU), zero-shot classification with
  labels `well_formed_arabic` / `garbled_ocr_error`; confidence = P(well_formed_arabic).
- **Script:** `scripts/experiment_gliner2_verifier.py` (torch is not a repo dep — see reproduce below).
- **Data:** existing predictions in `results/20260913T093935Z` (kitab_hindawi; no new OCR calls).
  Scores `cleaned_text`, skips errored rows, micro-CER via the repo's `score_pair`/`aggregate`
  (overall CER matches `summary.csv`: 0.2718 and 0.4323).
- **Cost:** none (local CPU); 270 rows x 3 texts scored.
- **Output:** `results/gliner2_verifier/20260920T144005Z/{summary.csv,confidences.jsonl}`

## Result (CER raw, micro)

| model | n | overall | thr 0.9: accepted (n) | thr 0.9: refused (n) | AUROC conf vs low-CER |
|---|---|---|---|---|---|
| paligemma2-3b-bf16 | 92 | 0.272 | 0.239 (57) | 0.328 (35) | 0.68 |
| exp3090-paligemma2-3b-ft-bf16 | 178 | 0.432 | 0.413 (88) | 0.454 (90) | 0.65 |

At threshold 0.5 the gate refuses 1-2 rows (does nothing); at 0.7 it refuses 6 / 14 rows and the
refused CER is higher (0.41 vs 0.26 base; 0.57 vs 0.42 ft).

Controls: mean confidence is 0.95 on ground truth, 0.87-0.89 on the model output, 0.84-0.85 on
character-shuffled output. AUROC ground-truth vs shuffled is 0.91-0.92.

## Notes

- **Direction is right, effect is small.** Accepted CER is 2-3 points below overall and refused is
  higher, but the AUROC is 0.65-0.68 and n is 92-178; no significance test was run, so treat the
  gap as suggestive, not established.
- **Overconfident, as the model card warns.** Even shuffled gibberish scores ~0.85, so useful
  thresholds sit in a narrow 0.85-0.95 band. A first smoke test scored a hand-garbled line *higher*
  (0.966) than the clean one (0.940).
- **Not checked:** length confound (long pages truncated to 1500 chars before scoring) and whether
  the gate beats a trivial baseline such as output length or repetition rate. Do that before
  investing more.
- **Inherent limit:** the gate never sees the image, so it cannot catch fluent-but-wrong text.
- **Ruled out, don't re-investigate:**
  - Jev (TypeSafe `/v1/systemone`): `TYPESAFE_AI_API_KEY` returned 401 on 2026-09-20; not on OpenRouter.
  - Von-1.0 (`wfzyx/von-1.0`, the open-weight Jev-style model): its vocab has 0 Arabic tokens
    (Arabic becomes byte fragments) and it was trained on English NLI.

## Reproduce

    uv venv /tmp/gliner_venv --python 3.11
    uv pip install --python /tmp/gliner_venv/bin/python "gliner2[local]" jiwer regex
    PYTHONPATH=src /tmp/gliner_venv/bin/python scripts/experiment_gliner2_verifier.py \
        --run results/20260913T093935Z --models paligemma2-3b-bf16 exp3090-paligemma2-3b-ft-bf16

Model confidences are cached in `results/gliner2_verifier/_cache.jsonl`, so reruns are instant.
