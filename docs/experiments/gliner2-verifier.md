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

## Second model: qari-ocr-0.2 on muharaf_lines (short handwritten lines)

Run `results/20260831T185739Z`, all 1319 non-errored lines; output
`results/gliner2_verifier/20260920T151011Z/`. Overall CER 0.798 (matches `summary.csv`).

| threshold | accepted (n) | accepted CER | refused (n) | refused CER |
|---|---|---|---|---|
| 0.5 | 1287 | 0.795 | 32 | 0.910 |
| 0.7 | 1221 | 0.794 | 98 | 0.843 |
| 0.9 | 1076 | 0.784 | 243 | 0.848 |

**The gate is uninformative here.** Mean confidence is 0.924 on ground truth and 0.925 on
character-shuffled text (AUROC 0.53, chance), and confidence predicts low-CER lines at AUROC 0.45
(below chance). The small accepted/refused CER gaps should not be read as signal. Qari is also close
to failing outright on this set (CER 0.80), so there is little good output to separate from bad.
Short lines give the classifier little to work with compared with the long kitab_hindawi pages.

## Third family: Qwen3-VL-4B base vs fine-tuned (short lines)

Runs `results/20260909T184119Z` (nakba_lines, 200 lines/model) and `results/20260909T174517Z`
(kitab_khatt, 180 lines/model); outputs `results/gliner2_verifier/20260920T151739Z/` and
`.../20260920T151852Z/`. Overall CERs match `summary.csv`. Threshold 0.9, CER raw micro:

| dataset | model | overall | accepted (n) | refused (n) | AUROC conf vs low-CER | AUROC GT vs shuffled |
|---|---|---|---|---|---|---|
| nakba_lines | qwen3-vl-4b base | 0.768 | 0.707 (129) | 0.836 (71) | 0.72 | 0.65 |
| nakba_lines | qwen3-vl-4b ft | 0.326 | 0.315 (124) | 0.338 (76) | 0.64 | 0.56 |
| kitab_khatt | qwen3-vl-4b base | 0.771 | 0.705 (108) | 0.860 (72) | 0.72 | 0.76 |
| kitab_khatt | qwen3-vl-4b ft | 0.300 | 0.266 (110) | 0.349 (70) | 0.62 | 0.72 |

Here the direction holds on all four: accepted CER is below overall, refused above. The gap is
largest for the weak base model (about 6 points below overall, 6-9 above at 0.9) and for the
fine-tuned model on kitab_khatt (3.4 below); it is negligible for the fine-tuned model on
nakba_lines. The controls are only moderate (GT-vs-shuffled AUROC 0.56-0.76, and on ft/khatt the
shuffled text scores higher on average, 0.892, than the model output, 0.872), so the gate
separates "clearly bad" output rather than certifying good output. n is 180-200 per cell with no
significance test.

## Notes

- **Direction is right on PaliGemma / kitab_hindawi, effect is small, and it does not carry over
  to Qari / Muharaf (above).** Accepted CER is 2-3 points below overall and refused is
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

For the Qari run use `--run results/20260831T185739Z --models qari-ocr-0.2 --dataset muharaf_lines`.

Model confidences are cached by text hash in `results/gliner2_verifier/_cache.jsonl` (committed;
6403 texts, covering every run above), so reruns and any other model that produced the same text are
instant.
