# 2026-09-21 — reference-free confidence signals for OCR lines

- **Question:** which cheap, ground-truth-free signal best predicts which OCR lines are wrong, so we
  can accept/refuse lines? Follows `gliner2-verifier.md` (text-only GLiNER2 gate: weak).
- **Data:** existing predictions only (latest run per model+line across `results/2*/`); no new OCR
  calls. `kitab_khatt` (short handwritten lines; 35 models, 8 evaluated) and `kitab_hindawi` (long
  printed pages; 31 models, 5 evaluated). A model is evaluated if it has >= 100 lines.
- **Scripts:** `scripts/experiment_confidence_baselines.py` (main comparison),
  `scripts/experiment_agreement_ablation.py` (robustness of the agreement signal).
- **Outputs:** `results/confidence_baselines/<ts>_<dataset>/{signals.jsonl,summary.csv,per_model.csv}`
  and `agreement_ablation_kitab_khatt.json`. Per-line signals are saved (nothing needs recomputing);
  LM scores cached in `results/confidence_baselines/_lm_cache.jsonl`, GLiNER2 in
  `results/gliner2_verifier/_cache.jsonl`.

## Signals (none use the ground truth; higher = more likely correct)

| signal | definition |
|---|---|
| agree | mean normalized Levenshtein similarity to the other models' outputs for the same line |
| lm | negative per-character NLL under `Qwen/Qwen2.5-0.5B` |
| gliner | GLiNER2-Arabic P(well-formed), as in `gliner2-verifier.md` |
| len_dev | negative abs. log ratio of output length to the median other-model output length |
| repeat / nonarabic | negative fraction of repeated char 4-grams / of non-Arabic letters |
| random | control |
| oracle | ranks by true CER (the ceiling) |

Model log-probs are not in the stored predictions, so the recognizer's own confidence, the standard
baseline, is NOT tested here (needs a rerun of the local models with `output_scores`).

## Method

Evaluated **within each model** and averaged over models, never pooled, so a signal that only
separates good models from bad ones gets no credit. Metrics: Spearman(signal, -CER), and
`gain@c` = mean per-line CER over all lines minus mean CER over the top-`c` share of lines by the
signal (CER points; bigger is better). 95% CIs bootstrap lines within each model (1000 draws); they
do not capture variance across models (only 5-8).

## Result: kitab_khatt (8 models, 1472 model-lines)

| signal | rho | gain@0.5 (95% CI) | gain@0.8 |
|---|---|---|---|
| **agree** | **0.76** | **15.3 pts (13.3-18.0)** | 7.2 |
| len_dev | 0.37 | 9.1 (6.9-11.9) | 5.2 |
| lm | 0.19 | 0.0 (-2.8-2.3) | 0.6 |
| gliner | 0.11 | 1.5 (-1.5-3.6) | 0.5 |
| repeat | 0.08 | 2.8 (1.2-6.0) | 2.3 |
| nonarabic | 0.17 | -0.4 | 0.6 |
| random | 0.00 | 0.1 | 0.1 |
| oracle | 1.00 | 20.8 | 10.3 |

Agreement recovers about 74% of the oracle gain at 50% coverage; the GLiNER2 gate and the LM are
indistinguishable from random. Per model, agreement's gain@0.5 is positive for all 8 (8.5-21.8
pts). The LM signal is negative for some models (pixtral: -17.8 pts): bad outputs can be fluent.

## Replication: kitab_hindawi (5 models, 974 model-lines, long pages)

| signal | rho | gain@0.5 (95% CI) |
|---|---|---|
| agree | 0.48 | 8.3 pts (7.0-9.3) |
| len_dev | 0.34 | 4.7 |
| repeat | -0.42 | -4.2 (sign flips: truncated outputs have fewer repeats) |
| nonarabic | 0.00 | 1.0 |
| random | 0.03 | 0.9 |
| oracle | 1.00 | 13.7 |

Same ordering, weaker (agreement recovers ~61% of oracle). GLiNER2/LM not run on this set.

## Ablation: how many other models, and does family matter (kitab_khatt)

| variant | rho | gain@0.5 |
|---|---|---|
| all other models | 0.761 | 15.3 |
| drop same-family models | 0.756 | 15.1 |
| 1 random other model | 0.726 | 14.4 |
| 3 random other models | 0.755 | 15.2 |
| 5 random other models | 0.754 | 15.1 |

The signal is not an artifact of base/fine-tuned siblings agreeing, and it saturates at about 3
models. Even ONE other model keeps ~94% of the gain, which suggests it mostly measures how hard the
line is (different models fail on the same lines), not consensus on the exact answer. (The k=10
rows have too few qualifying models to read.)

## Conclusions

- **The useful signal is cross-model agreement, not a text-only judge.** It works within a model,
  survives dropping same-family models, and replicates on printed pages, all from outputs we already
  have. Text-only plausibility (GLiNER2, LM) adds nothing measurable.
- **It costs extra inference.** A second model per line is the price; the ablation says one cheap
  second model may already capture most of it. This is a natural cascade/triage signal.

## Caveats

- Only 5-8 evaluated models per dataset, all from one benchmark family; no significance test across
  models. Datasets with <=5 models (nakba, rasam) were not tested, and they are the realistic
  2-model setting.
- Mean per-line CER includes lines with CER > 1 (pixtral averages 1.22), so gains are sensitive to
  outlier lines. For tie-heavy signals (nonarabic) the bootstrap rho CI can miss the point estimate.
- Not tested: model log-probs, VLM-as-judge with the image, and whether agreement helps with
  fluent-but-wrong output shared by all models (shared errors would look confident).

## Reproduce

    # venv from docs/experiments/gliner2-verifier.md (torch, transformers, gliner2, jiwer)
    PYTHONPATH=src python scripts/experiment_confidence_baselines.py --dataset kitab_khatt
    PYTHONPATH=src python scripts/experiment_confidence_baselines.py --dataset kitab_hindawi --no-gliner --no-lm
    PYTHONPATH=src python scripts/experiment_agreement_ablation.py --dataset kitab_khatt
