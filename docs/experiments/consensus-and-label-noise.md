# 2026-09-21 — consensus transcription and label-noise detection from model agreement

Follows `confidence-signal-baselines.md` (cross-model agreement predicts which lines are wrong).
Two uses of the same idea, both computed from existing predictions (no new OCR calls):

1. **Consensus transcription (MBR):** per line, pick the model output most similar to all the
   others (the medoid) and score it against the ground truth.
2. **Label-noise detection:** flag lines whose ground truth is far from that consensus.

- **Scripts:** `scripts/experiment_consensus_transcription.py`,
  `scripts/experiment_label_noise_detection.py` (shared loaders in
  `experiment_confidence_baselines.py`).
- **Outputs:** `results/consensus/<ts>_<dataset>/{per_model.csv,picks.jsonl}`,
  `results/label_noise/<ts>_<dataset>/{flagged.jsonl,injected_noise.json}`.
- **Pool:** latest non-errored prediction per (model, line) across all `results/2*/` runs; a line
  needs >= 5 models. Text similarity = normalized Levenshtein on diacritic-stripped text.

## 1. Consensus transcription

CER raw, micro, on the same lines. "best single" is the best model *chosen with the ground truth*,
a deliberately strong baseline that a real deployment could not pick in advance. `oracle` picks the
best candidate per line by true CER (unreachable ceiling); `random` picks a random candidate.

| dataset | lines | best single | MBR (all models) | MBR - best single (95% CI) | oracle | random |
|---|---|---|---|---|---|---|
| kitab_khatt | 200 | 0.285 (gemini-flash-lite) | **0.239** | **-4.6 pts (-7.7 to -1.7)** | 0.187 | 0.620 |
| kitab_khatt, 3-model committee | 180 | 0.278 (best member) | 0.240 | -3.8 (-6.6 to -1.2) | 0.201 | 0.320 |
| kitab_khatt, 5-model committee | 180 | 0.278 | 0.234 | -4.4 (-7.2 to -1.7) | 0.200 | 0.497 |
| kitab_hindawi (long pages) | 198 | 0.432 (paligemma-ft) | 0.555 | **+10.3 (+5.8 to +14.9)** | 0.284 | 0.652 |
| kitab_muharaf | 78 | 0.365 (gemini-flash-lite) | 0.396 | +3.1 (-2.0 to +8.1) | 0.302 | 0.956 |
| kitab_historyar | 60 | 0.310 (gemini-flash-lite) | 0.342 | +3.2 (-3.1 to +10.1) | 0.229 | 0.677 |
| kitab_onlinekhatt | 40 | 0.115 (gemini-flash-lite) | 0.143 | +2.8 (-0.6 to +6.8) | 0.090 | 0.468 |

The 3-model committee is gemini-flash-lite + the fine-tuned Qwen3-VL-4B + the fine-tuned PaliGemma;
the 5-model one adds gemini-flash and base Qwen3-VL-4B, which are weak on this set (CER ~0.77), and
MBR still does not get worse.

- MBR is far better than most individual models everywhere (the weaker ones improve by 20 to 200+
  CER points; the strongest by about 3-5 on khatt), so it is a safe way to avoid having to know
  which model is best.
- It **beats the best single model only on kitab_khatt** (many diverse, moderately good models).
  On the other four sets it merely ties the hindsight-best model (CIs include 0), and on long pages
  (kitab_hindawi) it is clearly worse: many models emit truncated or partial pages, so the medoid
  lands on the majority failure mode. Restricting to the more-agreeing half of the models
  (`mbr_top`) helps there (0.500) but hurts elsewhere; it is not a consistent fix.
- The gap to the oracle (0.19-0.28) shows selection headroom that plain similarity does not reach.

## 2. Label-noise detection

**Validation on injected noise.** 5% of ground-truth lines corrupted (30 random seeds), detector =
CER between the (possibly corrupted) ground truth and the consensus, versus the same distance from
the hindsight-best single model (what a one-model, CER-based cleaner does). AUROC / precision@k
(random: 0.5 / 0.05):

| dataset | noise | consensus AUROC | single-model AUROC | consensus P@k | single P@k |
|---|---|---|---|---|---|
| kitab_khatt | swap with another line | 0.994 | 0.975 | 0.82 | 0.54 |
| | drop 30% of words | 0.961 | 0.905 | 0.62 | 0.30 |
| | replace 30% of words | 0.865 | 0.782 | 0.27 | 0.09 |
| kitab_hindawi | swap | 0.736 | 0.994 | 0.15 | 0.83 |
| | drop 30% | 0.641 | 0.906 | 0.05 | 0.38 |
| kitab_muharaf / historyar / onlinekhatt | all three noise types | within ~0.05 AUROC of the single model, no consistent winner (40-78 lines, noisy) | | | |

Same pattern as section 1: the consensus detector wins clearly on kitab_khatt, is much worse on
long pages, and is on par elsewhere. Its advantage is that it needs no hindsight choice of model.

**Real data.** Models rarely truly agree on these hard sets (highest line agreement 0.82 on khatt;
only 30 of 200 khatt lines >= 0.6). Among the lines with agreement >= 0.6, the largest
ground-truth-vs-consensus gap is CER 0.15 on khatt, 0.07 on muharaf, 0.19 on historyar and 0.16 on
onlinekhatt (where models agree most: 21 of 40 lines >= 0.6), with one exception on hindawi (below).
The onlinekhatt gaps I read are ordinary model errors. Checked by eye or against the images:

- `kitab_hindawi/train/000016`: flagged with CER 0.99 at agreement 0.62. The image (viewed) is a full
  page and the ground truth matches it; the models' consensus is only the page title. **Shared model
  failure, not a label error.** This is the failure mode noted in the previous write-up: models
  agreeing does not mean they are right.
- `kitab_khatt/train/000095`: the handwriting contains a scribbled glyph and the annotators wrote `#`
  for it (viewed). **11 of 200 khatt ground-truth lines contain `#`** (none in the other four
  datasets), and those lines sit farther from the consensus (0.295 vs 0.192 mean CER). The repo's
  `normalize.py` does not strip `#`, so scoring penalizes models for not writing a placeholder.
  Not fixed here; worth deciding whether to normalize it out.
- The other top-flagged khatt lines read as sensible ground truth against garbled consensus text
  (I only read the text, not the images).

**No genuine label error was confirmed in these five datasets.**

## Caveats

- Small evaluation sets (40-200 lines) and one benchmark family; the injected-noise results on the
  40-78 line sets are noisy.
- The MBR pool includes each compared model itself, and per-line pool membership varies (some
  models cover only 10-40 lines). Only the committee rows are cost-matched; the all-models rows
  need every model's inference.
- Injected noise is synthetic and coarse; real label errors (a wrong word, a misread) are subtler
  than swaps.
- Only two flagged lines were checked against the images.

## Reproduce

    # scratch venv from docs/experiments/gliner2-verifier.md, plus: uv pip install pillow datasets pyyaml pandas tqdm
    PYTHONPATH=src python scripts/experiment_consensus_transcription.py --dataset kitab_khatt
    PYTHONPATH=src python scripts/experiment_consensus_transcription.py --dataset kitab_khatt \
        --committee gemini-flash-lite exp3090-qwen3vl4b-ft-bf16 exp3090-paligemma2-3b-ft-bf16
    PYTHONPATH=src python scripts/experiment_consensus_transcription.py --dataset kitab_muharaf --min-lines 40
    PYTHONPATH=src python scripts/experiment_label_noise_detection.py --dataset kitab_khatt
