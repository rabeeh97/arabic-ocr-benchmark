# 2026-09-21 — AIN-7B + NAKBA-specific LoRA (in progress: ViT-unfreeze stage pending)

- **Milestone:** greenfield areas 1 (Arabic-native encoder) and 3 (script-family
  coverage) — opening experiment of Phase P1, see
  `docs/engineer-guide-arabic-ocr.md` §1.3 for phasing.
- **Config:** trained directly on `U4RASD/omar-al-saleh-manuscripts-segments`
  (NAKBA) train split, no Muharaf pretraining stage — config not yet committed
  to `configs/training/` (run predates this write-up; back-fill before the
  next checkpoint).
- **Base model / method:** AIN-7B (MBZUAI, full-parameter Arabic-tuned
  Qwen2-VL-7B) / LoRA r=16, α=32, ViT **frozen** (stage 1, this entry) → ViT
  **unfrozen with a differential LR** (stage 2, in flight, not yet scored).
- **Data:** NAKBA train split only. Contamination: not yet re-run against this
  narrower recipe; the suite_v1 no-leak guard covers the wider Muharaf/AraMS/RASAM
  mix, not this NAKBA-only config specifically — confirm before citing this as
  a clean result outside the blind_test protocol.
- **Hardware / cost:** RTX 3090 (Windows box); stage 1 to checkpoint-1936
  (1 epoch) scored 2026-09-21.
- **Run dir(s):** `suite-v1-artifacts/ain7b_nakba/` (local backup, not in git —
  see memory `suite-v1-artifacts-location`; box path
  `C:\Users\User\a-experiment\suite-v1\ain7b_nakba\`).

## Result — stage 1 (ViT frozen), official Systems-Track protocol

Protocol: HF `U4RASD/omar-al-saleh-manuscripts-segments` split `blind_test`
(n=2,671, all with ground truth), corpus CER = `cer_micro_raw` (sum of edits /
sum of reference chars, Unicode codepoints, no normalization) — this is the
contest's actual ranking metric, not our usual CER-light. See
`suite-v1-artifacts/ain7b_nakba/official/README.md` for the full metric
reconciliation (two public "contest CER" definitions, and why our Baseer
re-score doesn't match the published number).

| model | n | corpus CER (raw) | per-line CER (raw) | WER (raw) | CER-light |
|---|---|---|---|---|---|
| AIN-7B+NAKBA LoRA, ckpt-1600 | 2670 | **10.78%** | 14.56% | 34.28% | 8.63% |
| AIN-7B+NAKBA LoRA, ckpt-1936 (epoch end) | 2670 | 10.82% | 14.64% | 34.22% | 8.67% |
| nakba-misraj-baseer (our re-score, same harness) | 2671 | 9.38% | 12.72% | 29.30% | 7.28% |
| Baseer-Nakba, published (NAKBA-2026 winner) | 2671 | **7.9%** | — | 24.4% | — |

Baseline = published Baseer-Nakba (0.079 corpus CER). Our AIN result is 1.4
points behind Baseer's number *as re-scored in our harness* (9.4%), and about
2.9 points behind the published number — but the published-vs-our-harness gap
for Baseer itself is ~1.5 points and attributed to prompt/decoding
(`line-v2` + greedy here vs Baseer's contest prompt +
`repetition_penalty=1.1`), not the test set. AIN's gap to Baseer under
*identical* decoding is therefore likely smaller than 1.4 points.

Training curve on the easier public 200-line `nakba_lines` (not the ranking
set — kept only to watch the trend):

| ckpt | CER-light | CER-raw |
|---|---|---|
| 400 | 7.92% | 9.03% |
| 800 | 7.28% | 8.32% |
| 1200 | 6.95% | 8.02% |
| 1600 | 6.17% | 7.13% |

Still improving at 1600/1936, not saturating.

## Notes

- A single-stage LoRA on an already Arabic-tuned 7B base (AIN), trained only
  on NAKBA's own data, gets within ~1.4 points of Baseer-Nakba's two-stage
  (Muharaf-then-NAKBA, full-parameter, weight-averaged) pipeline — a
  substantially cheaper recipe for a comparable result. Consistent with the
  suite_v1 finding that base-model choice, not recipe sophistication, has been
  the dominant lever so far (see `findings-and-notes.md`, "suite_v1" entry).
- **Stage 2 (in flight):** unfreeze the ViT with a differential LR (decoder
  faster, vision encoder slower — same shape as Baseer's own stage-2 recipe
  and the exp3090 `qwen3vl4b_visft` config), retrained on NAKBA. This is a
  same-model, same-data before/after test of the vision-encoder-adaptation
  hypothesis that suite_v1's cross-model comparison could only suggest —
  cleaner evidence because architecture and data are held fixed and only the
  freeze setting changes.
- **Next experiment this suggests / what to report when stage 2 lands:**
  paired per-sample CER (frozen vs unfrozen ViT, same blind_test lines), the
  same four failure-mode categories used in the suite_v1 per-sample read
  (hallucination, dot/letter-shape confusion, foreign-token fidelity,
  short-line noise) — specifically whether unfreezing measurably reduces
  category 2 (dot confusion) more than the others, which would be the
  sharpest evidence yet for the Arabic-native-encoder hypothesis (greenfield
  area 1).
- Housekeeping before closing this experiment: commit the training config to
  `configs/training/`, run the standard no-leak/contamination check against
  this NAKBA-only recipe, and confirm whether `train_vision` differential LR
  values match Baseer's disclosed 1e-4 (decoder) / 9e-6 (encoder) or were
  chosen independently.
