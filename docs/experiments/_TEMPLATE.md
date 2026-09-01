# <date> — <experiment name>

- **Milestone:** M?  (see docs/training-plan.md)
- **Config:** `configs/training/<exp>.yaml`
- **Base model / method:** e.g. Qari-OCR v0.3 / QLoRA r=16
- **Data:** sources + line counts; contamination report:
  `results/contamination_report.json` (flagged: ?)
- **Hardware / cost:** T4 or 3090; wall-clock; GPU-hours; $ (hosted)
- **Run dir(s):** `results/<run_id>/`

## Result

| metric | baseline | this run | Δ |
|---|---|---|---|
| CER·light micro (held-out test) | | | |
| CER raw | | | |
| CER diac-insensitive | | | |
| diacritic F1 | | | |
| WER·light | | | |
| exact-match % | | | |

Baseline = <prior best model on this dataset>.

## Notes

- What was tried, what changed vs the previous run, surprises.
- Failure modes seen in the predictions (repetition, markup, dropped diacritics…).
- Next experiment this suggests.
