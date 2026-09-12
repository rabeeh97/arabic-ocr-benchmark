# Benchmark summary — `20260912T125612Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-gemma-4-26b-free | kitab_hindawi | 1 | 11 | 0.0647 | 0.0416 | 0.0256 | 0.9584 | 0.3399 | 0.3355 | 0.0 | 0.8571 | 9358.5 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_hindawi | 1 | 11 | 0.0662 | 0.0094 | 0.0068 | 0.9906 | 0.1439 | 0.1439 | 0.0 | 0.8333 | 11305.9 | 0.0 |
| pixtral-12b | kitab_hindawi | 99 | 1 | 0.4238 | 0.385 | 0.3537 | 0.615 | 0.7024 | 0.661 | 0.0 | 0.2392 | 6378.8 | 0.016893 |
| gemini-flash | kitab_hindawi | 62 | 10 | 0.9405 | 0.9386 | 0.9381 | 0.0614 | 0.9493 | 0.9489 | 0.0 | 0.1106 | 8743.9 | 0.121427 |
| mistral-medium | kitab_hindawi | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| mistral-small | kitab_hindawi | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-llama-4-maverick | kitab_hindawi | 0 | 100 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_hindawi | 0 | 100 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 0 | 100 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_hindawi | 0 | 100 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
