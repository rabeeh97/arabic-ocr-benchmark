# Benchmark summary — `20260902T142223Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_muharaf | 10 | 0 | 0.2936 | 0.2822 | 0.2705 | 0.7178 | 0.6566 | 0.625 | 0.0 | 0.8 | 1590.6 | 0.001216 |
| openrouter-qwen3-vl-235b | kitab_muharaf | 10 | 0 | 0.5436 | 0.5379 | 0.461 | 0.4621 | 0.8283 | 0.8119 | 0.0 | 0.6095 | 3546.4 | 0.000884 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 10 | 0 | 0.589 | 0.5833 | 0.5181 | 0.4167 | 0.9495 | 0.8704 | 0.0 | 0.6 | 1646.7 | 0.001204 |
| openrouter-llama-4-maverick | kitab_muharaf | 10 | 0 | 0.6477 | 0.6402 | 0.5886 | 0.3598 | 0.9394 | 0.9118 | 0.0 | 0.52 | 6380.2 | 0.001204 |
| mistral-small | kitab_muharaf | 8 | 2 | 0.9158 | 0.9022 | 0.7446 | 0.0978 | 1.0145 | 1.0 | 0.0 | 0.125 | 3175.9 | 0.000368 |
| pixtral-12b | kitab_muharaf | 7 | 3 | 0.9188 | 0.911 | 0.6385 | 0.089 | 0.9853 | 0.9306 | 0.0 | 0.4354 | 2651.3 | 0.000204 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
