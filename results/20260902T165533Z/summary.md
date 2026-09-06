# Benchmark summary — `20260902T165533Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-gemma-4-31b-free | kitab_muharaf | 2 | 8 | 0.6479 | 0.6197 | 0.6197 | 0.3803 | 1.0 | 0.8125 | 0.0 | 1.0 | 6117.4 | 0.0 |
| mistral-medium | kitab_muharaf | 10 | 0 | 0.6936 | 0.6834 | 0.6713 | 0.3166 | 1.0753 | 0.9615 | 0.1 | 0.6 | 1248.5 | 0.000991 |
| nvidia-llama-3.2-90b-vl | kitab_muharaf | 10 | 0 | 1.0289 | 1.027 | 0.9114 | 0.0 | 1.2366 | 0.9829 | 0.1 | 0.5057 | 134676.4 | 0.0 |
| mistral-medium | muharaf_lines | 10 | 0 | 0.6986 | 0.6928 | 0.6767 | 0.3072 | 0.9818 | 0.9818 | 0.0 | 0.5 | 1200.5 | 0.000874 |
| nvidia-llama-3.2-90b-vl | muharaf_lines | 10 | 0 | 1.8571 | 1.8464 | 1.8357 | 0.0 | 1.3455 | 1.0 | 0.0 | 0.5056 | 137994.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
