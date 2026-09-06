# Benchmark summary — `20260903T165059Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-minimax-m3-free | kitab_historyar | 20 | 0 | 0.681 | 0.6734 | 0.642 | 0.3266 | 1.1127 | 0.9595 | 0.0 | 0.3727 | 10495.4 | 0.0 |
| openrouter-minimax-m3-free | kitab_onlinekhatt | 20 | 0 | 0.6011 | 0.5849 | 0.5124 | 0.4151 | 1.0123 | 0.8632 | 0.0 | 0.55 | 4990.6 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
