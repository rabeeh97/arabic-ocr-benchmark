# Benchmark summary — `20260912T101014Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_onlinekhatt | 15 | 0 | 0.1051 | 0.1009 | 0.0839 | 0.8991 | 0.4098 | 0.3846 | 0.0667 | 0.8444 | 1930.6 | 0.001859 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
