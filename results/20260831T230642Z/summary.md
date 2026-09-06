# Benchmark summary — `20260831T230642Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mistral-small | kitab_khatt | 12 | 0 | 0.7989 | 0.7931 | 0.6883 | 0.2069 | 0.9795 | 0.9728 | 0.0 | 0.0833 | 10913.6 | 0.00111 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
