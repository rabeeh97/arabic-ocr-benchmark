# Benchmark summary — `20260904T165925Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm-5.3-flash-paid | kitab_muharaf | 2 | 0 | 0.5385 | 0.5192 | 0.5192 | 0.4808 | 0.9 | 0.9 | 0.0 | 1.0 | 18870.4 | 0.000937 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
