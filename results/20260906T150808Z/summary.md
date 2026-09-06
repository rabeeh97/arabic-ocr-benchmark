# Benchmark summary — `20260906T150808Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trocr-large-arabic | muharaf_lines | 20 | 0 | 0.8302 | 0.8074 | 0.8085 | 0.1926 | 1.1032 | 1.0 | 0.0 | 0.4 | 1252.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
