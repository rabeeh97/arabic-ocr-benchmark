# Benchmark summary — `20260831T185739Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.2 | muharaf_lines | 1319 | 15 | 0.7981 | 0.779 | 0.7097 | 0.221 | 1.0262 | 0.9569 | 0.0099 | 0.5603 | 2131.1 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
