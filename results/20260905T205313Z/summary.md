# Benchmark summary — `20260905T205313Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2-vl-2b | muharaf_lines | 9 | 1 | 0.8082 | 0.7946 | 0.7017 | 0.2054 | 0.9773 | 0.9773 | 0.1111 | 0.3374 | 2160.3 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
