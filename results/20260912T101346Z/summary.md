# Benchmark summary — `20260912T101346Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite-640tok | kitab_hindawi | 15 | 0 | 0.4575 | 0.4343 | 0.4283 | 0.5657 | 0.5437 | 0.5417 | 0.0 | 0.4437 | 2584.2 | 0.002945 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
