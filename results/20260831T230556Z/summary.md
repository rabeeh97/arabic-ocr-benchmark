# Benchmark summary — `20260831T230556Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixtral-12b | kitab_khatt | 12 | 0 | 1.2394 | 1.2399 | 0.8139 | 0.0 | 1.1027 | 0.9938 | 0.0 | 0.0 | 10912.9 | 0.000758 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
