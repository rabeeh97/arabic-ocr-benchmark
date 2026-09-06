# Benchmark summary — `20260905T131733Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-vl-3b | kitab_khatt | 10 | 0 | 0.5501 | 0.5373 | 0.4006 | 0.4627 | 0.8099 | 0.784 | 0.0 | 0.4 | 44185.8 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
