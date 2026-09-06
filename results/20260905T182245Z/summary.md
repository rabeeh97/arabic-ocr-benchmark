# Benchmark summary — `20260905T182245Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash | kitab_khatt | 6 | 30 | 0.6941 | 0.6941 | 0.6923 | 0.3059 | 0.806 | 0.806 | 0.0 | 0.8333 | 10074.4 | 0.00588 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
