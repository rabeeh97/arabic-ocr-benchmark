# Benchmark summary — `20260831T230940Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-llama-4-maverick | kitab_khatt | 12 | 0 | 0.6006 | 0.5862 | 0.482 | 0.4138 | 0.9726 | 0.9404 | 0.0 | 0.2604 | 9372.7 | 0.002722 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
