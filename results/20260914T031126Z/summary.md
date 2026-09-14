# Benchmark summary — `20260914T031126Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-visft-ft-bf16 | rasam1 | 200 | 0 | 0.4318 | 0.4307 | 0.4169 | 0.5693 | 0.8566 | 0.8125 | 0.01 | 0.5877 | 3753.7 | 0.0 |
| qwen3-vl-4b-bf16 | rasam1 | 200 | 0 | 1.3695 | 1.3512 | 0.7571 | 0.0 | 1.1965 | 0.9886 | 0.01 | 0.0487 | 10965.1 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
