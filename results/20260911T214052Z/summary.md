# Benchmark summary — `20260911T214052Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-baseer-nakba-ft-bf16 | rasam1 | 200 | 0 | 0.4571 | 0.4543 | 0.4393 | 0.5457 | 0.8451 | 0.7967 | 0.005 | 0.6042 | 3507.6 | 0.0 |
| baseer-nakba-bf16 | rasam1 | 200 | 0 | 0.5313 | 0.511 | 0.5015 | 0.489 | 0.9378 | 0.8736 | 0.01 | 0.875 | 3470.2 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
