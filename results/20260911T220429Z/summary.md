# Benchmark summary — `20260911T220429Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-baseer-nakba-ft-bf16 | rasam2 | 195 | 5 | 0.5061 | 0.5016 | 0.4722 | 0.4984 | 0.8933 | 0.8328 | 0.0 | 0.5692 | 3402.3 | 0.0 |
| baseer-nakba-bf16 | rasam2 | 196 | 4 | 0.5184 | 0.4949 | 0.487 | 0.5051 | 0.9538 | 0.8769 | 0.0051 | 0.8929 | 3258.8 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
