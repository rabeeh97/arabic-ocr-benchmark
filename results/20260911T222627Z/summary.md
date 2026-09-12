# Benchmark summary — `20260911T222627Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseer-nakba-bf16 | kitab_hindawi | 199 | 1 | 0.8912 | 0.8858 | 0.8804 | 0.1142 | 0.9335 | 0.9323 | 0.0 | 0.0395 | 5171.9 | 0.0 |
| exp3090-baseer-nakba-ft-bf16 | kitab_hindawi | 199 | 1 | 0.9008 | 0.896 | 0.8913 | 0.104 | 0.947 | 0.9464 | 0.0 | 0.111 | 4838.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
