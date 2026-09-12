# Benchmark summary — `20260911T230005Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseer-nakba-bf16 | kitab_historicalbooks | 10 | 0 | 0.8454 | 0.8389 | 0.8331 | 0.1611 | 0.9331 | 0.9217 | 0.0 | 0.4118 | 5323.7 | 0.0 |
| exp3090-baseer-nakba-ft-bf16 | kitab_historicalbooks | 10 | 0 | 0.8728 | 0.8721 | 0.8672 | 0.1279 | 0.944 | 0.9411 | 0.0 | 0.3 | 3576.5 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
