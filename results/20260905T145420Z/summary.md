# Benchmark summary — `20260905T145420Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-pro | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash | kitab_khatt | 3 | 29 | 0.8624 | 0.8551 | 0.8551 | 0.1449 | 0.9111 | 0.9111 | 0.0 | 0.6667 | 5981.7 | 0.002887 |
| gemini-pro | kitab_khatt | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
