# Benchmark summary — `20260908T104458Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash | kitab_muharaf | 66 | 30 | 0.7501 | 0.7451 | 0.7396 | 0.2549 | 0.8857 | 0.8611 | 0.0455 | 0.6957 | 9145.3 | 0.065951 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
