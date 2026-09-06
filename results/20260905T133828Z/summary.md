# Benchmark summary — `20260905T133828Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.3 | kitab_historicalbooks | 10 | 0 | 0.6791 | 0.6607 | 0.6327 | 0.3393 | 0.9533 | 0.9387 | 0.0 | 0.4304 | 10278.2 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
