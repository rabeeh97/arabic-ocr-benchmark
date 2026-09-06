# Benchmark summary — `20260901T090841Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.3 | kitab_khatt | 10 | 0 | 0.5071 | 0.4955 | 0.4112 | 0.5045 | 0.9754 | 0.844 | 0.0 | 0.44 | 89873.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
