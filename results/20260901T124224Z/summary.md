# Benchmark summary — `20260901T124224Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.3 | kitab_muharaf | 10 | 0 | 0.7071 | 0.6931 | 0.5689 | 0.3069 | 1.0323 | 0.9697 | 0.0 | 0.42 | 104982.5 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
