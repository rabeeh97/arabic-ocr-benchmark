# Benchmark summary — `20260914T233521Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-visft-ft | muharaf_lines | 200 | 0 | 0.29 | 0.2718 | 0.2542 | 0.7282 | 0.6234 | 0.598 | 0.05 | 0.6625 | 2019.2 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
