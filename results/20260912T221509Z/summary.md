# Benchmark summary — `20260912T221509Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-ft-bf16 | kitab_khatt | 180 | 0 | 0.3902 | 0.3774 | 0.3663 | 0.6226 | 0.7859 | 0.7644 | 0.0 | 0.6693 | 2323.2 | 0.0 |
| paligemma2-3b-bf16 | kitab_khatt | 180 | 0 | 0.6744 | 0.6638 | 0.6298 | 0.3362 | 0.941 | 0.9148 | 0.0 | 0.8267 | 1798.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
