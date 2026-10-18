# Benchmark summary — `20260912T223736Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-ft-bf16 | rasam1 | 199 | 1 | 0.5201 | 0.5195 | 0.5066 | 0.4805 | 0.9315 | 0.887 | 0.0151 | 0.6734 | 2591.1 | 0.0 |
| paligemma2-3b-bf16 | rasam1 | 200 | 0 | 0.7555 | 0.7513 | 0.7258 | 0.2487 | 0.9809 | 0.9637 | 0.005 | 0.85 | 1600.9 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
