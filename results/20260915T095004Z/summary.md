# Benchmark summary — `20260915T095004Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-visft-ft-bf16 | rasam1 | 200 | 0 | 0.5456 | 0.5451 | 0.529 | 0.4549 | 0.9391 | 0.8855 | 0.01 | 0.6103 | 2758.4 | 0.0 |
| paligemma2-3b-bf16 | rasam1 | 200 | 0 | 0.9836 | 0.9836 | 0.9837 | 0.0164 | 1.0075 | 1.0 | 0.0 | 0.925 | 483.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
