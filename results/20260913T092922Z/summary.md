# Benchmark summary — `20260913T092922Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-ft-bf16 | rasam2 | 149 | 7 | 0.5509 | 0.547 | 0.5281 | 0.453 | 0.9448 | 0.8907 | 0.0 | 0.604 | 2315.0 | 0.0 |
| paligemma2-3b-bf16 | rasam2 | 149 | 7 | 0.7994 | 0.7945 | 0.7428 | 0.2055 | 0.9917 | 0.9646 | 0.0 | 0.8255 | 1546.7 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
