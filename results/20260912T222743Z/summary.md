# Benchmark summary — `20260912T222743Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-ft-bf16 | nakba_lines | 200 | 0 | 0.3406 | 0.318 | 0.3118 | 0.682 | 0.7657 | 0.7404 | 0.11 | 0.82 | 2109.6 | 0.0 |
| paligemma2-3b-bf16 | nakba_lines | 199 | 1 | 0.9134 | 0.9068 | 0.8928 | 0.0932 | 0.9932 | 0.9834 | 0.0905 | 0.9397 | 814.5 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
