# Benchmark summary — `20260912T225147Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-ft-bf16 | rasam2 | 196 | 4 | 0.5404 | 0.538 | 0.5215 | 0.462 | 0.9142 | 0.8759 | 0.0051 | 0.6224 | 2350.2 | 0.0 |
| paligemma2-3b-bf16 | rasam2 | 196 | 4 | 0.7889 | 0.7829 | 0.7404 | 0.2171 | 0.9783 | 0.9633 | 0.0 | 0.8316 | 1490.3 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
