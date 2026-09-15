# Benchmark summary — `20260915T093041Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-visft-ft-bf16 | kitab_khatt | 199 | 1 | 0.4048 | 0.3921 | 0.3801 | 0.6079 | 0.8146 | 0.7835 | 0.005 | 0.696 | 2468.0 | 0.0 |
| paligemma2-3b-bf16 | kitab_khatt | 200 | 0 | 0.9499 | 0.9497 | 0.9496 | 0.0503 | 1.0103 | 1.0 | 0.0 | 0.935 | 634.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
