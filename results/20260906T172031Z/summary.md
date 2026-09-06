# Benchmark summary — `20260906T172031Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nakba-baseline-qwen3vl | muharaf_lines | 20 | 0 | 0.7642 | 0.6443 | 0.6317 | 0.3557 | 0.9859 | 0.9333 | 0.05 | 0.65 | 11399.7 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
