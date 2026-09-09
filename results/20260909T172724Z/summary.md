# Benchmark summary — `20260909T172724Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-ft-bf16 | rasam2 | 20 | 0 | 0.4939 | 0.4912 | 0.4729 | 0.5088 | 0.8889 | 0.8703 | 0.0 | 0.6 | 4413.2 | 0.0 |
| qwen3-vl-4b-bf16 | rasam2 | 20 | 0 | 1.0342 | 1.0228 | 0.5622 | 0.0 | 1.0214 | 0.9795 | 0.0 | 0.05 | 8530.1 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
