# Benchmark summary — `20260909T170822Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-ft-bf16 | kitab_khatt | 20 | 0 | 0.3941 | 0.3837 | 0.3679 | 0.6163 | 0.823 | 0.7937 | 0.0 | 0.35 | 3982.1 | 0.0 |
| qwen3-vl-4b-bf16 | kitab_khatt | 20 | 0 | 0.8681 | 0.8691 | 0.4824 | 0.1309 | 0.9835 | 0.9409 | 0.0 | 0.025 | 8001.8 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
