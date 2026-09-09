# Benchmark summary — `20260909T171608Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-ft-bf16 | rasam1 | 20 | 0 | 0.5207 | 0.5192 | 0.4996 | 0.4808 | 0.8796 | 0.8638 | 0.0 | 0.25 | 4344.7 | 0.0 |
| qwen3-vl-4b-bf16 | rasam1 | 20 | 0 | 1.1982 | 1.1871 | 0.6297 | 0.0 | 1.0657 | 0.9766 | 0.0 | 0.1291 | 11157.2 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
