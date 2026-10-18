# Benchmark summary — `20260914T040044Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-visft-ft-bf16 | rasam2 | 195 | 5 | 0.4857 | 0.4821 | 0.4651 | 0.5179 | 0.9052 | 0.8559 | 0.0 | 0.5436 | 3601.1 | 0.0 |
| qwen3-vl-4b-bf16 | rasam2 | 196 | 4 | 1.5067 | 1.4859 | 0.8834 | 0.0 | 1.2595 | 0.9871 | 0.0 | 0.0665 | 10467.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
