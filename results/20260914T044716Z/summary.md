# Benchmark summary — `20260914T044716Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-visft-ft-bf16 | kitab_hindawi | 180 | 0 | 0.8108 | 0.7958 | 0.7907 | 0.2042 | 1.0695 | 0.9633 | 0.0 | 0.2844 | 41906.6 | 0.0 |
| qwen3-vl-4b-bf16 | kitab_hindawi | 180 | 0 | 0.8691 | 0.8591 | 0.8152 | 0.1409 | 1.0014 | 0.9983 | 0.0 | 0.163 | 42485.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
