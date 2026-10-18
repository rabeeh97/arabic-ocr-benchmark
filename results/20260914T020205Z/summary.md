# Benchmark summary — `20260914T020205Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-visft-ft-bf16 | kitab_khatt | 180 | 0 | 0.2531 | 0.2421 | 0.2296 | 0.7579 | 0.6573 | 0.6395 | 0.0167 | 0.6378 | 3345.8 | 0.0 |
| qwen3-vl-4b-bf16 | kitab_khatt | 180 | 0 | 0.9606 | 0.9442 | 0.5309 | 0.0558 | 1.0269 | 0.8856 | 0.0 | 0.1327 | 7904.3 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
