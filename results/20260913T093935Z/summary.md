# Benchmark summary — `20260913T093935Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paligemma2-3b-bf16 | kitab_hindawi | 92 | 2 | 0.2718 | 0.2297 | 0.1942 | 0.7703 | 0.5623 | 0.5424 | 0.0 | 0.316 | 27290.0 | 0.0 |
| exp3090-paligemma2-3b-ft-bf16 | kitab_hindawi | 178 | 2 | 0.4323 | 0.395 | 0.3662 | 0.605 | 0.7128 | 0.7031 | 0.0 | 0.3372 | 23504.1 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
