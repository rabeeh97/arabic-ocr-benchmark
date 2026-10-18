# Benchmark summary — `20260909T172833Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-ft-bf16 | kitab_hindawi | 20 | 0 | 0.8537 | 0.8435 | 0.833 | 0.1565 | 0.9879 | 0.9876 | 0.0 | 0.2006 | 14166.4 | 0.0 |
| qwen3-vl-4b-bf16 | kitab_hindawi | 20 | 0 | 0.9083 | 0.9029 | 0.8989 | 0.0971 | 0.9993 | 0.9993 | 0.0 | 0.1545 | 14954.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
