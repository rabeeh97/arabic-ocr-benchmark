# Benchmark summary — `20260909T184119Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-ft-bf16 | nakba_lines | 200 | 0 | 0.3257 | 0.3014 | 0.292 | 0.6986 | 0.7781 | 0.7503 | 0.11 | 0.72 | 3131.3 | 0.0 |
| qwen3-vl-4b-bf16 | nakba_lines | 200 | 0 | 0.7682 | 0.7464 | 0.4578 | 0.2536 | 0.9722 | 0.935 | 0.095 | 0.3228 | 5752.9 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
