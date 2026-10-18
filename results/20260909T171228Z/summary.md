# Benchmark summary — `20260909T171228Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-ft-bf16 | nakba_lines | 20 | 0 | 0.3042 | 0.2861 | 0.2749 | 0.7139 | 0.7421 | 0.7231 | 0.25 | 0.69 | 3228.7 | 0.0 |
| qwen3-vl-4b-bf16 | nakba_lines | 20 | 0 | 0.9043 | 0.8899 | 0.4928 | 0.1101 | 1.0158 | 0.9279 | 0.2 | 0.3543 | 7465.9 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
