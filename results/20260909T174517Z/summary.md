# Benchmark summary — `20260909T174517Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-ft-bf16 | kitab_khatt | 180 | 0 | 0.3002 | 0.2883 | 0.2749 | 0.7117 | 0.6969 | 0.6764 | 0.0111 | 0.5156 | 3351.7 | 0.0 |
| qwen3-vl-4b-bf16 | kitab_khatt | 180 | 0 | 0.7711 | 0.7636 | 0.4177 | 0.2364 | 0.9264 | 0.8748 | 0.0 | 0.1327 | 6945.2 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
