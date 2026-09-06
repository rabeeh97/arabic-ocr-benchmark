# Benchmark summary — `20260901T160440Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | kitab_khatt | 10 | 0 | 0.2826 | 0.2695 | 0.1853 | 0.7305 | 0.5868 | 0.5772 | 0.0 | 0.4 | 1646.8 | 0.001197 |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 10 | 0 | 0.292 | 0.2759 | 0.238 | 0.7241 | 0.6942 | 0.6512 | 0.0 | 0.5 | 3212.9 | 0.002408 |
| gemini-flash-lite | kitab_khatt | 10 | 0 | 0.3878 | 0.3812 | 0.3275 | 0.6188 | 0.6364 | 0.562 | 0.0 | 0.7 | 11663.7 | 0.001274 |
| openrouter-llama-4-maverick | kitab_khatt | 10 | 0 | 0.5761 | 0.5662 | 0.3994 | 0.4338 | 0.9835 | 0.8623 | 0.0 | 0.1 | 2909.4 | 0.002283 |
| mistral-small | kitab_khatt | 10 | 0 | 0.8179 | 0.815 | 0.6981 | 0.185 | 1.0 | 0.9837 | 0.0 | 0.2 | 1411.6 | 0.000833 |
| pixtral-12b | kitab_khatt | 10 | 0 | 1.2543 | 1.2616 | 0.77 | 0.0 | 1.0992 | 1.0 | 0.0 | 0.1038 | 2541.0 | 0.000564 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
