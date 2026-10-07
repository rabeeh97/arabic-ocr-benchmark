# Benchmark summary — `20260915T122718Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | kitab_khatt | 20 | 0 | 0.343 | 0.3303 | 0.2617 | 0.6697 | 0.7654 | 0.7266 | 0.0 | 0.375 | 8154.1 | 0.002507 |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 20 | 0 | 0.3858 | 0.3736 | 0.3027 | 0.6264 | 0.7942 | 0.7366 | 0.0 | 0.6 | 12205.1 | 0.005355 |
| openrouter-gemma-4-26b-free | kitab_khatt | 5 | 7 | 0.4036 | 0.3971 | 0.3971 | 0.6029 | 0.8308 | 0.806 | 0.0 | 1.0 | 4945.7 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_khatt | 1 | 11 | 0.45 | 0.4375 | 0.4026 | 0.5625 | 0.8 | 0.75 | 0.0 | 0.0 | 3537.1 | 0.0 |
| openrouter-llama-4-maverick | kitab_khatt | 20 | 0 | 0.5763 | 0.563 | 0.4523 | 0.437 | 0.9753 | 0.8943 | 0.0 | 0.2062 | 6543.0 | 0.00456 |
| glm-5.3-flash-paid | kitab_khatt | 20 | 0 | 0.8112 | 0.8073 | 0.7985 | 0.1927 | 0.9877 | 0.9524 | 0.0 | 0.7 | 66414.8 | 0.028418 |
| mistral-small | kitab_khatt | 20 | 0 | 0.8228 | 0.8165 | 0.7065 | 0.1835 | 0.9918 | 0.9797 | 0.0 | 0.15 | 7148.8 | 0.001813 |
| mistral-medium | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
