# Benchmark summary — `20260912T145604Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixtral-12b | kitab_hindawi | 80 | 1 | 0.4118 | 0.3702 | 0.316 | 0.6298 | 0.6771 | 0.6531 | 0.0 | 0.2408 | 5254.3 | 0.013937 |
| gemini-flash | kitab_hindawi | 3 | 9 | 0.9152 | 0.909 | 0.9074 | 0.091 | 0.9136 | 0.9136 | 0.0 | 0.0 | 8172.8 | 0.005854 |
| mistral-medium | kitab_hindawi | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| mistral-small | kitab_hindawi | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_hindawi | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_hindawi | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
