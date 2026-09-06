# Benchmark summary — `20260902T082546Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_muharaf | 10 | 0 | 0.5645 | 0.5463 | 0.4449 | 0.4537 | 0.914 | 0.8333 | 0.1 | 0.7133 | 4069.2 | 0.001242 |
| openrouter-qwen3-vl-235b | kitab_muharaf | 10 | 0 | 0.605 | 0.5927 | 0.4862 | 0.4073 | 0.9462 | 0.9167 | 0.1 | 0.5654 | 2134.9 | 0.001018 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 10 | 0 | 0.6705 | 0.6602 | 0.628 | 0.3398 | 0.9247 | 0.9053 | 0.1 | 0.5226 | 2162.1 | 0.001224 |
| openrouter-llama-4-maverick | kitab_muharaf | 10 | 0 | 0.7476 | 0.7413 | 0.5965 | 0.2587 | 1.0 | 0.9588 | 0.1 | 0.4226 | 1710.4 | 0.001342 |
| gemini-flash | kitab_muharaf | 7 | 3 | 0.8388 | 0.8306 | 0.8263 | 0.1694 | 0.9403 | 0.9403 | 0.1429 | 0.75 | 8510.7 | 0.006778 |
| mistral-small | kitab_muharaf | 10 | 0 | 0.9403 | 0.9382 | 0.8248 | 0.0618 | 1.086 | 0.9902 | 0.0 | 0.0507 | 1087.6 | 0.00046 |
| pixtral-12b | kitab_muharaf | 10 | 0 | 1.1464 | 1.1178 | 0.7618 | 0.0 | 1.1183 | 0.9905 | 0.1 | 0.2329 | 1301.3 | 0.000296 |
| gemini-flash-lite | muharaf_lines | 10 | 0 | 0.4983 | 0.4799 | 0.4647 | 0.5201 | 0.9545 | 0.8468 | 0.0 | 0.5 | 1470.6 | 0.00124 |
| openrouter-qwen2.5-vl-72b | muharaf_lines | 10 | 0 | 0.6655 | 0.6632 | 0.6502 | 0.3368 | 1.0455 | 0.9914 | 0.0 | 0.4 | 2298.7 | 0.001091 |
| openrouter-llama-4-maverick | muharaf_lines | 10 | 0 | 0.723 | 0.7225 | 0.6873 | 0.2775 | 0.9909 | 0.9732 | 0.0 | 0.3316 | 2412.4 | 0.001133 |
| openrouter-qwen3-vl-235b | muharaf_lines | 10 | 0 | 0.7979 | 0.7906 | 0.6396 | 0.2094 | 1.0636 | 1.0 | 0.0 | 0.4141 | 1849.5 | 0.00108 |
| mistral-small | muharaf_lines | 10 | 0 | 0.8432 | 0.8377 | 0.7968 | 0.1623 | 1.0 | 1.0 | 0.0 | 0.0578 | 1254.2 | 0.000426 |
| gemini-flash | muharaf_lines | 8 | 2 | 0.8854 | 0.8852 | 0.8856 | 0.1148 | 0.9891 | 0.9891 | 0.0 | 0.375 | 6873.4 | 0.007802 |
| pixtral-12b | muharaf_lines | 10 | 0 | 1.1794 | 1.171 | 0.7686 | 0.0 | 1.0909 | 1.0 | 0.0 | 0.1372 | 1051.5 | 0.000277 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
