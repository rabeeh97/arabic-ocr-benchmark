# Benchmark summary — `20260902T082546Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_muharaf | 10 | 0 | 0.5645 | 0.5463 | 0.4449 | 0.914 | 0.1 | 0.7133 | 4069.2 | 0.001242 |
| openrouter-qwen3-vl-235b | kitab_muharaf | 10 | 0 | 0.605 | 0.5927 | 0.4862 | 0.9462 | 0.1 | 0.5654 | 2134.9 | 0.001018 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 10 | 0 | 0.6705 | 0.6602 | 0.628 | 0.9247 | 0.1 | 0.5226 | 2162.1 | 0.001224 |
| openrouter-llama-4-maverick | kitab_muharaf | 10 | 0 | 0.7476 | 0.7413 | 0.5965 | 1.0 | 0.1 | 0.4226 | 1710.4 | 0.001342 |
| gemini-flash | kitab_muharaf | 10 | 3 | 0.8863 | 0.8803 | 0.878 | 0.957 | 0.1 | 0.725 | 6051.5 | 0.006778 |
| mistral-small | kitab_muharaf | 10 | 0 | 0.9403 | 0.9382 | 0.8248 | 1.086 | 0.0 | 0.0507 | 1087.6 | 0.00046 |
| pixtral-12b | kitab_muharaf | 10 | 0 | 1.1464 | 1.1178 | 0.7618 | 1.1183 | 0.1 | 0.2329 | 1301.3 | 0.000296 |
| gemini-flash-lite | muharaf_lines | 10 | 0 | 0.4983 | 0.4799 | 0.4647 | 0.9545 | 0.0 | 0.5 | 1470.6 | 0.00124 |
| openrouter-qwen2.5-vl-72b | muharaf_lines | 10 | 0 | 0.6655 | 0.6632 | 0.6502 | 1.0455 | 0.0 | 0.4 | 2298.7 | 0.001091 |
| openrouter-llama-4-maverick | muharaf_lines | 10 | 0 | 0.723 | 0.7225 | 0.6873 | 0.9909 | 0.0 | 0.3316 | 2412.4 | 0.001133 |
| openrouter-qwen3-vl-235b | muharaf_lines | 10 | 0 | 0.7979 | 0.7906 | 0.6396 | 1.0636 | 0.0 | 0.4141 | 1849.5 | 0.00108 |
| mistral-small | muharaf_lines | 10 | 0 | 0.8432 | 0.8377 | 0.7968 | 1.0 | 0.0 | 0.0578 | 1254.2 | 0.000426 |
| gemini-flash | muharaf_lines | 10 | 2 | 0.9042 | 0.904 | 0.9046 | 0.9909 | 0.0 | 0.5 | 5552.3 | 0.007802 |
| pixtral-12b | muharaf_lines | 10 | 0 | 1.1794 | 1.171 | 0.7686 | 1.0909 | 0.0 | 0.1372 | 1051.5 | 0.000277 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
