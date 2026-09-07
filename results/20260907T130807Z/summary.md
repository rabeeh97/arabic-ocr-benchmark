# Benchmark summary — `20260907T130807Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-minimax-m3-free | kitab_hindawi | 40 | 0 | 0.6896 | 0.67 | 0.6524 | 0.33 | 0.8467 | 0.8404 | 0.0 | 0.1891 | 7005.8 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_hindawi | 5 | 35 | 0.774 | 0.7645 | 0.7423 | 0.2355 | 0.9017 | 0.8991 | 0.0 | 0.2758 | 5255.2 | 0.002414 |
| openrouter-llama-4-maverick | kitab_hindawi | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_historyar | 40 | 0 | 0.6037 | 0.5944 | 0.4846 | 0.4056 | 1.0329 | 0.9286 | 0.0 | 0.3692 | 8607.5 | 0.0 |
| openrouter-llama-4-maverick | kitab_historyar | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_historyar | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_historyar | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_khatt | 2 | 10 | 0.2677 | 0.2619 | 0.2619 | 0.7381 | 0.8148 | 0.8148 | 0.0 | 1.0 | 5772.9 | 0.0 |
| openrouter-minimax-m3-free | kitab_khatt | 40 | 0 | 0.5269 | 0.5147 | 0.4726 | 0.4853 | 0.9203 | 0.8644 | 0.0 | 0.75 | 3541.9 | 0.0 |
| pixtral-12b | kitab_khatt | 30 | 6 | 1.0928 | 1.0855 | 0.759 | 0.0 | 1.0502 | 0.9865 | 0.0 | 0.0679 | 1310.2 | 0.001879 |
| openrouter-qwen3-vl-235b | kitab_khatt | 20 | 20 | 3.2109 | 3.2308 | 3.1841 | 0.0 | 0.8565 | 0.7036 | 0.0 | 0.65 | 2782.2 | 0.004035 |
| mistral-medium | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| mistral-small | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-llama-4-maverick | kitab_khatt | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_muharaf | 20 | 20 | 0.4584 | 0.4439 | 0.3893 | 0.5561 | 0.8188 | 0.7871 | 0.05 | 0.6 | 2189.5 | 0.001551 |
| openrouter-minimax-m3-free | kitab_muharaf | 37 | 3 | 0.8322 | 0.8249 | 0.7982 | 0.1751 | 1.1782 | 0.9848 | 0.0 | 0.6216 | 3429.8 | 0.0 |
| openrouter-llama-4-maverick | kitab_muharaf | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_onlinekhatt | 40 | 0 | 0.6126 | 0.6019 | 0.4892 | 0.3981 | 0.9525 | 0.8515 | 0.0 | 0.5011 | 2889.9 | 0.0 |
| openrouter-llama-4-maverick | kitab_onlinekhatt | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_onlinekhatt | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_onlinekhatt | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
