# Benchmark summary — `20260903T154131Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | kitab_hindawi | 20 | 0 | 0.5758 | 0.5543 | 0.5258 | 0.7059 | 0.0 | 0.4268 | 8282.4 | 0.011455 |
| openrouter-minimax-m3-free | kitab_hindawi | 20 | 0 | 0.7126 | 0.6978 | 0.6811 | 0.8455 | 0.0 | 0.1973 | 7071.7 | 0.0 |
| pixtral-12b | kitab_hindawi | 20 | 0 | 0.7327 | 0.7189 | 0.7039 | 0.8241 | 0.0 | 0.1693 | 1901.2 | 0.002615 |
| mistral-medium | kitab_hindawi | 13 | 7 | 0.7408 | 0.7279 | 0.7091 | 0.8369 | 0.0 | 0.0592 | 2978.7 | 0.006487 |
| gemini-flash-lite | kitab_hindawi | 20 | 0 | 0.7572 | 0.7471 | 0.7374 | 0.7699 | 0.0 | 0.2904 | 14005.3 | 0.003003 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 20 | 0 | 0.7727 | 0.7625 | 0.7474 | 0.8265 | 0.0 | 0.2339 | 4193.3 | 0.011679 |
| mistral-small | kitab_hindawi | 10 | 10 | 0.8589 | 0.8527 | 0.8469 | 0.901 | 0.0 | 0.0645 | 1465.6 | 0.001799 |
| openrouter-llama-4-maverick | kitab_hindawi | 20 | 0 | 0.8909 | 0.8866 | 0.8809 | 0.9223 | 0.0 | 0.1545 | 2954.6 | 0.005204 |
| openrouter-gemma-4-31b-free | kitab_hindawi | 4 | 16 | 0.9042 | 0.9031 | 0.9017 | 0.9176 | 0.0 | 0.3 | 71559.7 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_hindawi | 3 | 17 | 0.9594 | 0.9561 | 0.9555 | 0.963 | 0.0 | 0.0 | 3103.3 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_historicalbooks | 10 | 0 | 0.3271 | 0.3124 | 0.2545 | 0.6967 | 0.0 | 0.3482 | 9791.0 | 0.004796 |
| openrouter-minimax-m3-free | kitab_historicalbooks | 10 | 0 | 0.6501 | 0.6327 | 0.6037 | 1.0871 | 0.0 | 0.4229 | 10762.8 | 0.0 |
| gemini-flash-lite | kitab_historicalbooks | 10 | 0 | 0.7218 | 0.7156 | 0.7059 | 0.8243 | 0.0 | 0.5 | 9730.1 | 0.001332 |
| mistral-medium | kitab_historicalbooks | 10 | 0 | 0.792 | 0.7832 | 0.7517 | 0.9658 | 0.0 | 0.5095 | 3883.3 | 0.00477 |
| openrouter-qwen2.5-vl-72b | kitab_historicalbooks | 10 | 0 | 0.864 | 0.8607 | 0.8576 | 0.9393 | 0.0 | 0.5615 | 4177.7 | 0.005535 |
| pixtral-12b | kitab_historicalbooks | 10 | 0 | 0.8822 | 0.8689 | 0.8414 | 1.0824 | 0.0 | 0.1108 | 4343.2 | 0.001366 |
| openrouter-llama-4-maverick | kitab_historicalbooks | 10 | 0 | 0.8956 | 0.8946 | 0.8914 | 0.9767 | 0.0 | 0.25 | 4615.9 | 0.002397 |
| mistral-small | kitab_historicalbooks | 10 | 0 | 0.913 | 0.9099 | 0.914 | 0.9984 | 0.0 | 0.2198 | 3347.9 | 0.001816 |
| openrouter-gemma-4-26b-free | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
