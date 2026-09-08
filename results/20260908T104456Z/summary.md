# Benchmark summary — `20260908T104456Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixtral-12b | kitab_hindawi | 20 | 0 | 0.7773 | 0.7656 | 0.7489 | 0.2344 | 0.8683 | 0.8648 | 0.0 | 0.1561 | 3563.2 | 0.002625 |
| nvidia-llama-3.2-11b-vl | kitab_hindawi | 20 | 0 | 0.8179 | 0.8071 | 0.784 | 0.1929 | 0.9395 | 0.9387 | 0.0 | 0.1232 | 36955.1 | 0.0 |
| openrouter-nemotron-omni-free | kitab_hindawi | 15 | 5 | 32.5717 | 30.3958 | 29.6122 | 0.0 | 32.3394 | 0.996 | 0.0 | 0.0832 | 358924.8 | 0.0 |
| openrouter-llama-4-maverick | kitab_hindawi | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_hindawi | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_hindawi | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-nemotron-omni-free | kitab_historicalbooks | 6 | 4 | 78.6072 | 77.2736 | 56.7598 | 0.0 | 43.1989 | 0.9996 | 0.0 | 0.1822 | 484160.9 | 0.0 |
| pixtral-12b | kitab_historyar | 20 | 0 | 1.1368 | 1.1198 | 0.811 | 0.0 | 1.128 | 1.0 | 0.0 | 0.2027 | 4852.4 | 0.001186 |
| nvidia-llama-3.2-11b-vl | kitab_historyar | 20 | 0 | 2.3355 | 2.3241 | 1.5655 | 0.0 | 1.9909 | 0.9955 | 0.05 | 0.2083 | 52218.6 | 0.0 |
| openrouter-nemotron-omni-free | kitab_historyar | 8 | 12 | 1823.8128 | 1310.1283 | 1664.2765 | 0.0 | 1569.7903 | 1.0 | 0.0 | 0.3568 | 565977.6 | 0.0 |
| openrouter-llama-4-maverick | kitab_historyar | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_historyar | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_historyar | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_historyar | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_khatt | 3 | 9 | 0.5163 | 0.5099 | 0.5099 | 0.4901 | 0.8421 | 0.8 | 0.0 | 1.0 | 4394.3 | 0.0 |
| pixtral-12b | kitab_khatt | 20 | 0 | 1.1724 | 1.1686 | 0.7453 | 0.0 | 1.0465 | 0.9712 | 0.0 | 0.2 | 1349.3 | 0.001374 |
| nvidia-llama-3.2-11b-vl | kitab_khatt | 20 | 0 | 1.3943 | 1.391 | 1.1712 | 0.0 | 1.8347 | 0.9737 | 0.0 | 0.4074 | 12774.0 | 0.0 |
| openrouter-nemotron-omni-free | kitab_khatt | 15 | 5 | 765.6089 | 715.1206 | 680.5449 | 0.0 | 659.994 | 0.9999 | 0.0 | 0.2667 | 370400.2 | 0.0 |
| mistral-medium | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| mistral-small | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-dots3-note-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-llama-4-maverick | kitab_khatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_khatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_khatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| pixtral-12b | kitab_muharaf | 15 | 5 | 1.6682 | 1.7944 | 0.9733 | 0.0 | 1.4375 | 1.0 | 0.0 | 0.013 | 1420.3 | 0.000439 |
| nvidia-llama-3.2-11b-vl | kitab_muharaf | 20 | 0 | 2.1561 | 2.1456 | 1.4588 | 0.0 | 1.872 | 0.995 | 0.0 | 0.1732 | 16636.5 | 0.0 |
| openrouter-nemotron-omni-free | kitab_muharaf | 8 | 12 | 745.347 | 707.7172 | 686.6987 | 0.0 | 674.9722 | 0.9999 | 0.0 | 0.2501 | 395519.2 | 0.0 |
| openrouter-llama-4-maverick | kitab_muharaf | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_muharaf | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_muharaf | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_onlinekhatt | 20 | 0 | 0.5885 | 0.5729 | 0.498 | 0.4271 | 0.9706 | 0.9167 | 0.0 | 0.3558 | 16262.2 | 0.0 |
| pixtral-12b | kitab_onlinekhatt | 17 | 3 | 1.2563 | 1.2394 | 0.8391 | 0.0 | 1.2056 | 0.9923 | 0.0 | 0.0588 | 1469.7 | 0.000841 |
| openrouter-llama-4-maverick | kitab_onlinekhatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_onlinekhatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_onlinekhatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_onlinekhatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
