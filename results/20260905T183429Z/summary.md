# Benchmark summary — `20260905T183429Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | kitab_hindawi | 20 | 0 | 0.637 | 0.6195 | 0.6093 | 0.3805 | 0.7447 | 0.7434 | 0.0 | 0.4072 | 5331.3 | 0.010956 |
| nvidia-llama-3.2-11b-vl | kitab_hindawi | 20 | 0 | 0.6608 | 0.6418 | 0.6205 | 0.3582 | 0.8286 | 0.8241 | 0.0 | 0.2142 | 14920.7 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 20 | 0 | 0.7524 | 0.7407 | 0.7355 | 0.2593 | 0.7924 | 0.791 | 0.0 | 0.1688 | 3712.3 | 0.011717 |
| openrouter-minimax-m3-free | kitab_hindawi | 20 | 0 | 0.7534 | 0.738 | 0.7296 | 0.262 | 0.8872 | 0.8851 | 0.0 | 0.1988 | 5558.9 | 0.0 |
| pixtral-12b | kitab_hindawi | 20 | 0 | 0.8133 | 0.8034 | 0.7952 | 0.1966 | 0.894 | 0.8898 | 0.0 | 0.0868 | 2141.0 | 0.002566 |
| openrouter-llama-4-maverick | kitab_hindawi | 20 | 0 | 0.8939 | 0.8886 | 0.8865 | 0.1114 | 0.934 | 0.9326 | 0.0 | 0.1761 | 2271.8 | 0.005231 |
| nvidia-llama-3.2-90b-vl | kitab_hindawi | 20 | 0 | 0.9385 | 0.9367 | 0.9335 | 0.0633 | 0.9855 | 0.9851 | 0.0 | 0.025 | 86652.2 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_historicalbooks | 10 | 0 | 0.907 | 0.8961 | 0.8717 | 0.1039 | 1.1788 | 0.9781 | 0.0 | 0.2861 | 9051.6 | 0.0 |
| nvidia-llama-3.2-90b-vl | kitab_historicalbooks | 10 | 0 | 1.2398 | 1.2186 | 1.1576 | 0.0 | 1.3997 | 0.9858 | 0.0 | 0.327 | 62952.6 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_historyar | 20 | 0 | 0.4354 | 0.4239 | 0.3093 | 0.5761 | 0.7364 | 0.7168 | 0.1 | 0.6085 | 2875.3 | 0.00198 |
| openrouter-qwen2.5-vl-72b | kitab_historyar | 20 | 0 | 0.5008 | 0.4926 | 0.3947 | 0.5074 | 0.8909 | 0.8448 | 0.05 | 0.4655 | 3051.6 | 0.00434 |
| openrouter-llama-4-maverick | kitab_historyar | 20 | 0 | 0.6448 | 0.6367 | 0.4269 | 0.3633 | 0.9091 | 0.8696 | 0.0 | 0.424 | 2911.1 | 0.003129 |
| nvidia-llama-3.2-90b-vl | kitab_historyar | 20 | 0 | 1.2113 | 1.2071 | 1.1752 | 0.0 | 1.6948 | 0.9627 | 0.0 | 0.4754 | 158336.2 | 0.0 |
| pixtral-12b | kitab_historyar | 18 | 2 | 1.3437 | 1.152 | 0.7964 | 0.0 | 1.019 | 0.9907 | 0.0 | 0.1092 | 2808.4 | 0.000932 |
| nvidia-llama-3.2-11b-vl | kitab_historyar | 20 | 0 | 1.3721 | 1.3636 | 1.2021 | 0.0 | 2.2958 | 0.9839 | 0.0 | 0.3938 | 5853.1 | 0.0 |
| openrouter-minimax-m3-free | kitab_historyar | 20 | 0 | 2.0 | 1.9918 | 2.0854 | 0.0 | 2.2 | 0.9719 | 0.0 | 0.3457 | 5048.2 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 20 | 0 | 0.3787 | 0.3702 | 0.2957 | 0.6298 | 0.6951 | 0.668 | 0.05 | 0.65 | 3258.0 | 0.006422 |
| openrouter-qwen3-vl-235b | kitab_khatt | 20 | 0 | 0.4632 | 0.4545 | 0.2504 | 0.5455 | 0.7561 | 0.7381 | 0.1 | 0.3625 | 1578.8 | 0.003243 |
| openrouter-llama-4-maverick | kitab_khatt | 20 | 0 | 0.6061 | 0.5991 | 0.4051 | 0.4009 | 0.9512 | 0.8699 | 0.0 | 0.3668 | 1824.7 | 0.00501 |
| nvidia-llama-3.2-90b-vl | kitab_khatt | 20 | 0 | 0.6719 | 0.6614 | 0.4758 | 0.3386 | 0.9588 | 0.9209 | 0.0 | 0.4069 | 397811.4 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_khatt | 20 | 0 | 0.9769 | 0.9783 | 0.7174 | 0.0217 | 1.1358 | 0.9787 | 0.0 | 0.3 | 9711.2 | 0.0 |
| openrouter-minimax-m3-free | kitab_khatt | 20 | 0 | 1.0396 | 1.0292 | 1.0059 | 0.0 | 1.3539 | 0.919 | 0.0 | 0.5833 | 3533.7 | 0.0 |
| pixtral-12b | kitab_khatt | 20 | 0 | 1.2272 | 1.2286 | 0.7372 | 0.0 | 1.0614 | 0.9767 | 0.0 | 0.05 | 1383.1 | 0.001301 |
| mistral-medium | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| mistral-small | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-dots3-note-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-nemotron-omni-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 20 | 0 | 0.6415 | 0.6339 | 0.5123 | 0.3661 | 0.9512 | 0.8966 | 0.0 | 0.4378 | 1571.4 | 0.002368 |
| openrouter-qwen3-vl-235b | kitab_muharaf | 20 | 0 | 0.6525 | 0.6437 | 0.5123 | 0.3563 | 0.9939 | 0.8763 | 0.05 | 0.3577 | 1297.8 | 0.001911 |
| openrouter-llama-4-maverick | kitab_muharaf | 20 | 0 | 0.7235 | 0.7169 | 0.5817 | 0.2831 | 1.0122 | 0.9171 | 0.0 | 0.3329 | 1752.7 | 0.002573 |
| openrouter-minimax-m3-free | kitab_muharaf | 20 | 0 | 0.7689 | 0.76 | 0.7222 | 0.24 | 1.1302 | 0.9602 | 0.0 | 0.6581 | 3312.7 | 0.0 |
| pixtral-12b | kitab_muharaf | 20 | 0 | 1.3092 | 1.3006 | 0.8879 | 0.0 | 1.2248 | 1.0 | 0.0 | 0.0619 | 1400.9 | 0.000608 |
| nvidia-llama-3.2-11b-vl | kitab_muharaf | 20 | 0 | 1.9444 | 1.9368 | 1.2733 | 0.0 | 1.8743 | 0.9968 | 0.0 | 0.2759 | 4913.8 | 0.0 |
| nvidia-llama-3.2-90b-vl | kitab_muharaf | 20 | 0 | 2.353 | 2.3235 | 2.1174 | 0.0 | 2.378 | 0.9824 | 0.0 | 0.3676 | 127381.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_onlinekhatt | 20 | 0 | 0.4062 | 0.3893 | 0.2497 | 0.6107 | 0.7206 | 0.6533 | 0.05 | 0.5065 | 1871.5 | 0.001948 |
| openrouter-llama-4-maverick | kitab_onlinekhatt | 20 | 0 | 0.4792 | 0.4648 | 0.3535 | 0.5352 | 0.9191 | 0.8065 | 0.0 | 0.3452 | 1343.3 | 0.003188 |
| openrouter-qwen2.5-vl-72b | kitab_onlinekhatt | 20 | 0 | 0.6224 | 0.6146 | 0.4415 | 0.3854 | 0.9265 | 0.8571 | 0.05 | 0.4033 | 1392.1 | 0.004274 |
| pixtral-12b | kitab_onlinekhatt | 20 | 0 | 1.0506 | 1.0284 | 0.7313 | 0.0 | 1.1329 | 0.9939 | 0.0 | 0.1709 | 799.5 | 0.000993 |
| nvidia-llama-3.2-90b-vl | kitab_onlinekhatt | 20 | 0 | 1.1473 | 1.0828 | 1.0194 | 0.0 | 1.5309 | 0.8857 | 0.0 | 0.6833 | 30398.9 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_onlinekhatt | 20 | 0 | 1.2731 | 1.2677 | 0.9752 | 0.0 | 1.6481 | 0.9368 | 0.0 | 0.55 | 1969.2 | 0.0 |
| openrouter-minimax-m3-free | kitab_onlinekhatt | 20 | 0 | 1.6732 | 1.6562 | 1.5401 | 0.0 | 1.8971 | 0.9485 | 0.0 | 0.3636 | 2675.1 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
