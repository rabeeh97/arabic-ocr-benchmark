# Benchmark summary — `20260914T131518Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | kitab_hindawi | 20 | 0 | 0.637 | 0.6195 | 0.6093 | 0.3805 | 0.7447 | 0.7434 | 0.0 | 0.4072 | 5331.3 | 0.010956 |
| nvidia-llama-3.2-11b-vl | kitab_hindawi | 20 | 0 | 0.7291 | 0.7139 | 0.6853 | 0.2861 | 0.8974 | 0.8951 | 0.0 | 0.1969 | 19420.7 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 20 | 0 | 0.7524 | 0.7407 | 0.7355 | 0.2593 | 0.7924 | 0.791 | 0.0 | 0.1688 | 3712.3 | 0.011717 |
| openrouter-llama-4-maverick | kitab_hindawi | 20 | 0 | 0.8939 | 0.8886 | 0.8865 | 0.1114 | 0.934 | 0.9326 | 0.0 | 0.1761 | 2271.8 | 0.005231 |
| pixtral-12b | kitab_hindawi | 1 | 0 | 0.9766 | 0.9738 | 0.9757 | 0.0262 | 1.0 | 1.0 | 0.0 | 0.1818 | 1244.7 | 0.000116 |
| openrouter-dots3-note-free | kitab_hindawi | 20 | 0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 5171.5 | 0.0 |
| openrouter-dots3-note-free | kitab_historicalbooks | 1 | 0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | 8970.5 | 0.0 |
| openrouter-dots3-note-free | kitab_historyar | 6 | 6 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.1667 | 8443.9 | 0.0 |
| pixtral-12b | kitab_historyar | 12 | 0 | 1.0646 | 1.0484 | 0.7923 | 0.0 | 1.1037 | 1.0 | 0.0 | 0.1412 | 4190.8 | 0.001063 |
| nvidia-llama-3.2-11b-vl | kitab_historyar | 20 | 0 | 1.1457 | 1.1396 | 0.8748 | 0.0 | 1.3232 | 0.9813 | 0.0 | 0.2519 | 7872.1 | 0.0 |
| openrouter-llama-4-maverick | kitab_historyar | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_historyar | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_historyar | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-dots3-note-free | kitab_khatt | 20 | 0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.8 | 4546.3 | 0.0 |
| nvidia-llama-3.2-90b-vl | kitab_khatt | 11 | 9 | 1.0756 | 1.0613 | 0.8036 | 0.0 | 1.8621 | 0.9408 | 0.0 | 0.3712 | 267480.7 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_khatt | 20 | 0 | 1.4018 | 1.4029 | 1.143 | 0.0 | 1.4498 | 0.9459 | 0.0 | 0.2 | 5205.3 | 0.0 |
| mistral-medium | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| mistral-small | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-31b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-llama-4-maverick | kitab_khatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_khatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-dots3-note-free | kitab_muharaf | 20 | 0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.6 | 4434.2 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_muharaf | 20 | 0 | 1.1101 | 1.1013 | 0.7909 | 0.0 | 1.2042 | 0.9787 | 0.05 | 0.3625 | 5622.7 | 0.0 |
| openrouter-llama-4-maverick | kitab_muharaf | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_muharaf | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_onlinekhatt | 20 | 0 | 0.651 | 0.6433 | 0.5347 | 0.3567 | 0.943 | 0.9141 | 0.0 | 0.35 | 3746.2 | 0.0 |
| pixtral-12b | kitab_onlinekhatt | 20 | 0 | 1.1972 | 1.1828 | 0.7926 | 0.0 | 1.141 | 0.9944 | 0.0 | 0.0124 | 1097.3 | 0.001636 |
| openrouter-llama-4-maverick | kitab_onlinekhatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_onlinekhatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_onlinekhatt | 0 | 20 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
