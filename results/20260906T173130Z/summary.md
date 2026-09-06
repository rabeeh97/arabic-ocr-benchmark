# Benchmark summary — `20260906T173130Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixtral-12b | kitab_hindawi | 40 | 0 | 0.7387 | 0.7247 | 0.7131 | 0.2753 | 0.8397 | 0.8345 | 0.0 | 0.184 | 2839.1 | 0.005327 |
| openrouter-minimax-m3-free | kitab_hindawi | 19 | 17 | 0.7677 | 0.7528 | 0.7346 | 0.2472 | 0.9123 | 0.9078 | 0.0 | 0.1854 | 7871.5 | 0.0 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 17 | 23 | 0.8422 | 0.8339 | 0.8259 | 0.1661 | 0.8812 | 0.8812 | 0.0 | 0.1102 | 3882.9 | 0.009545 |
| pixtral-12b | kitab_historyar | 39 | 1 | 1.1665 | 1.1517 | 0.8035 | 0.0 | 1.1425 | 1.0 | 0.0 | 0.1418 | 3998.5 | 0.002313 |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 2 | 38 | 0.1154 | 0.098 | 0.0915 | 0.902 | 0.4643 | 0.4194 | 0.0 | 1.0 | 1900.4 | 0.000321 |
| openrouter-gemma-4-31b-free | kitab_khatt | 1 | 11 | 0.45 | 0.4375 | 0.4026 | 0.5625 | 0.8 | 0.75 | 0.0 | 0.0 | 3537.1 | 0.0 |
| openrouter-minimax-m3-free | kitab_khatt | 40 | 0 | 0.5059 | 0.4897 | 0.4187 | 0.5103 | 0.8745 | 0.8229 | 0.0 | 0.475 | 4749.0 | 0.0 |
| pixtral-12b | kitab_khatt | 40 | 0 | 1.2908 | 1.2928 | 0.818 | 0.0 | 1.1211 | 0.9941 | 0.0 | 0.0277 | 2308.0 | 0.002567 |
| mistral-medium | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| mistral-small | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| openrouter-minimax-m3-free | kitab_muharaf | 39 | 1 | 0.8817 | 0.8709 | 0.7869 | 0.1291 | 1.1821 | 0.9655 | 0.0256 | 0.4301 | 3166.7 | 0.0 |
| pixtral-12b | kitab_muharaf | 37 | 3 | 1.3233 | 1.318 | 0.8102 | 0.0 | 1.1927 | 0.985 | 0.0541 | 0.135 | 1239.0 | 0.001013 |
| openrouter-qwen2.5-vl-72b | kitab_muharaf | 0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| pixtral-12b | kitab_onlinekhatt | 36 | 4 | 1.1092 | 1.0935 | 0.7344 | 0.0 | 1.1095 | 0.9632 | 0.0 | 0.0591 | 998.5 | 0.001813 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
