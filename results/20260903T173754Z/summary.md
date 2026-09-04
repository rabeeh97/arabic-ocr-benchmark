# Benchmark summary — `20260903T173754Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_hindawi | 80 | 0 | 0.8249 | 0.8169 | 0.812 | 0.8456 | 0.0 | 0.1407 | 5714.8 | 0.011561 |
| gemini-flash-lite | kitab_historyar | 80 | 0 | 0.3058 | 0.299 | 0.2173 | 0.7013 | 0.025 | 0.5172 | 8719.8 | 0.009904 |
| gemini-flash-lite | kitab_khatt | 80 | 0 | 0.2891 | 0.2797 | 0.2561 | 0.5947 | 0.0375 | 0.8689 | 5791.7 | 0.009952 |
| gemini-flash-lite | kitab_muharaf | 79 | 1 | 0.3543 | 0.3436 | 0.3223 | 0.7273 | 0.0633 | 0.7263 | 4791.7 | 0.009735 |
| gemini-flash-lite | kitab_onlinekhatt | 56 | 0 | 0.12 | 0.1156 | 0.1017 | 0.3904 | 0.1429 | 0.7723 | 8649.6 | 0.006794 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
