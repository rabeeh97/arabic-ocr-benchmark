# Benchmark summary — `20260905T111156Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixtral-12b | kitab_hindawi | 20 | 0 | 0.7522 | 0.7392 | 0.7318 | 0.8572 | 0.0 | 0.1448 | 1873.4 | 0.002656 |
| pixtral-12b | kitab_khatt | 20 | 0 | 1.1048 | 1.0987 | 0.747 | 1.0691 | 0.0 | 0.1046 | 1048.3 | 0.001463 |
| pixtral-12b | kitab_muharaf | 15 | 5 | 1.3371 | 1.3078 | 0.8462 | 1.25 | 0.0 | 0.1846 | 1004.3 | 0.000418 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
