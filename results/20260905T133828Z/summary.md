# Benchmark summary — `20260905T133828Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.3 | kitab_historicalbooks | 10 | 0 | 0.6796 | 0.6607 | 0.6327 | 0.9533 | 0.0 | 0.4356 | 10278.2 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
