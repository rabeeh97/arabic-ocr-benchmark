# Benchmark summary — `20260905T204013Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.3 | kitab_onlinekhatt | 10 | 0 | 0.3868 | 0.3736 | 0.3392 | 0.85 | 0.0 | 0.3 | 6833.1 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
