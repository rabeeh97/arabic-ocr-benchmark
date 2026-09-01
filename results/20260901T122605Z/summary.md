# Benchmark summary — `20260901T122605Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.2 | kitab_muharaf | 10 | 0 | 0.7206 | 0.7162 | 0.6831 | 1.1505 | 0.1 | 0.7 | 94555.7 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
