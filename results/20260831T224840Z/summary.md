# Benchmark summary — `20260831T224840Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixtral-12b | synthetic | 10 | 0 | 0.1379 | 0.1379 | 0.0129 | 0.2381 | 0.6 | 0.8 | 2653.0 | 0.000389 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
