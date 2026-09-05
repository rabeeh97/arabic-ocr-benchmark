# Benchmark summary — `20260905T131106Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.2 | kitab_khatt | 10 | 0 | 0.6287 | 0.6056 | 0.5466 | 0.9174 | 0.0 | 0.8 | 10299.5 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
