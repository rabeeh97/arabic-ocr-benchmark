# Benchmark summary — `20260831T231319Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mistral-small | synthetic | 10 | 0 | 0.1336 | 0.1336 | 0.0 | 0.2143 | 0.8 | 0.8 | 3813.1 | 0.000585 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
