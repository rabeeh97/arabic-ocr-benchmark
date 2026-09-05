# Benchmark summary — `20260905T132549Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trocr-large-arabic | kitab_khatt | 10 | 0 | 0.8043 | 0.7981 | 0.7935 | 1.0744 | 0.0 | 1.0 | 6661.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
