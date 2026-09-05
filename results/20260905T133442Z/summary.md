# Benchmark summary — `20260905T133442Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trocr-large-arabic | kitab_muharaf | 10 | 0 | 0.803 | 0.7973 | 0.7924 | 1.0808 | 0.0 | 0.8 | 2342.6 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
