# Benchmark summary — `20260906T003316Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trocr-large-arabic | kitab_historyar | 20 | 0 | 0.8658 | 0.8609 | 0.8843 | 1.1818 | 0.0 | 0.6 | 2425.3 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
