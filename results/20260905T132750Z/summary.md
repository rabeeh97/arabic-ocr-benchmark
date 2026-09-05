# Benchmark summary — `20260905T132750Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.2 | kitab_muharaf | 9 | 1 | 0.723 | 0.721 | 0.7174 | 0.9368 | 0.0 | 0.7778 | 4488.3 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
