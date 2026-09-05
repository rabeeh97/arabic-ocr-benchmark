# Benchmark summary — `20260905T204359Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-vl-3b | kitab_hindawi | 10 | 0 | 0.6191 | 0.602 | 0.5905 | 0.7299 | 0.0 | 0.3041 | 20000.2 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
