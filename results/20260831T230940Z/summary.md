# Benchmark summary — `20260831T230940Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-llama-4-maverick | kitab_khatt | 12 | 0 | 0.6006 | 0.5862 | 0.482 | 0.9726 | 0.0 | 0.2604 | 9372.7 | 0.002722 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
