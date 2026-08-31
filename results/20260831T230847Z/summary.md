# Benchmark summary — `20260831T230847Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | kitab_khatt | 12 | 0 | 0.364 | 0.3491 | 0.3059 | 0.8425 | 0.0 | 0.375 | 12437.9 | 0.001489 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
