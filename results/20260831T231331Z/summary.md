# Benchmark summary — `20260831T231331Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen2.5-vl-72b | synthetic | 10 | 0 | 0.2802 | 0.2802 | 0.1379 | 0.4048 | 0.6 | 0.8 | 7707.1 | 0.001629 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
