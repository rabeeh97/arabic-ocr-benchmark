# Benchmark summary — `20260831T231356Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | synthetic | 10 | 0 | 0.0043 | 0.0043 | 0.0043 | 0.0238 | 0.9 | 1.0 | 3798.8 | 0.000571 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
