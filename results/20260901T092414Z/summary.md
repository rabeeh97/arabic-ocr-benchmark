# Benchmark summary — `20260901T092414Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2-vl-2b | kitab_khatt | 10 | 0 | 0.7979 | 0.8036 | 0.5725 | 0.9918 | 0.0 | 0.41 | 79935.6 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
