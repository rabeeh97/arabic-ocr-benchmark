# Benchmark summary — `20260905T121042Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-pro | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash | kitab_khatt | 3 | 27 | 0.7283 | 0.7253 | 0.7238 | 0.8158 | 0.0 | 0.6667 | 14277.8 | 0.003199 |
| gemini-pro | kitab_khatt | 0 | 25 |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
