# Benchmark summary — `20260904T034451Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-pro | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash-lite | kitab_khatt | 3 | 28 | 0.3615 | 0.3476 | 0.3476 | 0.7292 | 0.0 | 1.0 | 8614.3 | 0.000378 |
| gemini-flash | kitab_khatt | 3 | 26 | 0.8497 | 0.8366 | 0.8105 | 0.9688 | 0.0 | 0.6667 | 8129.9 | 0.002907 |
| gemini-pro | kitab_khatt | 0 | 25 |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
