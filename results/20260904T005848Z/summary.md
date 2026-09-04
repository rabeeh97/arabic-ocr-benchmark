# Benchmark summary — `20260904T005848Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash | kitab_historicalbooks | 1 | 0 | 0.907 | 0.907 | 0.9017 | 0.931 | 0.0 | 0.0 | 22859.4 | 0.000983 |
| gemini-pro | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash-lite | kitab_khatt | 1 | 45 | 0.1127 | 0.087 | 0.087 | 0.2857 | 0.0 | 1.0 | 5676.5 | 0.000128 |
| gemini-flash | kitab_khatt | 3 | 32 | 0.7473 | 0.768 | 0.7624 | 0.8919 | 0.0 | 1.0 | 26087.5 | 0.002909 |
| gemini-pro | kitab_khatt | 0 | 25 |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
