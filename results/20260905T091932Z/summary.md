# Benchmark summary — `20260905T091932Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_hindawi | 7 | 0 | 0.7174 | 0.7046 | 0.7008 | 0.739 | 0.0 | 0.4214 | 2275.8 | 0.001106 |
| gemini-flash-lite | kitab_khatt | 2 | 0 | 0.2435 | 0.2368 | 0.2193 | 0.3182 | 0.0 | 0.5 | 2613.5 | 0.000239 |
| gemini-flash-lite | kitab_muharaf | 49 | 1 | 0.4111 | 0.3913 | 0.3642 | 0.7748 | 0.0816 | 0.6815 | 2213.7 | 0.006148 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
