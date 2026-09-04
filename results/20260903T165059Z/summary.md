# Benchmark summary — `20260903T165059Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-minimax-m3-free | kitab_historyar | 20 | 0 | 0.681 | 0.6734 | 0.642 | 1.1127 | 0.0 | 0.3727 | 10495.4 | 0.0 |
| openrouter-minimax-m3-free | kitab_onlinekhatt | 20 | 0 | 0.6011 | 0.5849 | 0.5124 | 1.0123 | 0.0 | 0.55 | 4990.6 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
