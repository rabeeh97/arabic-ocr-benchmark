# Benchmark summary — `20260905T100451Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_hindawi | 50 | 0 | 0.854 | 0.8467 | 0.8404 | 0.8736 | 0.0 | 0.1275 | 2497.3 | 0.007052 |
| gemini-pro | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash-lite | kitab_historyar | 50 | 0 | 0.287 | 0.2788 | 0.1872 | 0.6628 | 0.0 | 0.4844 | 3187.2 | 0.006114 |
| gemini-flash | kitab_khatt | 23 | 25 | 0.7883 | 0.7822 | 0.7806 | 0.9097 | 0.0 | 0.7826 | 5814.2 | 0.022471 |
| gemini-pro | kitab_khatt | 0 | 25 |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash-lite | kitab_muharaf | 22 | 0 | 0.326 | 0.3171 | 0.3106 | 0.6702 | 0.1818 | 0.7727 | 1822.3 | 0.002665 |
| gemini-flash-lite | kitab_onlinekhatt | 50 | 0 | 0.0997 | 0.0957 | 0.0866 | 0.3103 | 0.26 | 0.92 | 1772.6 | 0.006035 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
