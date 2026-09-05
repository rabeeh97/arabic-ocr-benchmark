# Benchmark summary — `20260905T091838Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm-5.3-flash-paid | kitab_khatt | 10 | 0 | 0.7855 | 0.782 | 0.7736 | 0.959 | 0.0 | 0.6 | 59775.1 | 0.013909 |
| glm-5.3-flash-paid | kitab_muharaf | 10 | 0 | 0.6936 | 0.6795 | 0.6713 | 0.957 | 0.1 | 0.6 | 34989.2 | 0.007753 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
