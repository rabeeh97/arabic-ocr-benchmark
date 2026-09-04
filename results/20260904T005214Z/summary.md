# Benchmark summary — `20260904T005214Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_khatt | 32 | 3 | 0.3048 | 0.2949 | 0.2876 | 0.6345 | 0.0 | 0.8333 | 6028.6 | 0.003978 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
