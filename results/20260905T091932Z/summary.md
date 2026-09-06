# Benchmark summary — `20260905T091932Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_hindawi | 7 | 0 | 0.7176 | 0.7046 | 0.7008 | 0.2954 | 0.739 | 0.7384 | 0.0 | 0.4214 | 2275.8 | 0.001106 |
| gemini-flash-lite | kitab_khatt | 2 | 0 | 0.2435 | 0.2368 | 0.2193 | 0.7632 | 0.3182 | 0.28 | 0.0 | 0.5 | 2613.5 | 0.000239 |
| gemini-flash-lite | kitab_muharaf | 49 | 1 | 0.4111 | 0.3891 | 0.362 | 0.6109 | 0.7748 | 0.7002 | 0.0816 | 0.6815 | 2213.7 | 0.006148 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
