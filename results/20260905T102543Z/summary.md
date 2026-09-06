# Benchmark summary — `20260905T102543Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_hindawi | 1 | 0 | 0.7663 | 0.7594 | 0.7619 | 0.2406 | 0.8086 | 0.7988 | 0.0 | 0.56 | 2157.4 | 0.000154 |
| gemini-pro | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash-lite | kitab_historyar | 38 | 0 | 0.3136 | 0.3041 | 0.2324 | 0.6959 | 0.7209 | 0.6873 | 0.0 | 0.3801 | 4385.1 | 0.004757 |
| gemini-flash | kitab_khatt | 0 | 25 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-pro | kitab_khatt | 0 | 25 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| gemini-flash-lite | kitab_onlinekhatt | 50 | 0 | 0.1412 | 0.1322 | 0.1164 | 0.8678 | 0.3836 | 0.3641 | 0.1 | 0.8631 | 2234.6 | 0.006062 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
