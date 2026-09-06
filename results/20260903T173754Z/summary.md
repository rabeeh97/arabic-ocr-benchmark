# Benchmark summary — `20260903T173754Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_hindawi | 80 | 0 | 0.8249 | 0.8169 | 0.812 | 0.1831 | 0.8456 | 0.843 | 0.0 | 0.1407 | 5714.8 | 0.011561 |
| gemini-flash | kitab_historicalbooks | 9 | 1 | 0.9507 | 0.9501 | 0.9484 | 0.0499 | 0.9521 | 0.9505 | 0.0 | 0.5556 | 33367.2 | 0.009834 |
| gemini-flash-lite | kitab_historyar | 80 | 0 | 0.3058 | 0.299 | 0.2173 | 0.701 | 0.7013 | 0.6686 | 0.025 | 0.5172 | 8719.8 | 0.009904 |
| gemini-flash-lite | kitab_khatt | 80 | 0 | 0.2891 | 0.2797 | 0.2561 | 0.7203 | 0.5947 | 0.537 | 0.0375 | 0.8689 | 5791.7 | 0.009952 |
| gemini-flash | kitab_khatt | 6 | 25 | 0.8382 | 0.8316 | 0.8302 | 0.1684 | 0.963 | 0.963 | 0.0 | 0.6667 | 19796.4 | 0.005241 |
| gemini-flash-lite | kitab_muharaf | 79 | 1 | 0.3543 | 0.3436 | 0.3223 | 0.6564 | 0.7273 | 0.6539 | 0.0633 | 0.7263 | 4791.7 | 0.009735 |
| gemini-flash-lite | kitab_onlinekhatt | 80 | 0 | 0.1239 | 0.1194 | 0.1062 | 0.8806 | 0.3765 | 0.3588 | 0.15 | 0.8073 | 7501.8 | 0.009731 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
