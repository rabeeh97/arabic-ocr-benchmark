# Benchmark summary — `20260904T140500Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_hindawi | 42 | 8 | 0.7373 | 0.7262 | 0.7207 | 0.2738 | 0.774 | 0.7707 | 0.0 | 0.2631 | 48453.9 | 0.006622 |
| gemini-flash-lite | kitab_historyar | 12 | 0 | 0.273 | 0.2574 | 0.2019 | 0.7426 | 0.6475 | 0.5852 | 0.0 | 0.6095 | 44365.2 | 0.001489 |
| gemini-flash-lite | kitab_khatt | 22 | 28 | 0.2298 | 0.2224 | 0.219 | 0.7776 | 0.4847 | 0.4483 | 0.0 | 0.8182 | 107667.2 | 0.002718 |
| gemini-flash-lite | kitab_muharaf | 30 | 20 | 0.3886 | 0.3797 | 0.3652 | 0.6203 | 0.7695 | 0.6854 | 0.0667 | 0.6571 | 82363.0 | 0.003662 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
