# Benchmark summary — `20260908T120417Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash | kitab_muharaf | 3 | 33 | 0.7931 | 0.7816 | 0.7901 | 0.2184 | 0.9375 | 0.9375 | 0.3333 | 0.8205 | 31537.6 | 0.003279 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
