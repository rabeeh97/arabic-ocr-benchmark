# Benchmark summary — `20260905T203855Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trocr-large-arabic | kitab_hindawi | 10 | 0 | 0.9534 | 0.9519 | 0.9482 | 0.0481 | 1.0 | 1.0 | 0.0 | 0.0 | 1757.8 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
