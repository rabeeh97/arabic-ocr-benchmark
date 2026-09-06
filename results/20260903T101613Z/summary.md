# Benchmark summary — `20260903T101613Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nvidia-gemma-4-31b | kitab_muharaf | 7 | 3 | 0.2908 | 0.274 | 0.2581 | 0.726 | 0.7843 | 0.7407 | 0.1429 | 0.7143 | 492680.5 | 0.0 |
| mistral-medium | kitab_muharaf | 10 | 0 | 0.5795 | 0.5758 | 0.5676 | 0.4242 | 0.9697 | 0.932 | 0.0 | 0.7 | 1088.8 | 0.001034 |
| mistral-medium | muharaf_lines | 10 | 0 | 0.7264 | 0.6925 | 0.6364 | 0.3075 | 0.9858 | 0.972 | 0.0 | 0.4 | 1427.7 | 0.001244 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
