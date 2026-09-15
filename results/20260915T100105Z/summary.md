# Benchmark summary — `20260915T100105Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-visft-ft-bf16 | rasam2 | 196 | 4 | 0.5601 | 0.5561 | 0.5366 | 0.4439 | 0.9338 | 0.8788 | 0.0 | 0.5765 | 2489.7 | 0.0 |
| paligemma2-3b-bf16 | rasam2 | 196 | 4 | 0.9701 | 0.9701 | 0.9701 | 0.0299 | 1.006 | 1.0 | 0.0 | 0.9949 | 434.5 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
