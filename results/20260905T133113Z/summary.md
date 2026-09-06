# Benchmark summary — `20260905T133113Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2-vl-2b | kitab_muharaf | 9 | 1 | 0.723 | 0.721 | 0.7174 | 0.279 | 0.9368 | 0.9271 | 0.0 | 0.7778 | 3491.3 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
