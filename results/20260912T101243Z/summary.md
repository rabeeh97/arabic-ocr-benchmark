# Benchmark summary — `20260912T101243Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-lite | kitab_onlinekhatt | 15 | 0 | 0.1037 | 0.0895 | 0.0811 | 0.9105 | 0.3525 | 0.3468 | 0.0 | 1.0 | 1751.0 | 0.00185 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
