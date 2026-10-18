# Benchmark summary — `20260914T023601Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-qwen3vl4b-visft-ft-bf16 | nakba_lines | 200 | 0 | 0.2078 | 0.1819 | 0.1737 | 0.8181 | 0.5931 | 0.5742 | 0.1 | 0.7453 | 3075.2 | 0.0 |
| qwen3-vl-4b-bf16 | nakba_lines | 200 | 0 | 0.98 | 0.9544 | 0.5694 | 0.0456 | 1.0612 | 0.9676 | 0.05 | 0.1803 | 7496.1 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
