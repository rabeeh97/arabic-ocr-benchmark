# Benchmark summary — `20260911T212318Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseer-nakba-bf16 | nakba_lines | 200 | 0 | 0.0122 | 0.0104 | 0.0101 | 0.9896 | 0.0445 | 0.0442 | 0.72 | 0.9933 | 2566.8 | 0.0 |
| exp3090-baseer-nakba-ft-bf16 | nakba_lines | 200 | 0 | 0.0537 | 0.0341 | 0.033 | 0.9659 | 0.2097 | 0.206 | 0.27 | 0.9633 | 2661.9 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
