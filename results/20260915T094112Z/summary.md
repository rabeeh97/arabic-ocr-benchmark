# Benchmark summary — `20260915T094112Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-visft-ft-bf16 | nakba_lines | 200 | 0 | 0.3294 | 0.3059 | 0.2983 | 0.6941 | 0.7451 | 0.7192 | 0.11 | 0.8083 | 2182.8 | 0.0 |
| paligemma2-3b-bf16 | nakba_lines | 200 | 0 | 0.9919 | 0.9918 | 0.9918 | 0.0082 | 1.0094 | 0.9978 | 0.005 | 0.98 | 445.9 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
