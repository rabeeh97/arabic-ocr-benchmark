# Benchmark summary — `20260915T101112Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-visft-ft-bf16 | kitab_hindawi | 180 | 0 | 0.5415 | 0.5094 | 0.4861 | 0.4906 | 0.7611 | 0.7527 | 0.0 | 0.2875 | 18411.1 | 0.0 |
| paligemma2-3b-bf16 | kitab_hindawi | 180 | 0 | 0.998 | 0.9979 | 0.9978 | 0.0021 | 1.0 | 1.0 | 0.0 | 0.0167 | 620.3 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
