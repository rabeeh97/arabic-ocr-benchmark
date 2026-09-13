# Benchmark summary — `20260913T091549Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-ft-bf16 | rasam1 | 199 | 1 | 0.4976 | 0.4969 | 0.4816 | 0.5031 | 0.9001 | 0.8606 | 0.0 | 0.6072 | 2575.2 | 0.0 |
| paligemma2-3b-bf16 | rasam1 | 200 | 0 | 0.7711 | 0.7669 | 0.7527 | 0.2331 | 0.9808 | 0.9582 | 0.005 | 0.8511 | 1384.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
