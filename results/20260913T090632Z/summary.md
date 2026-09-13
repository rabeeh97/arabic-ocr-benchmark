# Benchmark summary — `20260913T090632Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp3090-paligemma2-3b-ft-bf16 | nakba_lines | 200 | 0 | 0.3106 | 0.2866 | 0.2787 | 0.7134 | 0.7288 | 0.6979 | 0.14 | 0.87 | 1942.1 | 0.0 |
| paligemma2-3b-bf16 | nakba_lines | 199 | 1 | 0.8511 | 0.8429 | 0.8381 | 0.1571 | 0.987 | 0.9705 | 0.1206 | 0.9648 | 706.5 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
