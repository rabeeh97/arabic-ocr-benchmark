# Benchmark summary — `20260905T132931Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qari-ocr-0.3 | kitab_muharaf | 10 | 0 | 0.6837 | 0.6686 | 0.4933 | 0.3314 | 0.9596 | 0.9223 | 0.0 | 0.5119 | 4787.8 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
