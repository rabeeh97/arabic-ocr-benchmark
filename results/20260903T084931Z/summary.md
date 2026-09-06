# Benchmark summary — `20260903T084931Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nvidia-llama-3.2-90b-vl | kitab_muharaf | 10 | 0 | 1.9261 | 1.6951 | 1.3257 | 0.0 | 1.6162 | 0.9877 | 0.0 | 0.6118 | 170664.1 | 0.0 |
| nvidia-llama-3.2-11b-vl | kitab_muharaf | 10 | 0 | 3.8921 | 3.8861 | 2.9272 | 0.0 | 3.914 | 0.9973 | 0.0 | 0.2082 | 6906.1 | 0.0 |
| nvidia-llama-3.2-90b-vl | muharaf_lines | 10 | 0 | 2.1862 | 2.1862 | 1.9261 | 0.0 | 2.0071 | 0.9965 | 0.0 | 0.4186 | 173142.1 | 0.0 |
| nvidia-llama-3.2-11b-vl | muharaf_lines | 10 | 0 | 2.2683 | 2.2635 | 1.4258 | 0.0 | 1.7727 | 1.0 | 0.0 | 0.3213 | 4143.4 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
