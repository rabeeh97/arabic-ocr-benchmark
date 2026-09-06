# Benchmark summary — `20260905T165958Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-flash | kitab_khatt | 37 | 23 | 0.7686 | 0.763 | 0.7614 | 0.237 | 0.8522 | 0.8486 | 0.0 | 0.8829 | 10043.6 | 0.036618 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
