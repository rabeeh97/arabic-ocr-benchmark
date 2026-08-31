# Benchmark summary — `20260831T230728Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen2.5-vl-72b | kitab_khatt | 12 | 0 | 0.4348 | 0.4224 | 0.3333 | 0.8151 | 0.0 | 0.6667 | 17939.4 | 0.003296 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
