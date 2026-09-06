# Benchmark summary — `20260906T155935Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixtral-12b | kitab_hindawi | 40 | 0 | 0.7524 | 0.7386 | 0.7259 | 0.2614 | 0.8528 | 0.8459 | 0.0 | 0.107 | 3200.6 | 0.005195 |
| glm-5.3-flash-paid | kitab_hindawi | 15 | 0 | 0.9847 | 0.9842 | 0.984 | 0.0158 | 0.991 | 0.991 | 0.0 | 0.0644 | 83913.2 | 0.027079 |
| pixtral-12b | kitab_historyar | 40 | 0 | 1.3715 | 1.3584 | 0.9917 | 0.0 | 1.263 | 0.9977 | 0.0 | 0.2318 | 9401.4 | 0.002538 |
| glm-5.3-flash-paid | kitab_khatt | 15 | 0 | 0.7713 | 0.7674 | 0.7589 | 0.2326 | 0.9527 | 0.9148 | 0.0 | 0.7333 | 80898.4 | 0.021851 |
| openrouter-nemotron-omni-free | kitab_khatt | 16 | 8 | 907.3099 | 835.3225 | 804.3717 | 0.0 | 798.6718 | 0.9999 | 0.0 | 0.2768 | 421384.2 | 0.0 |
| openrouter-gemma-4-26b-free | kitab_khatt | 0 | 12 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| glm-5.3-flash-paid | kitab_muharaf | 15 | 0 | 0.6709 | 0.6645 | 0.6658 | 0.3355 | 0.9859 | 0.9333 | 0.0 | 0.8 | 45733.9 | 0.01355 |
| pixtral-12b | kitab_muharaf | 35 | 3 | 1.5515 | 1.5408 | 0.934 | 0.0 | 1.3423 | 0.9955 | 0.0286 | 0.0939 | 2124.3 | 0.001083 |
| pixtral-12b | kitab_onlinekhatt | 37 | 3 | 1.3061 | 1.2881 | 0.7707 | 0.0 | 1.1756 | 1.0 | 0.0 | 0.1115 | 1478.0 | 0.001884 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
