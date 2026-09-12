# Benchmark summary — `20260912T215127Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-paid | kitab_hindawi | 10 | 0 | 0.685 | 0.6729 | 0.69 | 0.3271 | 0.7394 | 0.7351 | 0.0 | 0.3091 | 17755.8 | 0.23263 |
| claude-sonnet-paid | kitab_hindawi | 10 | 0 | 0.8143 | 0.8073 | 0.7973 | 0.1927 | 0.8494 | 0.8408 | 0.0 | 0.2792 | 9184.2 | 0.039239 |
| claude-haiku-paid | kitab_hindawi | 10 | 0 | 0.8933 | 0.8883 | 0.8864 | 0.1117 | 0.9359 | 0.9359 | 0.0 | 0.1099 | 8994.1 | 0.012776 |
| gpt-4o-paid | kitab_hindawi | 10 | 0 | 0.9233 | 0.9201 | 0.9138 | 0.0799 | 0.9427 | 0.9427 | 0.0 | 0.0948 | 11617.9 | 0.025212 |
| gpt-4.1-paid | kitab_hindawi | 9 | 1 | 0.9587 | 0.9573 | 0.9543 | 0.0427 | 0.9754 | 0.9741 | 0.0 | 0.0381 | 9075.3 | 0.018962 |
| gpt-4o-mini-paid | kitab_hindawi | 9 | 1 | 0.9637 | 0.9624 | 0.9618 | 0.0376 | 0.9832 | 0.9832 | 0.0 | 0.117 | 9576.9 | 0.051398 |
| deepseek-vl-flash | kitab_hindawi | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| deepseek-vl-flash | kitab_historicalbooks | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| claude-opus-paid | kitab_historyar | 10 | 0 | 0.1586 | 0.1511 | 0.0908 | 0.8489 | 0.479 | 0.475 | 0.0 | 0.6531 | 52312.1 | 0.095586 |
| claude-sonnet-paid | kitab_historyar | 10 | 0 | 0.2205 | 0.219 | 0.1193 | 0.781 | 0.563 | 0.5583 | 0.0 | 0.4 | 45091.8 | 0.012051 |
| gpt-4.1-paid | kitab_historyar | 10 | 0 | 0.3202 | 0.3157 | 0.2303 | 0.6843 | 0.6387 | 0.6387 | 0.0 | 0.46 | 58853.4 | 0.014993 |
| gpt-4o-paid | kitab_historyar | 10 | 0 | 0.3459 | 0.3429 | 0.2706 | 0.6571 | 0.6218 | 0.6167 | 0.0 | 0.4143 | 65173.5 | 0.018403 |
| claude-haiku-paid | kitab_historyar | 9 | 1 | 0.6228 | 0.6053 | 0.511 | 0.3947 | 1.0 | 0.9821 | 0.0 | 0.4444 | 48048.5 | 0.004711 |
| gpt-4o-mini-paid | kitab_historyar | 10 | 0 | 0.6828 | 0.6722 | 0.5815 | 0.3278 | 0.9328 | 0.9328 | 0.0 | 0.3611 | 61012.5 | 0.037869 |
| deepseek-vl-flash | kitab_historyar | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| claude-opus-paid | kitab_khatt | 10 | 0 | 0.1695 | 0.163 | 0.1475 | 0.837 | 0.5372 | 0.4887 | 0.0 | 0.9 | 13730.4 | 0.135547 |
| claude-sonnet-paid | kitab_khatt | 10 | 0 | 0.1895 | 0.1755 | 0.1615 | 0.8245 | 0.595 | 0.5625 | 0.0 | 0.9 | 11543.4 | 0.012578 |
| gpt-4o-paid | kitab_khatt | 10 | 0 | 0.4052 | 0.3866 | 0.3742 | 0.6134 | 0.7107 | 0.688 | 0.0 | 0.9 | 11468.2 | 0.020792 |
| gpt-4.1-paid | kitab_khatt | 10 | 0 | 0.4145 | 0.4053 | 0.3866 | 0.5947 | 0.7934 | 0.7273 | 0.0 | 0.7 | 6541.2 | 0.016931 |
| claude-haiku-paid | kitab_khatt | 10 | 0 | 0.5833 | 0.5817 | 0.5552 | 0.4183 | 0.9478 | 0.9478 | 0.0 | 0.5 | 8089.3 | 0.006498 |
| gpt-4o-mini-paid | kitab_khatt | 10 | 0 | 0.7381 | 0.7283 | 0.6351 | 0.2717 | 0.9421 | 0.9048 | 0.0 | 0.4 | 10540.5 | 0.043525 |
| deepseek-vl-flash | kitab_khatt | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| claude-opus-paid | kitab_muharaf | 10 | 0 | 0.3428 | 0.3258 | 0.3257 | 0.6742 | 0.7778 | 0.713 | 0.0 | 0.9 | 14354.1 | 0.079018 |
| claude-sonnet-paid | kitab_muharaf | 10 | 0 | 0.3674 | 0.3636 | 0.36 | 0.6364 | 0.7374 | 0.73 | 0.0 | 0.9 | 7675.6 | 0.012106 |
| gpt-4.1-paid | kitab_muharaf | 9 | 1 | 0.4234 | 0.4191 | 0.4115 | 0.5809 | 0.7841 | 0.7753 | 0.0 | 0.7778 | 6534.3 | 0.010445 |
| gpt-4o-paid | kitab_muharaf | 9 | 1 | 0.5128 | 0.5064 | 0.5011 | 0.4936 | 0.8295 | 0.7849 | 0.0 | 0.7778 | 5681.6 | 0.01296 |
| claude-haiku-paid | kitab_muharaf | 10 | 0 | 0.7231 | 0.7154 | 0.704 | 0.2846 | 1.0 | 0.9315 | 0.0 | 0.6 | 5958.6 | 0.003165 |
| gpt-4o-mini-paid | kitab_muharaf | 8 | 2 | 0.7522 | 0.7217 | 0.5643 | 0.2783 | 0.954 | 0.9432 | 0.0 | 0.125 | 7754.2 | 0.024277 |
| deepseek-vl-flash | kitab_muharaf | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |
| claude-sonnet-paid | kitab_onlinekhatt | 10 | 0 | 0.1516 | 0.1495 | 0.1459 | 0.8505 | 0.4878 | 0.4545 | 0.0 | 0.8 | 6912.7 | 0.011523 |
| claude-opus-paid | kitab_onlinekhatt | 10 | 0 | 0.1579 | 0.1537 | 0.1332 | 0.8463 | 0.439 | 0.4 | 0.1 | 0.9 | 7241.7 | 0.078727 |
| gpt-4.1-paid | kitab_onlinekhatt | 10 | 0 | 0.2821 | 0.2758 | 0.2622 | 0.7242 | 0.6585 | 0.6067 | 0.0 | 0.7667 | 4861.4 | 0.014771 |
| gpt-4o-paid | kitab_onlinekhatt | 10 | 0 | 0.3305 | 0.3221 | 0.2706 | 0.6779 | 0.7683 | 0.6923 | 0.0 | 0.9 | 11177.2 | 0.018561 |
| claude-haiku-paid | kitab_onlinekhatt | 10 | 0 | 0.4829 | 0.4707 | 0.4621 | 0.5293 | 0.8784 | 0.8784 | 0.0 | 0.8 | 6289.4 | 0.004867 |
| gpt-4o-mini-paid | kitab_onlinekhatt | 10 | 0 | 0.5221 | 0.5158 | 0.3594 | 0.4842 | 0.9146 | 0.8929 | 0.0 | 0.4569 | 6274.4 | 0.037824 |
| deepseek-vl-flash | kitab_onlinekhatt | 0 | 10 |  |  |  |  |  |  |  |  | 0.0 | 0.0 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
