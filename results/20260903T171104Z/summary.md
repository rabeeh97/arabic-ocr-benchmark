# Benchmark summary — `20260903T171104Z`

CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | wer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-fable-5-1-paid | kitab_hindawi | 10 | 0 | 0.049 | 0.0105 | 0.0062 | 0.0716 | 0.0 | 0.8216 | 18801.4 | 0.6548 |
| claude-opus-paid | kitab_hindawi | 10 | 0 | 0.3253 | 0.2947 | 0.2928 | 0.35 | 0.0 | 0.5879 | 15857.3 | 0.28405 |
| gpt-5-paid | kitab_hindawi | 10 | 0 | 0.7009 | 0.6832 | 0.6739 | 0.8299 | 0.0 | 0.3656 | 3931.8 | 0.02298 |
| claude-fable-paid | kitab_hindawi | 10 | 0 | 0.733 | 0.7221 | 0.7217 | 0.7507 | 0.0 | 0.2894 | 9272.8 | 0.27215 |
| gpt-4o-paid | kitab_hindawi | 10 | 0 | 0.7456 | 0.7346 | 0.7272 | 0.8112 | 0.0 | 0.2303 | 3465.9 | 0.0298 |
| claude-haiku-paid | kitab_hindawi | 10 | 0 | 0.8251 | 0.8178 | 0.813 | 0.8687 | 0.0 | 0.0 | 2710.7 | 0.01256 |
| claude-sonnet-paid | kitab_hindawi | 10 | 0 | 0.8388 | 0.8323 | 0.8311 | 0.85 | 0.0 | 0.148 | 4586.8 | 0.02605 |
| gpt-4.1-paid | kitab_hindawi | 10 | 0 | 0.8431 | 0.8377 | 0.8332 | 0.8813 | 0.0 | 0.2101 | 1806.4 | 0.020624 |
| gpt-4o-mini-paid | kitab_hindawi | 10 | 0 | 0.9159 | 0.9129 | 0.909 | 0.9493 | 0.0 | 0.0834 | 1457.5 | 0.03857 |
| claude-fable-5-1-paid | kitab_historicalbooks | 10 | 0 | 0.109 | 0.0862 | 0.0516 | 0.493 | 0.0 | 0.5 | 18128.8 | 0.51583 |
| claude-opus-paid | kitab_historicalbooks | 10 | 0 | 0.3945 | 0.3652 | 0.341 | 0.7123 | 0.0 | 0.5 | 16361.4 | 0.25709 |
| claude-sonnet-paid | kitab_historicalbooks | 10 | 0 | 0.467 | 0.4595 | 0.4418 | 0.7278 | 0.0 | 0.5 | 8889.1 | 0.036676 |
| claude-fable-paid | kitab_historicalbooks | 10 | 0 | 0.5382 | 0.525 | 0.5084 | 0.7403 | 0.0 | 0.5 | 11554.6 | 0.29833 |
| gpt-4o-paid | kitab_historicalbooks | 10 | 0 | 0.7669 | 0.7584 | 0.7495 | 0.93 | 0.0 | 0.5 | 4187.0 | 0.02297 |
| gpt-4.1-paid | kitab_historicalbooks | 10 | 0 | 0.8068 | 0.7951 | 0.7557 | 1.042 | 0.0 | 0.4104 | 3433.1 | 0.021952 |
| gpt-5-paid | kitab_historicalbooks | 10 | 0 | 0.843 | 0.8361 | 0.8299 | 0.958 | 0.0 | 0.4 | 4880.0 | 0.01158 |
| gpt-4o-mini-paid | kitab_historicalbooks | 10 | 0 | 0.8653 | 0.8586 | 0.8297 | 0.9907 | 0.0 | 0.2206 | 4186.9 | 0.033759 |
| claude-haiku-paid | kitab_historicalbooks | 10 | 0 | 0.9163 | 0.9145 | 0.9116 | 0.9907 | 0.0 | 0.5 | 3784.6 | 0.009388 |
| claude-fable-paid | kitab_historyar | 10 | 0 | 0.2053 | 0.1996 | 0.0901 | 0.5745 | 0.0 | 0.5461 | 7987.5 | 0.10954 |
| claude-fable-5-1-paid | kitab_historyar | 10 | 0 | 0.2319 | 0.23 | 0.0878 | 0.5851 | 0.0 | 0.5501 | 8720.0 | 0.10599 |
| claude-opus-paid | kitab_historyar | 10 | 0 | 0.289 | 0.2738 | 0.1396 | 0.7234 | 0.0 | 0.5357 | 13876.1 | 0.173595 |
| claude-sonnet-paid | kitab_historyar | 10 | 0 | 0.308 | 0.3061 | 0.1667 | 0.734 | 0.0 | 0.3 | 5621.2 | 0.012168 |
| gpt-4.1-paid | kitab_historyar | 10 | 0 | 0.4335 | 0.424 | 0.2883 | 0.8404 | 0.0 | 0.3873 | 5354.4 | 0.014144 |
| gpt-4o-paid | kitab_historyar | 10 | 0 | 0.4411 | 0.4335 | 0.3243 | 0.8723 | 0.0 | 0.3125 | 3954.0 | 0.01758 |
| gpt-5-paid | kitab_historyar | 10 | 0 | 0.4658 | 0.4544 | 0.3086 | 0.883 | 0.0 | 0.2479 | 6206.5 | 0.0095 |
| claude-haiku-paid | kitab_historyar | 10 | 0 | 0.6027 | 0.5913 | 0.5293 | 1.0106 | 0.0 | 0.3 | 3779.6 | 0.005719 |
| gpt-4o-mini-paid | kitab_historyar | 10 | 0 | 0.6217 | 0.6084 | 0.518 | 0.9468 | 0.0 | 0.4184 | 2979.8 | 0.028306 |
| claude-fable-paid | kitab_onlinekhatt | 10 | 0 | 0.0791 | 0.0725 | 0.0617 | 0.2875 | 0.2 | 1.0 | 6032.1 | 0.11567 |
| claude-fable-5-1-paid | kitab_onlinekhatt | 10 | 0 | 0.0879 | 0.0813 | 0.0705 | 0.3625 | 0.0 | 1.0 | 5759.9 | 0.05817 |
| claude-opus-paid | kitab_onlinekhatt | 10 | 0 | 0.1209 | 0.1165 | 0.1035 | 0.375 | 0.0 | 1.0 | 6435.7 | 0.08906 |
| claude-sonnet-paid | kitab_onlinekhatt | 10 | 0 | 0.1451 | 0.1385 | 0.1145 | 0.4875 | 0.0 | 0.94 | 3939.2 | 0.011624 |
| gpt-4.1-paid | kitab_onlinekhatt | 10 | 0 | 0.2571 | 0.2549 | 0.2423 | 0.6 | 0.0 | 0.7 | 1449.4 | 0.013988 |
| gpt-4o-paid | kitab_onlinekhatt | 10 | 0 | 0.3099 | 0.3055 | 0.2996 | 0.65 | 0.0 | 0.9 | 1734.6 | 0.017415 |
| gpt-5-paid | kitab_onlinekhatt | 10 | 0 | 0.3165 | 0.3055 | 0.2863 | 0.75 | 0.0 | 0.7 | 10627.9 | 0.009375 |
| claude-haiku-paid | kitab_onlinekhatt | 10 | 0 | 0.5143 | 0.4945 | 0.4868 | 0.8875 | 0.0 | 0.9 | 1644.5 | 0.005357 |
| gpt-4o-mini-paid | kitab_onlinekhatt | 10 | 0 | 0.5626 | 0.5538 | 0.4141 | 1.0875 | 0.0 | 0.3 | 1226.7 | 0.027481 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
