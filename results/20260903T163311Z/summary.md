# Benchmark summary — `20260903T163311Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openrouter-qwen3-vl-235b | kitab_hindawi | 20 | 0 | 0.5754 | 0.5542 | 0.5256 | 0.4458 | 0.7056 | 0.7034 | 0.0 | 0.4081 | 8282.4 | 0.011455 |
| pixtral-12b | kitab_hindawi | 20 | 0 | 0.7329 | 0.7189 | 0.7039 | 0.2811 | 0.8241 | 0.8199 | 0.0 | 0.1693 | 1901.2 | 0.002615 |
| mistral-medium | kitab_hindawi | 20 | 0 | 0.7338 | 0.7194 | 0.7034 | 0.2806 | 0.8428 | 0.8344 | 0.0 | 0.0564 | 3128.4 | 0.009952 |
| gemini-flash-lite | kitab_hindawi | 20 | 0 | 0.7567 | 0.747 | 0.7372 | 0.253 | 0.7685 | 0.7643 | 0.0 | 0.2979 | 14005.3 | 0.003003 |
| openrouter-qwen2.5-vl-72b | kitab_hindawi | 20 | 0 | 0.7724 | 0.7625 | 0.7473 | 0.2375 | 0.8265 | 0.824 | 0.0 | 0.2361 | 4193.3 | 0.011679 |
| mistral-small | kitab_hindawi | 20 | 0 | 0.8866 | 0.8822 | 0.8762 | 0.1178 | 0.9278 | 0.9265 | 0.0 | 0.0632 | 1467.1 | 0.003562 |
| openrouter-llama-4-maverick | kitab_hindawi | 20 | 0 | 0.8909 | 0.8866 | 0.8809 | 0.1134 | 0.9219 | 0.9194 | 0.0 | 0.1502 | 2954.6 | 0.005204 |
| openrouter-gemma-4-31b-free | kitab_hindawi | 4 | 16 | 0.9042 | 0.9031 | 0.9017 | 0.0969 | 0.9176 | 0.9176 | 0.0 | 0.3 | 71559.7 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_historicalbooks | 10 | 0 | 0.3271 | 0.3124 | 0.2545 | 0.6876 | 0.6967 | 0.6677 | 0.0 | 0.3482 | 9791.0 | 0.004796 |
| gemini-flash-lite | kitab_historicalbooks | 10 | 0 | 0.7218 | 0.7156 | 0.7059 | 0.2844 | 0.8243 | 0.7946 | 0.0 | 0.5 | 9730.1 | 0.001332 |
| mistral-medium | kitab_historicalbooks | 10 | 0 | 0.792 | 0.7832 | 0.7517 | 0.2168 | 0.9658 | 0.9367 | 0.0 | 0.5095 | 3883.3 | 0.00477 |
| openrouter-qwen2.5-vl-72b | kitab_historicalbooks | 10 | 0 | 0.864 | 0.8607 | 0.8576 | 0.1393 | 0.9393 | 0.9335 | 0.0 | 0.5615 | 4177.7 | 0.005535 |
| pixtral-12b | kitab_historicalbooks | 10 | 0 | 0.8822 | 0.8689 | 0.8414 | 0.1311 | 1.0824 | 0.9943 | 0.0 | 0.1108 | 4343.2 | 0.001366 |
| openrouter-llama-4-maverick | kitab_historicalbooks | 10 | 0 | 0.8956 | 0.8938 | 0.8906 | 0.1062 | 0.9767 | 0.9752 | 0.0 | 0.25 | 4615.9 | 0.002397 |
| mistral-small | kitab_historicalbooks | 10 | 0 | 0.913 | 0.9099 | 0.914 | 0.0901 | 0.9984 | 0.9984 | 0.0 | 0.2198 | 3347.9 | 0.001816 |
| openrouter-gemma-4-31b-free | kitab_historicalbooks | 1 | 9 | 0.9169 | 0.9169 | 0.9107 | 0.0831 | 0.8889 | 0.8889 | 0.0 | 0.0 | 3268.6 | 0.0 |
| gemini-flash-lite | kitab_historyar | 20 | 0 | 0.3359 | 0.3283 | 0.2435 | 0.6717 | 0.7981 | 0.7083 | 0.0 | 0.5181 | 9906.8 | 0.002478 |
| openrouter-gemma-4-31b-free | kitab_historyar | 2 | 10 | 0.34 | 0.32 | 0.2184 | 0.68 | 0.7647 | 0.6842 | 0.0 | 0.5 | 5536.9 | 0.0 |
| openrouter-qwen3-vl-235b | kitab_historyar | 20 | 0 | 0.3594 | 0.3468 | 0.2358 | 0.6532 | 0.7653 | 0.7512 | 0.0 | 0.5497 | 5581.1 | 0.002595 |
| openrouter-llama-4-maverick | kitab_historyar | 20 | 0 | 0.415 | 0.4108 | 0.2955 | 0.5892 | 0.8263 | 0.8111 | 0.0 | 0.377 | 4885.2 | 0.004284 |
| mistral-medium | kitab_historyar | 13 | 7 | 0.4688 | 0.4575 | 0.4012 | 0.5425 | 0.9023 | 0.8889 | 0.0 | 0.2308 | 3830.7 | 0.002552 |
| openrouter-qwen2.5-vl-72b | kitab_historyar | 20 | 0 | 0.5362 | 0.5295 | 0.3831 | 0.4705 | 0.8122 | 0.7972 | 0.0 | 0.4334 | 5142.2 | 0.005589 |
| mistral-small | kitab_historyar | 14 | 6 | 0.9119 | 0.9069 | 0.7518 | 0.0931 | 1.0511 | 0.9863 | 0.0 | 0.2945 | 3381.4 | 0.001263 |
| pixtral-12b | kitab_historyar | 20 | 0 | 1.1288 | 1.1103 | 0.794 | 0.0 | 1.1127 | 1.0 | 0.0 | 0.1933 | 3853.5 | 0.001248 |
| gemini-flash-lite | kitab_onlinekhatt | 20 | 0 | 0.0946 | 0.0925 | 0.0787 | 0.9075 | 0.3519 | 0.3413 | 0.05 | 0.9333 | 7023.6 | 0.002438 |
| openrouter-qwen3-vl-235b | kitab_onlinekhatt | 20 | 0 | 0.3054 | 0.3011 | 0.206 | 0.6989 | 0.6728 | 0.6124 | 0.0 | 0.46 | 2187.3 | 0.002204 |
| openrouter-llama-4-maverick | kitab_onlinekhatt | 20 | 0 | 0.471 | 0.457 | 0.3312 | 0.543 | 0.8148 | 0.763 | 0.0 | 0.453 | 1583.5 | 0.003642 |
| mistral-medium | kitab_onlinekhatt | 12 | 8 | 0.4898 | 0.4823 | 0.471 | 0.5177 | 0.8542 | 0.8039 | 0.0 | 0.8889 | 1566.4 | 0.002236 |
| openrouter-qwen2.5-vl-72b | kitab_onlinekhatt | 20 | 0 | 0.5581 | 0.5505 | 0.4358 | 0.4495 | 0.821 | 0.8012 | 0.0 | 0.6377 | 2367.9 | 0.005023 |
| mistral-small | kitab_onlinekhatt | 14 | 6 | 0.8179 | 0.8117 | 0.6749 | 0.1883 | 0.9825 | 0.9739 | 0.0 | 0.2857 | 4294.4 | 0.001214 |
| pixtral-12b | kitab_onlinekhatt | 18 | 2 | 1.283 | 1.2588 | 0.7949 | 0.0 | 1.1096 | 1.0 | 0.0 | 0.0556 | 1639.5 | 0.001068 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
