# Benchmark summary — `20260903T130126Z`

CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. `micro` = corpus-weighted, `macro` = per-sample mean.
`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).
Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) are in `summary.csv`; see `docs/metrics.md` for definitions.

| model | dataset | n | errors | cer_micro_raw | cer_micro_light | cer_micro_diac_insensitive | car_micro_light | wer_micro_raw | mer_micro_raw | exact_raw | diac_f1 | latency_ms_mean | cost_usd_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-fable-paid | kitab_khatt | 10 | 0 | 0.2465 | 0.2306 | 0.2156 | 0.7694 | 0.6311 | 0.5662 | 0.1 | 0.8 | 9025.1 | 0.19683 |
| claude-opus-paid | kitab_khatt | 10 | 0 | 0.3014 | 0.2847 | 0.2754 | 0.7153 | 0.623 | 0.5891 | 0.1 | 0.44 | 8305.4 | 0.11849 |
| claude-fable-5-1-paid | kitab_khatt | 10 | 0 | 0.3209 | 0.3063 | 0.2935 | 0.6937 | 0.6885 | 0.6087 | 0.1 | 0.65 | 8488.0 | 0.09433 |
| claude-sonnet-paid | kitab_khatt | 10 | 0 | 0.344 | 0.3279 | 0.3207 | 0.6721 | 0.8443 | 0.8047 | 0.0 | 0.9 | 3305.5 | 0.012826 |
| gpt-4.1-paid | kitab_khatt | 10 | 0 | 0.3936 | 0.3748 | 0.3496 | 0.6252 | 0.7541 | 0.736 | 0.0 | 0.7167 | 2683.7 | 0.015748 |
| gpt-4o-paid | kitab_khatt | 10 | 0 | 0.4699 | 0.4486 | 0.4275 | 0.5514 | 0.8525 | 0.8387 | 0.0 | 0.4 | 1805.5 | 0.019355 |
| gpt-5.6-sol-paid | kitab_khatt | 10 | 0 | 0.4823 | 0.4775 | 0.4656 | 0.5225 | 0.9016 | 0.873 | 0.0 | 0.88 | 60027.6 | 0.653728 |
| gpt-5.6-luna-paid | kitab_khatt | 10 | 0 | 0.5035 | 0.4919 | 0.4692 | 0.5081 | 0.8689 | 0.848 | 0.0 | 0.7 | 39472.7 | 0.03831 |
| gpt-5-paid | kitab_khatt | 10 | 0 | 0.5053 | 0.4829 | 0.433 | 0.5171 | 0.9426 | 0.9127 | 0.0 | 0.2 | 3232.6 | 0.010895 |
| gpt-5.6-terra-paid | kitab_khatt | 10 | 0 | 0.5195 | 0.5117 | 0.5054 | 0.4883 | 0.8607 | 0.84 | 0.0 | 0.9 | 47914.8 | 0.295964 |
| claude-haiku-paid | kitab_khatt | 10 | 0 | 0.5745 | 0.5586 | 0.5417 | 0.4414 | 0.9754 | 0.937 | 0.0 | 0.8 | 3057.9 | 0.006133 |
| gpt-4o-mini-paid | kitab_khatt | 10 | 0 | 0.7801 | 0.7766 | 0.5743 | 0.2234 | 0.9918 | 0.9918 | 0.0 | 0.1 | 1618.2 | 0.030969 |
| claude-fable-paid | kitab_muharaf | 10 | 0 | 0.316 | 0.3031 | 0.2874 | 0.6969 | 0.6452 | 0.6186 | 0.2 | 0.825 | 7507.8 | 0.13197 |
| claude-fable-5-1-paid | kitab_muharaf | 10 | 0 | 0.4027 | 0.3919 | 0.3819 | 0.6081 | 0.8065 | 0.6944 | 0.1 | 0.825 | 7734.6 | 0.06697 |
| claude-opus-paid | kitab_muharaf | 10 | 0 | 0.4066 | 0.39 | 0.3799 | 0.61 | 0.7957 | 0.7475 | 0.1 | 0.8 | 7302.2 | 0.083985 |
| claude-sonnet-paid | kitab_muharaf | 10 | 0 | 0.4759 | 0.4653 | 0.4488 | 0.5347 | 0.871 | 0.8438 | 0.1 | 0.7 | 3002.7 | 0.010034 |
| gpt-5-paid | kitab_muharaf | 10 | 0 | 0.5588 | 0.5425 | 0.5197 | 0.4575 | 0.8925 | 0.8737 | 0.1 | 0.7118 | 2655.9 | 0.00779 |
| gpt-4.1-paid | kitab_muharaf | 9 | 1 | 0.5753 | 0.5686 | 0.556 | 0.4314 | 0.9121 | 0.9022 | 0.1111 | 0.7407 | 2511.1 | 0.011112 |
| gpt-4o-paid | kitab_muharaf | 9 | 1 | 0.593 | 0.5784 | 0.546 | 0.4216 | 1.0 | 0.9681 | 0.0 | 0.5556 | 2063.6 | 0.01367 |
| claude-haiku-paid | kitab_muharaf | 10 | 0 | 0.6455 | 0.6351 | 0.6201 | 0.3649 | 0.9892 | 0.9485 | 0.0 | 0.7 | 1755.5 | 0.003192 |
| gpt-5.6-sol-paid | kitab_muharaf | 10 | 0 | 0.6821 | 0.6776 | 0.6201 | 0.3224 | 0.9462 | 0.9167 | 0.1 | 0.7182 | 72170.6 | 0.829748 |
| gpt-5.6-terra-paid | kitab_muharaf | 10 | 0 | 0.6956 | 0.6873 | 0.6732 | 0.3127 | 0.9892 | 0.9388 | 0.1 | 0.8 | 35248.8 | 0.216482 |
| gpt-4o-mini-paid | kitab_muharaf | 9 | 1 | 0.7808 | 0.7686 | 0.61 | 0.2314 | 0.967 | 0.967 | 0.1111 | 0.1457 | 2929.0 | 0.019863 |
| gpt-5.6-luna-paid | kitab_muharaf | 10 | 0 | 0.8092 | 0.7992 | 0.6988 | 0.2008 | 1.043 | 0.97 | 0.1 | 0.5 | 30466.4 | 0.02859 |

Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.
