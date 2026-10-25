# RASAM leaderboard (published vs ours)

There is no official public leaderboard for RASAM. This table collects every published number we found and our own results on the official Hugging Face test
splits (`calfa-ai/RASAM-1` test, 734 lines; `calfa-ai/RASAM-2` test, 372 lines). The live version is the "RASAM leaderboard" tab of the dashboard
(`scripts/colab_dashboard.py`), which picks up new results automatically.

**Comparability.** The papers score their own test lines and do not say how CER is averaged (on our predictions, micro vs mean-per-page differ by about 1 point),
so a gap under about 1 point is not a clear win. Our numbers are micro CER, raw profile (diacritics kept), one seed unless noted.

## RASAM-1 test (target: 4.8 %)
| CER | system | source |
|---|---|---|
| 3.23 % | meta-word CRNN, specialised in-domain model, different split | Lucas et al. 2022, arXiv 2211.16147 |
| 4.8 % | word-based CRNN trained on RASAM-1 | RASAM, ICDAR 2021 workshops |
| **5.35 %** | **ours: A3 char CRNN (RASAM-1+2, half width, 64 px) + character LM** | this project, 2026-10-25 |
| 5.50 % | word-based CRNN trained on RASAM-1 + RASAM-2 | CHR 2024 (ceur-ws Vol-3834 paper35) |
| 6.55 % | ours: A3, greedy decoding | this project |
| 6.82 % | ours: A1 (RASAM-1 only), greedy | this project |
| 23.1 % | Phoenix (Kraken), re-scored by us on the HF crops (reports 7.67 % on its own split) | arXiv 2608.19385 |

## RASAM-2 test (target: 6.79 %)
| CER | system | source |
|---|---|---|
| 6.79 % | word-based CRNN trained on RASAM-1 + RASAM-2 | CHR 2024 |
| **10.45 %** | **ours: A3 + character LM** | this project |
| 12.84 % | ours: A3, greedy | this project |
| 18.47 % | ours: A2 (RASAM-2 only), greedy | this project |
| 18.7 % | Phoenix re-scored by us on the HF crops | arXiv 2608.19385 |
| 30.91 % | RASAM-1-only model, unseen manuscripts | CHR 2024 |

**Where we stand.** RASAM-1 is at the published level (5.35 % vs 4.8 % / 5.50 %); RASAM-2 is the gap (10.45 % vs 6.79 %). The plan in
`designs/colab-experiment-plan.md` targets RASAM-2 first.
