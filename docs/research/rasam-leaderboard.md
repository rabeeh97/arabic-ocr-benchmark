# RASAM leaderboard (published vs ours)

There is no official public leaderboard for RASAM. This table collects every published number we found and our own results on the official Hugging Face test
splits (`calfa-ai/RASAM-1` test, 734 lines; `calfa-ai/RASAM-2` test, 372 lines). The live version is the "RASAM leaderboard" tab of the dashboard
(`scripts/colab_dashboard.py`), which picks up new results automatically.

**Comparability.** The papers score their own test lines and do not say how CER is averaged (on our predictions, micro vs mean-per-page differ by about 1 point),
so a gap under about 1 point is not a clear win. Our numbers are micro CER, raw profile (diacritics kept), one seed unless noted.

Literature sweep 2026-10-25 (OpenAlex citations of the three RASAM papers plus web search): no public leaderboard, benchmark server or shared task exists for
RASAM or Maghrebi HTR (the nearest, NakbaNLP 2026 on CodaBench, uses the Omar Al-Saleh manuscripts). No paper says it scored on the Hugging Face test splits.
Three Springer papers that cite RASAM could not be read (paywall) and may hold further numbers.

## RASAM-1 test (target: 4.8 %)
| CER | system | source |
|---|---|---|
| 3.23 % | meta-word CRNN, specialised in-domain model, different split (32 pages) | [Lucas et al. 2022, arXiv 2211.16147](https://arxiv.org/abs/2211.16147), Table 3 |
| 4.8 % | word-based CRNN trained on RASAM-1 | [RASAM, ICDAR 2021 workshops](https://halshs.archives-ouvertes.fr/halshs-03430697) |
| 4.95 % | HAFITH (SigLIP2 + RoBERTa, 642 M; Muharaf + KHATT + RASAM), "RASAM" version not stated, own 906-line test split | [HF model card mdnaseif/hafith](https://huggingface.co/mdnaseif/hafith) |
| **5.35 %** | **ours: A3 char CRNN (RASAM-1+2, half width, 64 px) + character LM** | this project, 2026-10-25 |
| 5.50 % | word-based CRNN trained on RASAM-1 + RASAM-2 | [CHR 2024, ceur-ws Vol-3834 paper35](https://ceur-ws.org/Vol-3834/paper35.pdf), Table 4 |
| 6.55 % | ours: A3, greedy decoding | this project |
| 6.82 % | ours: A1 (RASAM-1 only), greedy | this project |
| 23.1 % | Phoenix (Kraken), re-scored by us on the HF crops (reports 7.67 % on its own split) | arXiv 2608.19385 |

## RASAM-2 test (target: 6.79 %)
| CER | system | source |
|---|---|---|
| 6.79 % | word-based CRNN trained on RASAM-1 + RASAM-2 | [CHR 2024](https://ceur-ws.org/Vol-3834/paper35.pdf), Table 4 |
| **10.45 %** | **ours: A3 + character LM** | this project |
| 12.84 % | ours: A3, greedy | this project |
| 18.47 % | ours: A2 (RASAM-2 only), greedy | this project |
| 18.7 % | Phoenix re-scored by us on the HF crops | arXiv 2608.19385 |
| 30.91 % | RASAM-1-only model, unseen manuscripts | CHR 2024 |

**Where we stand.** RASAM-1 is at the published level (5.35 % vs 4.8 % / 5.50 %); RASAM-2 is the gap (10.45 % vs 6.79 %). The plan in
`designs/colab-experiment-plan.md` targets RASAM-2 first.

## Only systems scored on the official Hugging Face test splits (strictly comparable)
| | RASAM-1 test (734 lines) | RASAM-2 test (372 lines) |
|---|---|---|
| ours, A3 + character LM | **5.35 %** | **10.45 %** |
| ours, A3 greedy | 6.55 % | 12.84 % |
| Phoenix (Kraken), re-scored by us | 23.1 % | 18.7 % |
| TariMa Kraken model (RASAM unseen), re-scored by us | 42.6 % | 29.9 % |

On this strict view we are first on both sets, but it holds few entries: the strongest published systems (Calfa's word-based CRNNs) are not public, so
they cannot be re-scored. HAFITH is public and is the next candidate to re-score on the official split; its training lines must first be checked for
overlap with the official test lines (any overlap would inflate its score).

## HAFITH re-scored on the official split (2026-10-25)

HAFITH (public, 642 M) run on a Colab T4 on all 1,106 official test lines (transformers 4.57, its own `HAFITHRecognizer`), scored on the Mac with the project
metric (`scripts/rasam_score_preds.py`). Overlap check by normalised text (lines of at least 12 characters): **266 official RASAM-1 test lines are in HAFITH's
training data and 96 in its validation data** (87 more in its own test split); none of the RASAM-2 test lines are, so HAFITH's "RASAM" is RASAM-1.

| Official test | HAFITH | ours: 96 px greedy | ours: 96 px + LM |
|---|---|---|---|
| RASAM-1, all 734 lines | 6.27 % | 5.39 % | **4.58 %** |
| RASAM-1, 372 clean lines (not in HAFITH train/val) | 12.34 % | 6.75 % | **5.72 %** |
| RASAM-2, all 372 lines (no overlap) | 17.56 % | 12.14 % | **9.66 %** |

On lines it never saw, HAFITH's error is twice its overall figure; its reported 4.95 % is on its own split and benefits from memorised lines.

## Calfa's AMIDDA reports on the official test lines (2026-10-26)

Calfa's AMIDDA dataset ([HF](https://huggingface.co/datasets/calfa-ai/AMIDDA), created 2026-05-20 like the HF RASAM mirrors) has a test split that contains
**exactly** the official test lines: all 734 RASAM-1, 372 RASAM-2 and 229 TariMa test images are identical by MD5 (`scripts/amidda_split_check.py`,
`results/rasam/amidda_split_check.json`). So the per-corpus numbers on the [AMIDDA model card](https://huggingface.co/calfa-ai/amidda1.0Line4B)
(paper: [HIP 2026 @ ICDAR 2026](https://enc.hal.science/hal-05693582)) are the first published results on the same lines as ours:

| official test | Calfa CTC, one CRNN per corpus | ours (60 ep + char LM + AraGPT2) | AMIDDA 1.0 Line 4B (VLM) | Gemini 3 Flash 25-shot |
|---|---|---|---|---|
| RASAM-1 | **3.6 %** | 4.12 % | 25.0 % | 16.5 % |
| RASAM-2 | **4.9 %** | 8.92 % | 27.6 % | 29.0 % |
| TariMa | **3.9 %** | 7.04 % | 24.2 % | 25.6 % |

The Calfa CTC baselines are not public and the card does not state their training data, line height or CER averaging (the paper PDF is behind a
bot check from here), so this is "reported on the same lines", not reproduced. The gap is largest on RASAM-2 and TariMa. The AMIDDA VLM is public; re-running
it needs a Colab GPU (pending budget). Note: our local AMIDDA copy is incomplete (no validation files, some train shards missing), so only the test
comparison is established.

The full local leaderboard (strict board, other splits, our ideas, why scores differ) is `docs/research/rasam_leaderboard.html`, built by
`scripts/build_rasam_leaderboard.py`.
