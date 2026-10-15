# Design: adapting to a writer (CRNN few-shot, VLM in-context, calibration sentence)

Map items: `writer-adapt-crnn`, `vlm-incontext-writer`, `calibration-sentence`, `personalised-synthesis` (see
`shape-programme.md`), `E6`, `R0.4`, `P1.8`, `khatt-full`. Origin: the user's ideas of 2026-10-14 and 2026-10-15.

## Testbeds (we have no writer IDs for KHATT, so units are manuscripts, pages or a single-writer set)

| Testbed | Units | Lines per unit | Notes |
|---|---|---|---|
| RASAM-2 test | 15 manuscripts | 24–31 | Maghrebi; unseen by the standard-split CRNN |
| RASAM-1 test | 3 manuscripts | 110–353 | Maghrebi; split each into adaptation and test pages |
| NAKBA | 1 writer (Omar al-Saleh memoirs) | 15k train, 2,095 test, 2,671 blind | adapt with K lines from NAKBA train; test on test and blind |
| Muharaf test | pages unknown | — | page IDs not in the HF copy; skip unless R0.4 recovers them |
| KHATT v1.0 (Teklia) | no writer IDs | — | fixed-text paragraphs give writer-held-out text pairs only if writers can be inferred (P1.8) |

## A. CRNN few-shot adaptation (`writer-adapt-crnn`, GPU, short runs)
- **Base model:** the standard-split CRNN (seed 0). It never saw RASAM, NAKBA or KHATT.
- **Adaptation sets:** K ∈ {4, 8, 16} lines per unit (RASAM-2: K ≤ 8, keeping at least 16 test lines), chosen three ways:
  1. random (3 draws);
  2. **letter coverage:** greedy set cover over letter × positional form, the fewest lines covering the most forms. This is the
     user's "samples that capture most of the letters";
  3. **novelty:** the lines with the most high-novelty letters under the base model.
- **Methods:**
  - (i) fine-tune all weights, about 200 steps, small learning rate;
  - (ii) train only the BiLSTM and output layer;
  - (iii) no training: a prototype memory. Embed the writer's letters from the K lines (CNN features in their aligned spans), then
    re-score the per-column posteriors by nearest-prototype similarity, as retrieval-augmented LMs (kNN-LM) do.
- **Metrics:** the unit's remaining lines, with analyses A and B. For RASAM, also the ق→ف rate.
- **Keep rule:** coverage-chosen lines beat random by ≥ 1 point at K = 8 on most RASAM-2 manuscripts (sign test across
  manuscripts) → the coverage rule is useful (gate V5).
- **E6 shares this machinery.** The few-shot Maghrebi curve uses RASAM-1 manuscripts with K ∈ {16, 100, 300} and records the
  K at which ق→ف falls below 5 %.

## B. VLM in-context writer calibration (`vlm-incontext-writer`, GPU with a VLM)
**User's design:** one prompt (or two consecutive turns) holding a labelled reference line from the same writer (image plus its
transcription), then the test line. A user would write one known sentence once, and the system uses it as calibration.

- **Models:** Qwen3-VL-8B base, and the suite fine-tuned Qwen3-VL-8B (and 4B if time).
- **Prompts:**
  - (a) the test line alone (current prompt, the baseline);
  - (b) one reference image and label, then the test line;
  - (c) two or three references;
  - (d) the reference is the calibration sentence (C below) written by the same writer. Approximated on existing data by the
    unit's line with the best letter coverage.
- **Reference choice:** random vs letter-coverage vs nearest in style.
- **Controls:** a reference from a different writer (tests that the gain is writer-specific); the reference label alone without
  its image (tests that the gain is not just text priming).
- **Testbeds:** RASAM-2 manuscripts and NAKBA (reference from NAKBA train, test from blind).
- **Keep rule:**
  - the gain over (a) is ≥ 2 points on RASAM-2 manuscripts or NAKBA, for base or fine-tuned;
  - the different-writer control gains less;
  - the label-only control gains less.
- **Needs (setup):**
  - the base VLM downloaded on the 3090 from the Hub (no upload from the laptop);
  - the 142 MB adapter copied for the fine-tuned arm;
  - runner support for multi-image prompts;
  - token log-probabilities saved (`vlm-logprobs`).
- **Prior art:**
  - in-context writer adaptation without weight updates (arXiv 2603.29450, 8M-parameter model);
  - MetaHTR (one gradient step);
  - MetaWriter (prompt tuning on unlabelled images).

  Ours is in-context with a general VLM, in Arabic, with a designed calibration sentence.

## C. The calibration sentence (`calibration-sentence`, CPU)
- **Goal:** the shortest natural Arabic text (one or two lines) in which every letter appears in every positional form it can
  take, ideally with common ligatures (lam-alef) and dotted pairs.
- **Method:**
  1. Candidate sentences from the training texts and the lexicon.
  2. Greedy set cover over letter × form, then a small search for natural, short combinations.
  3. Check readability with a native speaker (the user).
- **Output:** 3–5 candidate sentences with their coverage. It feeds B(d) and the own-dataset copy-texts (S1).

## D. Writer and page structure (`R0.4`, `P1.8`)
- **R0.4:** try to recover writer or page IDs (Muharaf page structure, KHATT originals). Without them, the page clusters (NAKBA,
  RASAM) carry the clustered intervals. The original KHATT is free for research but needs a request; low priority.
- **P1.8:** in KHATT (full v1.0), fixed-text paragraphs are written by many writers. Hold the text constant and measure the
  between-line variance, which is writer and scan variance. Descriptive.
