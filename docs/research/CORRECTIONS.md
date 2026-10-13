# Corrections log

Dated list of problems found in earlier results, and what was done about them. The ledger's corrections block is built
from this file, so every number that was ever published can be traced. Newest first.

## 2026-10-13

| # | Problem | Found by | Effect on published numbers | Status |
|---|---|---|---|---|
| C1 | **TTA readings stored in reverse order** (`xai_geom_tta.py` did not reverse the CRNN's visual-order output) | planning agent; confirmed | Edit counts were right (computed on tokens); the TTA gallery in ledger v19 showed reversed text | Fixed: script, stored files, gallery (ledger v20) |
| C2 | **Two different "NAKBA blind" samples.** `eval_data.pkl` and `inkcrop_eval.pkl` share only 118 of their 600 lines | planning agent; confirmed | Blind-test numbers from the two packs are not comparable line for line | The store keys every reading by HF row index; the new panel scores all 2,671 lines |
| C3 | **The CRNN's KHATT set silently dropped 17 of 200 lines** (CTC length filter applied before the ink crop); they are the worst loose crops | planning agent; confirmed in code | Every CRNN KHATT number is on an easier 183-line subset; VLM rows use all 200 | New panel (`xai_prepare_panel.py`) scores every line and only flags `ctc_ok` |
| C4 | **The CRNN trained on Muharaf and RASAM-1 test lines** (all except a 300 / 100 held-out quota) | planning agent; confirmed in `xai_prepare_aligner_data.py` | X-T2's Muharaf rows are mostly training lines; no comparison with published Muharaf numbers is possible | `--standard-split` mode added; standard-split CRNNs (2 seeds) queued on the 3090 |
| C5 | **PaddleOCR (and TrOCR on KHATT) readings are in visual order.** Reversing them: PaddleOCR KHATT 85.6 → 62.7 % CER (crnn-native rung), Muharaf 87.8 → 77.6 %; TrOCR KHATT 80.6 → 51.9 % | `store_build.py` reversal detector (D22) | Leaderboard rows for these engines are too high; the ledger's "PaddleOCR is hurt by loose crops" (`vlm_ink_extent.py`) is unreliable | Open: fix the engine adapters' output order in the main checkout and re-score; until then exclude these rows from cross-model claims |
| C6 | AIN-7B on KITAB-KHATT far better than every other model (6.4 % vs ≥ 20.7 %) and copies GT whitespace quirks | planning agent | — | **Unverified** (AIN's paper names no training sets). Recorded only; AIN left out of cross-model claims |
| C7 | CRNN readings depend on batch padding and numerics (22–39 % of lines change between fp32-bs32 and fp16-bs1) | planning agent | Small net effect, but line-level comparisons across packs mix evaluation noise with model change | New evaluator `crnn_eval_panel.py` scores one line at a time in fp32 |
