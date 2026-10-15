# Design: data and label audit (labels, conventions, leakage, metrics policy)

Map items: `R4.1`, `R4.2`, `R4.3`, `R4.4`, `R2.7`, `R1.3-R1.4`, `R3.2`, `R3.3`, `X5-len`, `X6`, `X3-retrain`, `X7`, `C6`,
`aug-18`, `aug-20`, `khatt-full`, `detectors-open`, `detector-residual`, `main-checkout-sync`. Evidence so far: ledger §3, §4, §12,
§17 and the corrections log.

## 1. Label review with a human ceiling (`R4.1`, human, about 2 h)
- **Tool:** the label-review artifact (https://claude.ai/artifact/FAEdbYULz3g88KQhXoKiPQ).
- **Candidates:** 458 (all 218 CER-HV-reviewed Muharaf test lines, verdict hidden; the top 25 consensus suspects per block),
  **plus** the 20 KHATT hard-core lines (ledger §3).
- **Per line:** label correct / wrong / unclear, and **"can a human read it?"** (yes / partly / no).
- **Outputs:**
  - a label-error rate per set;
  - κ with CER-HV on the 218;
  - a human-readability ceiling for the hard core;
  - lines judged unreadable or mislabelled are flagged and reported separately in headline metrics.
- **Export:** ArtifactData `decisions` collection → `results/label_review/`.

## 2. Leakage and duplicates (`R4.2`, CPU)
- MinHash on normalised text plus a perceptual image hash, within and across sets.
- Also the ~6 % of `muharaf_lines` test lines that occur in train (from NEXT_STEPS X3).
- Already measured: NAKBA's official split repeats training text on 11 % of blind lines (§17); `kitab_muharaf` is 86.5 % in
  Muharaf train.
- **Output:** a leakage table per set; leaked lines flagged in the store.

## 3. Spelling and conventions (`R4.3`, `R4.4`, CPU)
- **R4.3:** the same word spelled several ways inside one set (hamza, tāʾ marbūṭa, yāʾ). Rates per set; examples.
- **R4.4:** a per-dataset conventions table from the dataset papers and our measured fingerprint (§17: hamza, digits, spacing,
  harakat). Note that NAKBA test and blind follow different conventions.

## 4. The diacritic scoring rule (`R2.7`, CPU)
- **Rule check:** strip marks on sets whose policy is "undiacritised" and compare model rankings (rank correlation) with and
  without.
- **Recommendation:** which rung of the ladder is the headline per dataset policy.
- **Includes** the NEXT_STEPS note that a third of Muharaf lines carry sparse marks.

## 5. Profiler outputs and datasheets (`R1.3`, `R1.4`, CPU)
- **R1.3:** share of lines with diacritics per set (the policy fingerprint).
- **R1.4:** a datasheet card per dataset on the Atlas: size, source, conventions, leakage, image profile (store_features) and
  difficulty.

## 6. Difficulty structure (`R3.2`, `R3.3`, `X5-len`, CPU)
- **R3.2:** IRT 2PL difficulty and discrimination per line, with systems as test-takers. Low-discrimination lines feed R4.1.
- **R3.3:** challenge subsets registered as dataset filters: hard core, family-split lines, dot-dense, OOV-heavy, fragments, loose
  crops, slant outliers.
- **X5-len:** CER per length band on the leaderboard; fragment lines (≤ 20 characters, digits, single words) as their own
  subset.

## 7. Pipeline fixes (`X6`, `X7`, `X3-retrain`, `main-checkout-sync`)
- **X6:**
  - add `max_tokens` (and a minimal thinking budget) for API and VLM runs whose outputs hit the cap;
  - re-score the truncated runs;
  - backfill `kitab_khatt` for Baseer-Nakba (NEXT_STEPS second X6).
- **X7:** check whether full-resolution Muharaf scans and line polygons exist (the public copy is stored at 60 px).
- **X3-retrain:** retrain without the 1,424 CER-HV-flagged Muharaf train lines; the effect on in-domain and transfer CER.
- **main-checkout-sync** (needs permission):
  - apply `patches/0001-paddleocr-reverse-output.diff` and re-score PaddleOCR;
  - refresh NEXT_STEPS.md from the master checklist;
  - update the Atlas tracker (`progress.json`) and republish the Atlas.

## 8. Full KHATT v1.0 (`khatt-full`)
- Teklia copy on the SSD: 4,672 train, 963 validation, 1,038 test lines.
- **Add to the panel** with as-stored and ink-cropped variants, canonical IDs `kitab_khatt_full/{split}/{index}`, and a
  fixed-text flag (the text occurs more than once in KHATT).
- 127 of the 200 KITAB-KHATT texts recur in it; mark that overlap.
- Score fixed-text and unique-text lines separately.

## 9. Detectors still to write (`detectors-open`, `detector-residual`, CPU)
- **D4** neighbour-line intrusion: edge-ink rows or several baselines.
- **D21** lam-alef ligature errors.
- **D23** memorisation or contamination signature: GT whitespace or typo quirks reproduced; exact-match rate ≥ 3× the median of
  systems with similar CER. This is also the test behind C6, AIN-7B, kept unverified.
- **D24** numeric and seed instability.
- **Residual review:** each analysis round, look at the worst-fifth lines no detector explains and write the next detector.

## 10. Augmentations tied to policy (`aug-18`, `aug-20`)
- **Augmentation 18:** harakat on/off renders with matching labels (the diacritic-policy mismatch).
- **Augmentation 20:** fragment crops for short lines.

Both run through the synthesis machinery.
