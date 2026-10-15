# Design: evaluation standards for every experiment

Map items: `fdr`, `wer-lines`, `decision-rules`, `seed-instability`, `gates`, `paper-groupings`, `R8`, `R6-R11`, `box-cleanup`.
These rules apply to every checklist item that produces a number.

## The contract (from PLAN-2026-10-13 §3.9, extended)
1. **Panel.** Score every test set in full: KHATT (KITAB 200, plus the full v1.0 test once added), NAKBA test 2,095, NAKBA blind
   2,671, Muharaf test 1,334, RASAM-1 734, RASAM-2 372. Score each as stored and ink-cropped, one line at a time in fp32
   (`crnn_eval_panel.py`). Lines a system trained on are flagged and excluded (`in_train`).
2. **Store.** Every reading enters `results/store/` (`store_build.py`). Its CRNN edits must reconcile exactly with the stored
   ones.
3. **Seeds.** At least 2 seeds for a training claim; 3 where the effect is under 1 point. Report seed-pooled hierarchical
   intervals (seeds, then lines). Lines count as better or worse only when the seed ranges are disjoint.
4. **Intervals.** Line bootstrap, page-cluster bootstrap where page IDs exist (NAKBA, RASAM), and seed-pooled.
5. **Many comparisons.** When scanning letter × form cells or detector tables, control the false-discovery rate
   (Benjamini–Hochberg, q = 0.1) and report which cells survive (`fdr`).
6. **Units.** CER at the CRNN-native and no-diacritics rungs; no-dots CER and the dot tax; the space tax; WER; and lines fixed
   or broken (`wer-lines`).
7. **Views.**
   - Analysis A: buckets, transitions, beyond-noise lines, line × system matrix.
   - Analysis B: letters, confusions, PAWs, words.
   - Detectors, with the residual.
   - Dots and conventions views.
8. **Gallery.** Best, worst and random lines for every result that moves a metric. Full-width images; correction C8 explains the
   old 640 px crop.
9. **Ledger.** A panel in the evidence block; the research map item updated (status, grade, evidence link); the checklist ticked.
10. **Decision rule first.** Write the keep/drop rule into the checklist item before the run (`decision-rules`). The rules for
    proposed items are in the design documents.
11. **Seed instability** (`seed-instability`): characterise the lines that flip across identical trainings and report the share of
    a comparison's "better" lines that are unstable lines.

## Validation gates for a characteristic (`gates`; characteristics-framework §7)
- **V1 reliable:** deterministic, tested, bootstrap CI.
- **V2 predictive:** joint model, leave-one-dataset-out and leave-one-family-out (done for the current features: §8).
- **V3 causal:** an intervention (geometry transfer, synthesis, renders) or a natural experiment (KHATT fixed text).
- **V4 generalises:** another Arabic-script language and the Latin analogue (R10).
- **V5 useful:** acting on it changes an outcome (R8: coverage- or novelty-based selection beats random; transformation choice).
- **V6 adoptable:** library, datasheets, documentation (R11).

## Claims register (`paper-groupings`)
Kept in `PLAN-2026-10-15-experiments.md`: for each future write-up, the findings already at A/B and what is missing. Update it
whenever a grade changes.

## Image layer second pass and release (`R6-R11`)
- **R6:** px per x-height, blur, bleed-through and line overlap per line (extends store_features). Includes the vertical-overlap
  and baseline measures from `lorigo2006survey`.
- **R9:** causal synthetic probes (gate V3), shared with the resolution probe.
- **R11:** release, with datasheet cards, the library (store and analyses as a package) and a write-up.

## Housekeeping (`box-cleanup`)
After each queue, back up new weights to the SSD (`aocr-box-backup/`, verified sizes). Delete regenerable pickles on the 3090
once no queued job needs them. Keep the footprint minimal.
