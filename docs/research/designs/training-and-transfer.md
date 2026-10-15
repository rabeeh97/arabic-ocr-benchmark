# Design: training, transfer and controls

Map items: `nakba-indomain`, `khatt-full` (in-domain part), `E1`, `E3`, `E4`, `E5-retsinas`, `E7`, `P-L5`, `P-L6`,
`aug-11`, `ketaba-dora`, `X5-print`, `R10`, `X3-retrain`, `C-track`. Detailed specs for E1–E4 and C0–C4 are in
[`../reference/NEXT_STEPS.md`](../reference/NEXT_STEPS.md) (E-track and C-track sections).

## In-domain controls (answer "everything is zero-shot")
- **`nakba-indomain`:**
  - CRNN (standard recipe) trained on NAKBA train (15,250 lines plus 517 validation; data verified on the 3090), 2 seeds;
  - scored on the panel; NAKBA test and blind scored on clean vs leaked lines separately (ledger §17);
  - repeat the diagnostics: crop, novelty battery, PAW-gap splits, hamza emission;
  - **rule:** a finding that reverses in-domain is restricted to "under domain shift".
- **`khatt-full` in-domain:**
  - a CRNN on KHATT v1.0 train (4,672 lines), tested on the KHATT v1.0 test;
  - fixed-text vs unique-text lines reported separately, because texts recur across splits.

## Architectures and references
- **`E5-retsinas`:** the Retsinas best-practice CRNN (column max-pool, CTC shortcut, median padding) as a second architecture;
  2 seeds; repeat the key diagnostics. Shares X-G3 settings (`dots-and-skeleton-studies.md` §8).
- **`E1`** (VLM; NEXT_STEPS E1):
  - frozen vs unfrozen ViT, controlled;
  - E1.1: bench the diff-LR arm on KHATT, NAKBA and RASAM-1/2 (about 1 h; the base VLM downloaded on the 3090 plus the 142 MB
    adapter);
  - E1.2: train arm (b), about 10 h; only if E1.1 gains ≥ 3 points on 2 of 4 held-out sets;
  - E1.3: compare, with the primary outcome letter error on medial and dot-ambiguous letters.
- **`ketaba-dora`:** DoRA / rsLoRA arm; gated until E1 says whether the encoder or the decoder is the bottleneck.

## Transfer and data
- **`E7`:** multilingual pre-training with a freeze sweep.
  - Pre-training: CRNN on Latin (IAM, READ, NorHand), on Arabic-script (KHATT, NUST, PHTD), and on both.
  - Fine-tune at K = 100, 500 and 1,000 with no freezing, CNN 1–3 frozen, all CNN frozen, or BiLSTM only.
  - Priors: `aradillas2021boosting`, `alazzawi2026seqmodel`, `alazzawi2026crossling`.
  - Needs new data downloads.
- **`E3`:** data scaling on the K grid (100, 500, 1k, 2k, 4k, full), reported in characters per letter × form (E3′ on their grid,
  next to the CRNN curve of `alazzawi2026gap`).
- **`aug-11`:** cross-lingual / cross-dataset mixing; part of E7.
- **`R10`:** cross-script check (gate V4).
  - Run the store, analyses and detectors on Persian PHTD (on the Hub), Urdu NUST-UHWR and Pashto if available.
  - Also the Latin analogue on IAM (`marti2002iam`), to reproduce the 30 % vs 15 % confusion gap.
- **`X5-print`:** add a line-level printed Arabic set and a page track; a printed regression check (Baseer).
- **`X3-retrain`:** retrain without the CER-HV-flagged Muharaf train lines (`data-and-label-audit.md` §7).

## VLM training tricks (from Baseer-Nakba)
- **`P-L5`:** checkpoint weight averaging of the AIN-7B NAKBA adapters (400 … 1,936 steps); NAKBA CER of the averaged vs the
  final adapter.
- **`P-L6`:** per-sample loss logging during a VLM training run (dataset cartography, `swayamdipta2020cartography`); compare
  hard-to-learn lines with the CER-HV flags.
- **`E4`:** Baseer replay (lower rank, fewer steps, NAKBA-domain lines).

## Curriculum (`C-track`, parked; NEXT_STEPS §C)
- **C0:** render isolated letters in every positional form; baseline.
- **C1:** curriculum vs controls (the main test): equal total steps, 3 seeds.
- **C2:** letter-incremental probe: add shape families one at a time.
- **C3:** factorised head (same as X-G9).
- **C4:** checkpoint averaging.

Parked because the shape programme covers the same question ("teach shapes first") with real glyphs. Revisit after
`synthesis-v1`.
