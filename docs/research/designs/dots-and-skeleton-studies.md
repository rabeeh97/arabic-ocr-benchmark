# Design: dots, skeleton features, spacing and geometry studies

Map items: `PAW-gap-width`, `dot-styles`, `loops-teeth`, `vertical-stacking`, `D21`, `X-G7`, `X-G9`, `X-A6`, `aug-14`,
`aug-16`, `aug-17`, `pen-contrast-aug`, `X-G3`, `R9`, `X-G12`, `geom-matched-aug`, `slant-density`, `geometry-lit`, `pix2pix`,
`vlm-geometry`, `E5-retsinas`. Evidence so far: ledger §5, §7–§11, §15, §18 and the geometry sections (d-geom, d-xfer).

## 1. PAW gap widths (`PAW-gap-width`, CPU)
- From the letter dumps (spans) and the ink, measure for every word-internal PAW boundary and every word boundary the horizontal
  ink gap in pixels. Normalise by stroke width and by x-height.
- **Questions:**
  - Are the PAW gaps where systems insert spaces wider than PAW gaps where they do not?
  - Do they overlap the word-gap distribution?
- **Decision:** overlap → spacing is visually ambiguous (a characterisation result). Narrower → the models over-split.

## 2. Dot-writing styles (`dot-styles`, CPU)
- Classify the dot components of dotted letters:
  - separate dots;
  - a dash (two dots joined: elongated component, aspect ratio > 2);
  - a caret (three dots as a ˆ shape);
  - displaced (horizontal offset from the letter's span beyond a threshold);
  - touching (no separate component; the dot merged with the body).
- Prevalence per dataset. Dot-error rate per style for each family.
- **Keep rule:** a style with ≥ 2× the dot-error rate of separate dots (interval excluding 1) becomes a finding and a synthesis
  requirement. Three-dot letters are already the worst (§15).

## 3. Skeleton features (`loops-teeth`, CPU)
- **Feature classes:**
  - loops: ف ق و م ه ع ص ض ط ظ;
  - teeth: medial and initial ب ت ث ن ي (one tooth), س ش (three);
  - descending tails: ر ز و ن ي ع (final);
  - verticals: ا ل ك ط.
- Confusions between and within classes per family.
- **Tooth-count detector:** in runs of consecutive tooth letters, insertion or deletion of a tooth letter; error against run
  length. This is the Arabic analogue of minim confusion in Latin ("minimum").
- **Keep rule:** a class whose error exceeds the letter baseline in every family becomes a T3 finding.

## 4. Vertical stacking (`vertical-stacking`, CPU)
- Letters written above the previous letter (e.g. initial م or ح stacked, as in محمد or لمح).
- **Detect:** consecutive letters whose span columns overlap by more than half, and whose ink centroids differ vertically by
  more than the x-height.
- Measure reading-order errors (transpositions) and deletions on them.

## 5. Lam-alef and ligatures (`D21`, CPU)
- A detector for ligature-class edits (lam-alef). Add it to `run_detectors.py`.

## 6. Dots at low resolution (`X-G7`, GPU) and the factorised head (`X-G9`, GPU; same as curriculum C3)
- **X-G7:** CRNN at 32, 40 and 48 px with and without (a) dot-preserving resize (max-pool over dot components) and (b) a dot input
  channel. Outcome: dot tax and CER. The proxy says dots are lost at 32 px and kept at 64 px.
- **X-G9:** two CTC heads (skeleton and dot pattern) against one joint head. Related to the two-stage reader (`decoding-and-restoration.md` §4).

## 7. Attribution on dots (`X-A6`, GPU)
- Gradient mass on dot components against other ink, per dotted letter, on Qwen3-VL-4B with the E1 adapter.

## 8. Normalisation and resolution (`X-G3`, `R9`, `E5-retsinas`, GPU)
- **X-G3:** height {48, 64, 96} × {pad, resize}, plus deslant and binarisation, with the crop on; 2 seeds. Adopt only a setting
  that wins on ≥ 2 targets without in-domain loss.
- **E5-retsinas:** column max-pool, CTC shortcut, median padding; 2 seeds. A second CRNN architecture for every diagnostic.
- **R9:** KHATT downscaled to 60 px (Muharaf's storage height) and three noise levels, for the CRNN and 3 VLMs. If CER rises
  ≥ 3 points, Muharaf's 60 px storage becomes a claim.
- **Prior art:** `retsinas2022best`, `liebl2020crnn`, `vinciarelli2001norm`, `rice1995isri`, `hegghammer2021noisy`,
  `rethinking2021`, and `Qari-OCR` (a three-tier design for renders).

## 9. VLM geometry check (`vlm-geometry`, GPU with a VLM)
- Fine-tuned Qwen3-VL-8B on KHATT as stored, ink-cropped and slant-sheared (the X-T2 transforms).
- **Decision:** a VLM |Δ| < 0.5 points (interval) where the CRNN moves ≥ 1 → the effect belongs to fixed-height CNNs.

## 10. Augmentation items (GPU; mostly through the synthesis machinery)
- **Augmentation 14:** PAW-gap jitter.
- **Augmentation 16:** dot minimal-pair renders with new labels.
- **Augmentation 17:** Maghrebi dot swap.
- **Pen-contrast augmentation:** an anisotropic erode/dilate to imitate a cut nib (RASAM pen contrast is an outlier for 50–68 %
  of lines).
- **Ablation A1:** items 13, 14, 16 and 17 against the 1–4 baseline. Each must move its target error without raising total CER.

## 11. Geometry long tail (CPU or GPU, low priority)
- **`X-G12`:** geometry-prediction pretext pretraining on unlabelled scans.
- **`geom-matched-aug`:** shift augmentation to the target's measured geometry. Slant augmentation already hurt, so low priority.
- **`slant-density`:** select training lines by the rarity of their slant and pen proportions.
- **`geometry-lit` follow-ups:**
  - a second baseline estimator (`abuain2013baseline`);
  - AHTID/MW baseline ground truth, availability and licence (`mezghani2013baseline`);
  - a polygon-curvature descriptor (`abdi2010contour`);
  - read the contour-codebook original (`ahmed2023codebooks`);
  - FATF curve features (`parvez2013fatf`);
  - validate dot detection on CTC spans (`arabic-secondary-components`).
- **`pix2pix`:** learned image normalisation, only after X-G3 shows that normalisation helps.
- **`pen-trajectory`:** parked long shot (`plamondon-kinematic`).
