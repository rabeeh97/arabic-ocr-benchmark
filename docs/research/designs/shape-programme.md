# Design: the shape programme (allograph library, synthesis, shape tokens, personalised synthesis)

Map items: `novel-clusters`, `shape-library`, `synthesis-v1`, `render-refine`, `shape-tokens`, `personalised-synthesis`, `X-G11`,
`shape-nakba`, `hard-example-control`, `aug-9`, `aug-10`, `aug-13`, `aug-19`, `aug-21`, `P-L3`. Origin: the user's ideas of
2026-10-14 and 2026-10-15, built on X-G10 (novelty) and X-G11L (shape-targeted repeats).

## Why

- Letter-shape novelty is the strongest letter-level finding. Within a line, novel letters have about 16× the odds of a misreading
  for CRNNs and about 4.4× for fine-tuned VLMs (ledger §14).
- About 90 % of error is in the skeleton, not the dots (§15).
- Adding real training lines with similar shapes already helps on KHATT, though only there (§13).
- The next step is to work with shapes directly: know them, synthesise them, and read by them.

## What we know before starting (ledger §18)

- **CTC letter spans are anchors, not cut lines.** At the 64 px input the median span is 12 px, 9 % are a single 4 px column,
  and spans often cover only part of a letter, because Arabic letters overlap sideways.
- **Handwritten PAWs touch.** Large ink components match the label's PAWs exactly on only 6–9 % of lines.
- **Extraction must therefore combine the alignment with the ink:** the alignment gives which letter and its order; the ink
  skeleton gives gaps and thin joins.
- **Prior art:**
  - Elarian et al. 2015 (KFUPM): character segments joined with Extended-Glyph and Synthetic-Extension models; better on IFN/ENIT.
  - A KFUPM patent family: glyph choice by thickness and direction at the connection points.
  - Dinges et al.: active shape models with controlled slant and connection length.
  - StackMix (Shonenkov et al. 2021): weakly supervised character boundaries, cut and restacked; Latin and Cyrillic.
  - `ahmad2019subcore`: HMMs over sub-core shapes beat character models.
  - `alazzawi2026gap`: positional-form labels did not help a CRNN. Caution for shape tokens.
- **What would be ours:**
  - automatic extraction from real multi-dataset handwriting;
  - skeleton, dots and join points as separate parts;
  - selection aimed at measured novelty;
  - evaluation with the per-line, per-letter and cross-family framework.

## Step 1. Novel-shape clusters (`novel-clusters`, CPU)
1. **Input:** the letter dumps (`results/panel/letters/*/letters.json`): span, letter, form, novelty and pixel novelty for every
   letter of the panel, plus the training bank.
2. **Embed** every letter by the mean CNN feature in its span, as X-G10 does.
3. **Cluster** per letter × form, k-means with k chosen by silhouette, on the training bank plus the KHATT and NAKBA letters.
4. **Report:** clusters populated by KHATT or NAKBA letters but rare in Muharaf train ("missing allographs"), with medoid crops.
5. **Done when:** an atlas v0 page with clusters per letter × form, their dataset mix and medoid images.

## Step 2. The allograph library (`shape-library`, CPU, a few days)
**Unit:** a letter glyph inside its PAW. Fields:
- letter, positional form, dotless class;
- dot pattern: count, side, and style (separate, dash or caret);
- skeleton polyline and ink mask;
- entry and exit join points on the baseline, with stroke direction and thickness there;
- height relative to the baseline, slant, stroke width;
- source (dataset, manuscript or page), novelty, cluster ID.

**Extraction pipeline:**
1. Ink-crop the line. Binarise (Otsu) and skeletonise.
2. Forced alignment (CTC, the base CRNN, or the standard-split CRNN for test sets) gives the letter order and approximate column
   ranges.
3. Build the skeleton graph. Candidate cut points are thin, near-horizontal skeleton segments near the baseline, lying between the
   column ranges of two neighbouring joined letters.
4. Assign dot components to the nearest letter body above or below, by alignment order and position. Classify dot style from the
   component's aspect ratio and the number of sub-blobs.
5. Cut each glyph; record its join points at the cut ends.
6. **Quality gate (cycle consistency):** re-join each glyph with its real neighbours, then check that an independent model (the
   standard-split CRNN if the line came from the aligner set, and vice versa) still reads the PAW as before. Keep only passing
   glyphs. Also reject glyphs with implausible width, height or ink mass for their letter × form.

**Gate to continue:** at least 30 % of letters pass, and every letter × form has at least 20 glyphs from at least 2 sources.

**Outputs:**
- `results/shapes/library/` (glyph images, masks and JSON metadata);
- an atlas page in the ledger: shapes per letter × form per dataset, and the missing ones;
- per-dataset allograph counts, the measured counterpart of the "82–110 shapes" literature count.

## Step 3. Synthesis v1 (`synthesis-v1`, CPU to build, GPU to test)
**Composer:**
1. Text → sequence of (letter, form) slots, using the joining rules.
2. Draw one glyph per slot. Policies:
   - (a) random;
   - (b) novelty-targeted: weighted towards clusters missing from training;
   - (c) coverage-balanced: augmentation 19;
   - (d) coherent style: all glyphs from one source cluster or manuscript.
3. Join: align glyph *i*'s exit point with glyph *i+1*'s entry point on a common baseline. Bridge gaps with a stroke matching the
   thickness and direction at both ends, with optional kashida stretching (augmentation 13).
4. Dots: place the dot pattern from the dot library. Styles vary (separate, dash, caret, small displacement within the measured
   range).
5. PAW gaps and word gaps are drawn from their measured distributions per source (also tests T1). Optional PAW-gap jitter
   (augmentation 14).
6. Geometry and appearance:
   - our measured slant, stroke width and pen contrast;
   - real paper backgrounds and ruled lines cut from real lines;
   - blur and noise (augmentations 1–4);
   - style mixing (augmentation 21).

**Text sources:** Muharaf and NAKBA train text, a modern Arabic sample, the lexicon (`designs/lexicon.md`) including proper names,
and coverage-balanced text.

**Realism checks before training:**
- a real-vs-synthetic classifier must stay at ≤ 75 % accuracy;
- our geometry measures (store_features) must fall within the real ranges;
- the standard-split CRNN's CER on synthetic lines must be in the range of real lines (not near 0, which would mean the lines are
  too easy).

**Training experiment (3 seeds):**
- **Arms:** Muharaf train + K synthetic lines (K = 5k, 20k, 50k), crossed with policies (a), (b), (c) and (d).
- **Controls:**
  - real shape-targeted repeats (X-G11L);
  - a hard-example control (repeat the highest-loss real lines);
  - a StackMix-style naive cut (rectangles cut at span boundaries);
  - font renders (augmentation 8).
- **Readouts:** the panel with analyses A, B and D, especially KHATT novelty bands 4–5, unseen words, and in-domain loss.

**Keep rule:** novelty-targeted synthesis beats real repeats and the StackMix-style control on KHATT bands 4–5 by ≥ 1.5 points,
pooled over 3 seeds with the interval below 0, and with no in-domain loss above 0.3.

## Step 4. Render, then refine (`render-refine`, gated on step 3)
- A small conditional image model (pix2pix or ControlNet-like) trained on our own pairs: skeleton/mask → real ink crop.
- It turns stitched skeletons into realistic ink with the label kept exact.
- Only if seams are visible to the real-vs-synthetic classifier, or a "seam detector" shows the recogniser exploits them.

## Step 5. Shape-token recognition (`shape-tokens`, GPU)
**Vocabularies to compare** (CTC targets):
- (0) letters (the baseline);
- (1) dotless class × positional form, about 19 × 4 (the two-stage reader of `designs/decoding-and-restoration.md` is this case);
- (2) discovered allograph clusters from step 1 (finer, data-driven, many-to-many with letters);
- (3) a shape BPE: merge frequent adjacent shape tokens into PAW-sized units, as LLM tokenisers merge characters. P-L3 (the PAW
  coverage curve) says whether the vocabulary stays manageable.

**Decoder:** shape lattice (top-k per column) → letters. Options, in order:
1. lexicon plus character n-gram, as a WFST;
2. a learned transducer;
3. a small LLM constrained to letter strings consistent with the lattice (prefix-constrained decoding).

Arabic's joining grammar is a hard constraint: an initial or medial form must be followed by a medial or final form, and a
non-joiner ends the PAW.

**Borrowed from LLMs:** top-k beams with calibrated scores; speculative decoding (a cheap reader drafts, a larger model verifies;
see the cross-family oracle).

**Keep rule:** a shape vocabulary that lowers KHATT high-novelty letter error by ≥ 1.5 points, with no overall CER loss
(2 seeds).

## Step 6. Personalised synthesis (`personalised-synthesis`, gated on steps 2–3 and writer adaptation)
- From a writer's K labelled lines, extract their glyphs (step 2) and synthesise hundreds of lines in their own allographs.
- Fine-tune on them; compare with fine-tuning on the K lines alone (`designs/writer-adaptation.md`).
- **Keep rule:** ≥ 1 point better than the K-line fine-tune at the same K.

## Related checks
- `shape-nakba`: shape targeting with selection on NAKBA blind half A and testing on half B; 2 seeds against random.
  **Keep rule:** pooled Δ ≤ −0.5 with the interval below 0 → shape targeting is a general method.
- `hard-example-control`: repeat the highest-loss training lines (equal count to the targeted set); 2 seeds. It separates
  "similar shapes" from "hard lines".
- `P-L3`: PAW-inventory coverage curve. Distinct PAWs and dotless PAWs needed for 95 % and 99 % token coverage per dataset (text
  only).
