# 02 · Error taxonomy (v0.1, 2026-11-02)

Each error type has a stable ID (used in code, flags, votes and the paper), what it means, what a detector needs, its status, and its
**fix class**: how far the fix may be automated.

**Status values:**
- **built**: in `src/arabic_ocr_bench/dataqa.py`, validated in `docs/research/dataqa.md`. The check name is given in brackets.
- **planned**: specified here, not built.
- **future**: needs a new modality or data.

**Fix classes** (see [03-toolkit-spec.md](03-toolkit-spec.md) §6):

| class | meaning |
|---|---|
| **A0** | Lossless and deterministic: always safe to apply automatically (e.g. remove a BOM). |
| **A1** | Automatic only when ≥ 2 independent signals agree and the fix is local (e.g. remove a pasted copy that both readers confirm). Written as a patch, applied by policy. |
| **S** | A suggested patch: a person accepts or rejects it. |
| **H** | Flag only: a person decides what to do (drop, re-transcribe, re-cut). |

**What the detectors need** (*needs*): `gt` text · `img` line image · `page` page image and geometry · `read` reading(s) from ≥ 1 model ·
`lm` character LM · `lex` lexicon · `ds` the whole dataset · `meta` file and metadata.

## C · Characters and encoding (text only, highest precision)

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| C01 | empty or whitespace GT | no letters and no digits | gt | built (`empty_gt`) | H |
| C02 | invisible characters: BOM, ZWJ/ZWNJ outside joining contexts, LRM/RLM, soft hyphen, NBSP | Unicode category Cf and a whitelist | gt | planned | A0 |
| C03 | presentation forms (U+FB50–FDFF, FE70–FEFF) | code-point ranges | gt | planned | A0 (NFKC on those code points only) |
| C04 | look-alike letters from other scripts: ی/ي, ک/ك, ہ/ه, ۀ, Persian/Urdu digits | dataset majority + per-line minority | gt, ds | planned | S (A1 when the dataset is ≥ 99 % one form) |
| C05 | decomposed hamza / madda (ا + U+0654) vs precomposed (أ) | NFC differs from the raw text | gt | planned | A0 (NFC) |
| C06 | mixed digit systems in one line (٣ and 3) | per-line digit scripts | gt | planned | S |
| C07 | mixed punctuation (`,` vs `،`, `?` vs `؟`, `;` vs `؛`) | per-line plus dataset convention | gt, ds | planned | S |
| C08 | rare characters (frequency < 1e-4 of lines in the dataset) | character inventory | gt, ds | planned | H |
| C09 | runs of one character (≥ 6, not tatweel or space) | regex | gt | built (`char_run`) | H |
| C10 | tatweel (kashida) in a dataset that otherwise has none | dataset convention | gt, ds | planned | S |
| C11 | double, leading or trailing spaces | regex | gt | planned | A0 (only if the dataset policy says spaces are not meaningful) |
| C12 | mojibake (UTF-8 read as Latin-1 / CP1256: "Ø§Ù„") | byte-pattern test, re-decode round trip | gt | planned | A1 (when the round trip yields valid Arabic) |
| C13 | markup and editor marks left in the GT (`#`, `[ ]`, `{ }`, `<…>`, `&amp;`, `…`, `؟؟`) | inventory per dataset; markup lexicon | gt, ds | planned | H |
| C14 | text off the dataset's script (Latin, Persian-only letters پ چ ژ گ in an Arabic set) | letter-script share | gt | built (`off_script`, partial) | H |

## O · Orthography and lexicon (text plus language resources)

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| O01 | impossible letter positions: ة or ى inside a word, tanween not word-final, hamza seat errors | positional rules | gt | planned | S |
| O02 | misplaced harakat: on a space or digit, stacked invalidly, at word start | rules | gt | planned | S |
| O03 | missing space (merged words) | token longer than p99, or splits into two lexicon words; O01 hits | gt, lex | planned | S |
| O04 | extra space (split word) | a lone letter that is not a valid one-letter word; two non-words that join into one lexicon word | gt, lex | planned | S |
| O05 | **dotting error** (same rasm, different dots) | an out-of-vocabulary word whose rasm skeleton matches a frequent lexicon word that differs only in dots | gt, lex | planned | S |
| O06 | typo (one edit from a frequent word) | after clitic stripping, edit distance 1 to a lexicon word | gt, lex | planned | S |
| O07 | out-of-vocabulary word (no near neighbour) | lexicon miss after clitic stripping and variant folding | gt, lex | planned | H (reported as a rate per page or annotator) |
| O08 | unlikely as Arabic (garbled, uncorrected HTR) | char-LM bits per character, as an outlier in the dataset | gt, lm, ds | planned (RASAM version in `rasam_clean_lists.py`) | H |
| O09 | stored in visual order (reversed string) | LM or lexicon score of the reversed text ≫ the original | gt, lm/lex | planned | A1 (whole-line reversal when the reversed text is far more likely and all its words are in the lexicon) |
| O10 | reversed digits inside numbers | the reading disagrees in digit order only | gt, read | planned | S |
| O11 | convention drift: harakat present in some lines only; أ/ا, ة/ه, ى/ي written differently by page or annotator | per-group outliers in convention ratios | gt, ds | planned | H (report a convention table) |
| O12 | the wrong variety for the dataset: Persian, Urdu or Ottoman lines; Maghrebi ڢ/ڧ vs Eastern ف/ق mixed | variety-specific letters and words | gt, ds | planned | H |
| O13 | text pasted twice | repeat of ≥ 15 characters, or the GT is X+X | gt | built (`repeated_gt`) | H (A1 with reader confirmation, see R03) |

## I · Image only

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| I01 | degenerate crop (< 8 px) | size | img | built (`degenerate`) | A1 (re-cut when page geometry exists and a reader matches the GT) |
| I02 | vertical or marginal strip | h ≥ 1.8 w | img, gt | built (`vertical`) | H |
| I03 | blank or empty crop | flat image or < 0.3 % ink against the local background | img, gt | built (`blank`) | H |
| I04 | more than one line in the crop | deskewed projection peaks | img | built (`multi_line`) | A1 (re-cut to the polygon when page geometry exists) |
| I05 | neighbour line cut at the edge | edge peaks | img | built (`neighbour_clip`, informational) | – |
| I06 | text cut at the left or right edge (truncated crop) | ink density in the edge columns; side-aware with J01 | img | planned | A1 (re-cut with a margin) |
| I07 | rotated 180°, mirrored, or white-on-black | profile asymmetry; reader on the flipped image (R07) | img (read) | planned | A1 (when the reader on the flipped image matches the GT) |
| I08 | too small to read dots (core height < 8 px), heavy blur, JPEG blocking, upscaling | core height, Laplacian, block-artefact score, as dataset outliers | img, ds | planned | H |
| I09 | scanner border, colour chart, ruler, fingers | large dark or saturated regions touching the border | img | planned | S (crop) |
| I10 | polygon-mask artefacts (white filler cutting through letters) | flat filler regions crossing ink | img | planned | H |

## J · Image and text together, no model

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| J01 | GT length does not fit the ink | robust z of log(letters) − log(core aspect) | img, gt, ds | built (`length_mismatch`) | H |
| J02 | number of PAWs (connected pieces) implied by the GT ≠ main blobs in the image | Arabic joining classes vs connected components | img, gt, ds | planned | H |
| J03 | dot count implied by the GT ≠ small blobs in the image | dot table per letter vs small components | img, gt, ds | planned (print first) | H |
| J04 | harakat in the GT but no marks in the image (or the reverse) | mark density vs small components above and below the core | img, gt, ds | planned | S (strip harakat when the dataset policy is undiacritised) |
| J05 | which end of the line is missing | I06 edge ink combined with an R01/R02 alignment | img, gt, read | planned | H |

## R · Evidence from readers (any recogniser; more readers, more power)

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| R01 | GT truncated (the image shows more) | sub + del ≤ 15 %, ins ≥ 15 % + 5 | gt, read | built (`gt_truncated`) | S (the reading proposes the missing text) |
| R02 | image truncated (the GT has extra text) | sub + ins ≤ 15 %, del ≥ 30 % + 5 | gt, read | built (`image_truncated`) | H |
| R03 | doubled GT confirmed by a reader | deleting the copy cuts the edit distance by ≥ 60 % of its length | gt, read | built (`doubled_gt`) | A1 (with 2 independent readers) |
| R04 | GT belongs to a neighbouring line | another line's reading on the same page matches far better | gt, read, page | built (`shifted`) | A1 (reassign within the page when the match is one-to-one) |
| R05 | unreadable / mismatched | CER ≥ 60 % | gt, read | built (`unreadable`) | H |
| R06 | **readers agree with each other, not with the GT** (likely a GT error) vs **readers disagree with each other** (a hard line) | pairwise CER between readers vs CER to the GT; flag the GT span | gt, read×2+ | planned (RASAM version: cross-fitted folds) | S on the span |
| R07 | orientation (see I07) | reader on rotated or mirrored copies | img, read | planned | A1 |
| R08 | low likelihood of the GT under the model | CTC forced-alignment score of the GT vs the best path, per character | img, gt, model | planned | S on the span |
| R09 | never learned during training | training-dynamics loss in late epochs (dataset cartography) | train logs | planned | H |
| R10 | systematic substitution by page or annotator (always ه where the image has ة) | confusion matrix GT→reading per group | gt, read, ds | planned | H (convention report) |

## P · Page and structure

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| P01 | whole page shifted by one line (all GTs offset) | line i's GT matches line i±1's reading across the page | read, page | planned | A1 (shift the page when ≥ 80 % of lines match better shifted) |
| P02 | wrong reading order of lines or regions | LM continuity of consecutive GTs | gt, lm, page | planned | H |
| P03 | polygon or baseline geometry broken (self-intersecting, outside the page, zero area) | geometry validation | page | planned | H |
| P04 | page-level quality outliers (rates of O07, O08 and R05 per page) | aggregation | ds | planned | H (batch review) |

## D · Dataset and splits

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| D01 | duplicate image (same or near-same), possibly with differing GT | quantised hash (built); perceptual hash with tolerance (planned) | img, ds | built (exact) / planned (near) | H |
| D02 | duplicate GT text | exact match (built); fuzzy ≥ 90 % (planned) | gt, ds | built / planned | H |
| D03 | train/test text overlap | GT match across splits and datasets | gt, ds | built (`dataqa_overlap.py`) | H (report) |
| D04 | page, writer or manuscript leakage across splits | metadata, or image near-duplicates at page level | meta, ds | planned | H |
| D05 | test characters never seen in training; split distribution shift | inventory and KL divergence | gt, ds | planned | H (report) |
| D06 | ID collisions, missing IDs, unstable ordering | schema checks | meta | planned | A0 / H |

## M · Files and metadata

| ID | error | detector | needs | status | fix |
|---|---|---|---|---|---|
| M01 | corrupt or truncated image file | decode test | meta | planned | H |
| M02 | EXIF orientation set (viewers rotate, loaders do not) | EXIF tag 274 | meta | planned | A0 (bake the rotation in, recorded in the patch) |
| M03 | alpha channel with transparent background (turns black on RGB conversion) | mode RGBA/LA with alpha < 255 | meta | planned | A0 (composite on white) |
| M04 | unusual mode: CMYK, 16-bit, palette | PIL mode | meta | planned | A0 (convert, recorded) |
| M05 | licence or provenance missing or inconsistent | dataset card fields | meta | planned | H |

## Future modalities (same framework, new joint checks)

| ID | error | needs |
|---|---|---|
| S01 | speech: transcript length vs audio duration (like J01) | audio, gt |
| S02 | speech: forced-alignment confidence per word (like R08) | audio, gt, model |
| N01 | NLP labels: classifier ensemble agrees with each other, not with the label (like R06) | text, label, models |
| N02 | parallel corpora: length ratio, language ID, untranslated copy-through | src, tgt |

## How new types enter the taxonomy

People report a flag with "none of these" plus a short description. Reports are grouped (text and evidence embeddings plus the detector signals
that fired). When a group has ≥ 20 confirmed examples, it gets a new ID (`X01`, …) and its examples become a test set. A detector
for it can then be proposed and scored. See the product spec for the workflow.
