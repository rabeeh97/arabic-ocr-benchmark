# Design: decoding studies, dot restoration, the two-stage reader, cross-family combination

Map items: `lm-decoding`, `P-L1`, `P-L2`, `two-stage-reader`, `cross-family-ensemble`, `vlm-logprobs`, `E2`, `perea-root`,
`X-A3`. Evidence so far: ledger §15 (no-dots CER, cross-family oracle) and §16 (undotted ambiguity).

## 0. Posterior dump (GPU, 15–30 min) — the enabler
- **Models:** base, combo and standard-split CRNNs.
- **Data:** the panel plus the full KHATT test.
- **Saved per line:** CTC log-posteriors (T × C, float16, compressed) and the charset.
- Everything below then runs on the laptop.

## 1. Language-model decoding (`lm-decoding`, CPU)
- **Character n-gram LMs** (KenLM-style, order 6–8) trained on:
  - (a) Muharaf train text;
  - (b) NAKBA train text;
  - (c) a modern Arabic sample (Arabic Wikipedia or news, downloaded to the SSD);
  - (d) (a) + lexicon words.
- **Beam search** over CTC posteriors with LM weight and insertion bonus tuned on a held-out half.
- **Decision rule:** if (c) − (a) gains ≥ 1 point on KHATT and ≤ 0.3 on NAKBA, text distance carries part of KHATT's gap
  (finding T12 moves from D to B). If no LM helps, the residual gap is visual or writer-related.
- **Lexicon-constrained variant:** words must come from the lexicon, or from out-of-lexicon characters with a penalty. Report OOV
  handling separately; proper names are where it fails.

## 2. PAW-aware spacing (`P-L2`, CPU)
- **Rule version:** merge a predicted space after a non-joining letter when the joined form is in the lexicon and the parts are
  not.
- **Decoder version:** in the beam search, a space after a non-joiner costs extra unless the LM supports it.
- **Decision rule:** adopt if WER falls and CER does not rise on every competent pair.

## 3. Dot restoration ceiling (`P-L1`, CPU)
- **Input:** each system's reading mapped to its skeleton.
- **Restore** the dots with:
  - (a) the most frequent word for that skeleton;
  - (b) an n-gram over words;
  - (c) a character LM constrained to the skeleton;
  - (d) a small LLM constrained to skeleton-consistent words.
- **Measures:**
  - share of dot errors fixed vs dots broken, for known and unseen words separately;
  - H(dots | skeleton, context) as the Shannon-style counterpart.

  First evidence (§16): (a) restores 95 % of known words; 19 % of tokens are unseen.

## 4. The two-stage reader (`two-stage-reader`, GPU 1 h + CPU)
- **Train** a skeleton-only CRNN: the standard-split recipe, labels mapped to dotless classes (19 skeleton letters plus the
  non-letters). One seed first.
- **Questions:**
  - (i) Does it read skeletons better than the normal CRNN reads them? Compare no-dots CER of both.
  - (ii) Does skeleton reading plus constrained restoration (§3, (c) or (d)) beat the normal CRNN on full CER, for known and
    unseen words?
- **Keep rule:** full CER ≤ the normal CRNN on two sets, or a win on unseen words without a loss on known ones.
- **Why it might work:** dots are a small share of error; separating them may free capacity for skeletons, and context restores
  most dots.
- **Why it might not:** dots also help the model identify the skeleton letter in context.

## 5. Cross-family combination (`cross-family-ensemble`, CPU after C3)
- **The oracle (§15):** the line-wise better of best CRNN and best fine-tuned VLM is KHATT 14.3 % (17.4 / 18.1), NAKBA test 4.1 %
  (10.2 / 18.7), blind 4.8 % (11.1 / 6.4).
- **Combiners:**
  - (a) pick per line by calibrated confidence (CRNN mean max-probability; VLM mean token log-probability, needs `vlm-logprobs`);
  - (b) ROVER-style word alignment and voting across families;
  - (c) the VLM corrects the CRNN under a constraint: it may only change words whose skeleton or posterior lattice allows it.
    This is the safe version of E8.
- **Keep rule:** a combiner closing ≥ 30 % of the gap between the best single family and the oracle on two sets.

## 6. Side studies
- **`E2` tokenizer view:** do the Qwen and PaliGemma tokenizers give dot-pair words (ب/ت, ف/ق swaps) unrelated token sequences?
  Fertility per line against line CER. CPU.
- **`perea-root`:** do real-word substitutions share the root (CAMeL Tools morphological analysis) more often than chance? CPU.
- **`X-A3` image reliance:** per-token log-probability with and without the image, per VLM. GPU with a VLM. It shows how much of
  a VLM reading is language prior.

## 7. `vlm-logprobs` (code change in the benchmark runner, main checkout)
- Save token log-probabilities for local HF models.
- Changes cache keys only for new runs.
- Needed for §5 and for calibration and selective reading.
