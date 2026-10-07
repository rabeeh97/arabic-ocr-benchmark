# Findings & standing notes — Arabic OCR fine-tuning

Living document. Read before designing the next experiment. Each entry: what we
saw, what it means, what to do about it. Newest section first within each area.

---

## Benchmark validity

### Line-end errors are a pipeline artefact; several models are scored on truncated output

**Seen (2026-10-07, R2.8, `scripts/line_end_check.py`).** The last tenth of the
line is the worst in 9 of 10 complete blocks. Split by edit type, the excess is
deletions (+3 to +10 points) and insertions after the line's end (overrun, up to
+0.9 per character on `kitab_muharaf`). Substitutions rise by only 0.1–3.7 points.
**No line is cut short by a majority of models** (0–1 lines per block), so the
images do not crop line ends. Tail cuts (the last ≥ 10 % of a line deleted
outright) are model-specific: `gemini-flash` 44–46 % of lines, PaddleOCR 31 %
(plus 23 % head cuts, consistent with its text detector), Qari-0.2 11 %; most
models 0.5–6 %.

The `gemini-flash` cause: it has no `max_tokens` in `configs/models.yaml`, so the
litellm client default of 256 applies. Gemini 3.6 Flash is a thinking model, so
its reasoning tokens use up the budget: 74 % of its outputs stop at 252 tokens,
and some contain reasoning text ("Wait, let's"). A scan of all models for
outputs piling up at one token count also flags `openrouter-dots3-note-free`
(100 % at 256), `glm-5.3-flash-paid` (39 % at its 4,000 cap),
`nvidia-llama-3.2-11b-vl` (25 % at 256, mostly rambling) and
`openrouter-qwen3-vl-235b` (12 % at 256).

**Meaning.** The end-of-line effect is not a script property. Those models'
leaderboard rows understate them; `gemini-flash` most of all.

**To do.** X6: set `max_tokens` (and a minimal thinking budget) for the capped
models, flag their current rows, and re-score within free quota.

### The public Muharaf copy is stored at 60 px line height

**Seen (2026-10-07, R6.0, `scripts/image_layer.py`).** Every image in
`aamijar/muharaf-public`, train and test, is 60 px tall (widths vary): a
letter core of about 12 px, strokes of about 2 px, about 11 px of ink per
character. KHATT lines are about 146 px tall (38 px core, 33 px per character),
HistoryAr about 230 px (57 px core). Muharaf is the hardest handwriting set for
every model (best single model 34 % CER vs 5.5 % on KHATT). Every fine-tune so
far learned from these 60 px lines.

**Meaning.** Low resolution is a strong candidate explanation for Muharaf's
difficulty, and a D1 question for our training data. It is a between-dataset
comparison (resolution is constant within Muharaf), so it is a hypothesis, not
a measured effect.

**To do.** X7: find out whether full-resolution Muharaf scans (with line
polygons) are available; if so, re-crop at native resolution and re-score a
few models. R9 causal probe: downscale KHATT lines to 60 px and re-score.

### Image features explain part of what text cannot

**Seen (2026-10-07, R6.0, 7 of 10 blocks).** Adding 11 image features (size,
aspect, ink density, core height, stroke width, ink per character, contrast,
blur, skew, edge ink) to the nine text features raises the rank-regression R²
from 0.06–0.32 (text) to **0.15–0.58** (text + image): `kitab_historyar`
0.32 → 0.58, `kitab_onlinekhatt` 0.16 → 0.43, `kitab_khatt` 0.32 → 0.39,
`rasam2` 0.10 → 0.25, `muharaf_lines` 0.06 → 0.15. Individual image coefficients
are unstable (features are collinear), so no single image feature is claimed yet.

**Meaning.** The image layer helps, but most of Muharaf's within-dataset
difficulty is still unexplained, which points to handwriting style or writer.
That needs writer IDs (R0.4).

### What makes a line hard: per-sample characteristics over complete model blocks

**Seen (2026-10-07, `scripts/sample_characteristics.py`, Atlas §10).** Ten
complete blocks, meaning sets of models that all read the same lines:
`kitab_khatt` 18×180, `rasam1` 13×177, `rasam2` 13×174, `nakba_lines` 18×62,
`muharaf_lines` 5×472, `nakba_blind` 6×2,669 (closely related models, so its
reliability is optimistic), plus `kitab_historyar`, `kitab_hindawi`,
`kitab_muharaf` and `kitab_onlinekhatt`. Line CER is diacritic-insensitive and
capped at 1.

1. **Difficulty belongs to the line.** Split-half reliability of consensus
   difficulty is 0.80 (`kitab_khatt`), 0.81 (`nakba_blind`), 0.76 / 0.62
   (RASAM-2 / 1), 0.65 (`nakba_lines`) and 0.54 (`muharaf_lines`). On
   page-scale `kitab_hindawi` it is 0.23: there, model failures are
   idiosyncratic (pairwise ρ 0.07).
2. **Text explains little of it.** A rank regression on nine GT features
   (length, word length, PAWs/word, dot-ambiguous share, OOV share,
   non-Arabic share, diacritics, lam-alef, hamza) explains **6–32 %** of the
   variance (51 % on the 62-line `nakba_lines`). Most line difficulty is
   visual or writer-related.
3. **The features that survive, all others held fixed:** OOV share +
   (significant in 3 sets, positive in 7 of 10, never significantly negative);
   dot-ambiguous share + (3 significant positive, 1 negative); non-Arabic
   content + (2); short words − (2). **PAWs per word drops out** once word
   length is controlled: the raw correlation was word length.
4. **Letter forms.** Medial is the hardest positional form in 9 of 10 blocks.
   Isolated is the easiest in 8 of 10. Dot-ambiguous letters fail 10–17 points
   more often than other letters in every handwriting block, at every
   position. The hardest shapes (error ÷ dataset mean, in ≥ 4 sets) are
   medial ف ح ض ث خ ج ش (1.3–1.5×), isolated ء, final ى, and the small final
   non-joiners ز ذ د. The easiest are the tall strokes ا and ل in every
   position (0.6–0.8×) and و / م. That is the Arabic analogue of the Latin
   result that ascenders are easy and minims hard.
5. **Errors pile up at the end of the line.** The last tenth of the line is
   the worst in 9 of 10 blocks (1.10–1.32× the line average); the first tenth
   is usually the best.
6. **Short lines are disproportionately hard.** The shortest length quintile
   (≈1–20 characters: numbers, single words, marginalia) has the highest
   consensus CER on the historical sets (`nakba_lines` 0.66 vs 0.18 mid-length;
   `kitab_historyar` 0.70 vs 0.38). Whole-line failures concentrate on short
   lines (ρ −0.54 / −0.40 / −0.26).
7. **Ensembling will not help much.** Picking the best model per line beats
   the best single model by only 0.8–3.6 CER points on line sets (7 on
   Hindawi).
8. **About half of word substitutions produce a real Arabic word** (43–63 %,
   against the Muharaf-train vocabulary). When both words are in the
   vocabulary, the model's word is only slightly more often the more frequent
   one (46–55 %). A lexicon post-corrector can address at most half of the
   word errors.
9. **Label noise.** Against CER-HV's human-verified Muharaf flags, consensus
   difficulty ranks all flags at chance (AUC 0.52). That is because their
   largest category, "script mismatch (non-Arabic)", is read well by VLMs
   (median consensus CER 0.43 vs 0.58 unflagged). Those labels are correct;
   only the CRNN's Arabic charset failed on them. On the real-noise flags
   (segmentation, non-text), AUC is **0.94**, but on only 7 lines.

**Meaning.** The R3 difficulty measure passes V1 (reliable) on handwriting line
sets. The text layer, our P1 profiler, cannot explain most difficulty, so the
image layer (R6) is the main gap. The letter-level results point to *where*
models fail (medial, dotted, small final strokes, line ends). The line-level
ladder said dots are a small tax; both hold, because dotted letters fail mostly
by losing their skeleton too, not by a clean dot swap. The end-of-line and
short-line effects are D3/D4 issues (cropping, truncation, how CER is
averaged), not script properties, and need a check before they are quoted as
such.

**To do.** See NEXT_STEPS "Now": the end-of-line check (R2.8), the image layer
(R6.0), length-stratified reporting (X5), challenge subsets (R3.3) and
consensus label-noise review (R4.1).

### Muharaf label noise does not move our test comparisons

**Seen (2026-10-07, X2, `scripts/rescore_muharaf_clean.py`).** The CER-HV review
logs (`alazzawi2026labelerr`, github.com/SanaNGU/CER-HV @ a39544d, **no licence
file**) flag 126 of 1,334 Muharaf test lines (9.4 %) and 1,424 train lines as
label errors: script mismatch, segmentation, orientation, wrong transcription,
signature or non-text. Lines judged "valid but hard" are not counted. Their IDs
are our `muharaf_lines/test/NNNNNN` (all 218 reviewed lines match our GT
exactly). Dropping the flagged lines changes CER·light by −0.4 to +0.9 points
across 13 models and **changes no rank**.

**Meaning.** Test-side label noise is not distorting model comparisons. Our
samples are mostly the first 200 lines, about 19 of them flagged. The logs
are flags, not corrected text, so they cannot fix labels, only exclude them.

**To do.** X3 (retrain without the flagged train lines) is low priority. Use the
flags as ground truth when validating R4.1's label-noise detector.

### Error decomposition: dots are a small tax, PAW gaps attract spaces, diacritics are a policy tax

**Seen (2026-10-07, characterisation P2, `scripts/decompose_errors.py`).** Every
canonical prediction (`characterize/runs.py`: standard prompt, latest success
per sample, no smoke runs) was re-scored on the normalisation ladder and
aligned edit by edit. 194 model × dataset pairs have n ≥ 20; 91 are competent
(diacritic-insensitive CER ≤ 40 %).
- **Dot tax.** CER(−diacritics) − rasm-CER has a median of **1.8 CER points**,
  or **8 %** of diacritic-insensitive error (IQR 7–9 %). For the best models it
  is 0.1–1.3 points.
- **PAW-gap spaces.** Of the 10,655 spaces wrongly inserted inside GT words,
  **60 %** sit at a PAW boundary, after a letter that cannot join forward. The
  chance rate, the share of word-internal letter boundaries that are PAW gaps,
  is **35 %** (31–38 % across datasets).
- **Diacritic tax.** Base Qwen3-VL-4B inserts **10.5 marks per 100 letters**
  on `kitab_patsocr` (3.3 on `kitab_isippt`). Its raw CER is 18.8 %, against
  7.1 % with marks folded. The suite-v1 fine-tune is at about 0.1 marks per
  100. Claude Opus is at 8 points of tax on `kitab_historyar`.
- **Taxonomy over competent pairs:** other letter errors 58 %, whole-line
  failures 10 %, space 9 %, diacritic 7 %, orthographic 6 %, dot 5 %,
  shape-similar 3 %, lam-alef 3 %. Over *all* pairs, whole-line failures
  (line CER > 80 %) are 57 % of edits, so pooled numbers have to be restricted
  to competent pairs.
- One smoke run (`qwen3vl4b_visft_difflr/bench/smoke`) wrote a different
  checkpoint's output under the model name `exp3090-qwen3vl4b-ft-bf16`, so
  "latest run wins" would have silently swapped models. Smoke runs are now
  excluded.

**Meaning.** For VLMs that already read the script, dots alone are not the
bottleneck. That is far below the ~30 % share of substitutions that
`alazzawi2026gap` reports for a CRNN. The two numbers measure different things
(share of all edits vs share of substitutions; VLMs vs CRNN), so this is a
hypothesis to test, not a contradiction. Word segmentation follows the ink: a
PAW gap looks like a word gap. The diacritic tax is a D4 (metric) question
more than a model fault. CER·light keeps harakat, so on undiacritised GT it
penalises a model for adding correct marks.

**To do.** R2.6: split the "other" bucket. R2.7: score undiacritised sets with
marks stripped and compare rankings. Feed the PAW-gap rate and dot tax into
the R7 validity model as candidate per-sample predictors.

### `kitab_muharaf` overlaps Muharaf train, so fine-tuned scores on it are leaked

**Seen (2026-10-07, characterisation P1, `scripts/profile_datasets.py`).** Among
`kitab_muharaf` lines of 20 characters or more, after diacritic-insensitive
normalisation, **86.5 % appear word for word in `aamijar/muharaf-public`
train**, the 22k lines every exp3090 and suite_v1 fine-tune samples from. Every
other benchmark set is at ≤ 6 % (`muharaf_lines` test 5.7 %, `kitab_evarest`
5.7 % on mostly very short lines).

**Meaning.** A fine-tune's `kitab_muharaf` row measures recall of training
lines, not recognition. Base-model rows are unaffected.

**To do.** ~~Exclude `kitab_muharaf` from base-vs-fine-tune comparisons, or label it
"in-train".~~ Done 2026-10-07 (X1): the leaderboard masks it for fine-tunes. Check the ~6 % of `muharaf_lines` test lines that also occur in train
(formulaic letter openings?) before quoting in-domain gains.

### `kitab_hindawi` / `kitab_historicalbooks` are page-scale, not lines — kept, with a raised token cap and a caveat

**Seen (2026-09-09, exp3090 smoke).** On `kitab_hindawi` both `qwen3-vl-4b-bf16`
and `exp3090-qwen3vl4b-ft-bf16` score ~85–90 % CER-light vs ~30 % on the
handwriting sets. The ground truth is **~860 chars ≈ ~350 tokens** — a whole
page / multi-paragraph block with tab-separated "lines". Predictions were
~194 chars (base) / ~284 chars (ft) — about **77 / 113 tokens**.

**Verified — two compounding causes:**
1. **Output cap.** `hf_local` defaults to `max_new_tokens=160`
   (`models/registry.py`). A complete transcription needs ~350+; the cap alone
   guarantees truncation on a full page.
2. **Prompt/content mismatch.** The predictions (77–113 tok) actually stop
   *below* the 160 cap — the model emits one paragraph then EOS because the
   prompt (`line-v2`) says "transcribe this single line" and the model treats one
   paragraph as the answer. So raising the cap is necessary but not sufficient.

**Done.** Kept both datasets in `configs/training/exp3090_bench.run.yaml`.
`run_3090_benchmark.py` now registers every row with **`--max-new-tokens 640`**
(new `bench-train register --max-new-tokens` flag). Line datasets stop at EOS
well before 640, so no slowdown and the existing line-dataset cache stays valid.

**How to read these two rows:** a "does the model volunteer a full-page
transcript from a line prompt" probe, not a clean line-OCR score, and **not
comparable to KITAB-Bench's own numbers** (page-level protocol). Watch whether
(a) the higher cap moves them and (b) other / stronger models — which transcribe
whole pages more readily — do better.

**To do.** (1) For a real printed-Arabic *line* regression check, add a
line-level printed set. (2) Add a `page` prompt/track for genuinely page-scale
data. (3) The runner should surface `empty_rate` and median `len(pred)/len(gt)`
per (model, dataset) — an all-models-truncated dataset should raise a loud "wrong
granularity" flag instead of just a mediocre CER.

---

## Fine-tuning results

### E1 preliminary: unfreezing the ViT (differential LR) helps letter identity, not dots

**Seen (2026-10-07, from the 2026-09-24 box run, recovered config
`configs/training/suite_v1/qwen3vl4b_visft_difflr.yaml`).** Same suite_v1 data,
seed and recipe as `suite-v1-qwen3vl4b` (frozen). Only `train_vision: true` and
`vision_lr: 9e-6` differ. On the same 200 Muharaf test lines:

| arm | CER·light | −diacritics | rasm-CER | dot share |
|---|---|---|---|---|
| frozen (`suite-v1-qwen3vl4b-ft`) | 36.3 % | 34.5 % | 32.0 % | 7.2 % |
| unfrozen, diff-LR | **28.7 %** | 26.6 % | 24.4 % | 8.3 % |

Paired bootstrap on the difference: −7.6 points, 95 % CI −9.1 to −6.2. The run
wrote its outputs as `exp3090-qwen3vl4b-ft` (launched under the `qwen3vl4b`
BASES entry); `characterize/runs.py` renames it.

**Meaning.** The frozen encoder *is* a bottleneck in-domain. But the gain is not
the dot-specific gain E1 hypothesised: the dot share holds steady while
skeleton-level errors fall. One seed, in-domain only.

**To do.** E1.1 bench this arm on the held-out panel (RASAM is the real test);
E1.2 train the same-LR arm; E1.3 compare all three.

### exp3090 — Qwen3-VL-4B + LoRA on 8k Muharaf lines: large, script-transferable gain

**Seen (2026-09-09).** bf16 LoRA (r=16, α=32, ViT frozen), 1 epoch, 8,000
Muharaf `train` lines. Held-out CER-light, base → fine-tune:

| dataset | script | base | ft | Δ |
|---|---|---|---|---|
| kitab_khatt | Naskh/Ruq'ah HW | 77 % | 30 % | −47 |
| nakba_lines | 20th-c. degraded HW | 89 % | 29 % | −60 |
| rasam1 | Maghrebi manuscript | 119 % | 52 % | −67 |
| rasam2 | Maghrebi manuscript | 102 % | 49 % | −53 |
| in-domain muharaf_lines | Ruq'ah HW | 90 % | 36 % | −54 |

**Meaning.**
- Base Qwen3-VL-4B is **unusable** on cursive Arabic (CER 77–119 %, char-accuracy
  ~0 % on Maghrebi). It mostly emits meta-commentary, wrong-script text, or
  fully-diacritised confabulation. The fine-tune's *first* win is simply
  producing a bare line transcript.
- The gain **transfers off-corpus**: KHATT and NAKBA were never in training and
  improve by roughly the same margin as in-domain Muharaf. This is script-level
  learning, not corpus memorisation — a publishable point ("a small LoRA on one
  handwriting corpus generalises across Arabic handwriting").
- **Maghrebi (RASAM) is the weakest** fine-tune result (~50 % CER vs ~30 % for
  Naskh/Ruq'ah). Different script family, under-represented in Muharaf, and the
  **ViT is frozen** — unfreezing it (`train_vision: true`) is the lever to test
  for degraded / distinct scripts.
- The fine-tune is also ~2× faster: the base runs to the token cap emitting
  filler; the LoRA stops at end-of-line.
- Diacritics: base diac-F1 ~0.03–0.35, LoRA 0.35–0.69, and its raw vs
  diac-insensitive CER are close — it is not just guessing skeletons.

**Open questions to design experiments around.**
- **Format vs vision.** How much of the gain is the *output-format prior*
  (line-only, no chat) vs. genuine visual grapheme learning? A control that only
  changes the prompt / system message on the *base* model would separate them.
- **Data scaling.** 8k lines was one arbitrary point. Re-run at 1k / 2k / 4k /
  full on the chosen base and trace where the held-out curve flattens. That
  number sizes the moat for a from-scratch Arabic HTR effort and tells us how
  much labelled handwriting we actually need.
- **Base model choice.** Only Qwen3-VL-4B tested. Next: `baseer` (Arabic-
  manuscript-specialised base) and `paligemma` (different arch family) on the
  *same* 8k Muharaf data — the controlled A/B/C that answers "what do we build
  on". If an Arabic-specialised base + LoRA barely beats generic-Qwen + LoRA,
  that itself is the headline.

### exp3090 — base-model A/B/C (Qwen3-VL-4B vs Baseer-Nakba vs PaliGemma-2-3B): the specialised base doesn't benefit from the shared recipe

**Seen (2026-09-13).** Same recipe for all three (bf16 LoRA r=16, α=32, ViT
frozen, 1 epoch, the same 8k Muharaf `train` lines), scored on the same four
held-out sets. CER-light, base → fine-tune (n = per-row sample count; small n
is called out — it's noise, not signal):

| dataset | Qwen3-VL-4B (generic) | Baseer-Nakba (Arabic-specialised) | PaliGemma-2-3B (different arch) |
|---|---|---|---|
| kitab_khatt | 77 % → 30 % (n=200) | *(not yet run)* | 66 % → 38 % (n=180) |
| nakba_lines | 76 % → 30 % (n=220) | **1.0 % → 3.4 %** (n=200, *worse*) | 88 % → 30 % (n=400) |
| rasam1 (Maghrebi) | 119 % → 52 % (n=20, noisy) | 51 % → 45 % (n=200) | 76 % → 51 % (n=400) |
| rasam2 (Maghrebi) | 102 % → 49 % (n=20, noisy) | 49 % → **50 %** (n=196, *flat/worse*) | 79 % → 54 % (n=345) |

**Meaning — this is the headline the open question above predicted might
happen, and it did.** Baseer-Nakba is Misraj's NAKBA-2026 Systems-Track winner
— already Arabic-manuscript-competent before any fine-tuning (0.079 CER on its
own eval; 1.0 % here on `nakba_lines`, its home domain). Running the *same*
generic 8k-Muharaf LoRA recipe on it does not help:
- On `nakba_lines` — the domain it's already best at — the fine-tune makes it
  **3× worse** (1.0 % → 3.4 %), even at a deliberately gentle LR (5e-5, chosen
  in `exp3090_baseer_nakba.yaml` specifically to avoid this).
- On `rasam2` it's flat-to-slightly-worse (49 % → 50 %); on `rasam1` it's a
  modest win (51 % → 45 %), nowhere near Qwen's or PaliGemma's ~25–30 point
  swings on the same two Maghrebi sets.

Meanwhile the two *generic* bases — Qwen3-VL-4B and PaliGemma-2-3B, architecturally
unrelated (Qwen's dynamic-resolution ViT vs PaliGemma's fixed-448px SigLIP,
different decoders, no shared pretraining) — start from comparably poor bases
(66–119 % CER) and converge to a similar post-fine-tune band (30–54 % CER)
across all four sets. That similarity, despite the architecture difference,
points at the 8k-line Muharaf recipe/data as the current ceiling, not base
architecture — echoed by the **"format vs vision"** open question above.

**Per-sample diagnosis of the Baseer regression (2026-09-13).** Paired base-vs-ft
CER on every sample id scored by *both* rows (not just the aggregate):

- `nakba_lines` (n=63 paired): **35 got worse, 5 got better, 23 flat** — a
  systematic decline, not a couple of outliers. Most of the damaged lines were
  at or near CER 0.000 on the base and picked up small-but-real errors after
  fine-tuning (e.g. `nakba_lines/test/000185` 0.000→0.300,
  `.../000171` 0.000→0.213, `.../000179` 0.000→0.200). This is genuine mild
  catastrophic forgetting on Baseer's own best domain, confirmed — not noise.
- `rasam2` (n=195 paired): 93 worse / 81 better / 21 flat — roughly balanced,
  with the mean barely moving (0.512→0.532). A few samples swing hard either
  direction (up to ±0.6 CER), but the aggregate "flat/worse" reading from the
  summary table is **within per-sample noise here**, unlike `nakba_lines`.

So the finding sharpens: Baseer's fine-tune reliably *hurts* its home domain
(NAKBA) and is a wash (not a proven regression) on RASAM2 — the recipe isn't
uniformly bad, it specifically erodes what the base was already good at.

**To do.**
1. ~~Diagnose the Baseer regression~~ — done above; it's real on `nakba_lines`,
   noise-level on `rasam2`.
2. Don't fine-tune Baseer with this recipe again as-is — right now the
   fine-tuned row is worse than just shipping the untuned base on its strongest
   domain. Worth trying fewer steps / a lower rank / a replay mix that includes
   some NAKBA-domain lines alongside the Muharaf data, so the LoRA doesn't
   overwrite what the base already knew.
3. `train_vision: true` (unfreeze the ViT) is still untested on any base and is
   the most promising lever for the still-weak Maghrebi (RASAM) scores (45–54 %
   CER post-tune across all three bases) — try it before the data-scaling sweep,
   since scaling data against a possibly wrong recipe wastes compute either way.
4. Backfill `kitab_khatt` for Baseer-Nakba — it's the one cell in this table
   with no data at all.

**External research (2026-09-16).** See `misraj-baseer-research.md` for the
full writeup: Misraj's own system paper confirms this table's finding from
their side — Baseer-Nakba is two stacked fine-tunes on Qwen2.5-VL-3B (general
Baseer, then a Muharaf→NAKBA HTR specialization with a differential
vision/decoder LR and epoch weight-averaging), and their own ethics section
admits the model is tied to NAKBA's specific calligraphic/historical
distribution. Their disclosed recipe is fully reproducible and directly
actionable for us: the `train_vision: true` + differential-LR idea in item 3
above is exactly what gave them their biggest single jump (CER 0.16→0.10).

---

## Cross-cutting research direction (from the project brief — treat as hypotheses)

- **Per-model vs per-product.** A purely per-model benchmark stays academic. The
  differentiator is measuring things a per-model board can't: effect of
  pre-filtering / post-processing, known-vocabulary priors (Syrian place names,
  dates), font/dialect robustness. Design at least one "product-shaped" axis.
- **What makes our datasets unique.** Muharaf + NAKBA are *real, degraded,
  20th-century* Arabic — most benchmarks test clean modern or synthetic. State
  what claims only our data supports (e.g. "model X collapses on natural
  diacritic use / ligature density Y").
- **Unicode & tokenisation.** Diacritic score gaps partly reflect *ambiguous
  Unicode representation* of Arabic harakat. Check whether tokenisation is
  inconsistent across the same rendered text, and whether we can influence the
  tokenisation seen by a vision-encoder architecture.
- **Architectures.** Catalogue what current Arabic OCR products actually use
  (RNN/CTC vs Transformer decoder vs VLM). That maps where an SFT or a
  from-scratch effort has leverage.
- **Synthetic data.** Quantify how base models degrade under controlled noise
  (blur, skew, bleed-through, JPEG) vs. our real degradation — tells us whether
  synthetic augmentation would move the needle.
- **Possible products to keep in view.** Digitising a private rare-word
  collection; structured book OCR (chapters, symbols); government forms (MSA +
  handwritten fields); domain-vocabulary-conditioned OCR; dialect / font
  generalisation ("is it a model per font?").
