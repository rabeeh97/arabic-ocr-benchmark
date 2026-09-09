# Findings & standing notes — Arabic OCR fine-tuning

Living document. Read before designing the next experiment. Each entry: what we
saw, what it means, what to do about it. Newest section first within each area.

---

## Benchmark validity

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
