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
data. (3) ~~The runner should surface `empty_rate` and median `len(pred)/len(gt)`
per (model, dataset)~~ — done below (2026-09-13), see `report.py`'s
`len_ratio_median` column and truncation warning.

### Same bug, almost every hosted model: only 6 of ~35 models were ever fixed, and one of those 6 was still broken for a different reason

**Seen (2026-09-13).** Checked which models actually got the `page-v2` /
raised-token-cap fix above. `run.page-v2-rollout.yaml`'s own header said
"Free/no-cost hosted models only for this pass" — literally 6 models
(`gemini-flash`, `pixtral-12b`, `mistral-small`, `mistral-medium`,
`openrouter-gemma-4-31b-free`, `openrouter-gemma-4-26b-free`). Every other run
config that lists `kitab_hindawi` / `kitab_historicalbooks` — `run.paid-*.yaml`,
`run.free-*.yaml`, `run.gemini-full.yaml` — still uses `prompt_version: line-v2`
with no token-cap override. Median prediction length vs ground truth across the
whole roster (`len(pred)/len(gt)`, cheap to check directly on `results/*/predictions.jsonl`):
almost every paid and free model comes in under 0.5, several under 0.15 (see
`report.py`'s new `len_ratio_median` column for the mechanism now built to
catch this automatically). No other dataset shows this pattern — every one
besides these two has a ~40-70 char mean ground truth (checked directly), well
within any reasonable line-level cap.

**A second, distinct bug hiding inside the "already fixed" list.** `gemini-flash`
was supposedly already fixed (page-v2 prompt, 640-token cap via
`run.page-v2-rollout.yaml`) but its cached `kitab_hindawi` results still showed
`len_ratio` ≈ 0.07 — worse than several *unfixed* models. Per-sample inspection:
`output_tokens` was consistently ~636/640 (i.e. it always used its whole
budget) while the actual visible answer (`raw_text`, not just `cleaned_text` —
ruled out a parsing artifact) was only 40-90 characters. This is not "cap too
low", it's Gemini 3.x's adaptive thinking consuming nearly the entire output
budget on hidden reasoning before the visible transcription starts — the exact
failure mode this repo already defends against for the `-paid` reasoning rows
(`gpt-5-paid`, `gpt-5.6-*-paid`, `claude-*-paid`: `reasoning_effort` set to
`minimal`/`none` and `max_tokens` raised to 4000-8000, see the "thinking/reasoning
models" comment block in `configs/models.yaml`) — just never applied to Gemini.

**Done.**
1. `runner.py`'s `max_tokens_override` now only *raises* a model's cap
   (`max(existing, override)`), never lowers it — so one run config can safely
   mix reasoning models (already at 4000-8000) with plain ones (at the
   256 default) without silently clobbering the reasoning models' headroom.
2. `report.py` now records a per-sample `len_ratio` (prediction length /
   ground-truth length) and a per-(model, dataset) `len_ratio_median`, and
   prints a loud `WARNING` when ≥60% of a dataset's scored models have a
   median ratio under 0.5 — this is the "wrong granularity" flag the original
   to-do above asked for, and it would have caught this whole class of bug
   automatically instead of it hiding as a uniformly bad CER.
3. New model rows `gemini-flash-page` / `gemini-flash-lite-page`:
   `reasoning_effort: none`, `max_tokens: 900`. New names, not edits to
   `gemini-flash` / `gemini-flash-lite` in place — the cache key is
   `(sample_id, model, prompt_version, CACHE_REV)`, so an in-place edit would
   have kept silently serving the already-cached, still-thinking-enabled
   predictions under the old name.
4. `run.page-v2-rollout.yaml` extended to cover every remaining genuinely-free
   hosted model (`nvidia-gemma-4-31b`, `nvidia-llama-3.2-11b-vl/90b-vl`,
   `openrouter-dots3-note-free`, `openrouter-minimax-m3-free`) and bumped
   640 → 900 tokens (640 wasn't enough for `kitab_hindawi`'s longer tail —
   p90 ≈ 530 tokens, max ≈ 660).
5. New `run.page-v2-paid.yaml` for the 14 paid models — not run without
   sign-off (real spend; ~$2 estimated for `limit: 10` per dataset across all
   14, dominated by `claude-fable-paid` / `claude-fable-5-1-paid`).

**To do.** Extend the same `reasoning_effort: none` treatment to `gemini-pro`
if it's ever used for OCR (same model family, same risk). The 3090-box
fine-tunes (`qwen3-vl-4b-bf16` etc.) also still register `prompt_version:
line-v2` for these two datasets, mitigated only by the 640-token cap already
described above — worth the same `page-v2` treatment eventually, but that's a
GPU-box change, tracked separately from this hosted-API fix.

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

### exp3090 — Qwen3-VL-4B, vision unfrozen (qwen3vl4b-visft): the frozen ViT was leaving real gain on the table

**Seen (2026-09-14).** Identical recipe and 8k-line Muharaf data to
exp3090-qwen3vl4b, only `train_vision` flips `false` → `true`. CER (raw),
frozen-ViT fine-tune → vision-unfrozen fine-tune, same four held-out sets:

| dataset | script | frozen ViT (n) | ViT unfrozen (n) | Δ |
|---|---|---|---|---|
| kitab_khatt | Naskh/Ruq'ah HW | 30.0 % (n=180) | 25.3 % (n=180) | −4.7 |
| nakba_lines | 20th-c. degraded HW | 32.6 % (n=200) | 20.8 % (n=200) | −11.8 |
| rasam1 (Maghrebi) | Maghrebi manuscript | 52.1 % (n=20, noisy) | 43.2 % (n=200) | −8.9 |
| rasam2 (Maghrebi) | Maghrebi manuscript | 49.4 % (n=20, noisy) | 48.6 % (n=195) | −0.8 |

**Meaning.** Unfreezing the ViT wins everywhere, not just on Maghrebi —
`nakba_lines` moved the most in absolute terms (−11.8 pts), with `kitab_khatt`
and `rasam1` both showing solid, clean gains at n≥180 (the frozen-ViT rasam1/
rasam2 numbers were n=20 and flagged noisy at the time; unfrozen has n≥195 on
both, so this is a much cleaner read than the original single-base writeup).
The hypothesis that motivated this run — that a frozen ViT trained only on
Ruq'ah-style Muharaf strokes can't learn Maghrebi-specific grapheme shapes —
is **partially confirmed**: `rasam1` closes a real chunk of the gap (52 %→43 %),
but `rasam2` barely moves (49.4 %→48.6 %), so whatever residual is capping
`rasam2` isn't (only) the frozen encoder. diac_f1 also improved across the
board (e.g. nakba_lines 0.69→0.75), consistent with genuinely better visual
features, not just better formatting.

**To do.**
1. Run the same `train_vision: true` flip on PaliGemma-2-3B
   (`paligemma-visft`, registered, not yet launched) — a second, unrelated
   vision tower (fixed-448 SigLIP vs Qwen's dynamic-res ViT) tells us whether
   "unfreezing helps, but doesn't fully close Maghrebi" is a property of the
   *recipe* or of Qwen's ViT specifically.
2. `rasam2`'s stubbornness despite unfreezing is now the most interesting open
   question in this whole programme — worth a per-sample diff (same method as
   the Baseer regression diagnosis above) against `rasam1` to see what's
   different about the errors that don't move.
3. This makes vision-unfrozen the new default recipe going forward, given the
   consistent (if partial) win and no observed downside — worth carrying into
   any future data-scaling sweep instead of testing frozen-ViT scaling first.

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
