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
- **Format vs vision — RESOLVED (2026-09-15), using data that already existed.**
  How much of the gain is the *output-format prior* (line-only, no chat) vs.
  genuine visual grapheme learning? A control that only changes the prompt /
  system message on the *base* model would separate them — and it turns out
  this control was already run, just never framed as one: every held-out
  `qwen3-vl-4b-bf16` (untuned base) row already used `prompt_version: line-v2`
  (`"Do NOT add HTML tags, Markdown... Do NOT repeat words... Output only the
  transcription — nothing before or after it"` — src/arabic_ocr_bench/prompting.py),
  the exact kind of anti-commentary, format-only instruction this question was
  asking about. The base *under that strict formatting prompt* still scores
  76–119% CER on `kitab_khatt`/`nakba_lines`/`rasam1`/`rasam2` — barely
  different from its behavior described earlier in this doc (`"mostly emits
  meta-commentary, wrong-script text, or fully-diacritised confabulation"`).
  **Telling the base model to format its output correctly does not close
  meaningfully any of the gap to the ~25–55% CER the LoRA fine-tune reaches.**
  The fine-tuning gain is overwhelmingly genuine visual/script learning, not
  output-format learning — no new GPU run needed to answer this, the answer
  was already sitting in the existing base-vs-fine-tune table.
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

### exp3090 — PaliGemma-2-3B, vision unfrozen (paligemma-visft): the opposite result from Qwen

**Seen (2026-09-15).** Same question as qwen3vl4b-visft, on a second,
architecturally unrelated base — PaliGemma's fixed-448 SigLIP + Gemma-2 vs
Qwen's dynamic-resolution ViT. Identical recipe and 8k-line Muharaf data to
exp3090-paligemma2-3b, only `train_vision` flips `false` → `true`. CER
(light), frozen-ViT fine-tune → vision-unfrozen fine-tune:

| dataset | script | frozen ViT (n) | ViT unfrozen (n) | Δ |
|---|---|---|---|---|
| kitab_khatt | Naskh/Ruq'ah HW | 37.7 % (n=180) | 39.2 % (n=199) | +1.5 (worse) |
| nakba_lines | 20th-c. degraded HW | 31.8 % (n=200) | 30.6 % (n=200) | −1.2 (better) |
| rasam1 (Maghrebi) | Maghrebi manuscript | 52.0 % (n=199) | 54.5 % (n=200) | +2.5 (worse) |
| rasam2 (Maghrebi) | Maghrebi manuscript | 53.8 % (n=196) | 55.6 % (n=196) | +1.8 (worse) |
| kitab_hindawi | clean printed | 39.5 % (n=178) | 50.9 % (n=180) | +11.4 (much worse) |

**Meaning — this directly contradicts the Qwen3-VL-4B result above.**
Unfreezing the ViT helped Qwen on every held-out set (biggest win on
`nakba_lines`, real gains on `rasam1`/`kitab_khatt`). For PaliGemma,
unfreezing SigLIP is flat-to-worse on four of five sets, and a large
regression on `kitab_hindawi` (clean printed Arabic — the easiest, most
in-distribution-for-a-pretrained-vision-encoder set, which is exactly where
you'd expect unfreezing to hurt least if it hurts at all). Only `nakba_lines`
improved, and only marginally. So "unfreeze the vision tower" is **not** a
universal recipe improvement — it's at best base-architecture-dependent, and
for PaliGemma this run it actively cost accuracy on the majority of sets
including the domain (Maghrebi) the intervention was designed to help.

Two architectural differences are plausible explanations, neither confirmed
here: (1) SigLIP-so400m (PaliGemma's vision tower) is already extensively
pretrained on OCR/document-adjacent data via PaliGemma's own multi-task
pretraining mixture, so there may be less headroom to gain and more to lose
by perturbing it with only 8k lines; Qwen3-VL's ViT has no equivalent
document-specific pretraining. (2) PaliGemma's LoRA setup here touches a
much smaller, more specialized vision tower (SigLIP-so400m, ~400M params)
relative to Gemma-2, vs Qwen3-VL's ViT being a larger share of the overall
dynamic-resolution architecture — the same nominal LoRA rank may be a
proportionally more disruptive perturbation for PaliGemma's vision tower.

Base-model note for this run: the untuned `paligemma2-3b-bf16` base scores
95–99 % CER across every held-out set here (vs. 66–88 % in the original
base-model A/B/C table) — checked the raw predictions and it's not a
pipeline bug, the base is emitting English image captions
(`"arabic calligraphy"`, `"the first line of a poem"`) rather than
attempting transcription at all, the same "unusable, emits meta-commentary"
failure mode already documented for untrained Qwen3-VL-4B. The gap from the
earlier 66–88 % reading is most likely just different specific lines being
sampled (accumulate-mode offsets differ run to run), not a behavior change —
a fundamentally non-OCR-following base's CER is inherently noisy across
different line samples.

**To do.**
1. Don't default to `train_vision: true` for PaliGemma the way we now do for
   Qwen — the frozen-ViT PaliGemma recipe is currently the better one on 4/5
   held-out sets.
2. ~~The `kitab_hindawi` regression is worth a per-sample diff~~ — done below;
   it's broad, systematic degradation, and the actual mechanism (premature
   EOS, not truncation) is now understood.
3. If a third vision-unfrozen base is ever run, this is now a real ambiguous
   result to break the tie on, not a confirmatory rerun.

**`kitab_hindawi` regression diagnosis (2026-09-15).** Used the new
`scripts/pair_diff.py` (generalizing the ad-hoc method used for the Baseer
diagnosis into a reusable tool) to compare frozen vs. vision-unfrozen
PaliGemma per sample_id on `kitab_hindawi`:

```
uv run python scripts/pair_diff.py \
    --model-a exp3090-paligemma2-3b-ft-bf16 \
    --model-b exp3090-paligemma2-3b-visft-ft-bf16 \
    --dataset kitab_hindawi
```

`n paired = 178, worse = 150 (84%), better = 27, flat = 1` — broad, systematic
degradation, not a handful of outliers dragging the mean (the same pattern as
Baseer's genuine `nakba_lines` forgetting, not Baseer's noise-level `rasam2`
result).

**Root cause: premature EOS, not the missing-token-budget bug.** Both frozen
and vision-unfrozen rows already carry the same explicit `max_new_tokens: 640`
(set in the same fix as the truncation bug above), so this looked at first
like it could be the same class of problem — it isn't. A median
`len(prediction)/len(ground_truth)` check confirmed vision-unfrozen's ratio
(0.70) is still meaningfully lower than frozen's (0.83) despite the equal cap,
and looking at the actual predicted text for the two biggest regressions
confirms why: the model is **stopping generation on its own, far short of the
token budget**, not being cut off by it.

- `kitab_hindawi/train/000171` (GT 879 chars): frozen predicts 736 chars (a
  near-complete, plausible transcription); vision-unfrozen predicts **11
  characters** — `"الطويل ولكن"` ("The long but"), the first two words of the
  line, then EOS.
- `kitab_hindawi/train/000173` (GT 541 chars): frozen predicts 465 chars;
  vision-unfrozen predicts 71.

**Interpretation.** The 8k-line Muharaf training set is exclusively short,
line-level handwriting (median ~50 chars per line). Unfreezing the vision
tower during that fine-tune appears to have made the model overfit more
strongly to "the response ends soon" as a length prior tied to what it was
actually shown, at the cost of generalizing that prior to genuinely longer,
page-scale printed content it never saw during training. The frozen-ViT
fine-tune, whose vision tower was never touched, evidently retains more of
its original pretrained length calibration for longer documents. This is a
plausible, secondary explanation for the `kitab_hindawi` regression on top of
the two architectural theories above — and it's specific to output *length*
control, not visual reading quality per se (both models get plenty of
individual words/phrases right in what they do produce, per the raw text
above).

**To do.** This EOS-timing effect is now the clearer of the two `train_vision`
questions raised on this base and worth a small targeted check: does it show
up on `kitab_historicalbooks` too (also page-scale, shorter GT so a milder
version of the same test), or only on the longest-GT dataset (`kitab_hindawi`
has the longest median ground truth of any dataset in the benchmark)?

### kitab_hindawi / kitab_historicalbooks — most models are being measured wrong, not performing badly

**Seen (2026-09-15).** Every model's CER on these two datasets looked
uniformly bad (`exact_light = 0.0` for essentially every model in the
leaderboard, including ones scoring 1-4% CER elsewhere) — worth checking
whether that's real model failure or a granularity mismatch, since both
datasets ship page/paragraph-scale ground truth, not line-scale, unlike
every other dataset in the benchmark. Pulled every prediction's
`len(cleaned_text) / len(ground_truth)` ratio and empty-output rate across
all ~40 models scored on either set. Confirmed: median ground truth is
**~880-920 characters** for `kitab_hindawi` and **~380 characters** for
`kitab_historicalbooks` — several paragraphs, not a line (a Muharaf line
ground truth is typically 30-80 characters). Two distinct, previously
undiagnosed failure modes explain most of the damage:

**1. Silent truncation from an undersized generation cap.** Every
`hf_local` model (the RTX-3090 fine-tunes, their untuned bases, `qari-ocr-*`)
defaults to `max_new_tokens=160` in `hf_local_client.py` unless a config row
explicitly overrides it — nowhere near enough for an 880-character target.
Only the two PaliGemma rows (`paligemma2-3b-bf16`,
`exp3090-paligemma2-3b-*-ft-bf16`) got an explicit `max_new_tokens: 640`
written into `configs/models.yaml`; Qwen3-VL and Baseer-Nakba's rows never
did, despite `run_3090_benchmark.py --max-new-tokens 640` defaulting to 640
for exactly this reason. Median prediction-length ratios on `kitab_hindawi`:

| model | ratio (pred len / gt len) |
|---|---|
| `baseer-nakba-bf16` / `exp3090-baseer-nakba-ft-bf16` | 0.10 |
| `paligemma2-3b-bf16` (base) | 0.02 |
| `qari-ocr-0.2` | 0.15 |
| `trocr-large-arabic` (own cap: 96) | 0.06 |
| `exp3090-qwen3vl4b-ft-bf16` (frozen) | 0.32 |
| `qwen3-vl-4b-bf16` (base) | 0.52 |
| `exp3090-qwen3vl4b-visft-ft-bf16` | 0.88 |
| `exp3090-paligemma2-3b-*-ft-bf16` (both, explicit 640 cap) | 0.70-0.83 |

A CER computed against a reference 5-10x longer than what the model was
even allowed to emit isn't measuring OCR quality, it's measuring the
truncation cap. `trocr-large-arabic` is a structurally different case worth
calling out separately: it's a line-level CRNN-style architecture, not a
VLM — a 96-token cap isn't a bug to fix, it's evidence this dataset is the
wrong task shape for that model entirely, and its number here should
probably be excluded rather than "fixed."

**2. High empty-output rates on several hosted/API models — a different bug,
not truncation.** Some models return nothing at all on a meaningful fraction
of `kitab_hindawi` samples, with `n` (successfully-scored count) already
excluding these, so this doesn't show up as a low CER — it shows up as a
quietly small sample count nobody had flagged as suspicious:

| model | empty rate |
|---|---|
| `deepseek-vl-flash`, `mistral-medium`, `mistral-small`, `openrouter-dots3-note-free` | 100% |
| `openrouter-gemma-4-26b-free`, `openrouter-gemma-4-31b-free` | 97% |
| `openrouter-minimax-m3-free` | 91% |
| `glm-5.3-flash-paid` | 87% |
| `openrouter-llama-4-maverick`, `openrouter-qwen2.5-vl-72b`, `openrouter-qwen3-vl-235b` | 71% |
| `gemini-flash-page` (the already-attempted page-scale fix, see below) | 64% |
| `gemini-flash-lite-page` | 29% |
| `gemini-flash` (vanilla, non-page) | 23% |

**This overlaps almost exactly with the "7 fully-broken free-tier models"
flagged from the interrupted free-accumulate sweep** (`mistral-medium`,
`mistral-small`, `openrouter-gemma-4-26b-free`, `openrouter-gemma-4-31b-free`,
`openrouter-llama-4-maverick`, `openrouter-qwen2.5-vl-72b`,
`openrouter-qwen3-vl-235b` — 7 for 7). That sweep's config
(`configs/run.free-accumulate.yaml`) includes `kitab_hindawi` in its dataset
list, so it's very plausible those models aren't broken across the board —
they may specifically choke on page-scale/long-context requests (a large
image, a long expected output, or both), which would make "100% error rate"
from the earlier free-tier investigation a measurement artifact of testing a
long-context-hostile model against a page-scale dataset, not a real API/auth
break. **Needs re-verification on a line-scale dataset before writing those
7 off** — they may be perfectly fine on `kitab_khatt` / `muharaf_lines`.

**GLM-5.3 specifically is worth flagging** given it's a named model of
interest for this project (frontier + trainable via Thinking Machines'
Tinker) — 87% empty on `kitab_hindawi` and outright `cer_light=1.0` (every
sample either empty or maximally wrong) on `kitab_historicalbooks`. Whatever
is causing this needs to be understood before GLM-5.3 can be evaluated fairly
at all, on *any* dataset, not just resolved for these two.

**A fix already exists for part of this and should be generalised.**
`gemini-flash-page` / `gemini-flash-lite-page` are a prior attempt at a
page-scale-specific config (`reasoning_effort: none` — Gemini 3.x's
"thinking" was consuming the whole token budget before the visible answer
even started — plus `max_tokens: 900`). `gemini-flash-lite-page` mostly
worked (93% length ratio) but `gemini-flash-page` still shows a 64% empty
rate, so even that fix is incomplete. The general shape of the fix (bump the
token budget, disable/limit any hidden "thinking" tokens for models that
have them) is very likely the same shape needed for the other
high-empty-rate models above.

**To do.**
1. ~~Set `max_new_tokens: 640`~~ — superseded by a proper fix: `runner.py` now
   resolves a real per-`(model, dataset)` generation budget from
   `configs/datasets.yaml`'s `max_tokens_hint` field, not a hardcoded
   per-model edit. See the "real per-dataset generation-token budget" entry
   below.
2. Debug `gemini-flash-page`'s remaining 64% empty rate — the existing page
   fix isn't fully working. Still open.
3. ~~Re-run the 7 free-tier models flagged as "fully broken" against a
   line-scale dataset~~ — done. **Confirmed: the "fully broken" verdict was a
   page-scale artifact for 5 of the 8 flagged models.** Re-ran all 8 against
   `kitab_khatt` (line-scale): `openrouter-qwen3-vl-235b` (33% CER, 0 errors),
   `openrouter-qwen2.5-vl-72b` (37%), `openrouter-llama-4-maverick` (56%),
   `glm-5.3-flash-paid` (81% CER — high, but producing real output, not
   empty), and `mistral-small` (82%) are all healthy on line-scale data. The
   two `openrouter-gemma-4-*-free` rows hit a genuine free-tier daily-quota
   wall mid-run (`5+ consecutive rate-limit errors`) — a real, separate,
   already-understood issue (see `skip-rate-limited-free-models` /
   `stalled-sweep-diagnosis`), not page-scale-related. `mistral-medium`
   failed with a live `RateLimitError` (429) that traces to running
   back-to-back with `mistral-small` in the same short test, sharing one
   Mistral account's rate limit — also not page-scale-related, just needs
   its own isolated run or a gap between the two Mistral models.
4. **GLM-5.3 is not actually broken** — it was returning empty output
   specifically on the page-scale datasets (now understood as a page-scale
   generation-budget problem, addressed by fix #1 above); on `kitab_khatt` it
   produces real predictions (81% CER — poor, but a real signal to act on,
   not a pipeline failure).
5. Once `gemini-flash-page` (#2) is fixed, every "-page" style CER on these
   two datasets should be treated as provisional and re-scored — several
   numbers currently on the leaderboard for these two datasets were measuring
   truncation/empty-output bugs, not OCR quality. Re-running the free-tier
   sweep (`configs/run.free-accumulate.yaml`) now that the real per-dataset
   budget mechanism is live should recover most of these models' `kitab_hindawi`
   coverage too.

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

### OCR-conditioned correction (2026-09-15) — a "per-product" post-processing axis, not yet run

Implemented per arXiv 2608.22366's finding that feeding a classical OCR
engine's rough pass to a VLM as a correction prior helps a lot on clean
printed text (up to 53% relative CER improvement over Tesseract alone on
their aged-print analog) but *hurts* on cursive/Maghrebi manuscripts, where
Tesseract's own output is unreliable enough to mislead the corrector more
than it helps — this matches our own independent struggles on RASAM/Muharaf
for unrelated reasons, so this technique is deliberately scoped to
`kitab_hindawi` / `kitab_historicalbooks` only.

- `src/arabic_ocr_bench/models/tesseract_client.py`: `TesseractClient` (a
  Tier-5 classical-OCR anchor, ignores the prompt) and `OCRCorrectedClient`
  (wraps any other registered `ModelClient`, runs Tesseract first, and if it
  succeeds, appends its raw output to the prompt as an explicitly-untrusted
  "for reference only" prior before delegating to the wrapped client — a
  Tesseract failure falls back to the wrapped client's plain-prompt behavior
  rather than erroring the whole call).
- New `client: tesseract` / `client: ocr_corrected` cases in
  `models/registry.py`'s `build_model()`.
- New rows in `configs/models.yaml`: `tesseract-ara` (the anchor) and
  `gemini-flash-lite-page-corrected` (wraps the already-fixed
  `gemini-flash-lite-page` row — thinking disabled, max_tokens: 900 — with a
  Tesseract Arabic prior).
- `pyproject.toml`: new `ocr` extra (`pytesseract`) — needs the system
  `tesseract-ocr` binary + Arabic traineddata installed separately (not a
  pip package); not installed in this dev environment, so this is
  code-complete but **not yet run** against real data.
- **To do**: install the system Tesseract binary + `ara` traineddata, run
  `gemini-flash-lite-page-corrected` vs. plain `gemini-flash-lite-page` and
  `tesseract-ara` alone on `kitab_hindawi` / `kitab_historicalbooks`, and
  confirm (or refute) the paper's finding on our own data. No GPU needed —
  Tesseract runs on CPU and the corrector reuses an existing hosted API call.
