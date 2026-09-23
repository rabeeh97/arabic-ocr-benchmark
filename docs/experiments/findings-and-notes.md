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

### suite_v1 (Qwen3-VL-4B/8B, AIN-7B, Fanar-2-Oryx): Arabic-tuned 7B bases crush generic Qwen zero-shot, and Fanar's fine-tune breaks the Maghrebi plateau

**Seen (2026-09-20/21, suite_v1 on the 3090, artifacts at
`/Volumes/PortableSSD/repos/arabic_ocr/suite-v1-artifacts/`, not in git — see
memory `suite-v1-artifacts-location`).** Same LoRA recipe for all four models
(bf16, r=16/α=32, ViT frozen, 1 epoch, 6k Muharaf + 6k AraMS + 3k RASAM-1 + 2k
RASAM-2). CER-light, base → fine-tune, on the held-out panel (n=200 unless noted):

| dataset | Qwen3-VL-4B | Qwen3-VL-8B | AIN-7B | Fanar-2-Oryx |
|---|---|---|---|---|
| kitab_khatt | 94%→28% | 48%→23% | **7%→7%** | 40%→22% |
| nakba_lines | 98%→32%(n=820) | 58%→28% | **13%→11%** | 49%→14% |
| rasam1 (Maghrebi) | 150%→27%(n=734) | 88%→25% | 39%→18% | 65%→**11%** |
| rasam2 (Maghrebi) | 139%→33%(n=372) | 85%→32% | 39%→24% | 67%→**17%** |

**Meaning.**
- **AIN-7B's base (zero-shot, before any of our fine-tuning) already beats
  every one of our own fine-tunes on printed/mixed KHATT and NAKBA** (7%/13%
  CER). It is a genuinely different regime from generic Qwen3-VL, not a small
  edge.
- **Fanar-2-Oryx's fine-tune takes RASAM-1 (Maghrebi) to 11% CER** — the
  45–54% Maghrebi plateau reported in the base-model A/B/C entry above is
  **not a hard limit of the script or data**. It was a base-model-choice
  problem. This substantially answers greenfield area 3 (script-family
  coverage): the fix so far has been "start from a better-pretrained base,"
  not a script-balanced data mix or a novel architecture.
- **Correction to an earlier claim in this session:** AIN and Fanar do *not*
  obviously keep an unchanged, web-caption-only vision encoder. AIN
  ([arXiv:2502.00094](https://arxiv.org/abs/2502.00094)) is a **full-parameter**
  fine-tune of Qwen2-VL-7B on 3.6M Arabic-English pairs; Fanar-2-Oryx
  ([arXiv:2603.16397](https://arxiv.org/abs/2603.16397)) came from "targeted
  continual pre-training" on 62M bilingual examples. Neither paper states the
  vision tower was frozen, so both plausibly adapted it. Combined with
  Baseer-Nakba's own best ablation being ViT-unfrozen + differential LR
  (0.16→0.10 CER), **every model we know of that adapted its vision encoder to
  Arabic ended up far ahead, and every one of our own frozen-ViT runs plateaus
  well behind.** This is now a repeated pattern across three independent
  sources, not a single result — treat it as the leading hypothesis for
  greenfield area 1, to be tested directly (see the AIN+NAKBA entry below)
  before considering a from-scratch encoder.

**Per-sample diagnosis (paired base→ft, same 200 lines, full predictions
recovered from `bench/bundle.zip` — bundle.zip's `full/<dataset>/` holds both
base and fine-tuned model rows in one run, unlike `eval/` which only has the
in-domain set):**

| run | worse | better | flat | mean CER base→ft |
|---|---|---|---|---|
| AIN-7B on rasam1 | 7 | 185 | 8 | 40.5%→19.9% |
| Fanar on rasam1 | 1 | 199 | 0 | 70.6%→12.9% |
| AIN-7B on nakba_lines | 46 | 71 | 83 | 21.6%→19.3% |
| Fanar on nakba_lines | 6 | 182 | 12 | 58.7%→22.2% |

AIN's nakba_lines result echoes the Baseer-Nakba mild-forgetting pattern from
the entry above (a fair number of individual regressions against an
already-decent base) — same mechanism, much milder, since AIN's base wasn't
already at Baseer's home-domain ceiling.

**Reading the actual text surfaced four distinct failure modes** (CER alone
hides all of these):

1. **Base models hallucinate fluent, unrelated Arabic.** AIN base on
   `rasam1/test/000142`: ground truth "خيمة ضربها حول" (a tent, pitched
   around); base output "مَنْ يُحِبّ طَهارةً تَعالى" — grammatical, fully
   diacritised, and invented. Fine-tuning fixes most of this class — it's a
   decoding-prior problem, not primarily a vision problem.
2. **Real dot/letter-shape confusion survives fine-tuning.** Fanar-ft on
   `rasam1/test/000058`: "الطيور" (the birds) → "الظيرون" — ط/ظ differ by one
   dot, ي/ر got scrambled too. This is the concrete, script-level error the
   Arabic-native-encoder hypothesis is actually about, and it's still present
   post-fine-tune on Maghrebi script specifically.
3. **Fine-tuning imposes a strong "always output Arabic" prior that breaks on
   genuinely non-Arabic content.** AIN nakba/test/000123: ground truth is the
   English header "TUESDAY"; base copied it correctly, the fine-tune
   translated it to "الثلاثاء". Worth its own metric (foreign-token fidelity)
   — invisible in aggregate CER, and a real risk for any deployed reader.
4. **Very short/fragment lines are noisy for everyone**
   (`rasam1/test/000199`, ground truth just "هم") — both base and fine-tune
   guess wildly. An eval-protocol note (down-weight or flag very short lines),
   not a modeling finding.

**To do.** Add a dot/diacritic-confusion rate and a foreign-token-fidelity
check to the standard eval, alongside CER.

**Qwen3-VL-4B/8B added (2026-09-21) — a fifth failure mode, and it's the
opposite direction from AIN's.** Paired base→ft, same recovery method:

| run | worse | better | flat | mean CER base→ft |
|---|---|---|---|---|
| Qwen3-VL-8B on rasam1 | 2 | 194 | 4 | 95.3%→27.5% |
| Qwen3-VL-8B on nakba_lines | 10 | 154 | 36 | 61.0%→31.4% |
| Qwen3-VL-4B on rasam1 | 0 | 198 | 2 | 131.5%→28.3% |
| Qwen3-VL-4B on nakba_lines | 4 | 182 | 14 | 95.5%→34.9% |

Reading the text: **generic Qwen3-VL base models do not hallucinate unrelated
fluent phrases the way AIN's base sometimes does (failure mode 1 above).**
Instead they attempt a real character-level reading and mangle it with heavy,
compulsive invented diacritics — e.g. `rasam1/test/000001`, ground truth
"سنة ست وخمسين من الهجرة..." (largely undiacritised, as real manuscripts
are), Qwen3-VL-8B base output "اسمُ سَتْ رَضِي مَا عَنْهُمَا..." — fully
voweled with shadda/fatha/damma on nearly every letter, unrelated to what's on
the page, while still tracking the rough consonant skeleton. This is **failure
mode 5: compulsive over-diacritization as part of the "fluent Arabic" prior**,
distinct from mode 1 (AIN inventing whole disconnected phrases) — different
base models default to different failure shapes even on the same kind of
error. Fine-tuning removes nearly all of it (`rasam1/test/000005`: base
"واجتمعت علييد اعل الساحل" vs ft "واجتمعت عليه اهل الساحل", an exact match).

**Relevance to area 5 (diacritics as a separate problem).** This is direct
evidence for the split: base models are guessing diacritics as a fluency
reflex, disconnected from the image, on datasets that are themselves mostly
undiacritised. A "read the letters only, no diacritics" task framing on the
base model — removing the temptation to sound fluent — is now a concretely
motivated first probe for that experiment, not just a hypothesis.

### AIN-7B + NAKBA-specific training: within ~1.4 CER points of Baseer, ViT-unfreeze run in flight

**Seen (2026-09-21), official Systems-Track protocol** (HF
`U4RASD/omar-al-saleh-manuscripts-segments` split `blind_test`, n=2,671, corpus
CER = micro raw, matching the contest's own ranking metric — see
`suite-v1-artifacts/ain7b_nakba/official/README.md` for the full metric
reconciliation against the published Baseer number):

| model | n | corpus CER (raw) | WER (raw) | CER-light |
|---|---|---|---|---|
| AIN-7B+NAKBA LoRA, ckpt-1600 | 2670 | **10.78%** | 34.28% | 8.63% |
| AIN-7B+NAKBA LoRA, ckpt-1936 (epoch end) | 2670 | 10.82% | 34.22% | 8.67% |
| nakba-misraj-baseer (our re-score, same harness) | 2671 | 9.38% | 29.30% | 7.28% |
| Baseer-Nakba, published | 2671 | **7.9%** | 24.4% | — |

**Meaning.** A single LoRA on AIN-7B, trained only on NAKBA's own train split
(no Muharaf pretraining stage), gets within **1.4 CER points of Baseer-Nakba**
in our harness — a much cheaper recipe than Baseer's two-stage
(Muharaf-then-NAKBA, full-parameter, weight-averaged) pipeline. The training
curve (ckpt 400→1600, CER-light 7.9%→6.2% on the easier public 200-line
`nakba_lines`) was still improving, not saturating, at the point this was
scored. Our own re-score of published Baseer (9.4% vs their 7.9%) has a known,
unclosed ~1.5-point gap attributed to prompt/decoding differences (Baseer's
contest prompt + `repetition_penalty=1.1` vs our `line-v2` greedy) — so AIN's
true gap to Baseer *on identical decoding* is likely smaller than 1.4 points,
not larger.

**In flight.** Next run: unfreeze AIN's ViT with a differential LR (decoder
faster, encoder slower — the same shape as Baseer's stage-2 recipe and the
exp3090 `visft` config), retrained on NAKBA. This is a same-model
before/after test of the vision-encoder-adaptation hypothesis above, cleaner
evidence than the cross-model comparison since architecture and data are held
fixed. **This is the opening experiment of P1** in the greenfield-areas design
(`docs/superpowers/specs/` — write-up pending); write full results here once
the checkpoint lands.

---

### Three quick re-analyses (no GPU needed) — diacritic base rate, dot-confusion rate, hallucination proxy

**Done (2026-09-23), reusing the suite_v1 predictions already recovered from
`bench/bundle.zip` for AIN-7B, Fanar, Qwen3-VL-4B/8B — base and fine-tuned,
across rasam1/rasam2/nakba_lines/kitab_khatt (+ muharaf_lines/arams_htr for
the base-rate check). All three are pure re-analysis of existing predictions
and ground truth, no training or inference run.**

**1. Diacritic base rate — the benchmark barely tests restoration at all.**
Fraction of ground-truth *characters* that are a diacritic mark, per corpus:

| dataset | diac rate (chars) | lines with ≥1 diacritic |
|---|---|---|
| arams_htr | 0.00% | 0.0% |
| rasam2 | 0.01% | 0.5% |
| nakba_lines | 0.03% | 2.0% |
| rasam1 | 0.13% | 7.5% |
| kitab_khatt | 0.40% | 18.9% |
| muharaf_lines | 1.42% | 36.0% |

**Meaning.** Every corpus is overwhelmingly undiacritised, and three of six
(arams_htr, rasam2, nakba_lines) have almost no diacritics to restore at all
— a diacritic-restoration experiment run against them would be measuring
noise. Only **muharaf_lines (36% of lines) and kitab_khatt (19%) have enough
diacritised lines to say anything meaningful about restoration** — any area-5
experiment should score on those two specifically, not the full panel.

This also reframes `diac_f1` as currently computed: `metrics.py::diacritic_prf`
special-cases an empty reference to P=R=F1=1.0 if the hypothesis is also
empty, and to F1=0.0 for a single invented mark otherwise. Since the
reference is empty on 63–100% of lines depending on corpus, **the aggregate
diac_f1 score is mostly measuring "does this model refrain from inventing
diacritics," not restoration quality** — which is a real and useful thing to
measure, but is not the same claim as "this model restores diacritics well,"
and explains the previously-unexplained cross-corpus diac_f1 swings (e.g.
0.97 on arams_htr, a corpus with literally zero ground-truth diacritics,
vs 0.55 on muharaf, the corpus with the most).

**2. Dot/letter-shape confusion rate — a big, consistent, and initially
counter-intuitive pattern.** Of all character-level *substitution* errors
(ref→hyp, via edit-distance alignment), the fraction that fall within a
one-dot-different confusable group (ب/ت/ث/ن/ي, ج/ح/خ, د/ذ, ر/ز, س/ش, ص/ض,
ط/ظ, ع/غ, ف/ق):

| model | rasam1 base→ft | rasam2 base→ft | nakba base→ft | khatt base→ft |
|---|---|---|---|---|
| Qwen3-VL-4B | 4.4%→18.6% | 4.0%→16.2% | 3.2%→11.0% | 5.7%→14.5% |
| Qwen3-VL-8B | 5.4%→16.5% | 5.7%→14.9% | 5.9%→12.7% | 8.9%→13.7% |
| AIN-7B | 15.0%→18.9% | 15.2%→17.7% | 11.9%→11.1% | 20.7%→20.8% |
| Fanar | 8.2%→29.3% | 8.5%→24.3% | 7.0%→12.3% | 10.4%→16.2% |

**Meaning.** In 14 of 16 model/dataset pairs, the dot-confusion rate *rises*
after fine-tuning, even though the *total* number of substitution errors
drops sharply everywhere (e.g. Fanar on rasam1: 2059→601 total substitutions,
a 3.4× drop, while the dot-confusable share of what's left nearly quadruples).
The direction is consistent across all four models and all four datasets —
not noise from a shrinking denominator alone. **Reading:** fine-tuning clears
the "easy" errors first (wrong words, hallucinated phrases, big misreads —
decoder/fluency-prior problems), and what's left behind is increasingly the
hard, fine-grained visual-discrimination errors — exactly the class a better
encoder should address. This is the sharpest evidence yet that **encoder
quality, not decoder exposure, is the binding constraint on the errors that
remain after a normal fine-tune** — directly motivating area 1's encoder
work as the next lever, not a replacement for what's already been tried.

**3. Hallucination-rate proxy (length-ratio + CER heuristic: prediction
>1.8× ground-truth length *and* CER >0.6) confirms failure mode 1 is
concentrated and fixable, not universal.**

| model | rasam1 | rasam2 | nakba | khatt |
|---|---|---|---|---|
| Qwen3-VL-4B base | 17.5% | 22.0% | 5.0% | 7.2% |
| Qwen3-VL-8B base | 8.5% | 9.5% | 2.5% | 1.0% |
| AIN-7B base | 0.5% | 2.0% | 1.0% | 0.0% |
| Fanar base | 1.5% | 3.5% | 1.0% | 0.5% |
| every fine-tuned model | ≤1.0% (mostly 0%) | ≤1.0% | ≤1.0% | 0% |

**Meaning.** The fluent-hallucination failure mode (mode 1) is concentrated
almost entirely in **generic Qwen3-VL's base model on the hardest script
(Maghrebi, 17–22% of lines)** — the already-Arabic-tuned bases (AIN, Fanar)
rarely show it even zero-shot (0.5–3.5%). Any of our fine-tuning recipes
essentially eliminates it. This is a crude proxy (length ratio + CER
threshold, not a real semantic check) — good enough to confirm the pattern,
not precise enough to be the area-4 hallucination metric itself.

**To do.** Build the real hallucination-rate metric for area 4 (this proxy
is a starting point, not the final version); the letters-only-prompt probe
for area 5 needs a new inference run (below).

**4. Per-diacritic-type breakdown (muharaf_lines/kitab_khatt only, the two
corpora with enough diacritics to score) — fine-tuning trades hallucination
for omission, it does not learn real restoration.** Precision/recall per mark
type, char-aligned via edit distance on the harakat-only subsequence, Qwen3-VL-4B
base → fine-tune on muharaf_lines:

| mark | support | base P / R | ft P / R |
|---|---|---|---|
| fatha | 113 | 0.02 / 0.86 | 0.26 / 0.24 |
| fathatan | 44 | 0.15 / 0.41 | 0.30 / 0.55 |
| shadda | 43 | 0.04 / 0.67 | 0.24 / 0.33 |
| sukun | 13 | 0.01 / 0.92 | 0.14 / 0.08 |
| kasra | 13 | 0.00 / 0.85 | 0.00 / 0.00 |
| damma | 6 | 0.00 / 0.33 | 0.00 / 0.00 |

**Meaning.** The base model has near-zero precision but fairly high recall on
every mark (it emits diacritics almost everywhere, so it "catches" most real
ones by flooding — fatha alone has 5,410 false positives on 200 lines).
Fine-tuning fixes precision a lot (still far from good) but **recall
collapses for the rarer marks** — kasra and damma go to 0.00 recall, sukun
drops to 0.08. Support is tiny per mark (6–113 instances across 200 lines,
consistent with the base-rate finding above), so the fine-tune's easiest way
to raise precision is to **stop emitting diacritics almost entirely**, not to
place them correctly. This is direct, quantified evidence for the two-stage
split (area 5, decision D1): a single fine-tune trained mostly on
undiacritised text has no real pressure to learn restoration — it learns
suppression instead, which looks like an improvement on aggregate diac_f1
without being one.

**5. Unicode/tokenization check — mostly clean, one small exception.** No
Arabic presentation-form characters (U+FB50–FDFF, U+FE70–FEFF) found in any
ground truth checked (kitab_khatt, muharaf_lines). NFC normalization changes
0/180 kitab_khatt lines and 6/400 muharaf_lines lines — a small, non-zero rate
worth a closer look (likely non-canonical combining-mark order in the source
transcriptions) but not the dominant source of diac_f1 noise; the base-rate
and per-type findings above explain most of it. Closes the long-open
cross-cutting research note on this question.

**Still open, needs a new inference run (waiting on the 3090):** the
letters-only-prompt ablation and the two-stage read/restore pipeline.

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
- **Unicode & tokenisation — checked 2026-09-23, see the "Three quick
  re-analyses" entry above.** No presentation-form artifacts found; NFC
  normalization changes 0/180 kitab_khatt and 6/400 muharaf_lines ground-truth
  lines. Not the dominant source of diac_f1 noise — the base-rate and
  per-diacritic-type findings in that entry explain most of it instead.
  Whether we can influence tokenisation seen by a vision-encoder architecture
  remains open, folded into greenfield area 1.
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
