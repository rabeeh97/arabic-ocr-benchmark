# AI/ML engineer guide — the Arabic OCR "greenfield" experiment program

Audience: an AI/ML engineer joining the project to run the five greenfield
experiments. This is both onboarding (why we are here) and a teaching path (what
you must be able to do, know, and decide). Read it top to bottom once, then use
sections 3–5 as a checklist while working.

Date written: 2026-09-21. Companion docs: `docs/experiments/findings-and-notes.md`
(what we have measured), `docs/experiments/misraj-baseer-research.md` (how the
current best model was built), `docs/training-plan.md` (rules and subsystem map),
`docs/metrics.md` (metric definitions).

---

## 1. How we got here

### 1.1 What we measured

We built a benchmark harness (`src/arabic_ocr_bench/`) and fine-tuned small
vision-language models (VLMs) on Arabic handwriting on a single RTX 3090. The
results that shaped this program:

1. **Base VLMs are unusable on cursive Arabic.** Qwen3-VL-4B scores 77–119 % CER
   on handwriting sets; it emits commentary, wrong-script text, or invented
   diacritics.
2. **A small LoRA fixes most of it, and the gain transfers.** 8k Muharaf lines
   take Qwen3-VL-4B to ~30 % CER on KHATT and NAKBA, which were never in training.
   This is script-level learning, not corpus memorisation.
3. **Maghrebi (RASAM) stays at ~45–54 % CER** for every fine-tune we ran,
   against ~30 % on Naskh and Ruq'ah.
4. **Two unrelated architectures converge to the same band.** Qwen3-VL-4B and
   PaliGemma-2-3B land at 30–54 % after the same recipe. That points at the
   *data or the vision encoder* as the ceiling, not the decoder or architecture.
5. **A specialised base can be harmed by a generic recipe.** Baseer-Nakba, already
   at ~1 % CER on its home domain, got 3× worse after our LoRA (35 of 63 paired
   lines worse, 5 better). We only understood this through paired per-sample
   analysis, not the aggregate table.
6. **The external evidence agrees.** Misraj's Baseer-Nakba (0.079 CER, NAKBA-2026
   winner) only got from 0.16 to 0.10 CER by *unfreezing the vision encoder* with a
   lower learning rate. Two independent results now say the encoder is the
   bottleneck for handwriting.

### 1.2 The five greenfield areas

From our research notes, the gaps no one has filled:

| # | Area | One-line claim to test |
|---|---|---|
| 1 | Arabic-native vision encoder | An encoder pretrained on Arabic script (dots, ligatures, script families) beats web-caption encoders on handwriting. |
| 2 | Handwriting data | Better or synthetic handwriting data moves CER more than model changes do. |
| 3 | Script-family coverage | The Maghrebi plateau is a data/encoder-coverage problem, not a hard limit. |
| 4 | Small literal line readers | A compact recogniser trained on good Arabic data is competitive, cheaper, and does not invent text. |
| 5 | Diacritics as a separate problem | Splitting "read the letters" from "restore diacritics" beats doing both in one pass. |

### 1.3 Phasing (decided)

All five at full depth, ordered by cost and payoff:

| Phase | Area | Compute |
|---|---|---|
| P1 | 5. Diacritics split; 3. Script-family coverage | 3090 |
| P2 | 4. Small literal line reader | 3090 |
| P3 | 2. Handwriting data / synthetic HTR | 3090, Colab for generation and parallel runs |
| P4 | 1. Arabic-native encoder | Decided after P3, see section 6 |

Each phase ends with a write-up in `docs/experiments/` using `_TEMPLATE.md`, and a
short entry in `findings-and-notes.md` for anything that should change how the next
experiment is designed.

---

## 2. Ground rules of this repo (non-negotiable)

- **No leakage.** Never train on a `(hf_id, split)` the benchmark scores on. This
  is enforced at config-load time in `training/data.py::check_source_not_scored`.
  Every training source is also run through `training/contamination.py` (perceptual
  hash and normalised-text overlap) and the report is cited per experiment.
- **Headline metric:** CER-light micro on the held-out test split, reported next to
  raw CER, diacritic-insensitive CER, diacritic F1, WER and exact match, plus
  training cost. See `docs/metrics.md`.
- **Shared environment on the 3090.** One venv and one HF cache for every run; no
  duplicated downloads; minimal disk.
- **Checkpoint often.** A crash must lose a few samples, not an epoch.
- **Colab runs must be crash-safe:** checkpoint every epoch, back checkpoints up
  off-Colab (Drive plus a private HF repo), and auto-resume. Notebooks add
  `<repo>/src` to `sys.path`; `arabic_ocr_bench` is a src-layout package and
  `pip install -e` is not reliable there.
- **Paid model APIs** must be cache-backed and cost-reviewed before spending.
- **Detached jobs:** whenever you hand someone a command that backgrounds a job, also
  give the command to watch it (log tail or `status` subcommand).
- **Runbook stays reproducible.** If you hit a setup issue following a runbook, fix
  the runbook and the launcher so a from-scratch run stays trivial.

---

## 3. What you must be able to do

Each item lists a self-test, an exercise in this repo, and what "good" looks like.

### 3.1 Turn a vague claim into a falsifiable experiment

*Skill.* "The encoder is the bottleneck" is not testable. "Unfreezing the ViT at
1/10 of the decoder LR lowers Maghrebi CER-light by ≥ 5 points on RASAM-1/2 at
equal training data" is.

*Every experiment plan must state, before running:*
1. Hypothesis and the single variable changed.
2. Baseline, with its own number of samples (n) and expected noise.
3. Success threshold and **kill criterion** (what result makes us stop).
4. Compute budget and expected wall-clock.
5. What the result would change in the next experiment.

*Exercise.* Rewrite the open questions at the end of the "Qwen3-VL-4B + LoRA"
entry in `findings-and-notes.md` (format vs vision; data scaling) into plans with
those five fields.

*Trap to avoid.* Running several changes at once. The exp3090 ViT-unfreeze run
(`configs/training/exp3090_qwen3vl4b_visft.yaml`) is the model of a one-variable
change.

### 3.2 Build and run a training loop end to end

*Skill.* LoRA and full fine-tuning on a 24 GB card, differential learning rates
(decoder vs vision encoder), gradient checkpointing, bf16, checkpoint and resume.

*Read first:* `src/arabic_ocr_bench/training/sft_vlm.py`, `sft_trocr.py`,
`checkpoints.py`, `config.py`, and the configs in `configs/training/`.

*Exercise.* Reproduce the Qwen3-VL-4B 8k Muharaf run at 1k lines. Confirm you can
kill the process mid-epoch and resume with a loss curve that continues, not resets.

### 3.3 Diagnose from predictions, not aggregates

*Skill.* Aggregate CER hides direction. Always compute **paired per-sample deltas**
(base vs fine-tune on the same ids: worse / better / flat), then read the worst 20
predictions by eye.

*Failure modes to look for:* repetition loops, meta-commentary, markup, dropped
or invented diacritics, truncation at the token cap, wrong-script output,
page-scale output from a line prompt.

*Exercise.* Reproduce the Baseer-Nakba per-sample diagnosis on `nakba_lines`
(35 worse / 5 better / 23 flat, n=63 paired). Then explain, in one paragraph, why
the rasam2 result (93 worse / 81 better) is *not* the same story.

### 3.4 Keep evaluation honest

*Skill.* Know exactly what each number means and which normalisation produced it.
Read `src/arabic_ocr_bench/normalize.py` and `metrics.py`. Be able to say why
CER-light, raw CER and diacritic-insensitive CER can disagree, and which one answers
which question.

*Also:* small n is noise. A row with n=20 (RASAM-1 in our first run) is a hint, not
evidence. Report n beside every number.

*Granularity check.* KITAB-Hindawi rows are page-scale, so a line prompt plus a 160
token cap truncates them. A model can look bad because the *task* was wrong. Check
`empty_rate` and median `len(pred)/len(gt)` before trusting any row.

---

## 4. What you must know

### 4.1 Modelling

- **Vision encoders.** ViT patching, why resolution and aspect ratio matter for
  long thin text lines, dynamic-resolution (Qwen) vs fixed-resolution (PaliGemma's
  448 px SigLIP), and what pretraining on web captions does and does not teach.
- **Freezing.** What a frozen encoder can and cannot adapt to. Why unfreezing needs
  a much smaller learning rate than the decoder (we use ~1e-4 vs ~9e-6 as the
  Baseer-Nakba reference point) and what happens if you don't.
- **Decoding.** Autoregressive decoders read fluently and can *invent* text; CTC
  models are monotonic and literal but weaker on context. Know when each hallucinates.
- **Forgetting.** Why a generic recipe can damage a specialised base (finding 5),
  and the mitigations: lower LR, LoRA rank, mixing in replay data, weight averaging
  (Baseer-Nakba averaged epoch-1 and epoch-5 weights).
- **Self-supervised objectives** (masked image modelling, contrastive, auxiliary
  heads) — needed for area 1.

### 4.2 Arabic script

- Letters take up to four shapes (isolated, initial, medial, final) and join
  right-to-left; six letters never join to the left.
- **Dots (i'jam)** distinguish letters that share a skeleton (ب ت ث ن ي). Dots are
  tiny and are the first thing low resolution destroys.
- **Diacritics (tashkeel/harakat):** short vowels, shadda, sukun, tanwin. Usually
  absent in real text. A model that emits them when they are not in the image is
  guessing, which is why the diacritic-insensitive CER exists.
- **Ligatures** (e.g. لا) and elongation (kashida).
- **Script families:** Naskh (printed and common handwriting), Ruq'ah (fast
  handwriting, Muharaf), Diwani (Ottoman chancery, dense and overlapping), **Maghrebi**
  (different letterforms, e.g. the placement of dots on ف and ق, and the reason RASAM
  behaves differently).
- **Unicode:** presentation forms vs base letters, normalisation of alef variants
  and taa marbuta, Arabic-Indic vs Western digits. Normalisation choices change CER
  by real amounts.

### 4.3 Data

- **Licences.** Muharaf is non-commercial. Record the licence of every source in
  `docs/datasets-catalog.md` before it goes into a mix.
- **Synthetic data failure modes:** font bias, a domain gap to real ink and paper,
  and unrealistic text. Qari's and Baseer's pipelines (Markdown → HTML → PDF → image
  with many fonts and augmentations) are the reference designs. Handwriting needs a
  different generator.
- **Splits.** Use the corpus's real split (AraMS-28k-HTR is book-based, and inventing
  a split cost us a fix). For held-out script families, hold out the *family*, not a
  random slice.

### 4.4 Research judgement

- Distinguish memorisation from script learning by testing on corpora never seen.
- Distinguish output-format learning from visual learning: a base model that just
  learns "emit one bare line" already gains a lot. A prompt-only control on the base
  model separates the two.
- Report negative results. "Specialised base + LoRA barely beats generic + LoRA" was
  a headline finding.

---

## 5. Decisions this work will put on you

For each: the options, our default, and what would change the default. Write your
choice and the reason into the experiment doc.

| # | Decision | Options | Default and why |
|---|---|---|---|
| D1 | Diacritics split design (area 5) | (a) separate restorer model, (b) same VLM, two-step prompt | (b) first: cheapest, and shows whether separation alone helps. Switch to (a) if (b) shows no gain. |
| D2 | Metric definition for diacritics | letters only / marks only / joint | Report all three; headline is letters-only CER for the reader stage and mark-F1 for the restorer stage. |
| D3 | Script-family training mix (area 3) | Muharaf only, Muharaf + RASAM, family-balanced | Family-balanced, with **one family held out** so transfer claims stay honest. |
| D4 | Encoder handling (area 3, then 1) | frozen, unfrozen at low LR, continued pretraining | Unfrozen with a differential LR first (evidence from Baseer-Nakba); pretraining only in P4. |
| D5 | Small reader architecture (area 4) | TrOCR-style encoder-decoder, CTC, both | Both, on identical data, compared on CER **and** hallucination rate (chars in prediction with no support in the image) and ms/line. |
| D6 | Synthetic handwriting approach (area 2) | font-based scripts, style-transfer/GAN, diffusion | Start with the cheapest that runs on Colab; decide by the scaling curve, not by novelty. |
| D7 | Encoder objective (area 1) | masked image modelling, dot/ligature auxiliary heads, contrastive line↔text | Decide after P3; whichever tests the claim with the least compute. |
| D8 | Scale-up rule | when to spend more | Only after a probe beats its baseline by the pre-registered threshold at n large enough to trust. |
| D9 | Stop rule | when to stop | The kill criterion from 3.1, written before the run. |

---

## 6. Compute plan (3090 + Colab subscription)

Decision: **3090 for P1–P3; decide P4 after P3's results.** We have a Colab
subscription and will not use the A100 (cost).

What each machine is for:

| Resource | Memory | Use it for | Avoid it for |
|---|---|---|---|
| RTX 3090 (Windows box) | 24 GB, bf16 | Anything long-running or sequential; the fine-tunes and evals in P1–P2; the reference runs | Jobs that need to run in parallel with another |
| Colab **L4** | 24 GB | Same memory class as the 3090 but slower; a second lane for parallel runs, synthetic data generation, small-encoder experiments | The longest single runs |
| Colab **T4** | 16 GB, no bf16 | QLoRA on ≤3B models, TrOCR-scale models (area 4), evaluation, data prep | Full bf16 fine-tunes |
| Colab A100 | 40 GB | Excluded for cost | — |

Check current compute-unit rates in the Colab UI before planning: the rate per GPU
type changes, so budget in *units per hour* and write the estimate in the experiment
doc.

Colab rules (from the repo): notebooks follow `docs/training-plan.md` — checkpoint
every epoch, back up to Drive and a private HF repo, auto-resume, and put
`<repo>/src` on `sys.path`. Sessions disconnect, so a plan that assumes an
uninterrupted 12 h run is a bad plan.

**P4 gate.** After P3, decide whether encoder pretraining fits the L4/T4 budget as
continued pretraining of a small encoder, or needs a costed, reviewed rented-GPU run.
The P3 scaling curve (how much labelled handwriting we have and how much it helps)
is the input to that decision.

---

## 7. Learning path (do these in order)

1. **Read** `findings-and-notes.md` and `misraj-baseer-research.md`. Write a
   half-page summary of what we know and what we don't.
2. **Reproduce** one existing number (Qwen3-VL-4B fine-tune on Muharaf at reduced
   size) using `docs/experiments/3090-experiment-runbook.md`. Note every place the
   runbook was unclear, and fix it.
3. **Score** existing predictions with `metrics.py` by hand for 20 lines, and explain
   each CER variant's value on them.
4. **Paired analysis:** reproduce the Baseer-Nakba per-sample diagnosis (3.3).
5. **Plan** P1 experiments using the five-field template (3.1) and get the plan
   reviewed *before* running anything.
6. **Run P1**, write up in `docs/experiments/` with `_TEMPLATE.md`, and update
   `findings-and-notes.md`.
7. Repeat 5–6 for each phase.

### Review checklist for any experiment write-up

- [ ] Hypothesis, single variable, baseline, threshold, kill criterion stated up front
- [ ] n shown next to every number; small-n rows flagged
- [ ] No-leak guard passed; contamination report cited
- [ ] Paired per-sample analysis, plus a read of the worst predictions
- [ ] All CER variants and diacritic F1 reported
- [ ] Cost and wall-clock recorded (GPU-hours, Colab units)
- [ ] Runbook updated if anything was unclear
- [ ] Next experiment named
