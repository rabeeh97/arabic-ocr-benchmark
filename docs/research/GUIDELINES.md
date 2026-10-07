# Research guidelines & standing context

> Read this at the start of any session on the research layer. It keeps the
> direction, rules, conventions and results so far in one place, so they don't
> depend on any one conversation. The backlog is in [`NEXT_STEPS.md`](NEXT_STEPS.md).
> Last updated 2026-10-07.

## 1. Direction (decided 2026-10-06/07)

The project moved from benchmark-and-fine-tune work into an **academic phase**.
The order:

1. **Literature first.** Six topics:
   - annotation and input schema;
   - diacritics;
   - architectures;
   - Arabic datasets;
   - datasets from other languages and what transfers;
   - HTR vs print vs old fonts vs book layouts.
2. **Understanding the text and data.** This means a deep characterisation of
   the letters, words and language: the letter's *locus* (positional form),
   dots, PAWs and so on, done the way English script and HTR were studied.
   This is the foundation layer, Wave 0.
3. **Controlled experiments**: frozen vs unfrozen ViT, tokenizer. The 3090 is
   the default; Colab is a secondary option, so don't build infrastructure
   around it.
4. **Stretch:** our own dataset, collected from schools and designed from what
   1–3 teach us.

## 2. The core rule for characterisation

From [`characteristics-framework.md`](characteristics-framework.md):

- **A characteristic earns its place only if it can change a decision:**
  - **D1** data: select, mix, augment or create;
  - **D2** model: base, output units, resolution, what to fine-tune;
  - **D3** transformations: normalisation and pre-/post-processing;
  - **D4** evaluation: metrics, splits, exclusions.
- **It is a hypothesis until it passes the gates:**
  - **V1** reliable;
  - **V2** predictive, beyond line length, and still holding when a dataset
    or model family is left out;
  - **V3** causal: a single-factor intervention reproduces it;
  - **V4** generalises to Persian, Urdu, Pashto and Latin;
  - **V5** useful: acting on it changes an outcome;
  - **V6** adoptable.
- **Always pair an Arabic property with its English/Latin analogue** and the
  classic study of it.
- Lineage we deliberately copy:
  - ISRI page quality groups (cross-engine median accuracy);
  - DIQA (predicting difficulty without ground truth);
  - IAM/KHATT datasheets;
  - benchmark-hygiene audits;
  - dataset cartography / IRT;
  - Chinese radicals and Urdu primary/secondary components, which map to
    rasm + i'jam + harakat;
  - Perso-Arabic graphemic normalisation.
- **Our Arabic-specific measures:**
  - the normalisation ladder;
  - rasm-CER and the dot tax;
  - the shape-coverage curve;
  - the diacritic-mismatch tax;
  - the PAW-boundary space error rate;
  - the fluency-hallucination rate.

## 3. Writing conventions for `docs/research/`

- **Topic notes** follow [`_TEMPLATE.md`](_TEMPLATE.md): Question → Key papers
  table → Synthesis → **Implications for us** → Open questions. Every
  implication is tagged:
  - **[hypothesis]**, which goes to an experiment config;
  - **[metric]**, which goes to `metrics.py` or `normalize.py`;
  - **[dataset]**, which goes to `dataset-design.md`.
- **Citations** are BibTeX keys in [`references.bib`](references.bib) (key:
  firstauthorYEARword). Mark anything not checked against the primary source
  **(unverified)**, and put the caveat in the bib `note`.
- **Findings** that affect results also go into
  `../experiments/findings-and-notes.md` (Seen → Meaning → To do).
- **New ideas** go into `NEXT_STEPS.md` as tasks with a **done when**, never only
  into a chat.

## 4. Where things live

| what | where |
|---|---|
| Backlog | `docs/research/NEXT_STEPS.md` |
| Tracker shown on the Atlas | `docs/research/progress.json` (phases, tasks, `updates` feed, `catalogue`) |
| Live page | Arabic Script Atlas, https://claude.ai/artifact/8Z6dsabo5FDFPW11XQn7yf (private) |
| Literature | `script-characteristics.md`, `characteristics-framework.md`, future topic files; `references.bib` |
| Characterisation plan | `characterization-plan.md` (layers L1–L7, revised phases P0–P11) |
| Own-dataset spec | `dataset-design.md` |
| Text analyzer | `src/arabic_ocr_bench/characterize/text.py` (positional forms, rasm, PAWs, diacritics, hygiene), tests in `tests/test_characterize_text.py` |
| Canonical predictions + error decomposition | `characterize/runs.py`, `characterize/errors.py`, `scripts/decompose_errors.py` → `results/characterize/canonical_runs.json`, `results/characterize/<stamp>/errors.json` |
| Per-sample characteristics | `characterize/samples.py`, `scripts/sample_characteristics.py` → `results/characterize/<stamp>/samples.json` |
| Profiler | `scripts/profile_datasets.py` → `results/characterize/<stamp>/{summary.json,lines.jsonl}` |
| Atlas builder | `scripts/build_atlas.py` + `docs/research/atlas/template.html` → `docs/research/atlas/index.html` |
| Experiment findings | `docs/experiments/findings-and-notes.md` |
| Misraj/Baseer research | `docs/experiments/misraj-baseer-research.md` |
| suite_v1 results (not in git) | `suite-v1-artifacts/<model>/{bench,eval}/` in the main checkout |

## 5. What we know so far (keep this short; details in the linked notes)

- **Arabic−Latin HTR gap.** About +26 CER at 100 training lines and +5.8 at
  full data, with the same model. The main reasons are 82 vs 50 letter shapes
  in a heavy-tailed distribution, and dot confusions: about 30 % of
  substitutions in Arabic vs 15 % in Latin (`alazzawi2026gap`).
- **Label noise.** Muharaf, our training set, has 6–9 % label errors
  (`alazzawi2026labelerr`). The flag logs are public (CER-HV, no licence).
  Removing flagged test lines moves CER by < 1 point and changes no rank.
- **Leakage.** **`kitab_muharaf` is 86.5 % in Muharaf train**, so fine-tuned
  scores on it are leaked. The other sets are at ≤ 6 %.
- **Diacritic policy differs.** About a third of Muharaf lines carry sparse
  marks; NAKBA, KHATT and RASAM are nearly undiacritised.
- **Dots and shapes.** About 50 % of letters in every dataset are
  dot-ambiguous, and the dots carry about 0.6–0.8 bits per letter. Each
  dataset has 112–131 distinct shapes, of which 63–85 cover 95 % of letters.
- **Where model errors come from** (P2, competent pairs). Dots alone are about
  8 % of diacritic-insensitive error. 60 % of wrongly inserted word-internal
  spaces fall at PAW gaps, against 35 % by chance. Base VLMs pay a large
  diacritic-policy tax on printed sets (up to 10 points of CER). Most error is
  plain wrong, missing or extra letters.
- **What makes a line hard** (R3, complete blocks). Difficulty is reliable
  across models (split-half up to 0.81), but text features explain only
  6–32 % of it. The image layer is the gap. Medial forms and dot-ambiguous
  letters fail most; tall strokes (ا ل) least. Errors rise at line ends.
  Short lines carry the whole-line failures. Ensembles gain ≤ 3.6 points.
- **Fine-tuning.** A Qwen3-VL-4B LoRA on 8k Muharaf lines transfers across
  handwriting (KHATT/NAKBA ~30 % CER). Maghrebi stays weak (~50 %).
  Baseer-Nakba gets worse on its home domain with the generic recipe. Unfreezing the
  ViT with a differential LR cut in-domain CER by 7.6 points (one seed). The
  gain is in letter identity, not dots; the held-out bench is pending (E1.1).
- **Children's handwriting.** Individuality emerges around grades 3–4
  (`srihari2016children`). No study links writer demographics to HTR error,
  which is an open gap our dataset can address.
- **No common characterisation standard** for Arabic OCR data exists that we
  could find. Filling that gap is a publishable contribution.

## 6. How to keep the live view current

After any research step:

1. Update `NEXT_STEPS.md`: tick the task and add new ones.
2. Update `progress.json`: task status, plus one `updates` entry (kind:
   `finding` or `done`).
3. Rebuild:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/profile_datasets.py --muharaf-train   # only if data or the analyzer changed (~4 min, CPU)
   PYTHONPATH=src .venv/bin/python scripts/decompose_errors.py                  # only if predictions or errors.py changed (~90 s)
   PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python scripts/sample_characteristics.py  # if predictions changed (~80 s)
   PYTHONPATH=src .venv/bin/python scripts/build_atlas.py
   ```
4. Republish `docs/research/atlas/index.html` to the **same** Atlas URL.

## 7. Standing working rules (from earlier sessions; still in force)

**Compute and runs**
- **3090 experiments:** one shared venv, no duplicate model or dataset
  downloads, minimal disk. Multi-model runners pass shared `HF_HOME` /
  `UV_CACHE_DIR` paths.
- **Runbook reproducibility:** when a step fails on the real box, fix the
  blocker *and* update the runbook and launcher, then commit, so `git archive
  HEAD` reproduces it.
- **Detached commands:** whenever you give one, also give the exact command
  to watch it (log tail and `status`).
- **Checkpoints:**
  - training must resume losing at most a few samples, so use a tight
    `save_steps` for LoRA;
  - on Colab, checkpoint every epoch or every N steps to Drive, plus an HF
    mirror;
  - Run-all must resume;
  - never commit weights.
- **Colab notebooks** put both the repo root and `<repo>/src` on `sys.path`;
  don't rely on `pip install -e .` succeeding.

**Model APIs**
- **Paid APIs:** always cache-backed (`results/_cache/`). Lay out the model
  list, sample set and cost estimate for approval before spending, after a
  tiny smoke and probe run.
- **Rate-limited free models:** skip a free model once it is clearly
  quota-exhausted. For a stalled sweep, probe the real 429 body before
  assuming why.

**Git**
- Commit or push only when asked. Branch first if on `main`.
