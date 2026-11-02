# 05 · Roadmap (agent-ready, v0.1, 2026-11-02)

Each task has an ID, inputs, what to do and **done when**. Work top to bottom inside a phase. Phases 1–3 serve the paper; phase 4+ serve the platform. Rules from `docs/research/GUIDELINES.md` apply:
- process data with scripts and read short summaries;
- one heavy job at a time on the Mac;
- downloads go to the SSD;
- show best, worst and random samples for any result that moves a metric.

## Phase 0: decisions (user)

- **T0.1** Answer [06-open-questions.md](06-open-questions.md): the name, licence, venue and v1 scope. **Done when** the answers are written into that file.

## Phase 1: package (2–3 weeks)

- **T1.1** Create the `tadqiq/` package. It can live in this repo under `src/tadqiq/` first, or in a new repo after T0.1, following the layout in [03-toolkit-spec.md §2](03-toolkit-spec.md).
  - Port `dataqa.py` without changing behaviour.
  - **Done when** the existing 16 tests pass against the new package, and `tadqiq scan` reproduces `results/dataqa/*.json` flag counts exactly.
- **T1.2** Implement the data model and formats: Sample, Flag, Patch; `flags.jsonl`, `patches.jsonl`, `card.json`. **Done when** the schemas are documented and round-trip tests pass.
- **T1.3** Write the IO adapters: hf_parquet (exists), folder (img + txt), PAGE-XML, ALTO, CSV/JSONL. **Done when** each adapter has a tiny fixture test, and RASAM's PAGE-XML gives line geometry.
- **T1.4** Write the C-family detectors (C01–C14) and the A0 fixers. **Done when**:
  - unit tests exist for each;
  - all local datasets have been scanned;
  - the flag rates are added to the summary.
- **T1.5** Build the O-family lexicon (OpenITI + train GT, folding, light clitics) and the detectors O05–O07. **Done when**:
  - precision is estimated on 50 flags each on RASAM test and Muharaf test (by-eye sheet);
  - the result is written to `docs/research/dataqa.md`.
- **T1.6** Move the garbled check to a pluggable char-LM detector (O08). **Done when** the RASAM garbled list is reproduced with the generic API.
- **T1.7** Write the J-family detectors J02 (PAW count) and J03 (dot count). The connected-component analysis runs on the deskewed ink. **Done when** they are validated on 100 lines against hand counts.

## Phase 2: benchmark (2–3 weeks)

- **T2.1** Build `eval/corrupt.py` with all the corruptions in spec §7. Each corruption is seeded, and the labels go to `corrupted/labels.jsonl`. **Done when** there are visual spot checks of 5 examples per type.
- **T2.2** Build TadqiqBench-syn: 4 datasets (RASAM-1, RASAM-2, Muharaf, NAKBA), on the cleaned lines only, at 1/5/10 %. **Done when** it is frozen, with a hash in the README.
- **T2.3** Build TadqiqBench-real v0. Collect the verified RASAM lines (by-eye, clean lists, PR #7). Verify 50 flags per check per dataset with a simple local sheet (HTML gallery plus keyboard labels, writing JSON). **Done when** there are ≥ 800 labelled flags and a second rater on 20 % of them (kappa reported).
- **T2.4** Write `tadqiq bench`, with per-type P/R/F1, CIs and cost. **Done when** the E2 table is produced.

## Phase 3: experiments and paper (3–4 weeks)

- **T3.1** E3, reader ensembles. Add Kraken/TariMa and HATFormer readings, plus one VLM on a sample of about 500 lines (cache-backed; review the cost first). Produce the scatter plot and AUROC. **Done when** the figure is made and the ablation table is filled.
- **T3.2** E4, A1 fix precision. **Done when** the table has CIs.
- **T3.3** E5, add one more dataset to the clean-vs-official benchmark (Muharaf or KHATT). **Done when** there are CIs and the ranking change is reported.
- **T3.4** E1, the audit of about 10 datasets. **Done when** the heat map and the per-dataset findings table are made, and issue drafts are written for the maintainers. The **user** sends the issues.
- **T3.5** Draft the paper in `docs/tadqiq/paper/`, using figures from scripts. **Done when** there is a complete draft with every number traceable to a results JSON.
- **T3.6** Release the package on PyPI (after T0.1), plus `tadqiq-bench` on HF. **Done when** a fresh pip install runs the quickstart.

## Phase 4: platform MVP (private repo; see the private product spec)

- **T4.1** Build a static report-card site from `card.json`: dataset page and flag explorer. No backend, so it can be hosted anywhere. This is the first public demo.
- **T4.2** Build the backend: upload and HF import, a job queue running the `tadqiq scan` worker, and storage of flags and patches.
- **T4.3** Build the verification UI: a one-line card with keyboard shortcuts, gold items and rater reliability (Dawid–Skene).
- **T4.4** Build the arena mode: two candidate transcriptions (GT vs the readers' consensus, or two patches), plus "both wrong" or "can't tell".
- **T4.5** Build new-type reports: free text plus an example. The reports are clustered, and when a cluster reaches about 20 confirmed examples it is promoted to a taxonomy entry with a test set.
- **T4.6** Build the detector leaderboard: plugins are submitted and scored on the hidden TadqiqBench-real split.
- **T4.7** Build patch export as a new dataset version (parquet), an HF PR or a downloadable diff.

## Phase 5: beyond line OCR (later)

- Page layout (P family, extended).
- ASR transcripts (S family).
- NLP label noise (N family).
- Persian, Urdu, Ottoman and Ajami.
- Vendor acceptance-testing mode (ISO 2859 sampling plans).

## Status board

Mirror this in `docs/research/NEXT_STEPS.md` (RS5 series) and `progress.json`.

| task | status |
|---|---|
| detectors v0 (17 checks, `dataqa.py`) | done (RS4m) |
| taxonomy v0.1, spec, paper plan | done (this folder) |
| T0.1 | waiting on the user |
| T1.x – T3.x | not started |
