# Tadqiq (تدقيق): Arabic dataset error finder (working name)

A toolkit, benchmark and paper (and later a platform) for **finding likely errors in Arabic datasets automatically**. It fixes an error only when the fix is provably safe, and routes everything else to people.

The idea grew out of the RASAM clean-up (RS4i–RS4m). Its first 17 detectors are built and validated in
`src/arabic_ocr_bench/dataqa.py` (see `docs/research/dataqa.md`).

## Read in this order (agents start here)

1. [01-assessment.md](01-assessment.md): is the idea valid, what is new, honest caveats, the verdict.
2. [02-error-taxonomy.md](02-error-taxonomy.md): the error types (IDs C/O/I/J/R/P/D/M/S/N/X), with status and fix class.
3. [03-toolkit-spec.md](03-toolkit-spec.md): architecture, data model, detector API, evidence fusion, fix policy, evaluation harness, CLI.
4. [04-paper-plan.md](04-paper-plan.md): titles, contributions, experiments E1–E7, figures, venues, threats.
5. [05-roadmap.md](05-roadmap.md): phased tasks T0–T5, each with a "done when".
6. [06-open-questions.md](06-open-questions.md): the user's decisions. Phase 1 waits on T0.1; defaults are marked ★.

## Status (2026-11-02)

- Detectors v0: done.
- Taxonomy, spec, paper plan and roadmap: drafted.
- Next: the user answers the open questions, then T1.1 (port to the `tadqiq` package).
- Website mock-up: a private claude.ai artifact, linked from the session report. It is an illustration only, with no backend.

## Conventions

- **Never mutate data.** Flags go in `flags.jsonl`, patches in `patches.jsonl`. "Apply" writes a new version with a changelog.
- **Reporting.** Each detector reports measured precision. Do not claim "100 % sure" except for A0 fixes.
- **Licences.** Check each dataset's licence before showing its images publicly. Flags keyed by line ID are fine.
- **Product planning** is kept outside this public repo, on purpose. Ask the user for access if a task needs it.
