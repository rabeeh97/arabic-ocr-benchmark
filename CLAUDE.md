# arabic_ocr — session entry point

Arabic OCR/HTR benchmark + fine-tuning harness (`src/arabic_ocr_bench/`, src
layout; run with `PYTHONPATH=src .venv/bin/python …`), now in an **academic
research phase** focused on characterising Arabic script, text and datasets.

Before research work, read:
- `docs/research/GUIDELINES.md` — direction, rules, conventions, what we know so far, standing working rules
- `docs/research/NEXT_STEPS.md` — the prioritised backlog ("Now" section first)

Keep both current: tick/add tasks in `NEXT_STEPS.md`, mirror status in
`docs/research/progress.json`, rebuild the Atlas (`scripts/build_atlas.py`)
and republish it to its existing URL (in GUIDELINES §4).

Other key docs: `docs/experiments/findings-and-notes.md` (results & caveats),
`docs/plan.md` (original benchmark plan), `docs/experiments/*-runbook.md`
(3090 / suite_v1 / char-CTC runs).
