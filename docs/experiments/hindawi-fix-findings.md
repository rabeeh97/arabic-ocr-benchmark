# hindawi-fix worktree: findings preserved before removal (2026-11-10)

Everything is committed on branch `worktree-hindawi-fix` (commit `4793aa0c`); the
worktree was removed to free ~13 GB (see `disk-space-and-worktrees.md`).
Restore with `git worktree add .claude/worktrees/hindawi-fix worktree-hindawi-fix`.

## What the work found (2026-09-13 page-v2 rollout on kitab_hindawi / kitab_historicalbooks)
- The KITAB sets have paragraph/page-scale ground truth; they had been scored with
  the single-line `line-v2` prompt at ~256-token caps, so predictions were
  truncated (median prediction length 10-55 % of GT) and CER looked like a model
  result when it was a granularity/cap artefact.
- `max_tokens_override` (run-level) lifts the cap for a run without editing registry
  rows; raised to 900 and made raise-only (never lowers a higher cap).
- Even 640 tokens was not enough for `gemini-flash`: Gemini 3.x adaptive thinking
  consumed ~636/640 tokens before the visible answer (40-90 visible chars vs ~970
  GT chars). New `gemini-flash-page` / `gemini-flash-lite-page` rows set
  `reasoning_effort: none` (new names, so a clean cache miss).
- Added free hosted models to the rollout: nvidia-gemma-4-31b,
  nvidia-llama-3.2-11b-vl/90b-vl, openrouter-dots3-note-free,
  openrouter-minimax-m3-free. Paid OpenRouter routes stay out ($0 credit then).
- The run got OOM-killed twice on the laptop; `concurrency` lowered 3 -> 1.
- `report.py` now records per-sample `len_ratio` and a per-row `len_ratio_median`,
  and prints a WARNING when >= 60 % of models on a dataset emit under half the GT
  length, so this failure cannot read as "models are bad at this text" again.
- `configs/run.page-v2-paid.yaml`: paid counterpart (14 models, est. ~$2 at
  limit 10 per dataset). Needs cost sign-off before running (paid-api rule).

## Where the data is
Run outputs `results/20260913T*` and the resume cache entries are in that commit.
`results/findings.json` was also regenerated there (+45k lines).
