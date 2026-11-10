# Disk space: why worktrees were huge, and what was done (2026-11-10)

## Root cause
The SSD (`/Volumes/PortableSSD`) is **exFAT with a 1 MiB allocation block**
(`diskutil info`: Allocation Block Size 1048576). Every file occupies a whole
number of MiB, so a 4 KB file costs 1 MiB. Consequences measured on 2026-11-10:

| Item | Real data | On disk |
|---|---|---|
| `results/_cache/` (10,289 JSON files, committed) | 19 MB | ~10 GB, **per checkout** |
| AppleDouble `._*` files macOS writes on exFAT (21,212 of them) | ~0 | ~21 GB |
| Each small worktree | mostly the cache above | 10-13 GB |

The disk was at 100 % (0 B free; git failed with "Out of diskspace").

## What was done
- Deleted all `._*` files and `__pycache__`/`.pytest_cache`/`.ruff_cache` (~21 GB).
  They come back whenever macOS writes to the volume: re-run
  `find . -name '._*' -type f -not -path './.git/*' -delete` when space gets tight.
- Removed worktrees `fix-sft-max-steps`, `gemma4-refill`, `misraj-baseer-research`
  (clean) and `hindawi-fix` (committed first; findings in
  `hindawi-fix-findings.md`). Branches `worktree-*` still exist with all commits.
- Replaced the per-file cache with one append-only `results/_cache.jsonl`
  (`src/arabic_ocr_bench/cache.py`, `.gitattributes` `merge=union`,
  `scripts/migrate_cache_to_jsonl.py`; legacy `_cache/<key>.json` is still read as
  a fallback, so old Colab notebooks keep working). 19 MB instead of ~10 GB.
- Free space went 0 -> ~68 GB.

## How to make a cheap worktree (until all history is slimmed)
```
git worktree add --no-checkout -b <branch> .claude/worktrees/<name> <base>
cd .claude/worktrees/<name>
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/results/20*/'
git checkout
```
(~360 MB instead of 10-13 GB.) With the JSONL cache the `_cache` exclusion is no
longer needed for new branches.

## Deferred (NEXT_STEPS X8, X9)
- `xai-attribution-spike` worktree is still 87 GB: `.geomvenv` 32 GB (in use by the
  gold-label app), `tadqiq-platform/app/workspace` 12 GB (gitignored, only copy),
  `results` 23 GB. Needs a decision, see X8.
- Run dirs cost 6 MiB each (6 files x 1 MiB). See X9.
