# How the multi-Colab-account sweep collaborates

Notebooks **`04_colab_sweep.ipynb`** and **`04b_colab_sweep_local_capped20.ipynb`**
are built so that **2–4 Google Colab accounts can run the exact same notebook at
the same time** and split the work between them without ever transcribing the
same line twice — with no coordination server, only GitHub and (optionally)
Google Drive.

This document describes what actually happens: the pieces, the lifecycle of one
unit of work, and precisely what each runner writes to Drive and to the GitHub
repo.

---

## 1. The moving parts

| Store | Where | Holds | Written by |
| --- | --- | --- | --- |
| **`main` branch** (GitHub) | the notebook's normal checkout | the benchmark **results** — `results/<run_id>/predictions.jsonl` + `scores.jsonl`, the per-line cache `results/_cache/`, `results/leaderboard.html/.json` | every runner, one commit per finished batch |
| **`sweep-claims` branch** (GitHub) | a second, throwaway shallow clone at `/content/.sweep-claims-repo` | the **lease ledger** — one tiny JSON file per claimed batch, under `claims/` | every runner, on every claim / complete / release |
| **Google Drive** | `MyDrive/arabic-ocr-bench/` on whichever account is running | a **mirror** of `results/<run_id>/` and `_cache/`, plus the latest leaderboard | every runner, after each batch (best-effort) |

Key point: **the ledger is on a git branch, not on Drive.** Two Colab sessions
signed in to *different* Google accounts do **not** share a `MyDrive` tree, and
even a folder shared between accounts syncs across accounts far too slowly (tens
of seconds to minutes) for leasing to be correct. Git is the one store every
runner sees consistently, and **GitHub serialises pushes to a single ref**, which
is exactly the mutex the scheme needs.

Drive is a *convenience only*: it warm-starts the line cache when a fresh VM
starts, and keeps a copy of results outside GitHub. Nothing about correctness
depends on it. If Drive is not mounted the notebook still refuses to start only
because the cache mirror is genuinely useful — the coordination itself would work
without it.

---

## 2. Runner identity

On startup each runner computes `(runner_id, session_id)` via
`arabic_ocr_bench.coordination.runner_identity()`:

- **`runner_id`** — *stable* across VM restarts of the same account. Resolved in
  order: the `SWEEP_RUNNER_ID` env var → the signed-in Colab **Google-account
  email** (obtained from the credentials the notebook already used to mount
  Drive) → a UUID persisted at `/content/.sweep_runner_id`. Sanitised to
  `[A-Za-z0-9._+@-]`, capped at 80 chars.
- **`session_id`** — a fresh 8-hex string on *every* run of the cell. Two
  sessions of the same account stay distinguishable in the audit, and it is the
  final tie-breaker when two runners race for the same batch.

Every prediction row written by that runner is stamped with `runner` and
`session`; every lease file it writes carries `runner_id` and `session_id`; and
its result commits are authored as `colab:<runner_id>`. So after the fact you can
always say which account produced which line.

*(Results produced before this system existed were retroactively stamped
`runner: "legacy"` by `scripts/backfill_runner_id.py`, so the audit never
confuses old work with new.)*

---

## 3. A unit of work

A **unit** is `(model, dataset, [i0, i1))` — a contiguous block of absolute
dataset row indices for one model on one dataset. Size is `BATCH` (default 10).
`04b` additionally caps each `(model, dataset)` pair at `TARGET` (default 20)
total scored lines.

Picking the next unit (`_free_indices`):

```
done  = scored_indices(results/, model, dataset)      # rows already scored locally
held  = LEDGER.covered_indices(model, dataset)        # rows any ACTIVE lease covers
want  = next_indices(done | held, BATCH)              # next BATCH rows in neither set
want  = [i for i in want if i < dataset_size]         # clamp to the split
```

`covered_indices` is read from a fresh fetch of `sweep-claims` and counts a lease
as *active* if it is `done`, **or** `reserved` and not older than the TTL. So a
runner will not pick rows that another runner has already **finished** (its
`done` lease still sits on the branch) *or* is **currently working on**.

---

## 4. The lifecycle of one batch

```
        ┌──────────────────────────── one runner, one loop turn ────────────────────────────┐

 pick   _free_indices(model, dataset)  ->  want = [i0 .. i1)          (or nothing -> pair exhausted)

 claim  LEDGER.claim(model, dataset, want, runner_id, session_id):
          • git fetch + reset to origin/sweep-claims
          • pre-check: does an active reserved lease overlap `want` that I don't win? -> give up, pick elsewhere
          • write claims/<model>__<dataset>__<i0>-<i1>__<session>.json  (status="reserved")
          • git add + commit + push origin HEAD:sweep-claims
                – push rejected  ->  fetch, rebase, retry (up to 6x); rebase conflict -> abort, return None
          • fetch again; re-check overlap; if I now lose the tie-break -> delete my file, return None

 confirm LEDGER.confirm(lease):   fetch once more, final overlap/tie-break check   -> False means "lost the race"

 run    runner.run({models:[model], datasets:[dataset], indices:want,
                     accumulate:true, cache:true, runner_id, session_id, ...})
          • for each row: consult results/_cache/<sha1(sample|model|prompt|rev)>.json first
          • only on a miss does it actually call the model
          • writes results/<run_id>/predictions.jsonl  (rows stamped runner + session)

 persist _persist(model, dataset, run_dir, n):
          • score_run  ->  scores.jsonl
          • export_run_samples  ->  samples.json (+ 3 sample images for the leaderboard)
          • rebuild results/leaderboard.html / .json
          • copy run_dir + _cache + leaderboard  ->  Drive        (best-effort)
          • git add -f results/<run_id> results/_cache leaderboard.*
          • git commit  "sweep[<runner_id>]: <model> <dataset> n=<n> (<run_id>)"
          • git pull --rebase --autostash origin main
          • git push origin HEAD:main        — RAISES if the push fails

 done   LEDGER.complete(lease, run_id, n):
          • flip the lease file to status="done", record run_id + n_scored
          • git commit + push origin HEAD:sweep-claims

        └──────────────────────────────────────────────────────────────────────────────────┘
              then: re-read everything from the top and grab the next unit
```

If `run` or `persist` throws, `_do` returns `False` and **the lease is left
untouched** — still `reserved`, still on the branch. It will expire after the TTL
and any runner (including this one, on its next start) can reclaim it. A `done`
lease therefore always means *the results are really in `main`*.

---

## 5. What each runner does to **Google Drive**

Per batch, best-effort (`_mirror_to_drive`), into `MyDrive/arabic-ocr-bench/`:

- `results/<run_id>/` — a full copy of this batch's run directory
  (`predictions.jsonl`, `scores.jsonl`, `samples.json`, config snapshot).
- `_cache/` — the growing per-line result cache, merged in
  (`shutil.copytree(..., dirs_exist_ok=True)`).
- `results/leaderboard.html` and `results/leaderboard.json` — overwritten with
  the latest rebuild.

On **startup** the notebook copies `MyDrive/arabic-ocr-bench/_cache/` back into
the local `results/_cache/` ("restored line cache from Drive"), so a brand-new VM
doesn't re-run lines that were already done in a previous session.

Drive is written by whichever account is running — each account writes to *its
own* `MyDrive`. It is a redundant safety copy and a cache warm-start. Two runners
never read each other's Drive and never coordinate through it. If you want a
single shared Drive copy, put a shortcut to one owner's shared
`arabic-ocr-bench` folder in each account's *My Drive* — but this is optional and
does not change how the work is divided.

---

## 6. What each runner does to the **GitHub repo**

### `main` branch — the results

One commit **per finished batch, per runner**:

```
sweep[omar@example.com]: qwen2.5-vl-3b nakba_lines n=10 (20260906T101502Z)
```

Each such commit adds/updates:

- `results/<run_id>/` (that runner's own run directories — **never** another
  runner's, so they never collide on these paths),
- `results/_cache/*.json` (new cache entries — additive; two runners adding
  different files rebase cleanly),
- `results/leaderboard.html` / `.json` (rebuilt from *all* results each time —
  this is the file most likely to cause a push race, see the footnote).

Before every push the runner does `git pull --rebase --autostash origin main`, so
it always builds on the newest results. A rejected push is retried after another
rebase. Because each runner owns a disjoint set of `results/<run_id>/` paths, the
only real merge surface is `_cache/` (additive) and the leaderboard files
(regenerated, so "theirs vs mine" is resolved by just regenerating again on the
next batch).

Commits are authored `colab:<runner_id>` / `colab@arabic-ocr-bench`.

### `sweep-claims` branch — the ledger

A separate shallow clone (`/content/.sweep-claims-repo`) pushes here. Traffic is
much higher than on `main` — a commit for **every claim, every completion, every
release, every reclaim** — but each commit is one ~400-byte JSON file:

```
claims/qwen2.5-vl-3b__nakba_lines__000010-000020__b88fe182.json
```

```jsonc
{
  "model": "qwen2.5-vl-3b", "dataset": "nakba_lines",
  "i0": 10, "i1": 20, "indices": [10, 11, ..., 19],
  "runner_id": "omar@example.com", "session_id": "b88fe182",
  "status": "reserved",              // -> "done" when the results are pushed
  "claimed_at": 1789000000.0, "updated_at": 1789000123.0,
  "run_id": "", "n_scored": 0        // filled in on completion
}
```

The branch is **created automatically** by the first runner (as an orphan branch
containing only `claims/`). It is disposable: delete it between big sweeps and it
will be recreated. `done` leases older than 7 days are pruned automatically at the
end of a run (`LEDGER.prune_done`).

### How simultaneous claims are resolved

1. **GitHub push serialisation.** Two runners cannot both fast-forward
   `sweep-claims` from the same parent. The loser's push is rejected; it fetches,
   rebases its one-file commit on top, and re-checks.
2. **Deterministic tie-break.** After rebasing, if the loser's lease *overlaps*
   an existing `reserved` lease and does **not** win
   `wins_over` — `(claimed_at, session_id)` compared as a tuple, earliest wins —
   it deletes its own lease file and picks a different unit.
3. **Idempotency backstop.** Even if a race slips through, `runner.run` consults
   `results/_cache/` before calling any model, so the same line is never
   *recomputed*; it would at worst produce a duplicate prediction row, which
   `scripts/sweep_audit.py` flags.

---

## 7. Crash / disconnect recovery

- **Same runner, restarted.** On startup a runner first runs
  `LEDGER.my_open(runner_id)` — its own still-`reserved` leases, i.e. batches it
  was interrupted mid-run before persisting — and re-does them before taking
  anything new.
- **A different runner died.** `LEDGER.reclaimable()` returns any `reserved`
  lease older than `LEASE_TTL_MIN` (default 30). The live runner `adopt`s it
  (deletes the stale file, writes a fresh one under its own identity), `confirm`s
  it, and runs it.
- **Nothing is ever lost**, because a batch only becomes `done` *after* its
  results are pushed to `main`. A half-finished batch's partial rows sit in an
  un-`done` `results/<run_id>/` and its cache entries are still valid, so the
  redo mostly hits cache.

---

## 8. Auditing a sweep

```
python scripts/sweep_audit.py results --claims /content/.sweep-claims-repo/claims
```

Reports, from **every** `results/*/predictions.jsonl` (nothing is overwritten):

- coverage per `(model, dataset)` — distinct rows with a real (non-cached,
  non-errored) result,
- lines contributed **per runner**,
- **duplicates** — any row really computed by more than one runner/run,
- stale leases still sitting in the claims directory.

Exit status is non-zero if any duplicate is found.

---

## 9. Setup checklist for a runner

1. Same repo, `Runtime → T4 GPU`.
2. Colab secret **`GH_TOKEN`** — a fine-grained PAT with **Contents: read *and
   write*** on the repo (needed to push results *and* claims). The *same* token
   on every account is fine; attribution comes from `runner_id`, not the token.
3. Colab secret **`HF_TOKEN`** — read scope; also click *Agree* once on any gated
   dataset page (e.g. the NAKBA set) with that account.
4. `Runtime → Run all`. Repeat on each account. Order doesn't matter; the first
   one to start creates `sweep-claims`.
5. Stop any time. A disconnected runner's batch is reclaimed after the TTL.

`LEASE_TTL_MIN` must be comfortably longer than one batch takes — a 7B/8B model
in 4-bit doing 10 lines on a T4 can be 10–15 min, so 30 is fine; bump to 45 if
you add slow models. `BATCH` trades commit volume against how much is lost on a
disconnect.

---

## Footnote — suggestions & recommendations

These are my (the implementer's) opinions on where this design is comfortable and
where it strains.

1. **Sweet spot is 2–4 runners.** Every claim is a push to one branch, so
   contention and GitHub's secondary rate limits grow with runner count. Past ~4
   you'd want either deterministic sharding (`hash(model, dataset, batch) % N`
   with a tiny registration step) or a real coordination service (a free Redis /
   Firebase). The lease scheme is deliberately "good enough for a handful of
   friends", not a cluster scheduler.

2. **The leaderboard rebuild is the main source of `main` push conflicts.**
   Every runner regenerates `results/leaderboard.html/.json` and pushes it every
   batch. Consider: (a) have runners push only `predictions/scores/_cache` and
   run a single separate "reduce" job (locally, or one designated runner on a
   timer) that rebuilds and pushes the leaderboard; or (b) `.gitignore` the
   generated leaderboard on `main` and only ever build it on demand. Either
   removes the one file that several runners fight over.

3. **`main` history gets noisy** — one commit per batch per runner, dozens per
   session. Options: push sweep results to a dedicated `sweep-results` branch and
   squash-merge to `main` periodically; or accept it and `git log
   --grep '^sweep\['` to filter. Don't rewrite already-pushed history while
   runners are live.

4. **Prune `sweep-claims` between big sweeps.** `prune_done` handles 7-day-old
   `done` leases, but the simplest reset is
   `git push origin --delete sweep-claims` when no sweep is running — the next
   run recreates it empty. Keeps the branch from accumulating thousands of dead
   lease files.

5. **Drive is optional; treat it that way.** The only thing it really buys you is
   the cache warm-start on a fresh VM. If Drive quota is tight, you could mirror
   *only* `_cache/` and skip the per-run copies (results are already on GitHub).
   Don't build anything that depends on Drive being consistent across accounts —
   it isn't.

6. **Run `sweep_audit.py` after each session**, not just when something looks
   wrong. A duplicate usually means a runner's clock was skewed (the tie-break
   uses wall-clock `claimed_at`) or a network partition let two claims both
   "win". It's cheap insurance and the fix is just deleting the redundant
   `results/<run_id>/`.

7. **Clock skew is a real (if small) risk.** `wins_over` compares
   `claimed_at` wall-clock timestamps from different VMs. Colab VMs are usually
   NTP-synced within a second, and `session_id` breaks exact ties, but a badly
   skewed VM could occasionally lose a race it "should" have won and redo work.
   The cache stops that from costing anything real; it just shows up in the
   audit.

8. **If you later want strict, zero-duplicate coordination**, the smallest change
   that gets you there is to move the ledger from "one file per lease on a
   branch" to a single append-only log consumed by one authoritative process, or
   to a hosted key-value store with atomic compare-and-set. The current design
   accepts rare, harmless, *visible* duplication as the price of having no
   server.
