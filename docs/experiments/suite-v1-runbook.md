# suite_v1 multi-model experiment — copy/run/finish/fetch cheat sheet

Everything needed to run the round-1 suite (`qwen3vl4b_suite`, `qwen3vl8b`,
`ain7b`, `fanar2_oryx`) end to end on the Windows 3090 box, start to finish,
in order. This is the condensed, copy-paste version of
`docs/experiments/3090-experiment-runbook.md` §10 — read that file (and
`docs/experiments/misraj-baseer-research.md` for *why* this data mix and
model lineup) if anything here needs more context than a command.

Server: `ssh user@10.100.0.3` (Windows, PowerShell or cmd both work).
Everything below uses one experiment root: **`C:\Users\User\a-experiment\suite-v1`**
— change every occurrence consistently if you use a different path.

---

## 0. One-time box prerequisites

Same as the single-experiment runbook §0: NVIDIA driver ≥ 550, ~60 GB free
disk (this suite runs 4 models sequentially, so budget more like 150-200 GB
if you don't clean up between models), Python 3.9+ available as `python` for
the stdlib-only launcher scripts, PowerShell available.

---

## 1. Copy the repo to the box (Mac → Windows)

From this Mac, in the repo root (make sure the suite_v1 commit is checked out
— `git log --oneline -1` should show the "Add a multi-model, multi-dataset
training suite (suite_v1)" commit or later):

```bash
git archive --format=zip HEAD -o /tmp/aocr.zip
scp /tmp/aocr.zip user@10.100.0.3:aocr.zip
ssh user@10.100.0.3 "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force C:\Users\User\a-experiment; Expand-Archive -Force \$HOME\aocr.zip C:\Users\User\a-experiment\aocr-repo; Remove-Item \$HOME\aocr.zip\""
```

Repo now at `C:\Users\User\a-experiment\aocr-repo`. Re-run this trio any time
you change something on the Mac and need it on the box (`git archive` only
ships committed files).

## 2. Drop in the HF token (Mac → Windows)

Muharaf, RASAM-1/2 and NAKBA all need an accepted-license `HF_TOKEN` (NAKBA
is gated with auto-approve — visit
`https://huggingface.co/datasets/U4RASD/omar-al-saleh-manuscripts-segments`
once while logged in and click "Agree and access repository" first).
Fanar-2-Oryx-IVU may also need one depending on its actual gating — check
when you get to that model.

```
# experiment.env — KEY=VALUE, no export, quotes optional
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

```bash
scp experiment.env user@10.100.0.3:experiment.env
ssh user@10.100.0.3 "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force C:\Users\User\a-experiment\suite-v1; Move-Item -Force \$HOME\experiment.env C:\Users\User\a-experiment\suite-v1\experiment.env\""
```

## 3. Fetch the shared training data ONCE (not once per model)

This needs `datasets`/`huggingface_hub`, so it must run inside a venv that
already has the repo's `[local,train]` extra — the simplest way is to let
`run_experiment_suite.py`'s first model bootstrap a venv, then reuse it. On
the box:

```powershell
cd C:\Users\User\a-experiment\aocr-repo
python scripts\run_3090_experiment.py --dir C:\Users\User\a-experiment\suite-v1\qwen3vl4b_suite --base qwen3vl4b_suite --only bootstrap
```

Wait for that to finish (a few minutes — it just builds the venv), then:

```powershell
C:\Users\User\a-experiment\suite-v1\qwen3vl4b_suite\venv\Scripts\python.exe scripts\fetch_training_corpus.py --dir C:\Users\User\a-experiment\suite-v1
```

This downloads Muharaf/RASAM-1/RASAM-2 into a shared HF cache under
`suite-v1\data_cache\hf` and AraMS-28k-HTR (~2.4 GB) into the repo-relative
`aocr-repo\data\arams_28k_htr\` that every `suite_v1/*.yaml` config already
points at. **After it finishes**, sanity-check the AraMS extraction landed
flat (`<id>.png` next to `<id>.gt.txt`, not nested a directory deeper) — see
`scripts/fetch_arams.py`'s docstring if `n_png == 0` in its own printed
summary; you may need `--strip-components 1`.

---

## 4. Launch the whole suite (detached)

```powershell
cd C:\Users\User\a-experiment\aocr-repo
python scripts\run_experiment_suite.py --dir C:\Users\User\a-experiment\suite-v1 --detach
```

This runs `qwen3vl4b_suite → qwen3vl8b → ain7b → fanar2_oryx` in that order,
one at a time — each one trains, merges, scores its own in-domain
(`muharaf_lines`) test, then runs the full cross-domain sweep against every
other safe benchmark dataset, before moving to the next model.
`fanar2_oryx` will be **skipped automatically** (logged, not queue-blocking)
unless `scripts/_experiment_bases.py`'s `fanar2_oryx["license_check"]` has
been flipped from `"pending"` to `"ok"` after actually checking QCRI's
fine-tuning/redistribution terms.

**Watch it** (Ctrl-C stops the tail, not the run):

```powershell
Get-Content C:\Users\User\a-experiment\suite-v1\logs\suite.log -Wait -Tail 50
```

or from `cmd`:

```bat
powershell -NoProfile -Command "Get-Content C:\Users\User\a-experiment\suite-v1\logs\suite.log -Wait -Tail 50"
```

---

## 5. Check on it / resume after any interruption

```powershell
python scripts\run_experiment_suite.py --dir C:\Users\User\a-experiment\suite-v1 status
```

Prints each model's status (`queued`/`running`/`done`/`failed`/`skipped`)
plus the last 40 log lines. If the box rebooted or the process died, just
run the **exact same launch command again**:

```powershell
python scripts\run_experiment_suite.py --dir C:\Users\User\a-experiment\suite-v1 --detach
```

Already-`done` models are skipped; the in-progress one resumes from its own
per-model state (training resumes from the newest checkpoint — at most a
handful of samples lost, not a partial epoch; the benchmark sweep resumes
from the last fully-scored batch per dataset). For a deeper look at one
model's own phase table (bootstrap/fetch/prepare/train/merge/eval/report):

```powershell
python scripts\run_3090_experiment.py --dir C:\Users\User\a-experiment\suite-v1\qwen3vl4b_suite status
```

(swap `qwen3vl4b_suite` for `qwen3vl8b` / `ain7b` / `fanar2_oryx` as needed.)

---

## 6. What "finished" looks like

`suite status` shows every enabled model as `done` (or `skipped` for
`fanar2_oryx` if its license was never confirmed), and the log ends with:

```
=== suite complete — rebuilding the top-level leaderboard ===
```

At that point `C:\Users\User\a-experiment\aocr-repo\results\leaderboard.html`
on the box already reflects every model's results — no manual leaderboard
step needed.

---

## 7. Retrieve the results (Windows → Mac)

Run these from the Mac once the suite reports complete:

```bash
mkdir -p ~/aocr-suite-v1-results && cd ~/aocr-suite-v1-results

# per-model merged weights, eval numbers, plots, and cross-domain bench results
for m in qwen3vl4b_suite qwen3vl8b ain7b fanar2_oryx; do
  mkdir -p "$m"
  scp -r "user@10.100.0.3:a-experiment/suite-v1/$m/merged/" "$m/merged/"
  scp -r "user@10.100.0.3:a-experiment/suite-v1/$m/eval/"   "$m/eval/"
  scp -r "user@10.100.0.3:a-experiment/suite-v1/$m/bench/"  "$m/bench/"   # curated summaries + that model's own leaderboard.html + bundle.zip
  scp    "user@10.100.0.3:a-experiment/suite-v1/$m/state.json" "$m/"
  scp    "user@10.100.0.3:a-experiment/suite-v1/$m/bench_state.json" "$m/"
done

# suite-level progress record
scp user@10.100.0.3:a-experiment/suite-v1/suite_state.json ./
scp user@10.100.0.3:a-experiment/suite-v1/logs/suite.log ./

# the rebuilt TOP-LEVEL leaderboard covering all 4 models together
scp -r user@10.100.0.3:a-experiment/aocr-repo/results/leaderboard.html ./
scp -r user@10.100.0.3:a-experiment/aocr-repo/results/leaderboard.json ./
```

(A model that was `skipped`, e.g. a still-license-gated `fanar2_oryx`, won't
have a `merged/`/`eval/`/`bench/` to fetch — that's expected, not an error.)

The per-model `merged/<experiment>/` directories are already independently
registerable elsewhere with `bench-train register --name <experiment>-ft
--path <that path> --arch <arch>` if you ever want to re-score them outside
this suite's own folder.

---

## 8. Clean up — leaves nothing behind

Per model (after you've fetched what you want from it):

```powershell
python scripts\run_3090_experiment.py --dir C:\Users\User\a-experiment\suite-v1\qwen3vl4b_suite clean --yes
```

repeat for `qwen3vl8b` / `ain7b` / `fanar2_oryx`. Or wipe the entire suite in
one shot once everything is fetched:

```powershell
Remove-Item -Recurse -Force C:\Users\User\a-experiment\suite-v1
```

The SSH-copied `C:\Users\User\a-experiment\aocr-repo` (and the shared
`data\arams_28k_htr\` inside it) is untouched by that — delete
`C:\Users\User\a-experiment` entirely to wipe both, or keep the repo copy
around if you plan to run another suite later (the AraMS download doesn't
need repeating).

---

## 9. What to expect (rough — verify against the real box, not a promise)

Each model trains on ~18k combined lines (Muharaf + AraMS + RASAM-1/2),
1 epoch. At the single-experiment runbook's own measured **~35-70 min per
1,000 training lines**, that's roughly:

| stage | rough time |
|---|---|
| bootstrap + fetch (once, shared) | ~30-60 min |
| training, per model | ~10.5-21 h |
| cross-domain benchmark sweep, per model (~15 datasets) | a few hours, varies with dataset sizes |
| **4-model sequential total** | **very roughly 2-4 days** of continuous box time |

This is exactly why round 1 uses a **capped** data mix rather than the full
combined corpora (~60-75k lines/model, which would multiply the above by
~4x) — see `docs/experiments/misraj-baseer-research.md` for that tradeoff.
If a faster first pass is wanted, lower every `limit:` in
`configs/training/suite_v1/*.yaml` before launching (e.g. halve each one),
or run just one model first with `run_3090_experiment.py --base
qwen3vl4b_suite` directly to sanity-check the whole pipeline before
committing to the full queue.
