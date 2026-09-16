# RTX 3090 fine-tuning experiment — runbook

One script, `scripts/run_3090_experiment.py`, takes the Windows 3090 box from
nothing to a **scored fine-tune**: it installs `uv`, builds a venv, downloads the
base model + data, runs the contamination gate, trains (resumably), merges the
LoRA, scores the result on the untouched Muharaf test split, and writes plots.
Everything lands in **one folder** you choose; deleting that folder leaves zero
trace on the machine.

- **Detached** — survives the SSH session dropping / you logging out.
- **Resumable** — re-run the same command; it continues from the last finished
  phase, and training itself resumes from the newest `checkpoint-*`.
- **Observable** — `status` prints the phase table, newest checkpoint, and the
  tail of the log.

Server: `ssh user@10.100.0.3` (Windows). Everything below assumes you are SSH'd
in with a normal shell (PowerShell or cmd both work).

---

## 0. One-time box prerequisites

| Need | Check | Notes |
|---|---|---|
| NVIDIA driver ≥ 550 | `nvidia-smi` | for the `cu124` torch wheels; older driver → pass `--torch-index https://download.pytorch.org/whl/cu121` |
| ~60 GB free disk on `C:` | `Get-PSDrive C` | venv + HF cache + base weights (~8 GB) + checkpoints |
| Python 3.9+ for the *launcher* only | `python --version` | launcher is **stdlib-only** (no pip installs needed); the venv it builds is separate and pulls its own 3.11 via uv |
| PowerShell available (for the uv installer) | `powershell -?` | present on every modern Windows |

Nothing else is installed globally — `uv`, torch, transformers, bitsandbytes all
go **inside the experiment folder**.

---

## 1. Copy the repo to the box (Mac → Windows)

The box has **no `rsync`**, and its stock Windows OpenSSH `sshd` rejects
MSYS-style `/d/...` paths — `scp` only reliably writes **relative to your Windows
home** (`C:\Users\User`). So build a clean zip straight from git (tracked files
only — this already excludes `.git/`, `.venv/`, `results/`, `__pycache__/`, …),
`scp` it into your home dir, then unpack it with PowerShell.

From this Mac, in the repo root:

```bash
git archive --format=zip HEAD -o /tmp/aocr.zip
scp /tmp/aocr.zip user@10.100.0.3:aocr.zip
```

Then unpack on the box, from the Mac:

```bash
ssh user@10.100.0.3 "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force C:\Users\User\a-experiment; Expand-Archive -Force \$HOME\aocr.zip C:\Users\User\a-experiment\aocr-repo; Remove-Item \$HOME\aocr.zip\""
```

(Or SSH in and run just the three PowerShell statements directly, with `$HOME`
un-escaped.)

Repo now at **`C:\Users\User\a-experiment\aocr-repo`**. The repo **code** is all
the box needs — the script fetches model weights and data itself.

> `git archive HEAD` ships only **committed** files — commit local edits first,
> or re-run the `git archive` / `scp` / `Expand-Archive` trio after each change.
> Installing `rsync` on the box (`choco install rsync`, or MSYS2) restores
> incremental `rsync -av --delete ./ user@10.100.0.3:a-experiment/aocr-repo/`.

## 2. Drop in the HF token (Mac → Windows)

The gated pieces (Muharaf dataset always; PaliGemma base if you pick it) need a
Hugging Face token. Create `experiment.env` **on the Mac** and copy it into the
*experiment folder* (not the repo) — same scp-to-home-then-move dance as the repo:

```
# experiment.env  — KEY=VALUE, no export, quotes optional
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

```bash
scp experiment.env user@10.100.0.3:experiment.env
ssh user@10.100.0.3 "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force C:\Users\User\a-experiment\exp; Move-Item -Force \$HOME\experiment.env C:\Users\User\a-experiment\exp\experiment.env\""
```

The token must have accepted the licences for `aamijar/muharaf-public` and, if
using PaliGemma, `google/paligemma2-3b-pt-448`. `Misraj/Baseer__Nakba` and
`Qwen/Qwen3-VL-4B-Instruct` are not gated.

---

## 3. Launch (detached)

On the box, from the repo:

```powershell
cd C:\Users\User\a-experiment\aocr-repo
python scripts\run_3090_experiment.py --dir C:\Users\User\a-experiment\exp --base qwen3vl4b --detach
```

`--base` is one of:

| `--base` | model | arch shim | notes |
|---|---|---|---|
| `qwen3vl4b` *(default)* | `Qwen/Qwen3-VL-4B-Instruct` | `qwen3_vl` (added) | needs `transformers>=4.57` — the script installs it into the venv |
| `baseer` | `Misraj/Baseer__Nakba` | `qwen2_5_vl` | Arabic-manuscript SOTA base (CER 0.079 on NAKBA); cc-by-nc-sa (research). The `fetch` phase prints the checkpoint's real `architectures` — if it's **not** a Qwen2.5-VL class, stop and use `--base qwen3vl4b` |
| `paligemma` | `google/paligemma2-3b-pt-448` | `paligemma` (added) | **gated.** Uses a different collator (`_make_collator_paligemma`) from the published PaliGemma-2 recipe — **watch the first training batches** (loss should start ≈ 2–4 and fall; labels should be all `-100` except the target span) |
| `qwen3vl4b-visft` | `Qwen/Qwen3-VL-4B-Instruct` | `qwen3_vl` (added) | Same recipe as `qwen3vl4b`, but `train_vision: true` (ViT unfrozen). Tests whether the frozen encoder caps generalisation to the Maghrebi RASAM script family — see `docs/experiments/findings-and-notes.md`. Costs more VRAM; drop `image.max_pixels` first if it OOMs |

`--detach` re-spawns the worker free of your SSH session (Windows
`DETACHED_PROCESS`), writes `C:\Users\User\a-experiment\exp\logs\run.pid`, prints the log path, and
returns. You can log out.

**One run per folder.** A launch while another worker is live in the same
`--dir` is **refused** (it prints the running pid and how to watch or stop it) —
so an accidental double-launch can't stack two trainers on the GPU. Override with
`--force` only if you know the recorded pid is actually dead. Each launch also
**rotates the log**: the old `run.log` becomes `run.<timestamp>.log` and a fresh
`run.log` starts, so `status` / `-Wait` never show last attempt's errors.

Useful flags: `--limit N` (training-line count; default 8000), `--epochs F`,
`--from PHASE` / `--only PHASE`, `--allow-contam` (only if the gate flags a
false positive), `--torch-index URL`, `--force` (re-run done phases *and* ignore
the run lock).

Watch it immediately after launch (Ctrl-C stops the tail, not the run):

```powershell
Get-Content C:\Users\User\a-experiment\exp\logs\run.log -Wait -Tail 50
```

or from `cmd`:

```bat
powershell -NoProfile -Command "Get-Content C:\Users\User\a-experiment\exp\logs\run.log -Wait -Tail 50"
```

---

## 4. Check on it (while away / after reconnecting)

```powershell
python scripts\run_3090_experiment.py --dir C:\Users\User\a-experiment\exp status
```

Prints: the phase table (`bootstrap → fetch → prepare → train → merge → eval →
report`), whether the worker pid is alive, the newest `checkpoint-NNN`, and the
last 40 lines of the **current** `run.log` (earlier sessions are in
`run.<timestamp>.log` beside it). For a live tail:

```powershell
Get-Content C:\Users\User\a-experiment\exp\logs\run.log -Wait -Tail 50
```

## 5. Resume after any interruption

Just run the **same launch command again** (with or without `--detach`):

```powershell
python scripts\run_3090_experiment.py --dir C:\Users\User\a-experiment\exp --base qwen3vl4b --detach
```

Finished phases are skipped; `train` re-enters `bench-train sft` which picks up
the newest checkpoint automatically. To force a phase to re-run, add `--force`
or start explicitly: `--from train`.

---

## 6. Retrieve the results (Windows → Mac)

**Run these from the Mac after the run reports `all phases complete`** — they are
*not* part of the script.

```bash
mkdir -p ~/aocr-exp-results && cd ~/aocr-exp-results

# paths below are relative to the box's Windows home (C:\Users\User)

# merged, standalone fine-tuned weights (safetensors + processor)
scp -r user@10.100.0.3:a-experiment/exp/merged/  ./merged/

# held-out eval: CER/WER numbers + per-line predictions
scp -r user@10.100.0.3:a-experiment/exp/eval/    ./eval/

# plots: SFT loss curve + eval-CER bar
scp -r user@10.100.0.3:a-experiment/exp/plots/   ./plots/

# training telemetry (loss history, LR schedule, config actually used)
scp    user@10.100.0.3:a-experiment/exp/effective_config.yaml ./
scp    "user@10.100.0.3:a-experiment/exp/runs/*/trainer_state.json" ./   # if the glob doesn't expand, add -O or name the run dir
scp    user@10.100.0.3:a-experiment/exp/state.json ./
scp    user@10.100.0.3:a-experiment/exp/contamination_report.json ./
```

To fold the fine-tune into this repo's leaderboard afterwards: copy
`merged/<experiment>/` somewhere under this repo, `bench-train register --name
<experiment>-ft --path <that path> --arch <arch>`, then
`bench run --config configs/training/exp3090_<base>.run.yaml --report` and
`scripts/make_leaderboard.py results`.

---

## 6a. One-command experiment — `run_3090_pipeline.py`

For a whole `train → merge → in-domain eval → held-out benchmark` in a single
detached command per base (`qwen3vl4b` | `qwen3vl4b-visft` | `baseer` | `paligemma`):

```bat
cd C:\Users\User\a-experiment\aocr-repo
python scripts\run_3090_pipeline.py --dir C:\Users\User\a-experiment\exp --base baseer --detach
```

- Watch: `powershell -NoProfile -Command "Get-Content C:\Users\User\a-experiment\exp\logs\pipeline.log -Wait -Tail 50"`
- Re-run the same line to **resume** (both sub-steps resume; each re-run also
  scores the next 200 lines/dataset of the benchmark's `full` stage).
- Fetch **one file**: `scp "user@10.100.0.3:a-experiment/exp/bench/bundle.zip" .` —
  curated per-batch summaries + `leaderboard.html`, no `_cache/` hash-file mess.
- `paligemma` is HF-gated — accept `google/paligemma2-3b-pt-448` once.

Runs the same base twice (bf16 base row vs bf16 fine-tune) on the held-out
datasets — four line sets (kitab_khatt, nakba_lines, rasam1, rasam2) plus two
page-scale sets (kitab_hindawi, kitab_historicalbooks) whose rows get
`--max-new-tokens 640` and a caveat (see findings doc).
`docs/experiments/findings-and-notes.md` is the standing log of what each
experiment taught us.

---

## 6b. Held-out benchmark — base vs fine-tune

`scripts/run_3090_benchmark.py` scores the **fine-tune vs its untuned base**
(both bf16) on datasets the fine-tune never saw (Muharaf is excluded — it was the
training data). It **reuses the training folder** (`--dir` = the same
`…\a-experiment\exp`): same venv, same HF cache, the merged weights already there.
Everything it writes goes under `…\exp\bench\` — **never inside the git repo**, so
re-syncing the repo can't wipe the cache or results.

Phases `check → register → smoke → full → collect` (tracked in `bench_state.json`),
same `--detach` / resume / one-run-lock / log-rotation as the trainer, plus:

- **`smoke`** — dataset by dataset, score both models on the first `--smoke-n`
  (20) lines, copy that summary to `…\exp\bench\smoke\<dataset>\`, next dataset.
- **`full`** — dataset by dataset in **accumulate** mode: each run scores the next
  `--full-n` (200) *un-scored* lines per dataset (into
  `…\exp\bench\full\<dataset>\`). **Re-run the same command to score the next
  batch**; a dataset is finished once a batch returns fewer new lines than asked.
- **`collect`** — regenerate `…\exp\bench\leaderboard.html` over everything scored
  so far (runs every pass).

Models and datasets are the two lists in
`configs/training/exp3090_bench.run.yaml`. `--with-4bit` adds a 4-bit fine-tune
row. `nakba_lines` is HF-gated (auto-approve): open
`https://huggingface.co/datasets/U4RASD/omar-al-saleh-manuscripts-segments` while
logged into the account behind `HF_TOKEN` and click **Agree and access
repository** once — otherwise that one dataset errors and the rest still run.

**Prereqs:** the training run finished at least `merge`
(`…\exp\merged\exp3090-qwen3vl4b\` exists).

**Sync the script (Mac):**

```bash
git archive --format=zip HEAD -o /tmp/aocr.zip
scp /tmp/aocr.zip user@10.100.0.3:aocr.zip
ssh user@10.100.0.3 "powershell -NoProfile -Command \"Remove-Item -Recurse -Force C:\Users\User\a-experiment\aocr-repo; Expand-Archive -Force \$HOME\aocr.zip C:\Users\User\a-experiment\aocr-repo; Remove-Item \$HOME\aocr.zip\""
```

**Launch (box, detached):**

```bat
cd C:\Users\User\a-experiment\aocr-repo
python scripts\run_3090_benchmark.py --dir C:\Users\User\a-experiment\exp --detach
```

The first launch runs `check → register → smoke → full (batch 1) → collect`.
Each later `python scripts\run_3090_benchmark.py --dir … --detach` does the next
`full` batch (200 more lines/dataset) + `collect`, until every dataset is
exhausted. `--only smoke` / `--from full` target one stage.

**Watch it (survives logout):**

```bat
powershell -NoProfile -Command "Get-Content C:\Users\User\a-experiment\exp\logs\bench.log -Wait -Tail 50"
```
or the phase table: `python scripts\run_3090_benchmark.py --dir C:\Users\User\a-experiment\exp status`

**Fetch the results (Mac):**

```bash
mkdir -p ~/aocr-exp-results && cd ~/aocr-exp-results
scp -r "user@10.100.0.3:a-experiment/exp/bench" ./
```

`bench/smoke/<ds>/<run_id>/` and `bench/full/<ds>/<run_id>/` hold the per-batch
`summary.csv/md` + `predictions.jsonl` + `scores.jsonl`; `bench/leaderboard.html`
is the cumulative view.

**Merge into this repo's leaderboard:** copy `merged/exp3090-qwen3vl4b/` under the
repo, re-register the fine-tune rows with **repo-relative** paths (the box's
`configs/models.yaml` has machine-local absolute paths — don't commit it), drop
the fetched `bench/full/*/*/` run dirs into `results/`, then
`uv run python scripts/make_leaderboard.py results -o results/leaderboard.html`.

## 7. Clean up — leaves nothing behind

```powershell
python scripts\run_3090_experiment.py --dir C:\Users\User\a-experiment\exp clean --yes
```

That is just `rmdir /s /q C:\Users\User\a-experiment\exp`. The venv, the uv cache
(`UV_CACHE_DIR` was pinned inside the folder), the HF cache (`HF_HOME` inside the
folder), `uv` itself (installed to `C:\Users\User\a-experiment\exp\bin`),
checkpoints, logs — all of it is under that one path. After deletion,
`nvidia-smi` is idle and no `arabic-ocr` process, service, scheduled task,
registry key, or file remains. The SSH-copied
`C:\Users\User\a-experiment\aocr-repo` is untouched — `rmdir /s /q
C:\Users\User\a-experiment` wipes both.

---

## 8. What to expect

**VRAM (24 GB, batch 1 × grad-accum 8, gradient checkpointing, seq ≤ 1024):**

| base | method | VRAM | OOM lever |
|---|---|---|---|
| Qwen3-VL-4B | bf16 LoRA | ~16–20 GB | `image.max_pixels` ↓, then `max_seq_len` ↓, then `method: qlora` |
| Baseer / NAKBA (~3B) | bf16 LoRA | ~12–18 GB | same |
| PaliGemma 2-3B | bf16 LoRA (batch 2) | ~10–16 GB | drop `batch_size` to 1 |

**Runtime** — ≈ **2–4 s / optimizer step** on a 3090, i.e. **~35–70 min per
1,000 training lines per epoch**. At the default `limit: 8000`, 1 epoch:

| base | 1-epoch train | + bootstrap & fetch |
|---|---|---|
| Qwen3-VL-4B | ~5–9 h | +25–45 min (deps + ~8 GB download) |
| Baseer / NAKBA | ~4–8 h | +20–35 min |
| PaliGemma 2-3B | ~3–6 h | +20–35 min |

`merge` ≈ 3–8 min (CPU). `eval` (200 test lines, `concurrency: 1`) ≈ 10–25 min.
`report` ≈ seconds.

**Lower `--limit`** for a fast first pass (e.g. `--limit 1000` ≈ 40–70 min of
training) to prove the whole pipeline before committing to the long run.

---

## 9. Gotchas

- **Launcher Python** — the launcher is stdlib-only and runs on **3.9+** (the box
  had 3.10). It does *not* need `pyyaml`; the config rewrite runs in the venv.
  Older `datetime` needs no `UTC` symbol — the script shims it.
- **Windows text encoding** — Python on Windows defaults file writes to `cp1252`,
  which cannot encode Arabic; every `read_text`/`write_text` in the package now
  passes `encoding="utf-8"` explicitly. Symptom if one is missed:
  `UnicodeEncodeError: 'charmap' codec can't encode…` while writing
  `contamination_report.json`, `summary.md`, prediction caches, etc.
- **`uv` installer 403** — the `astral.sh/uv/install.{ps1,sh}` URLs redirect to
  GitHub's asset CDN, which returns **HTTP 403** to `urllib`'s default agent. The
  script's `_download()` sends a `Mozilla/5.0` UA to get past it. If it still 403s
  the box is behind a proxy — install `uv` by hand (`winget install astral-sh.uv`
  or download the release zip) so `shutil.which("uv")` finds it, then re-run.
- **Training-stack pins** — `bootstrap` installs `transformers>=<floor>,<5` (floor
  is 4.57 for Qwen3-VL, 4.49 otherwise) and `trl>=0.21,<1.0`, after the editable
  install, so the resolver can't drift up. Unbounded `>=` had pulled
  transformers 5.x + trl 1.x, whose `SFTConfig` dropped `warmup_ratio` and rewrote
  the SFT pipeline (`TypeError: SFTConfig.__init__() got an unexpected keyword
  argument 'warmup_ratio'`). `pyproject.toml` now carries the `<5` / `<1.0` caps
  too. The core repo's transformers floor stays 4.49. The pin step uses
  `uv pip install -P transformers -P trl …` — a bare `-U` also upgrades `torch`,
  replacing the cu124 wheel (see next point). If you ever pin deps by hand, never
  pass `-U`; reinstall torch from the cu124 index if you do.
- **`torch` must come from the cu124 index, and WDAC** — the CUDA wheel is
  installed from `download.pytorch.org/whl/cu124`. A plain PyPI `torch` (what an
  unscoped `-U` resolve pulls) is either CPU-only or, on a box with Windows
  Defender Application Control / Smart App Control enabled, fails to load its
  DLLs: `OSError: [WinError 4551] An Application Control policy has blocked this
  file … torch\lib\shm.dll`. Fix: `uv pip install --python <venv-py>
  --reinstall-package torch --reinstall-package torchvision torch torchvision
  --index-url https://download.pytorch.org/whl/cu124`. If the cu124 wheel *also*
  hits 4551, the box's WDAC policy blocks PyTorch outright — needs an admin to
  allowlist it or disable Smart App Control; no pip/code change helps.
- **bitsandbytes on Windows** — 0.44+ ships CUDA wheels; the `bootstrap`
  self-check (`import bitsandbytes`) fails loudly if the wheel is CPU-only or the
  CUDA runtime mismatches. Only relevant if you switch a config to `qlora`.
- **PaliGemma collator is new** — from the standard PaliGemma-2 SFT recipe but
  not yet run on hardware in this repo. First run: confirm loss starts finite
  (~2–4) and the processor's label mask looks right (spot-check
  `batch["labels"]` in a debugger, or lower `--limit 50` and eyeball the loss
  trajectory). If it misbehaves, `_PALIGEMMA_PREFIX` and the `suffix=` handling
  in `src/arabic_ocr_bench/training/sft_vlm.py` are the two knobs.
- **Baseer arch probe** — the `fetch` phase prints `architectures:` from the
  checkpoint's `config.json`. The config assumes a Qwen2.5-VL class (matching the
  existing `nakba-misraj-baseer` entry in `configs/models.yaml`). If it prints
  something else, `--base qwen3vl4b` is the safe fallback.
- **Contamination gate** — `prepare` runs `bench-train check-contam` and
  **hard-stops** if a train/eval line overlap is flagged
  (`C:\Users\User\a-experiment\exp\contamination_report.json`). This is the no-leak guarantee; only
  override with `--allow-contam` after confirming a false positive. If
  `check-contam` *crashes* (no flagged report written) the launcher says so
  explicitly — that's a bug to fix, not something `--allow-contam` bypasses.
- **`--from` / `--only`** — if `eval` fails because a transient network hiccup
  broke the register step, `--from eval` re-runs just eval + report without
  re-training.

---

## 10. Multi-model suite — `run_experiment_suite.py`

Sequential `train → merge → in-domain eval → full cross-domain benchmark`
across the whole `suite_v1` round-1 model queue
(`scripts/_experiment_bases.round1_queue()` — currently
`qwen3vl4b_suite`, `qwen3vl8b`, `ain7b`, `fanar2_oryx`), one model at a time
on this one GPU. It's a thin orchestrator over the exact same
`run_3090_experiment.py` / `run_3090_benchmark.py` phase machines described in
§3/§6a/§6b above — nothing about crash-safety, resuming or venv handling is
reimplemented, only sequencing across models is new. See
`docs/experiments/misraj-baseer-research.md` for why this data mix (Muharaf +
AraMS-28k-HTR + RASAM-1/2, NAKBA deliberately excluded) and this model lineup.

**One-time prerequisite — fetch the shared training data once, not once per
model:**

```bat
cd C:\Users\User\a-experiment\aocr-repo
python scripts\fetch_training_corpus.py --dir C:\Users\User\a-experiment\suite-v1
```

(Needs the `datasets`/`huggingface_hub` libraries — run it inside a venv that
already has this repo's `[local,train]` extra installed, e.g. after any prior
`run_3090_experiment.py` bootstrap phase has built one.) This fetches
Muharaf/RASAM-1/2 into a shared HF cache and AraMS-28k-HTR into the
repo-relative `data/arams_28k_htr/` that every `suite_v1/*.yaml` config
already points at — so none of the four sequential model runs below
re-downloads the same corpora independently.

**Launch the whole queue, detached:**

```bat
python scripts\run_experiment_suite.py --dir C:\Users\User\a-experiment\suite-v1 --detach
```

- Watch: `powershell -NoProfile -Command "Get-Content C:\Users\User\a-experiment\suite-v1\logs\suite.log -Wait -Tail 50"`
- Status (phase-per-model table + log tail):
  `python scripts\run_experiment_suite.py --dir C:\Users\User\a-experiment\suite-v1 status`
- Re-run the same `--detach` line to **resume** — suite-level progress lives in
  `suite_state.json` (queued/running/done/failed/skipped per model); an
  already-`done` model is skipped, the in-progress one resumes via its own
  `<suite-dir>/<model>/state.json` exactly as in §5.
- `fanar2_oryx` is skipped automatically (logged, doesn't block the other
  three) until `scripts/_experiment_bases.py`'s `license_check` field for it
  is flipped from `"pending"` to `"ok"` — see that module's docstring before
  changing it.
- Each model's checkpoint cadence is `save_steps: 5` (not the older configs'
  `100`) — a crash loses at most a few samples, not the better part of an
  hour; cheap because these are small LoRA-adapter checkpoints, not full
  model saves.
- Per-model results land under `<suite-dir>/<model>/bench/` exactly as in
  §6b; after every model finishes, the suite also re-renders the **top-level**
  `results/leaderboard.html` (not just each model's own bench folder) so the
  whole queue's results are visible in one place without a manual step.
