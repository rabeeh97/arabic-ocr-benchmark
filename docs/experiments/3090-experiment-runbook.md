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
| ~60 GB free disk on the target drive | `dir D:\` | venv + HF cache + base weights (~8 GB) + checkpoints |
| Python 3.11+ **with `pyyaml`** for the *launcher* only | `python -c "import sys,yaml;print(sys.version)"` | the venv the script builds is separate; the launcher just needs stdlib + pyyaml. `python -m pip install pyyaml` if missing |
| PowerShell available (for the uv installer) | `powershell -?` | present on every modern Windows |

Nothing else is installed globally — `uv`, torch, transformers, bitsandbytes all
go **inside the experiment folder**.

---

## 1. Copy the repo to the box (Mac → Windows)

From this Mac, in the repo root. Pick a **clean** copy — exclude the big
local-only trees:

```bash
rsync -av --delete \
  --exclude '.git/' --exclude '.venv/' --exclude 'results/' \
  --exclude '.pytest_cache/' --exclude '.ruff_cache/' --exclude '.cache/' \
  --exclude '__pycache__/' --exclude '.superpowers/' \
  ./  user@10.100.0.3:/d/aocr-repo/
```

> No `rsync` on the box? `scp -r` works too, or zip locally and `scp` the zip:
> `git archive --format=zip HEAD -o /tmp/aocr.zip && scp /tmp/aocr.zip user@10.100.0.3:/d/`
> then unzip on the box into `D:\aocr-repo`.

The repo **code** is all the box needs — the script fetches model weights and
data itself.

## 2. Drop in the HF token (Mac → Windows)

The gated pieces (Muharaf dataset always; PaliGemma base if you pick it) need a
Hugging Face token. Create `experiment.env` **on the Mac** and copy it into the
*experiment folder* (not the repo):

```
# experiment.env  — KEY=VALUE, no export, quotes optional
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

```bash
ssh user@10.100.0.3 "mkdir -p /d/aocr-exp"
scp experiment.env user@10.100.0.3:/d/aocr-exp/experiment.env
```

The token must have accepted the licences for `aamijar/muharaf-public` and, if
using PaliGemma, `google/paligemma2-3b-pt-448`. `Misraj/Baseer__Nakba` and
`Qwen/Qwen3-VL-4B-Instruct` are not gated.

---

## 3. Launch (detached)

On the box, from the repo:

```powershell
cd D:\aocr-repo
python scripts\run_3090_experiment.py --dir D:\aocr-exp --base qwen3vl4b --detach
```

`--base` is one of:

| `--base` | model | arch shim | notes |
|---|---|---|---|
| `qwen3vl4b` *(default)* | `Qwen/Qwen3-VL-4B-Instruct` | `qwen3_vl` (added) | needs `transformers>=4.57` — the script installs it into the venv |
| `baseer` | `Misraj/Baseer__Nakba` | `qwen2_5_vl` | Arabic-manuscript SOTA base (CER 0.079 on NAKBA); cc-by-nc-sa (research). The `fetch` phase prints the checkpoint's real `architectures` — if it's **not** a Qwen2.5-VL class, stop and use `--base qwen3vl4b` |
| `paligemma` | `google/paligemma2-3b-pt-448` | `paligemma` (added) | **gated.** Uses a different collator (`_make_collator_paligemma`) from the published PaliGemma-2 recipe — **watch the first training batches** (loss should start ≈ 2–4 and fall; labels should be all `-100` except the target span) |

`--detach` re-spawns the worker free of your SSH session (Windows
`DETACHED_PROCESS`), writes `D:\aocr-exp\logs\run.pid`, prints the log path, and
returns. You can log out.

Useful flags: `--limit N` (training-line count; default 8000), `--epochs F`,
`--from PHASE` / `--only PHASE`, `--allow-contam` (only if the gate flags a
false positive), `--torch-index URL`.

---

## 4. Check on it (while away / after reconnecting)

```powershell
python scripts\run_3090_experiment.py --dir D:\aocr-exp status
```

Prints: the phase table (`bootstrap → fetch → prepare → train → merge → eval →
report`), whether the worker pid is alive, the newest `checkpoint-NNN`, and the
last 40 log lines. For a live tail:

```powershell
Get-Content D:\aocr-exp\logs\run.log -Wait -Tail 50
```

## 5. Resume after any interruption

Just run the **same launch command again** (with or without `--detach`):

```powershell
python scripts\run_3090_experiment.py --dir D:\aocr-exp --base qwen3vl4b --detach
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

# merged, standalone fine-tuned weights (safetensors + processor)
scp -r user@10.100.0.3:/d/aocr-exp/merged/  ./merged/

# held-out eval: CER/WER numbers + per-line predictions
scp -r user@10.100.0.3:/d/aocr-exp/eval/    ./eval/

# plots: SFT loss curve + eval-CER bar
scp -r user@10.100.0.3:/d/aocr-exp/plots/   ./plots/

# training telemetry (loss history, LR schedule, config actually used)
scp    user@10.100.0.3:/d/aocr-exp/effective_config.yaml ./
scp    "user@10.100.0.3:/d/aocr-exp/runs/*/trainer_state.json" ./
scp    user@10.100.0.3:/d/aocr-exp/state.json ./
scp    user@10.100.0.3:/d/aocr-exp/contamination_report.json ./
```

To fold the fine-tune into this repo's leaderboard afterwards: copy
`merged/<experiment>/` somewhere under this repo, `bench-train register --name
<experiment>-ft --path <that path> --arch <arch>`, then
`bench run --config configs/training/exp3090_<base>.run.yaml --report` and
`scripts/make_leaderboard.py results`.

## 7. Clean up — leaves nothing behind

```powershell
python scripts\run_3090_experiment.py --dir D:\aocr-exp clean --yes
```

That is just `rmdir /s /q D:\aocr-exp`. The venv, the uv cache
(`UV_CACHE_DIR` was pinned inside the folder), the HF cache (`HF_HOME` inside the
folder), `uv` itself (installed to `D:\aocr-exp\bin`), checkpoints, logs — all of
it is under that one path. After deletion, `nvidia-smi` is idle and no
`arabic-ocr` process, service, scheduled task, registry key, or file remains.
The SSH-copied `D:\aocr-repo` is untouched — delete it separately if you want.

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

- **`transformers>=4.57`** for Qwen3-VL — the script installs it into the venv
  after the editable install so the resolver can't downgrade it. The core repo
  stays at 4.49.
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
  (`D:\aocr-exp\contamination_report.json`). This is the no-leak guarantee; only
  override with `--allow-contam` after confirming a false positive.
- **`--from` / `--only`** — if `eval` fails because a transient network hiccup
  broke the register step, `--from eval` re-runs just eval + report without
  re-training.
