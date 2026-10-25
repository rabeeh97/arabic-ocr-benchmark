# Colab experiment plan: measure first, then run the real experiments

Created 2026-10-25. Nothing in this document has been run yet except the concurrency and billing probes in section 1. The budget cap is 10 compute units for the
benchmark stage (spent so far: 1.51; baseline balance 200.00 on 2026-10-24). All jobs run through `scripts/colab_job.py` (detached on the VM, incremental sync to the
Mac, budget and wall-clock interrupts, verified stop) and are described by a JSON spec in `configs/colab/`.

## 1. What the account allows (measured 2026-10-25)
| Fact | Measured |
|---|---|
| Concurrent GPU sessions | **2** (two T4 sessions READY; the third allocation was refused: "too many active sessions, or capacity") |
| Concurrent CPU sessions | at least **4** |
| Billing | per session, additive: T4 1.07 units/hour, CPU 0.08 units/hour; two T4 = 2.14/hour |
| Session start | about 6 s (`colab new`) |
| Data on the VM | Hugging Face download of RASAM-1 41.8 s, packing it with our prep script 163.9 s (CPU-bound) |
| Where results live | on the VM disk `/content` only, deleted when the session ends; the runner tars `/content/run` and `/content/crnn_out`, downloads them every 4 minutes and at the end |

So: up to **2 GPU groups plus several cheap CPU groups at once**. Each GPU group should be a whole queue of runs in one session so that the setup (about 3.5 minutes, about
0.06 units on a T4) is paid once. CPU-only work (LM decoding, error analysis, plots) should go to CPU sessions, not to a GPU.

## 2. Stage 1: measure speed, cost and result parity (same experiment everywhere)
The probe is the trainer we already ran on the Mac CPU: `xai_train_aligner.py`, RASAM-1 train (5,860 lines), geom-aug, seed 0, 3 epochs, with three variants inside one
session so setup is shared: half width bs 16 (the Mac recipe), full width bs 16, half width bs 64. Reference: Mac CPU about 546 s per epoch (half width, bs 16).

| Group | Sessions | Content | Expected cost | Why |
|---|---|---|---|---|
| **M1** | T4 and L4 at the same time (2 GPU sessions) | the 3-variant probe on each | about 0.3 + 0.5 units, about 15 min wall | speed, cost per epoch, and parity of the 3-epoch held-out CER across hardware |
| **M2** | A100 (1 GPU session) + 1 CPU session | the probe on the A100; on the CPU session the Mac-comparable LM-decode job (about 5 min on the Mac) | about 1 to 2 units for the A100 (rate unverified: sources say 5 to 12 units/hour), 0.01 for the CPU | does a faster GPU pay for itself in units per epoch; CPU-session speed vs the Mac |
| M3 (later, needs the VPN) | the 3090 | the same probe with `--device cuda` | no Colab units | the original comparison you asked for |

Outputs per run: seconds per epoch (steady state, epochs 2 and 3), setup seconds (VM clock, so network jitter does not count), units used, **units per epoch** and
**epochs per unit**, and held-out CER after 3 epochs on every hardware (it should agree within seed noise; if it does not, the speed-up is not a free lunch).
Decision it feeds: which GPU to default to (cheapest units per epoch), and what batch size and width are affordable.

## 3. Stage 2: the real experiments, grouped to run in parallel
All use RASAM (RASAM-1 and RASAM-2 published splits) and report CER with and without the character LM, so they feed the question "does better and more diverse data give
better results". Prerequisite code is listed in section 4. Costs are estimates to be replaced by the Stage 1 measurements.

| Group | Type | Runs (one session, shared setup) | Question |
|---|---|---|---|
| **R1: match the published setup** | GPU | train on RASAM-1 + RASAM-2 (A3 recipe) at full width, 30 epochs, 3 seeds; score both test sets; LM decode | like-for-like with the published 5.50 % and 6.79 %; first seed variance (we have none) |
| **R2: capacity and resolution** | GPU | RASAM-1 + RASAM-2, width {0.5, 1.0} x height {64, 96, 128}, 1 seed, 20 epochs | are thin letters and dense lines limited by input height and model size (error analysis: 12 % of errors are deleted or inserted thin letters, dense crops 8.8 % vs 6.0 %) |
| **R3: data diversity** | GPU | manuscript-disjoint RASAM-2 split, with extra training data added step by step: none; + RASAM-1; + TariMa (2.4 k lines); + Maghrebi AraMS books (8.9 k lines); + Muharaf pretrain | the thesis: held-out-manuscript CER vs number of manuscripts and lines |
| **R4: statistics** | GPU (cheap, L4/T4) | the best recipe from R1 and R2 x 5 seeds, plus bootstrap CI and paired tests | uncertainty on every number we will quote |
| **C1: decoding** | CPU sessions (about 0.08 units/hour each) | character LM order, LM text (RASAM train only vs OpenITI classical Arabic vs both), weights tuned on validation | how far the LM gain goes (so far 6.82 % -> 5.46 % on RASAM-1) |
| **C2: analysis** | CPU session | error analysis, per-manuscript tables, plots for R1 to R4 as they finish | writing up |

Which run together (at most 2 GPU groups at a time, any number of CPU groups):
- **Wave A:** R1 + R2 (two GPU sessions) + C1 (CPU).
- **Wave B:** R3 + R4 (two GPU sessions) + C2 (CPU).
Wave B depends on Wave A only through "best recipe", so R3 can start earlier if you want to run it with the current recipe.

## 4. What has to exist before Stage 2
1. `--height` in `xai_train_aligner.py` and `rasam_prepare_crnn_data.py` (the model pools to height 2 at 64 px; 96 and 128 give 3 and 4, so the projection layer size must follow). Small change, needs a CPU smoke test.
2. Data packs for R3: TariMa lines (cached on the SSD) and the Maghrebi AraMS books (the zip is not downloaded yet; the clean download was interrupted by the SSD dropout), packed into pickles and uploaded with `colab upload` (or pulled from the Hugging Face Hub).
3. Multi-seed and multi-run specs (the runner already takes a list of runs).
4. For C1: an OpenITI text file for the LM (cached on the SSD).

## 5. Safeguards (same for every job)
- The job runs detached on the VM with `timeout`; the Mac only supervises.
- Results are synced every 4 minutes and at the end to `~/colab_staging/jobs/<name>`, then copied to the SSD; checkpoints are written every epoch and every 200 steps.
- Interrupts with a final sync and a verified stop: job finished, job budget reached, overall budget floor reached (200 minus 10), wall clock, Ctrl-C, SIGTERM.
- Two jobs running at once both see the account balance fall, so each counts the other's spending against its own cap: conservative, it stops earlier, never later.
- If the Colab API is unreachable before a session exists, nothing starts (exit 75) and the wrapper retries every 2 minutes.
