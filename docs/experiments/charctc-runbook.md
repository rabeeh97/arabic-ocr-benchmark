# Character-level CTC — CPU experiment on the 3090 box

First experiment from the tokenizer discussion: **do not train a new Arabic
BPE**. Audit fertility on our own GT, then train a closed character-vocab
CNN-BiLSTM-CTC on the same 8k Muharaf `train` lines as `exp3090-qwen3vl4b`.

Pinned to **CPU** (`CUDA_VISIBLE_DEVICES=-1`) so suite_v1 keeps the GPU.

```powershell
cd C:\Users\User\a-experiment\aocr-repo
python scripts\run_charctc_experiment.py --dir C:\Users\User\a-experiment\charctc --detach
python scripts\run_charctc_experiment.py --dir C:\Users\User\a-experiment\charctc status
```

## 0. What this is, and what it is not

| this experiment | not this experiment |
|---|---|
| character CTC, ~150–200 classes | MorphBPE / Aranizer decoder-from-scratch |
| tiny CRNN (~1–2M params) | TrOCR English tokenizer fine-tune |
| CPU, own `--dir` | anything that touches the 3090 |
| same 8k Muharaf train leak-guard as exp3090 | Maghrebi-script unfreeze (`train_vision`) |

Success criteria are in `report.md` after `full` finishes, compared to:

- stock TrOCR on Muharaf in this repo: ~80% CER
- Qwen3-VL-4B LoRA on the same 8k lines: ~30% KHATT/NAKBA, ~36% in-domain Muharaf, ~50% RASAM

## 1. Phases

`install -> check -> prepare -> charset -> audit -> train -> smoke -> full -> report`

| phase | what | how to follow |
|---|---|---|
| `install` | locate suite_v1 venv, set `PYTHONPATH` to this repo. **No pip install** | log line `using venv python ...` |
| `check` | import torch, assert CPU, probe a 1-line charset + CRNN | `check ok` |
| `prepare` | write 8k line pngs + `cache/labels.jsonl` (resumes: existing pngs skipped) | `prepare: muharaf_train N lines` |
| `charset` | closed vocab from those labels | `n_classes ...` |
| `audit` | fertility of Qwen / TrOCR-English / Aranizer / PaliGemma vs character-level | `audit.md` |
| `train` | CTC + blank penalty 0.5, checkpoint every 50 steps, greedy sample + `blank_frac` every `log_every` | `step N loss=... blank_frac=... \| gt=... hyp=...` |
| `smoke` | 20 lines × 5 datasets | per-dataset `CER-light=` |
| `full` | 200-line batches, re-run until exhausted. **Skipped while smoke is still all-blank.** | same |
| `report` | `report.md` with the comparison table | path printed at the end |

Re-run the **same command** to resume. Finished phases are skipped; `train`
picks the newest `checkpoint-*` and continues if `cpu.yaml` `epochs` is higher
than `train_complete.json`. `full` offsets into each dataset from how
many prediction lines already exist.

Epoch 1 greedy-decoded all blank (CER=1.0). The loop now subtracts
`blank_penalty` from blank logits before CTC and greedy decode so a later
epoch can leave that local min. Watch `blank_frac` drop below 1.0 and `hyp=`
stop being `''`.

## 2. Watch / stop

```powershell
python scripts\run_charctc_experiment.py --dir C:\Users\User\a-experiment\charctc status
Get-Content C:\Users\User\a-experiment\charctc\logs\charctc.log -Wait -Tail 50
# stop (the GPU job is a different pid — check first):
# taskkill /F /PID <pid from status>
```

`--from train` / `--only audit` / `--force` work like the other launchers.

## 3. Layout under `--dir`

```
charctc/
  state.json
  logs/charctc.log
  cache/images/ ... labels.jsonl  manifest.json
  runs/charset.json  checkpoint-N/  train_state.json  train_complete.json
  audit/audit.md  audit.json
  bench/smoke/<dataset>/summary.md
  bench/full/<dataset>/summary.md
  report.md
```

## 4. Venv / cache reuse

Does **not** create a venv and does **not** `pip install` into the GPU job's
environment. Default venv:

`C:\Users\User\a-experiment\suite-v1\qwen3vl4b_suite\venv`

Default HF cache:

`C:\Users\User\a-experiment\suite-v1\data_cache\hf`

Override with `--venv` / `--hf-home` if those moved. The HF token is read
from `suite-v1\experiment.env` (Muharaf is gated).

## 5. Retrieve (Mac)

```bash
scp user@10.100.0.3:a-experiment/charctc/report.md ./
scp user@10.100.0.3:a-experiment/charctc/audit/audit.md ./
scp -r user@10.100.0.3:a-experiment/charctc/bench ./charctc-bench
scp user@10.100.0.3:a-experiment/charctc/runs/train_state.json ./
scp user@10.100.0.3:a-experiment/charctc/state.json ./
```
