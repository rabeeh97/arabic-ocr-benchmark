# XAI and geometry tasks: block for NEXT_STEPS.md, and the 3090 runbook

> Written 2026-10-11 on branch `worktree-xai-attribution-spike`. `NEXT_STEPS.md` is untracked in the main
> checkout, so this file holds the block to paste there (section 1) and the commands for the work that needs the
> 3090 (section 2). The 3090 was not reachable from the laptop on 2026-10-11 (SSH timed out), so nothing in
> section 2 has been run.

## 1. Block to paste into NEXT_STEPS.md

**XAI and geometry track** (plan: [`xai-attribution-plan.md`](xai-attribution-plan.md); results: Arabic OCR Paper Ledger)

- [x] (2026-10-10) **X-A1 Attribution maps validated** on Qwen3-VL-4B + E1 adapter. Three maps kept; gradient × (input − paper) puts 9× chance on ink.
- [x] (2026-10-10) **X-C1 LocateAnything-3B zero-shot probe.** No-go (whole-image or line-wide boxes on 42 lines).
- [x] (2026-10-11) **X-B1 CTC character spans** from a CRNN-CTC aligner (7.6 % held-out Muharaf CER). *Open:* validate against VML-HD sub-word boxes (data not yet obtained).
- [x] (2026-10-11) **X-B2 Pointing game.** About 3× chance for all three maps; wrong tokens point at the right place as often as correct ones.
- [x] (2026-10-11) **X-E1 CRNN generalization.** 13.8 % NAKBA test, 13.7 % blind test, 26.4 % KHATT, never trained on them.
- [x] (2026-10-11) **X-G1, X-G2, X-G4, X-G5, X-G6 Geometry measurements** (`scripts/geom_measure.py`, `geom_analyze.py`; `results/geometry/run1`). KHATT is far from the Muharaf training lines in slant (+11° against −2°) and pen proportions, not in contour histograms.
- [x] (2026-10-11) **X-G7 CPU proxy** (`scripts/geom_dots.py`): dots survive a plain resize at 64 px and are lost at 32 px.
- [x] (2026-10-11) **X-T1 Geometry transfer** (`scripts/geom_transfer.py`; page `results/geometry/transfer/index.html`): lines of one dataset transformed to the measured look of another (slant, stroke width, blur, tone, paper noise); mean gap to target over twelve measures falls from about 0.75 to 0.25 SD.
- [x] (2026-10-12) **X-T2 Transformed geometry on the CRNN.** KHATT as Muharaf 26.3 → 25.3 % (all from slant); Muharaf made upright 6.2 → 8.2 %; NAKBA stroke thinning 13.6 → 12.3 %; tone, noise and ruled lines no effect.
- [x] (2026-10-12) **X-G8 Slant sweep and confidence-selected TTA.** KHATT 26.3 → 25.4 %; on ink-cropped lines 23.3 → 22.4 % (the gains stack). NAKBA gets about 0.5 points worse, so gate TTA on measured slant.
- [x] (2026-10-12) **X-S1 Ink-band crop before resizing** (`geom_ink_extent.py`, `geom_inkcrop_export.py`). KHATT 26.3 → 23.2 % with no retraining; NAKBA unchanged; training on cropped lines adds nothing; VLMs are not affected by loose crops.
- [x] (2026-10-12) **X-T3 Slant and stroke augmentation in training.** Slant-robust, no gain on unseen sets (KHATT 26.9 %, NAKBA 14.4 %).
- [x] (2026-10-12) **X-G10 Letter-shape novelty** (`xai_letter_shapes.py`). Letter error rises from 1–5 % (typical) to 30–53 % (most novel); KHATT has 21 % unusual letters against 10 % for NAKBA; independent check passed.
- [x] (2026-10-12) **Hamza convention** (`geom_hamza.py`). About 2 points of NAKBA CER are annotators adding hamza (13.8 → 11.6 % folded).
- [x] (2026-10-13) **X-G11 lite Shape-targeted training repeats** (`xai_shape_select.py`, `shape_analyze.py`). Held-out KHATT half B: −0.8 points against a random control [−1.6, +0.1]; replicate drift +0.4. Error in the two most novel bands about 2 points lower.
- [x] (2026-10-13) **X-S2 Stacked fixes.** Ink-cropped training + shape repeats: KHATT 22.3 % cropped (original CRNN cropped 23.2 %), NAKBA unchanged. Combo + slant TTA: KHATT 21.5 % (NAKBA +0.5, so gate TTA).
- [x] (2026-10-12/13) **E8 and E8b CRNN features → Qwen3 LLM decoder.** No-go at 0.6B (KHATT 40.0 %) and at 1.7B (40.1 %); more fluent-but-wrong words. Closed in this form.
- [ ] **X-G11 second seed (3090).** Retrain targeted and random-control with another seed; settles whether the −0.8 on half B is real.
- [ ] **X-S2 second seed and TTA gate (3090/CPU).** Second seed of the combo model; apply slant TTA only to lines whose measured slant is far from training.
- [ ] **X-G3 Normalization ablation on the CRNN (3090).** Height 32/48/64/96, median padding against resize, deslant, binarization; two seeds.
- [ ] **X-G7 Dot-preserving resize at 32 and 48 px (3090).** Not at 64 px; the proxy shows no loss there.
- [ ] **X-A2b, X-A3, X-A6, X-A7, X-D0, X-D1, X-C2** as listed in the ledger; all need the 3090 or a VLM run.

## 2. Runbook for the 3090

> **Launching detached over SSH:** `Start-Process` does not survive the end of the SSH session on this box (Windows OpenSSH ends the session's processes). Use `Invoke-CimMethod -ClassName Win32_Process -MethodName Create`, which starts the process outside the session. Venv: `C:\Users\User\a-experiment\xai\.venv`; `<DIR>` = `C:\Users\User\a-experiment\xai\aligner`; `<RUN>` = `aligner_run`.

Assumptions: the aligner run folder (it holds `charset.json` and `model_final.pt`, from X-B1) and the evaluation
script `xai_eval_aligner.py` are where X-E1 left them. Replace `<RUN>` and `<DIR>` with those paths. Use the same
venv as X-E1. No new downloads are needed.

### Step 0: copy files (from the Mac, with the VPN up)

```
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike
scp results/geometry/transfer_eval.pkl scripts/xai_geom_tta.py scripts/geom_transfer_eval.py scripts/xai_train_aligner.py spring-win-server-3090:C:/Users/User/a-experiment/xai/
```

`transfer_eval.pkl` (about 220 MB) is rebuilt on the Mac with
`.geomvenv/bin/python scripts/geom_transfer_export.py results/geometry/transfer results/geometry/transfer_eval.pkl`.
It holds 183 KHATT, 300 NAKBA and 300 Muharaf test lines, each untouched and in every transformed version, with the same ids, plus single-step ablations (`_only_slant`, `_only_stroke`, `_only_paper`) for KHATT→Muharaf, Muharaf→KHATT and NAKBA→Muharaf. `xai_train_aligner.py` is copied because it gained a `--width` option (default 1.0, the 3090 model unchanged) that the evaluation script imports.

### Step 1: X-T2, CER on original and transformed lines (a few minutes)

```
cd <DIR>
python geom_transfer_eval.py --data transfer_eval.pkl --run <RUN> --ckpt model_final.pt --out transfer_cer.json --device cuda
```

It prints one line per dataset tag and, for each transformed tag, the paired change against the untouched lines with a 95 % bootstrap interval. Detached, with a live view:

```
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c cd /d <DIR> && ..\.venv\Scripts\python.exe -u geom_transfer_eval.py --data transfer_eval.pkl --run <RUN> --ckpt model_final.pt --out transfer_cer.json --device cuda > transfer_cer.log 2>&1'}"
powershell -NoProfile -Command "Get-Content <DIR>\transfer_cer.log -Wait -Tail 20"
```

How to read it (all against the untouched line set with the same ids):

| Comparison | If the geometry carries the gap | If it does not |
|---|---|---|
| `kitab_khatt__as_muharaf` against `kitab_khatt` (26.4 %) | CER drops clearly (say 4 points or more) | CER stays near 26 %: the gap is in letter shapes, writers or text, not in slant, tone or stroke weight |
| `muharaf_lines__as_khatt` against `muharaf_lines` (7.6 %) | CER rises | CER stays: the CRNN is already robust to these changes |
| `nakba_lines__norules` against `nakba_lines` (13.8 %) | CER drops by the ruled-line share | no change: ruled lines are not costing anything |
| the `_only_slant`, `_only_stroke` and `_only_paper` rows | whichever single step reproduces most of the full change is the one that carries the gap | |
| `nakba_lines__as_muharaf` and `nakba_lines__as_khatt` | small change expected (NAKBA is already near Muharaf) | |
| `muharaf_lines__as_nakba` | small rise | |

The contour-shape classifier did not move on the transformed lines, so letter shape is untouched by design. A drop for
KHATT therefore means slant, tone and stroke weight explain that part of the gap, and the remainder is letter shape.

### Step 2: X-G8, slant sweep and confidence selection (detached; about 1 to 2 hours at n = 300)

Try a small run first, in the foreground:

```
cd <DIR>
python xai_geom_tta.py --selftest
python xai_geom_tta.py --data eval_data.pkl --run <RUN> --ckpt model_final.pt --out tta_small.json --n 30
```

The model call in this script was copied from `xai_eval_aligner.py` and not run, so expect to fix a small error
at this stage. Then the full run:

```
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c cd /d <DIR> && ..\.venv\Scripts\python.exe -u xai_geom_tta.py --data eval_data.pkl --run <RUN> --ckpt model_final.pt --out tta_out.json --n 300 > tta.log 2>&1'}"
powershell -NoProfile -Command "Get-Content <DIR>\tta.log -Wait -Tail 20"
```

It writes, per dataset, the CER at each slant (−6, 0, 6, 12, 18 degrees), the CER when the most confident reading is
kept, and the oracle. Prediction: on KHATT the best fixed shear is around +10 to +14 degrees; on Muharaf, 0 to +6.

### Step 3: bring results back

```
scp spring-win-server-3090:C:/Users/User/a-experiment/xai/transfer_cer.json spring-win-server-3090:C:/Users/User/a-experiment/xai/tta_out.json results/geometry/
```

### Step 4: put the result in the ledger

```
.geomvenv/bin/python scripts/build_transfer_section.py results/geometry/transfer --cer results/geometry/transfer_cer.json --verdict results/geometry/transfer/verdict.html
```

Write the verdict paragraph first (HTML, in `verdict.html`), then republish `docs/research/paper-ledger.html` to the same artifact URL.

Note: a half-width replica could not be trained on the 8 GB Mac (swap thrashing, no finished epoch in 19 minutes), so the CER test needs the 3090.

## 3. The 2026-10-13 pipeline: panel, store, standard analyses (plan `PLAN-2026-10-13-analysis-and-next-steps.md`)

**Laptop (CPU), in order:**
1. `.geomvenv/bin/python scripts/xai_prepare_panel.py results/panel/panel.pkl` builds the standard evaluation panel: every line of six test sets, as stored and ink-cropped, canonical ids, no CTC filter (689 MB).
2. `HF_HUB_OFFLINE=1 .venv/bin/python scripts/xai_prepare_aligner_data.py results/panel/aligner_data_standard.pkl --standard-split` builds the standard-split training pickle (Muharaf train → train, validation → held-out).
3. After results come back: `.geomvenv/bin/python scripts/store_build.py`, then `analyze_lines.py`, `analyze_letters.py`, `run_detectors.py` and `build_evidence_section.py`. Publish the ledger.

**Upload is slow (about 0.6 MB/s from the laptop).** Do not upload what the box can derive:
- Standard split on the box: `python xai_derive_standard_split.py aligner_data.pkl aligner_data_standard.pkl`, then check that `python pickle_fingerprint.py` prints the same fingerprint on both machines (`450603de452a8195`).
- Upload the panel gzipped (373 MB), unpack it on the box, and check the fingerprint (`ed1dd4212f13619b`).

**Box queues** (`scripts/box/queue_p0.bat`, `queue_p1.bat`, launched with `Win32_Process Create`; `queue_p1` waits for "P0 ALL DONE"):
- P0: panel scores of the 7 existing CRNNs; standard-split CRNN seeds 0 and 1; seeds for shape, random control and base; panel scores of the new runs.
- P1: TTA with all shears stored (`--all-variants`); letter dumps (`xai_letter_shapes.py --dump-letters`); seeds for shape, random control and combo; panel scores.

Live view: `ssh spring-win-server-3090 "powershell Get-Content C:\Users\User\a-experiment\xai\aligner\queue_p0.log -Wait"` (or `queue_p1.log`).
Bring back: `scp "spring-win-server-3090:C:/Users/User/a-experiment/xai/aligner/panel_out/*.json" results/panel/out/`.

**Backups:** `/Volumes/PortableSSD/aocr-box-backup/xai-2026-10-13/` holds the `xai` folder from 2026-10-13, without `.venv`: weights, data pickles and outputs. After new runs, copy their `aligner_run_*` folders the same way, then delete the large pickles on the box once the queues are done. All of them can be regenerated from the laptop.

## 4. GPU night 2 (2026-10-18): VLM jobs 7.1-7.6, 7.2 and the CRNN item 4.8

Box folder `C:\Users\User\a-experiment\xai\vlm_n2` (VLM jobs), `xai\aligner` (CRNN). One venv (`xai\.venv`: torch 2.6 cu124,
transformers 5.19, peft 0.21.2, qwen_vl_utils) and one HF cache (`C:\Users\User\.cache\huggingface`, the default; HF_HOME unset).

**From scratch, in order (laptop commands from the worktree root):**
1. Base VLMs into the shared cache, detached: `scp scripts/box/n2_fetch_models.bat spring-win-server-3090:C:/Users/User/a-experiment/xai/`,
   then Win32_Process Create with `cmd /c C:\Users\User\a-experiment\xai\n2_fetch_models.bat` (log `vlm_n2\fetch.log`). `hf` 2.x
   keeps a content-addressed blob store (`hub\blobs\<xx>\<sha256>`) shared by all repos, so the 4B, already there, cost nothing
   and the 8B took 16.3 GB and 7 minutes. Check with the 8B's `snapshots\...\model-0000*-of-00004.safetensors` links.
2. Line images: `.geomvenv/bin/python scripts/vlm_n2_export.py results/vlm_n2/vlm_n2_lines.pkl` (205 MB, native resolution,
   benchmark ids; also writes `results/vlm_n2/icl_trials.json`). Needs the cached Arrow files of KITAB-KHATT, NAKBA test and
   blind, RASAM-1/2 test and Muharaf test.
3. The patched benchmark client: zip `src/arabic_ocr_bench` of the main checkout (branch research-characterization with
   `docs/research/patches/0002*` applied) without `__pycache__` as `aocr_pkg.zip`; the queue unpacks it to `vlm_n2\pkg`.
4. Upload `aocr_pkg.zip`, `scripts/vlm_n2.py`, `scripts/queue_n2.bat`, `results/vlm_n2/vlm_n2_lines.pkl` and `icl_trials.json`
   to `xai/vlm_n2/`; `scripts/queue_n2c.bat`, `scripts/xai_shape_select.py` (gained `--dataset`) and `scripts/xai_train_aligner.py`
   (gained `AOCR_CUDA_MEM_FRACTION`) to `xai/aligner/`. Upload ran at 0.1-0.6 MB/s, so the 205 MB pickle takes 6-30 minutes.
5. Laptop check first: `.geomvenv/bin/python scripts/vlm_n2.py selftest --pkg <unzipped pkg> --lines <export --limit 40>
   --trials <its icl_trials.json>` builds a tiny random Qwen3-VL and LoRA and runs gen, reliance and icl end to end
   (the laptop venv needs transformers 5.19, peft 0.21.2 and torchvision; without torchvision the Qwen3-VL processor fails).
6. Launch: Win32_Process Create with `cmd /c C:\Users\User\a-experiment\xai\vlm_n2\queue_n2.bat`. Live view:
   `ssh spring-win-server-3090 "powershell Get-Content C:\Users\User\a-experiment\xai\vlm_n2\queue_n2.log -Wait -Tail 30"`;
   the CRNN stream logs to `xai\aligner\queue_n2c.log`.

**Notes.**
- The suite fine-tuned Qwen3-VL-8B exists only as merged weights (16 GB, `suite-v1-artifacts/qwen3vl8b/merged`, no LoRA kept);
  at the measured upload speed it cannot reach the box in a night. The fine-tuned arm is the 4B diff-LR adapter
  (`xai\adapter`, sha256 d20417e0..., the same file as `suite-v1-artifacts/qwen3vl4b_visft_difflr/runs/.../adapter_model.safetensors`).
  To get the 8B arm: extract a rank-16 LoRA from merged minus base on the laptop, or upload the merged weights over a faster link.
- A CRNN training run holds about 13 GB of GPU memory by the end of an epoch although its peak is about 4.2 GB (caching
  allocator, varying widths). `AOCR_CUDA_MEM_FRACTION=0.30` caps it so a 4B VLM (about 11 GB) runs beside it; the 8B
  (about 19 GB) does not fit beside a CRNN, so queue_n2 waits for "N2C DONE" before loading the 8B.
- `n2_deadline.txt` (queue start + 8 h) stops every VLM job past it; delete it before relaunching on another day.
- Back up and trim afterwards: `scripts/box/backup_n2.sh` (checklist 7.7).
