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
- [ ] **X-T2 Does the transformed geometry carry the transfer gap? (3090)** Run the CRNN on the original and transformed lines (section 2, step 1). *Done when:* CER for `kitab_khatt` against `kitab_khatt__as_muharaf`, `muharaf_lines` against `muharaf_lines__as_khatt`, and `nakba_lines` against `nakba_lines__norules`, with paired per-line differences.
- [ ] **X-G3 Normalization ablation on the CRNN (3090).** Height 32/48/64/96, median padding against resize, deslant, binarization; two seeds.
- [ ] **X-G8 Slant sweep and confidence-selected TTA (3090).** `scripts/xai_geom_tta.py` (section 2, step 2).
- [ ] **X-G7 Dot-preserving resize at 32 and 48 px (3090).** Not at 64 px; the proxy shows no loss there.
- [ ] **X-A2b, X-A3, X-A6, X-A7, X-D0, X-D1, X-C2, E8** as listed in the ledger; all need the 3090 or a VLM run.

## 2. Runbook for the 3090

Assumptions: the aligner run folder (it holds `charset.json` and `model_final.pt`, from X-B1) and the evaluation
script `xai_eval_aligner.py` are where X-E1 left them. Replace `<RUN>` and `<DIR>` with those paths. Use the same
venv as X-E1. No new downloads are needed.

### Step 0: copy files (from the Mac, with the VPN up)

```
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike
scp results/geometry/transfer_eval.pkl scripts/xai_geom_tta.py spring-win-server-3090:C:/Users/User/a-experiment/xai/
```

`transfer_eval.pkl` (107 MB) is rebuilt on the Mac with
`.geomvenv/bin/python scripts/geom_transfer_export.py results/geometry/transfer results/geometry/transfer_eval.pkl`.
It holds 183 KHATT, 300 NAKBA and 300 Muharaf test lines, each untouched and in every transformed version, with the same ids.

### Step 1: X-T2, CER on original and transformed lines (a few minutes)

```
cd <DIR>
python xai_eval_aligner.py --data transfer_eval.pkl --run <RUN> --ckpt model_final.pt --out transfer_cer.json
```

It prints one line per dataset tag. Detached, with a live view:

```
powershell -NoProfile -Command "Start-Process -NoNewWindow python -ArgumentList 'xai_eval_aligner.py --data transfer_eval.pkl --run <RUN> --ckpt model_final.pt --out transfer_cer.json' -RedirectStandardOutput transfer_cer.log"
powershell -NoProfile -Command "Get-Content <DIR>\transfer_cer.log -Wait -Tail 20"
```

How to read it (all against the untouched line set with the same ids):

| Comparison | If the geometry carries the gap | If it does not |
|---|---|---|
| `kitab_khatt__as_muharaf` against `kitab_khatt` (26.4 %) | CER drops clearly (say 4 points or more) | CER stays near 26 %: the gap is in letter shapes, writers or text, not in slant, tone or stroke weight |
| `muharaf_lines__as_khatt` against `muharaf_lines` (7.6 %) | CER rises | CER stays: the CRNN is already robust to these changes |
| `nakba_lines__norules` against `nakba_lines` (13.8 %) | CER drops by the ruled-line share | no change: ruled lines are not costing anything |
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
powershell -NoProfile -Command "Start-Process -NoNewWindow python -ArgumentList 'xai_geom_tta.py --data eval_data.pkl --run <RUN> --ckpt model_final.pt --out tta_out.json --n 300' -RedirectStandardOutput tta.log"
powershell -NoProfile -Command "Get-Content <DIR>\tta.log -Wait -Tail 20"
```

It writes, per dataset, the CER at each slant (−6, 0, 6, 12, 18 degrees), the CER when the most confident reading is
kept, and the oracle. Prediction: on KHATT the best fixed shear is around +10 to +14 degrees; on Muharaf, 0 to +6.

### Step 3: bring results back

```
scp spring-win-server-3090:C:/Users/User/a-experiment/xai/transfer_cer.json spring-win-server-3090:C:/Users/User/a-experiment/xai/tta_out.json results/geometry/
```
