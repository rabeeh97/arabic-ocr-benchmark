# Attribution and localization: plan

> Drafted 2026-10-10 from the attribution spike (branch `worktree-xai-attribution-spike`).
> Not yet merged into `NEXT_STEPS.md` / `progress.json`; those live on `research-characterization`
> with uncommitted edits, so the task list in §7 must be ported there by hand.

## 1. Questions

1. **Where does an OCR/VLM model look** when it emits a token or a letter? Do the maps land on
   the right glyph, and on the dots?
2. **Can we locate a letter, PAW or word in the image** (text-to-region grounding), and is that
   good enough to score question 1 against a reference?
3. **Do errors have an attribution signature?** (diffuse maps, low confidence, a specific
   letter family, edge or border effects)
4. **Does any of this help training?** (attribution-guided LoRA freezing, protecting weights that
   carry confusable letters). Gated on 1 to 3.

## 2. Status (spike, Qwen3-VL-4B + our E1 LoRA adapter, run on the 3090)

- Pipeline works end to end: greedy decode, per-token maps by attention rollout, raw attention,
  gradient x input and Integrated Gradients, a deletion test with a random baseline, and a
  right-to-left centroid test. Script: `scripts/xai_attribute_qwen3vl.py`.
- Smoke result (one RASAM line, 4 tokens): **raw maps are not trustworthy.**
  IG and gradient x input are speckle over the paper, not the ink (known ViT gradient noise,
  worse in bf16). Rollout concentrates on the scan's black border. Raw attention shows an
  implausibly large deletion effect (9 nats at 10 % masked), most likely attention sinks.
- Fixes added and now being validated (3 lines x 6 tokens): SmoothGrad, relative attention
  (divided by the mean over all generated tokens), an ink-enrichment metric (share of
  attribution on ink pixels / ink share of the image), median-background fill for deletion.
- Staged and waiting on validation: a 20-line letter panel (alef/lam and jim/ha/kha lines
  from RASAM-1 and Muharaf) and a 23-line error panel (15 worst Muharaf lines of this model,
  8 controls; `scripts/xai_export_error_lines.py`, `--select errors`).
- Environment: `C:\Users\User\a-experiment\xai` on the 3090 (torch 2.6 cu124, transformers
  5.19, peft 0.21). The 8 GB base model does not fit the 8 GB Mac.

## 3. Literature map

| Area | What we found | What it means for us |
|---|---|---|
| Text-to-region grounding in VLMs | Q-Mask / TABench (arXiv 2604.00161): ground queried text to its region. PositionOCR (2602.19188): spotter + LLM for positional precision. InstructOCR (2412.15523). | The task is established for Latin and scene text. Nothing found for Arabic character level. |
| NVIDIA LocateAnything-3B (arXiv 2605.27365, `NVlabs/Eagle/Embodied`) | MoonViT-SO-400M + Qwen2.5-3B, parallel box decoding, coords in [0, 1000]; tasks include text grounding, scene-text detection, document layout/OCR localization. Weights: NVIDIA non-commercial research license (plus Qwen Research License). Code: Apache 2.0. | Usable for academic work. Derived weights inherit the restriction. |
| Fine-tuning LocateAnything | Official full fine-tune script (`locany_finetune_magi_stream.py`) and a LoRA recipe (`shell/locate-anything-lora-visual-prompt.sh`: LLM LoRA r=64, backbone frozen, projector trainable). JSONL annotation + recipe JSON. Pins transformers 4.57.1, peft 0.12.0. | LoRA fine-tuning is supported by the authors. Needs its own venv, separate from the attribution one. |
| Alignment without character boxes | CTC transcription alignment (Bullinger letters, 2508.07904); weakly supervised segmentation from transcripts (Chinese HTR, 2207.14801); HMM forced alignment (known to misplace boundaries). | A trained CTC model gives per-character frame spans for free. No Arabic-specific paper found, so we would be doing this ourselves. |
| Arabic localization data | VML-HD (sub-word boxes + character sequence, 1088-1451 manuscripts), KALIMA (word boxes), a 2024 Zenodo set (line/word/PAW boxes), MADCAT. A 2023 NISS paper states Arabic character-level ground truth is scarce. | No ready character-level reference. VML-HD sub-word boxes are the best independent check on any pseudo-labels. |
| Attribution evaluation | Deletion/insertion are fragile under distribution shift and can contradict each other. Pointing game compares two model-independent objects. Transformer-specific methods (Chefer et al.), metric-driven attribution (MDA, ICLR 2025), structured artifacts in ViT explanations (DAVE, 2026). | Our deletion test needs a fill that stays on-manifold, and must not be the only check. Add ink-enrichment and a reference-based check. Compare at least two methods. |
| Grounding of OCR output | Ancient Greek VLM study (2605.27750): some OCR models produce fluent text with little image reliance. | Attribution on an image-insensitive model is meaningless. Measure image reliance (with vs without the image) before interpreting maps. |

To read in full before design freeze: LocateAnything paper (training section and the box loss),
Q-Mask, the NISS 2023 character-segmentation dataset paper, Chefer et al. (transformer
interpretability), and one of the attribution-evaluation papers.

## 4. Tracks

**A. Validate attribution (running).** Gate: at least one method has ink-enrichment clearly above
1, beats random on deletion with an on-manifold fill, and shows the expected right-to-left
centroid order over a whole line. If none passes, stop and report that raw maps are unusable
on this model.

**B. Reference spans from CTC alignment.** Use our char-CTC model (`a-experiment/charctc` on
the box) to produce character spans for every line. Validate the spans against VML-HD sub-word
boxes on a hand-checked sample. Then score attribution maps by whether a token's mass falls
inside its aligned span (pointing game), with the circularity caveat in §6.

**C. LocateAnything.**
1. Install in a fresh venv on the 3090 (transformers 4.57.1, peft 0.12.0, trust_remote_code).
   Check VRAM on our line sizes (their 4K-image probe used 11.7 GB with `la_flash`, 35 GB with `sdpa`).
2. Zero-shot probe on 30 lines: text grounding of a word, a PAW, a single letter, and
   "the dots under the letter", scored against B's spans.
3. If word-level works, LoRA fine-tune on pseudo-labels (word and PAW boxes from B), starting at
   their recipe (LLM LoRA r=64, projector trainable). Hold out whole manuscripts/writers.
4. Only then try letter-level. Report IoU and the share of letters whose box centre lies in the
   CTC span.

**D. Attribution-guided training (deferred).** Parameter-level attribution (per-letter Fisher
or gradient information on the LoRA matrices, plus layer/head ablation) and a freeze-sweep.
Start only if A passes and B/C give a trustworthy reference. Expected gain is small; main
output is a negative or positive answer on whether letter-specific circuits exist.

## 5. Where the work lands

- Explorer page (private artifact) with overlays, metrics and the error panel.
- Atlas section after the explorer is reviewed, rebuilt from the research branch.
- Findings entry in `docs/experiments/findings-and-notes.md` after A completes.

## 6. Risks

- **Circular evaluation.** CTC spans come from one model, attributions from another. They are
  independent of each other, which is the point, but both can be wrong in the same direction on
  cursive joins. Check spans against VML-HD boxes and by hand.
- **Letters are not objects.** Initial, medial and final forms differ, and dots are separate
  marks. Start at PAW and word level, treat dots as their own target.
- **Low-resolution data.** The public Muharaf copy is stored at 60 px height; maps on it will be
  coarse whatever the method.
- **Licence.** LocateAnything weights are non-commercial research; anything fine-tuned from
  them is too. Record this next to any released artefact.
- **Compute.** One 24 GB GPU; the attribution runs and LocateAnything experiments share it.

## 7. Tasks to port into `NEXT_STEPS.md`

- [ ] **X-A1** Validate attribution methods (running; gate in §4A).
- [ ] **X-A2** Letter panel and error panel with the methods that pass; error-vs-correct token
  comparison (concentration, ink enrichment, model confidence).
- [ ] **X-A3** Image-reliance check (with vs without image) per model before reading maps.
- [ ] **X-B1** CTC character spans for the panel lines; validate on VML-HD sub-word boxes.
- [ ] **X-B2** Pointing-game score of each attribution method against B1 spans.
- [ ] **X-C1** LocateAnything environment and zero-shot probe.
- [ ] **X-C2** LoRA fine-tune on word/PAW pseudo-labels; letter level if C1/C2 justify it.
- [ ] **X-D1** Parameter-level attribution and freeze-sweep (gated).
- [ ] **S1** Script-styles inventory (separate design; see conversation 2026-10-10).
