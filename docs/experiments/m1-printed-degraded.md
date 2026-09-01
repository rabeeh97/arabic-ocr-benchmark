# M1 — Printed degraded Arabic

Plan + running log for milestone M1 (see `docs/training-plan.md`). Target: the
KITAB printed subsets `kitab_historicalbooks` (degraded historical print, high
headroom) and `kitab_hindawi` (clean digitised print, near-0 anchor — must not
regress).

## Decisions (from Q&A)

- **Synthetic corpus:** mix of Arabic Wikipedia + classical texts
  (Shamela/al-Waraq-style) + Hindawi prose. Hindawi overlaps the
  `kitab_hindawi` test set, so Hindawi-derived lines are **filtered** against the
  `kitab_hindawi` / `kitab_historicalbooks` ground truth before rendering
  (normalized-text match, reusing `contamination._match_key`).
- **Base models:** two runs — `Qwen2.5-VL-3B` (stronger general base) and
  `Qari-OCR v0.3` (already Arabic-OCR-tuned, 2B). Compare.
- **Scale:** start ~15k synthetic lines (1-2 h QLoRA on a T4). A proper
  15k/30k/60k scaling curve is a **later** experiment, run like the M3 flagship.
- **archive.org real scans:** the user wants these in the dataset eventually, but
  E1.3 runs **after** the synthetic-only E1.2 baseline.

## Experiments

| ID | What | Eval | Status |
|---|---|---|---|
| E1.1 | Build synthetic pipeline v2 (multi-font render + degradation stack) | — | todo |
| E1.2a | QLoRA `Qwen2.5-VL-3B` on ~15k synthetic printed | historicalbooks, hindawi | todo |
| E1.2b | QLoRA `Qari-OCR v0.3` on the same ~15k | historicalbooks, hindawi | todo |
| E1.3 | + archive.org hand-corrected historical-print gold set, continue-train | historicalbooks | after E1.2 |
| E1.4 | Ablation: synthetic-only vs synthetic+gold vs gold-only | historicalbooks | after E1.3 |
| E1.5 | Data-scaling curve 15k/30k/60k on the better base | historicalbooks | later |

Baseline to beat (partial numbers, `results/leaderboard.json`): no strong
printed-subset baseline recorded yet — first step is to score the two base
models zero-shot on `kitab_historicalbooks` / `kitab_hindawi` for the reference row.

## New code (this milestone)

```
src/arabic_ocr_bench/training/synth.py
  - corpus loaders: wikipedia (wikimedia/wikipedia 20231101.ar, streamed),
    classical (HF classical-Arabic dump), hindawi (HF) — each -> list[str]
  - sentences(pool, n, seed, min_len, max_len): clean, dedup, Arabic-ratio
    filter, deterministic sample
  - drop_test_overlap(sentences, test_datasets): remove lines whose
    contamination._match_key collides with a scored GT
  - fonts(): resolve .ttf files from assets/fonts/ (fallback: bundled Noto Naskh)
  - render_line(text, font, size, pad) -> PIL.Image  (reshaper+bidi, like
    datasets/synthetic.py)
  - degrade(img, rng, level) -> PIL.Image  (blur, resample, gaussian +
    salt/pepper noise, brightness/contrast jitter, small rotation/shear, JPEG
    recompress, paper-texture blend, morphological ink erosion; numpy + PIL only)
  - SyntheticPrinted dataset adapter: yields Sample(id, image, ground_truth,
    track="line", meta), deterministic from seed; optional on-disk image cache
    (point at Drive on Colab) so resume/re-runs are fast

scripts/fetch_fonts.py       # download OFL Arabic fonts into assets/fonts/ (git-ignored)
configs/training/m1_printed_synth_qwen25vl3b.yaml
configs/training/m1_printed_synth_qari.yaml
```

**Fonts.** `training.synth` renders through `arabic_reshaper` + Pillow (no
libraqm), which only works with fonts that carry the legacy *Arabic Presentation
Forms* cmap. Many modern faces (Scheherazade New, Lateef, Aref Ruqaa, Reem Kufi)
dropped that block and render tofu, so `fetch_fonts.py` pulls a curated set — the
**Amiri family** (Regular / Bold / Italic / BoldItalic) — and `synth.fonts()`
additionally probes every face with `fontTools` and skips any that lack the
forms. The bundled **Noto Naskh Arabic** is always included. Net: ~5 usable
faces; the degradation stack supplies most of the visual variety. `assets/fonts/`
is git-ignored. A raqm-based renderer (any font, proper shaping) is a possible
later upgrade.

`SyntheticPrinted` plugs into the existing `assemble_dataset` as a training
`source` (adapter = `arabic_ocr_bench.training.synth:SyntheticPrinted`), so no
trainer changes are needed.

## Config sketch (`m1_printed_synth_qwen25vl3b.yaml`)

```yaml
experiment: m1-printed-synth-qwen25vl3b
method: qlora
base_model: { hf_id: Qwen/Qwen2.5-VL-3B-Instruct, arch: qwen2_5_vl }
data:
  sources:
    - name: synthetic_printed
      adapter: arabic_ocr_bench.training.synth:SyntheticPrinted
      params:
        corpora: [wikipedia, classical, hindawi]
        n: 15000
        degrade_level: medium
        diacritics: mixed
      limit: 15000
  val_fraction: 0.03
  prompt_version: line-v2
train:
  epochs: 1
  save_strategy: steps
  save_steps: 250
  resume: true
  push_to_hub: false   # notebook 05 flips this on with HF_TOKEN
output_dir: results/training/m1-printed-synth-qwen25vl3b
eval_datasets: [kitab_historicalbooks, kitab_hindawi]
```

`m1_printed_synth_qari.yaml` is the same with
`base_model: { hf_id: NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct, arch: qwen2_vl }`
and `experiment: m1-printed-synth-qari`.

## Verification

1. `synth.py` unit tests (TDD): sentence filtering, test-overlap drop, render
   produces a non-blank image, `degrade` is deterministic per seed and changes
   pixels without changing size, `SyntheticPrinted` yields N deterministic
   non-empty samples.
2. `bench-train check-contam --config configs/training/m1_printed_synth_qwen25vl3b.yaml`
   → 0 flagged (synthetic is generated; Hindawi lines already filtered).
3. Colab: notebook `06` renders + caches ~15k lines to Drive, commits the
   contamination report; notebook `05` runs E1.2a/E1.2b, resumable.
4. Result rows for `<exp>-ft` on `kitab_historicalbooks` / `kitab_hindawi`
   appear on the leaderboard; hindawi CER stays ~baseline (no regression),
   historicalbooks CER drops. Log the deltas in the table above.
