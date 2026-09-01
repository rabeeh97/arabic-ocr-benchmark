# Training / fine-tuning plan

Living doc for the fine-tuning program (project plan "Phase 5"). The benchmark
stays a pure evaluation harness; everything here lives under
`src/arabic_ocr_bench/training/` and re-enters the benchmark only as ordinary
`configs/models.yaml` entries.

## Goal

Test the project hypothesis — *"cheap fine-tuning can close the gap between
general VLMs and cursive-Arabic HTR"* — with measurable CER reduction on the
**untouched** benchmark test splits. Baselines: `qari-ocr-0.2` at 0.78 CER·light
on Muharaf; best partial KHATT ~0.35.

## Rules (pragmatic-but-honest rigor)

- Train only on officially reserved splits (Muharaf `train`/`validation`),
  original-corpus train splits, synthetic data, or new in-house data.
- **Never** train on a `(hf_id, split)` the benchmark scores on. Enforced in
  code: `training/data.py::check_source_not_scored` derives the forbidden set
  from `configs/datasets.yaml` and fires at config-load time.
- Every training source is run through `training/contamination.py` (perceptual
  hash + normalized-text overlap) against every eval test split;
  `results/contamination_report.json` is committed and cited per experiment.
- Report CER·light micro on the held-out test split as the headline, alongside
  CER raw / diac-insensitive / diacritic-F1 / WER / exact, plus training cost.

## Hardware

QLoRA on a 2B/3B VLM is the workhorse (free Colab T4, notebook `05`). The
RTX 3090 (24 GB) is for bf16 LoRA, 7B QLoRA, and full 2B fine-tunes
(`docs/train_3090.md`).

## Disconnect resilience (Colab)

Notebook `05` is built to survive Colab drops:

- **Checkpoints + base-model cache live on Google Drive.** `05` sets
  `cfg.output_dir` to `/content/drive/MyDrive/arabic-ocr-bench/training/<exp>`
  and `HF_HOME` to Drive, so a reconnect re-uses both — no re-download, no lost
  progress.
- **Every checkpoint is mirrored to a private HF model repo**
  (`train.push_to_hub`, `hub_model_id`, `hub_strategy: every_save`) as a second
  backup and the reproducible way to reference the adapter later.
- **Resume is automatic.** `training/checkpoints.py::latest_checkpoint` finds the
  newest `checkpoint-<step>` under `output_dir`; the trainers pass it to
  `Trainer.train(resume_from_checkpoint=...)`. Re-running the notebook (Run all)
  continues where it stopped — every cell is idempotent.
- **Only results go to git.** `predictions.jsonl` / `scores` / `summary.md` /
  `leaderboard.*` / `contamination_report.json` are committed; weights never are
  (`results/training/` is git-ignored).
- For long runs set `train.save_strategy: steps` + `train.save_steps` so a
  checkpoint lands well before the ~12 h Colab cap.

## Subsystem map

| Piece | File | Runs where |
|---|---|---|
| No-leak guard + data assembly | `training/data.py` | core env |
| Contamination check | `training/contamination.py` | core env |
| Experiment schema + loader | `training/config.py`, `configs/training/*.yaml` | core env |
| VLM SFT (QLoRA/LoRA/full) | `training/sft_vlm.py` | T4 / 3090 |
| Encoder-decoder SFT | `training/sft_trocr.py` | T4 / 3090 |
| LoRA merge / export | `training/merge.py` | T4 / 3090 |
| Register into models.yaml | `training/registry.py` | core env |
| CLI | `training/cli.py` → `bench-train` | — |
| Colab: train | `notebooks/05_colab_train_lora.ipynb` | T4 |
| Colab: data prep / vet | `notebooks/06_colab_prepare_data.ipynb` | CPU |

## Milestones (easiest win first)

- **M0 — infra bring-up.** Build the subsystem; smoke experiment
  (`configs/training/smoke.yaml`) all the way through to a leaderboard row.
- **M1 — printed degraded.** Synthetic pipeline v2 (fonts + degradation) →
  QLoRA Qwen2.5-VL-3B → eval `kitab_historicalbooks` / `kitab_hindawi`; add an
  archive.org gold set.
- **M2 — KHATT modern handwriting.** Contamination-check the original KHATT
  corpus vs `kitab_khatt`; QLoRA Qari v0.3 and Qwen2.5-VL-3B; resolution + LoRA
  ablations; 3090 full-FT ceiling.
- **M3 — Muharaf historical (flagship).** Data-scaling curve
  (250→22k lines, 3 seeds); method matrix (QLoRA / bf16 LoRA / full / +ViT);
  base-model comparison; synthetic→real curriculum; diacritic-supervision
  ablation; TrOCR baseline; hosted OpenAI/Vertex FT; repetition-collapse check.
- **M4 — generalist + serving.** Union fine-tune + cross-transfer matrix;
  merge+quantize the best model, measure latency vs quality, permanent
  leaderboard entry.
- **M5 — data flywheel.** Ongoing archive.org ingest → weak-label → hand-correct
  → contamination-check → fold into the next round.

See `docs/experiments/` for per-run logs (`_TEMPLATE.md`).

## Typical loop

```bash
bench-train assemble     --config configs/training/<exp>.yaml   # sanity: counts
bench-train check-contam --config configs/training/<exp>.yaml   # exits 1 if flagged
bench-train sft          --config configs/training/<exp>.yaml   # T4 / 3090
bench-train merge  --adapter results/training/<exp> --out results/training/<exp>-merged
bench-train register --name <exp>-ft --path results/training/<exp>-merged --arch qwen2_vl
bench run --config <run-cfg with model <exp>-ft, dataset = held-out test> --report
```
