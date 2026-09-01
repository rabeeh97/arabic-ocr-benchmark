# Local RTX 3090 training runbook

The 3090 (24 GB) is the secondary trainer — use it for what a free Colab T4
can't do: bf16 LoRA on 2B/3B, QLoRA on 7B (Qwen2.5-VL-7B, AIN-7B), and full
fine-tunes of a 2B model. QLoRA on 2B/3B should still go on Colab (notebook `05`)
for reproducibility.

## Setup (once)

```bash
uv sync --extra local --extra train
export HF_HOME=/path/on/big/disk/hf_cache        # weights are large; keep off /
huggingface-cli login                            # for Muharaf + gated bases
```

`bitsandbytes` (QLoRA) needs a working CUDA build of torch — verify:

```bash
uv run python -c "import torch, bitsandbytes; print(torch.cuda.get_device_name(0))"
```

## Run

```bash
CFG=configs/training/<exp>.yaml

uv run bench-train check-contam --config $CFG        # exits 1 if flagged
uv run bench-train sft          --config $CFG        # -> results/training/<exp>/
uv run bench-train merge  --adapter results/training/<exp> --out results/training/<exp>-merged
uv run bench-train register --name <exp>-ft --path results/training/<exp>-merged --arch qwen2_vl
```

Then score on the untouched test split via the normal harness:

```bash
uv run bench run --config configs/training/<exp>.run.yaml --report
# a run.yaml listing models: [<exp>-ft], datasets: [<held-out test dataset>]
```

## VRAM guidance (24 GB, batch 1 + grad-accum, gradient checkpointing on)

| Setup | Fits | Notes |
|---|---|---|
| QLoRA 2B / 3B | comfortably | fast; prefer Colab for these |
| bf16 LoRA 2B / 3B | yes | ~16–20 GB; the quality-vs-QLoRA reference |
| QLoRA 7B | yes | ~18–22 GB; keep `max_seq_len` ≤ 1024 |
| full FT 2B | tight | `bf16`, gradient checkpointing, 8-bit optimizer; drop `max_pixels` if OOM |
| full FT 3B / any 7B non-quantized | no | out of scope on one 3090 |

If you OOM: lower `image.max_pixels`, then `train.max_seq_len`, then switch
`method: lora` → `qlora`.

## Checkpointing / resume (local)

Same mechanism as Colab: `sft` writes a checkpoint to `output_dir` every epoch
(or every `train.save_steps` steps with `train.save_strategy: steps`), keeps the
last `train.save_total_limit`, and re-running `bench-train sft` with
`train.resume: true` (the default) continues from the newest `checkpoint-*`.
Point `output_dir` at a stable disk. Set `train.push_to_hub: true` +
`train.hub_model_id` to also mirror checkpoints to the HF Hub.

## Method knobs (`configs/training/*.yaml`)

- `method`: `qlora` | `lora` | `full` | `trocr`
- `lora.r` / `lora.alpha` / `lora.target_modules` — rank/module ablations
- `lora.train_vision: true` — also adapt the ViT (degraded historical script)
- `image.min_pixels` / `image.max_pixels` — the OCR-specific resolution lever
- `base_model.adapter_id` — continue-training from an existing adapter
  (e.g. curriculum: synthetic warmup → real data)
