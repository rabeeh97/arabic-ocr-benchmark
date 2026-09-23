"""LoRA / QLoRA / full SFT for the Qwen2-VL-family OCR backbones.

Runs on a Colab T4 (QLoRA 2B/3B) or the RTX 3090 (bf16 LoRA, 7B QLoRA, full 2B).
Heavy deps (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`) are imported
lazily so the core benchmark environment stays light and importing this module
never fails.

Recipe (current OCR best practice): freeze the vision tower + merger, LoRA the
LLM attention + MLP projections, 4-bit nf4 double-quant base for QLoRA,
conversational format with the ground truth as the assistant turn, labels masked
everywhere except the assistant tokens.
"""

from __future__ import annotations

from pathlib import Path

from .checkpoints import resume_from
from .config import TrainingConfig, hub_trainer_kwargs
from .data import assemble_dataset

_ARCH_MODEL_CLS = {
    "qwen2_vl": ("transformers", "Qwen2VLForConditionalGeneration"),
    "qwen2_5_vl": ("transformers", "Qwen2_5_VLForConditionalGeneration"),
    # Qwen3-VL shares the Qwen chat template / <|image_pad|> tokens / qwen_vl_utils
    # pipeline with 2.5-VL, so only the model class differs. Needs transformers>=4.57.
    "qwen3_vl": ("transformers", "Qwen3VLForConditionalGeneration"),
    # PaliGemma 1 & 2 — different processor + collator (see _make_collator_paligemma).
    "paligemma": ("transformers", "PaliGemmaForConditionalGeneration"),
}

# arches that use the Qwen chat-template collator + qwen_vl_utils vision pipeline
_QWEN_FAMILY = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}


def _load_processor(cfg: TrainingConfig):
    from transformers import AutoProcessor

    if cfg.base_model.arch == "paligemma":
        # PaliGemma's processor has a fixed image size baked into the checkpoint;
        # it rejects the Qwen min_pixels / max_pixels kwargs.
        return AutoProcessor.from_pretrained(cfg.base_model.hf_id)
    return AutoProcessor.from_pretrained(
        cfg.base_model.hf_id,
        min_pixels=cfg.image.min_pixels,
        max_pixels=cfg.image.max_pixels,
    )


def _load_model(cfg: TrainingConfig):
    import torch

    mod_name, cls_name = _ARCH_MODEL_CLS.get(
        cfg.base_model.arch, _ARCH_MODEL_CLS["qwen2_vl"]
    )
    model_cls = getattr(__import__(mod_name, fromlist=[cls_name]), cls_name)

    kwargs: dict = {"torch_dtype": torch.bfloat16 if cfg.train.bf16 else torch.float16}
    if cfg.method == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    kwargs["device_map"] = "auto"

    model = model_cls.from_pretrained(cfg.base_model.hf_id, **kwargs)

    if cfg.base_model.adapter_id:  # continue-training from an existing adapter
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.base_model.adapter_id, is_trainable=True)
        return model

    if cfg.method in ("qlora", "lora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.method == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
            )
        targets = list(cfg.lora.target_modules)
        if cfg.lora.train_vision:
            targets += ["qkv", "proj"]  # Qwen2-VL ViT block projection names
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=targets,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    else:  # full fine-tune
        if not cfg.lora.train_vision and hasattr(model, "visual"):
            model.visual.requires_grad_(False)

    return model


def _make_collator(processor, max_seq_len: int):
    """Chat-template + vision-process a batch of rows, mask non-assistant labels."""
    from qwen_vl_utils import process_vision_info

    img_token_ids = {
        getattr(processor.tokenizer, "convert_tokens_to_ids", lambda _t: None)(t)
        for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
    }
    img_token_ids.discard(None)

    def collate(rows: list[dict]):
        texts, image_inputs = [], []
        for row in rows:
            texts.append(
                processor.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
            )
            # rebuild message list with the real PIL image spliced in for the util
            msgs = []
            imgs = iter(row["images"])
            for m in row["messages"]:
                content = [
                    dict(c, image=next(imgs)) if c.get("type") == "image" else c
                    for c in m["content"]
                ]
                msgs.append({"role": m["role"], "content": content})
            vi, _ = process_vision_info(msgs)
            image_inputs.append(vi)

        batch = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tid in img_token_ids:
            labels[labels == tid] = -100
        # mask everything before the assistant turn
        assistant_open = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        for i in range(labels.size(0)):
            ids = batch["input_ids"][i].tolist()
            cut = _find_subseq(ids, assistant_open)
            if cut is not None:
                labels[i, : cut + len(assistant_open)] = -100
        batch["labels"] = labels
        return batch

    return collate


def _find_subseq(seq: list[int], sub: list[int]) -> int | None:
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return None


# PaliGemma OCR prefix — short task tag, not the full line-v2 instruction (the
# SigLIP/Gemma stack does better with a terse prefix). Tune per experiment.
_PALIGEMMA_PREFIX = "transcribe the arabic text in this image"


_PALIGEMMA_IMAGE_TOKENS = 1024  # paligemma2-*-448: 448/14 patch size -> 32x32


def _make_collator_paligemma(processor, max_seq_len: int):
    """PaliGemma 2 SFT collator (HF `paligemma2` recipe).

    Each row from `assemble_dataset` carries a chat `messages` list + one PIL
    image; PaliGemma wants a flat (prefix, image, suffix) triple. Passing
    `suffix=` lets the processor build `input_ids` = prefix+<image>+suffix AND
    the label mask (everything but the suffix tokens set to -100), so no manual
    masking here. Verify the first few batches on the box — `labels` should be
    all -100 except the ground-truth span.
    """
    if max_seq_len < _PALIGEMMA_IMAGE_TOKENS + 64:
        raise ValueError(
            f"max_seq_len={max_seq_len} leaves no room for text after the "
            f"{_PALIGEMMA_IMAGE_TOKENS} image tokens PaliGemma prepends to every "
            "example — `truncation=True` below would cut into the image-token "
            "block itself (silently breaking the image, not just clipping "
            "labels). Raise train.max_seq_len in the config to at least "
            f"{_PALIGEMMA_IMAGE_TOKENS + 64}."
        )

    # The processor otherwise infers the image-token count itself and warns on
    # every single call ("You are passing both `text` and `images`...").
    # Prepending the exact count ourselves gives it what it expects up front.
    image_seq_length = getattr(processor, "image_seq_length", _PALIGEMMA_IMAGE_TOKENS)
    prefix_with_image_tokens = "<image>" * image_seq_length + _PALIGEMMA_PREFIX

    def _target(row: dict) -> str:
        for m in reversed(row["messages"]):
            if m["role"] == "assistant":
                return "".join(
                    c.get("text", "") for c in m["content"] if c.get("type") == "text"
                )
        return ""

    def collate(rows: list[dict]):
        images = [row["images"][0] for row in rows]
        prefixes = [prefix_with_image_tokens] * len(rows)
        suffixes = [_target(row) for row in rows]
        batch = processor(
            text=prefixes,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
        )
        return batch

    return collate


def _differential_lr_optimizer(model, cfg: TrainingConfig, args):
    """Build a two-param-group AdamW when `train.vision_lr` is set, so the
    (LoRA-adapted) vision tower trains at its own, usually much smaller, LR
    than the rest of the model — mirrors Baseer-Nakba's disclosed recipe
    (decoder 1e-4 / vision encoder 9e-6). Returns None to fall back to
    SFTTrainer's normal single-LR optimizer when `vision_lr` isn't set, or
    when `lora.train_vision` is off (there's no vision-tower LoRA params to
    split out).

    Vision-tower params are identified by name containing ".visual." — the
    attribute `_load_model` already checks (`hasattr(model, "visual")`) for
    the Qwen2-VL/Qwen2.5-VL/Qwen3-VL family PaliGemma isn't run through this
    path since `_load_model` freezes/unfreezes it differently and this repo
    hasn't unfrozen it with a differential LR yet.
    """
    if cfg.train.vision_lr is None or not cfg.lora.train_vision:
        return None
    import torch

    def _is_vision_param(name: str) -> bool:
        # PEFT wraps the model (e.g. "base_model.model.visual.blocks...."), so
        # "visual" is usually preceded by a dot — but check the unprefixed,
        # top-level case too ("visual.blocks....") rather than assume wrapping.
        return ".visual." in name or name.startswith("visual.")

    vision_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (vision_params if _is_vision_param(name) else other_params).append(p)
    if not vision_params:
        raise ValueError(
            "train.vision_lr is set but no trainable parameter name contains "
            "'.visual.' — lora.train_vision may not have produced vision-tower "
            "adapters for this base_model.arch, or the vision-tower attribute "
            "name differs from the Qwen2-VL family's 'visual'."
        )
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.train.lr},
            {"params": vision_params, "lr": cfg.train.vision_lr},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )


def train(cfg: TrainingConfig) -> Path:
    """Run SFT and write the adapter (or full weights) to ``cfg.output_dir``."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    processor = _load_processor(cfg)
    model = _load_model(cfg)

    bundle = assemble_dataset(cfg.data, seed=cfg.train.seed)
    train_ds = Dataset.from_list(bundle["train"])
    eval_ds = Dataset.from_list(bundle["val"]) if bundle["val"] else None

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resume_ckpt = resume_from(out, resume=cfg.train.resume)
    if resume_ckpt:
        print(f"[sft_vlm] resuming from {resume_ckpt}")

    args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.scheduler,
        warmup_ratio=cfg.train.warmup_ratio,
        bf16=cfg.train.bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        logging_steps=10,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        eval_strategy="epoch" if eval_ds is not None else "no",
        seed=cfg.train.seed,
        report_to=[],
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        **hub_trainer_kwargs(cfg),
    )
    collator = (
        _make_collator_paligemma(processor, cfg.train.max_seq_len)
        if cfg.base_model.arch == "paligemma"
        else _make_collator(processor, cfg.train.max_seq_len)
    )
    optimizer = _differential_lr_optimizer(model, cfg, args)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    (out / "training_config.yaml").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    if cfg.train.push_to_hub:
        trainer.push_to_hub()  # final adapter + card
    return out
