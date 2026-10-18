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


def _is_visual_param(name: str) -> bool:
    """True for Qwen-VL vision-tower weights (and their LoRA)."""
    parts = name.split(".")
    return "visual" in parts or "vision_tower" in parts or "vision_model" in parts


def _load_processor(cfg: TrainingConfig):
    from arabic_ocr_bench.regex_compat import ensure_regex

    ensure_regex()
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
    from arabic_ocr_bench.regex_compat import ensure_regex

    ensure_regex()
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

    if cfg.base_model.adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, cfg.base_model.adapter_id,
            is_trainable=not cfg.lora.train_vision,
        )
        # A frozen-ViT adapter has no vision LoRA. Merge it into the base so a
        # new LoRA can cover LLM + ViT (`qkv`/`proj`) without fighting PEFT's
        # single-adapter continue-train path.
        if cfg.lora.train_vision:
            model = model.merge_and_unload()
        else:
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


def _make_trainer_cls(cfg: TrainingConfig):
    """SFTTrainer, or a subclass that gives the vision tower its own LR."""
    from trl import SFTTrainer

    vis_lr = cfg.train.vision_lr
    if vis_lr is None:
        return SFTTrainer
    decoder_lr = cfg.train.lr

    class _VisionLrSFTTrainer(SFTTrainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            import torch

            decay_names = set(self.get_decay_parameter_names(self.model))
            vis_decay, vis_nodecay, oth_decay, oth_nodecay = [], [], [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                is_vis = _is_visual_param(name)
                is_decay = name in decay_names
                if is_vis and is_decay:
                    vis_decay.append(param)
                elif is_vis:
                    vis_nodecay.append(param)
                elif is_decay:
                    oth_decay.append(param)
                else:
                    oth_nodecay.append(param)
            wd = self.args.weight_decay
            groups = []
            if oth_decay:
                groups.append({"params": oth_decay, "lr": decoder_lr, "weight_decay": wd})
            if oth_nodecay:
                groups.append({"params": oth_nodecay, "lr": decoder_lr, "weight_decay": 0.0})
            if vis_decay:
                groups.append({"params": vis_decay, "lr": vis_lr, "weight_decay": wd})
            if vis_nodecay:
                groups.append({"params": vis_nodecay, "lr": vis_lr, "weight_decay": 0.0})
            if not groups:
                return super().create_optimizer()
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
            print(
                f"[sft_vlm] differential LR: decoder={decoder_lr:g} "
                f"vision={vis_lr:g} "
                f"(vis {len(vis_decay)+len(vis_nodecay)} tensors, "
                f"other {len(oth_decay)+len(oth_nodecay)})"
            )
            return self.optimizer

    return _VisionLrSFTTrainer


def train(cfg: TrainingConfig) -> Path:
    """Run SFT and write the adapter (or full weights) to ``cfg.output_dir``."""
    from datasets import Dataset
    from trl import SFTConfig

    processor = _load_processor(cfg)
    model = _load_model(cfg)
    trainer_cls = _make_trainer_cls(cfg)

    bundle = assemble_dataset(cfg.data, seed=cfg.train.seed)
    train_ds = Dataset.from_list(bundle["train"])
    eval_ds = Dataset.from_list(bundle["val"]) if bundle["val"] else None

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resume_ckpt = resume_from(out, resume=cfg.train.resume)
    if resume_ckpt:
        print(f"[sft_vlm] resuming from {resume_ckpt}")

    eval_strategy = "no"
    extra: dict = {}
    if cfg.train.max_steps is not None:
        extra["max_steps"] = int(cfg.train.max_steps)
        print(f"[sft_vlm] max_steps={cfg.train.max_steps} (overrides epochs for this run)")
    if eval_ds is not None:
        # load_best_model_at_end requires eval_strategy == save_strategy.
        eval_strategy = (
            cfg.train.save_strategy if cfg.train.load_best_model_at_end else "epoch"
        )
        if cfg.train.load_best_model_at_end:
            extra = {
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
            if cfg.train.save_strategy == "steps":
                extra["eval_steps"] = cfg.train.save_steps
            print(
                "[sft_vlm] load_best_model_at_end: picking the lowest eval_loss "
                f"checkpoint (val split, save/eval every {cfg.train.save_strategy}"
                f"/{cfg.train.save_steps})"
            )

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
        eval_strategy=eval_strategy,
        seed=cfg.train.seed,
        report_to=[],
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        **extra,
        **hub_trainer_kwargs(cfg),
    )
    collator = (
        _make_collator_paligemma(processor, cfg.train.max_seq_len)
        if cfg.base_model.arch == "paligemma"
        else _make_collator(processor, cfg.train.max_seq_len)
    )
    from transformers import TrainerCallback

    class _SaveLastStep(TrainerCallback):
        """``save_strategy=steps`` does not write a checkpoint at train end
        unless the last step is a multiple of ``save_steps``. Force one so an
        epoch boundary (e.g. 1936) is still on disk to score."""

        def on_step_end(self, args, state, control, **kwargs):
            if state.max_steps and state.global_step >= state.max_steps:
                control.should_save = True
            return control

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[_SaveLastStep()],
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    (out / "training_config.yaml").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    if cfg.train.push_to_hub:
        trainer.push_to_hub()  # final adapter + card
    return out
