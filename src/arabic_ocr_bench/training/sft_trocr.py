"""Full fine-tune of a VisionEncoderDecoder (TrOCR-style) OCR model.

The classical-architecture reference the benchmark wants to compare LLM/VLM
fine-tunes against (plan milestone E3.6). Small enough (~0.6B) to train on a
Colab T4. Heavy deps imported lazily.
"""

from __future__ import annotations

from pathlib import Path

from .checkpoints import resume_from
from .config import TrainingConfig, hub_trainer_kwargs
from .data import assemble_dataset


def _target_texts(rows: list[dict]) -> list[str]:
    return [r["messages"][2]["content"][0]["text"] for r in rows]


def train(cfg: TrainingConfig) -> Path:
    import evaluate
    import torch
    from datasets import Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    processor = TrOCRProcessor.from_pretrained(cfg.base_model.hf_id)
    model = VisionEncoderDecoderModel.from_pretrained(
        cfg.base_model.hf_id,
        torch_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
    )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    bundle = assemble_dataset(cfg.data, seed=cfg.train.seed)

    def _encode(rows: list[dict]) -> Dataset:
        images = [r["images"][0].convert("RGB") for r in rows]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        labels = processor.tokenizer(
            _target_texts(rows),
            padding="max_length",
            truncation=True,
            max_length=cfg.train.max_seq_len,
            return_tensors="pt",
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return Dataset.from_dict(
            {"pixel_values": list(pixel_values), "labels": list(labels)}
        ).with_format("torch")

    train_ds, eval_ds = _encode(bundle["train"]), (
        _encode(bundle["val"]) if bundle["val"] else None
    )
    cer = evaluate.load("cer")

    def _metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer.compute(predictions=pred_str, references=label_str)}

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resume_ckpt = resume_from(out, resume=cfg.train.resume)
    if resume_ckpt:
        print(f"[sft_trocr] resuming from {resume_ckpt}")

    args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.scheduler,
        warmup_ratio=cfg.train.warmup_ratio,
        bf16=cfg.train.bf16,
        predict_with_generate=True,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        logging_steps=10,
        seed=cfg.train.seed,
        report_to=[],
        **hub_trainer_kwargs(cfg),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=None,
        compute_metrics=_metrics if eval_ds is not None else None,
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    (out / "training_config.yaml").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    if cfg.train.push_to_hub:
        trainer.push_to_hub()
    return out
