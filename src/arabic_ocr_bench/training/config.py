"""Typed schema for a training experiment (`configs/training/*.yaml`).

Loading a config is also the point where the no-leak guard fires: every data
source is checked against the scored-split set before any weights are touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .data import check_source_not_scored

Method = Literal["qlora", "lora", "full", "trocr"]
Arch = Literal["qwen2_vl", "qwen2_5_vl", "qwen3_vl", "trocr", "paligemma", "minicpmv"]


class BaseModelSpec(BaseModel):
    hf_id: str
    arch: Arch
    adapter_id: str | None = None  # continue-training from an existing adapter


class LoraSpec(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    train_vision: bool = False  # unfreeze / LoRA the vision tower too


class SourceSpec(BaseModel):
    name: str
    adapter: str  # "pkg.mod:Class"
    hf_id: str | None = None
    split: str | None = None
    subset: str | None = None
    params: dict = Field(default_factory=dict)
    limit: int | None = None


class DataSpec(BaseModel):
    sources: list[SourceSpec]
    val_fraction: float = 0.05
    prompt_version: str = "line-v2"


class ImageSpec(BaseModel):
    min_pixels: int = 200_704  # 448*448
    max_pixels: int = 1_003_520  # ~1120*896


class TrainSpec(BaseModel):
    epochs: float = 1.0
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 8
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    seed: int = 0
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_seq_len: int = 1024

    # --- checkpointing / disconnect resilience (Colab) ----------------------
    # A checkpoint is written to output_dir (put it on Google Drive so it
    # survives a Colab recycle) every epoch, or every `save_steps` steps when
    # `save_strategy: steps`. Re-running with `resume: true` picks up the last
    # one. `push_to_hub` mirrors every checkpoint to a private HF model repo as
    # a second backup.
    save_strategy: str = "epoch"        # "epoch" | "steps" | "no"
    save_steps: int = 200              # used when save_strategy == "steps"
    save_total_limit: int = 3         # keep only the last N local checkpoints
    resume: bool = True               # resume from the last checkpoint if present
    push_to_hub: bool = False
    hub_model_id: str | None = None   # e.g. "rabeeh97/arabic-ocr-<experiment>"
    hub_private: bool = True
    hub_strategy: str = "every_save"  # push on every checkpoint save


class TrainingConfig(BaseModel):
    experiment: str
    method: Method
    base_model: BaseModelSpec
    data: DataSpec
    lora: LoraSpec = Field(default_factory=LoraSpec)
    image: ImageSpec = Field(default_factory=ImageSpec)
    train: TrainSpec = Field(default_factory=TrainSpec)
    output_dir: str
    eval_datasets: list[str] = Field(default_factory=list)


def hub_trainer_kwargs(cfg: TrainingConfig) -> dict:
    """HF Trainer args that mirror every checkpoint to a private HF model repo.

    Empty when `train.push_to_hub` is off. Shared by the VLM and TrOCR trainers.
    """
    if not cfg.train.push_to_hub:
        return {}
    return {
        "push_to_hub": True,
        "hub_model_id": cfg.train.hub_model_id or f"arabic-ocr-{cfg.experiment}",
        "hub_private_repo": cfg.train.hub_private,
        "hub_strategy": cfg.train.hub_strategy,
    }


def load_training_config(
    path: str | Path, *, datasets_cfg: str | Path = "configs/datasets.yaml"
) -> TrainingConfig:
    cfg = TrainingConfig.model_validate(yaml.safe_load(Path(path).read_text()))
    for src in cfg.data.sources:
        if src.hf_id and src.split:
            check_source_not_scored(src.hf_id, src.split, cfg_path=datasets_cfg)
    return cfg
