"""Typed config for the character-CTC CPU experiment."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from arabic_ocr_bench.training.config import DataSpec
from arabic_ocr_bench.training.data import check_source_not_scored, resolve_source_hf_id


class CharCTCTrainSpec(BaseModel):
    epochs: float = 1.0
    lr: float = 1e-3
    batch_size: int = 4
    seed: int = 0
    save_steps: int = 50
    save_total_limit: int = 6
    resume: bool = True
    img_height: int = 32
    max_width: int = 512
    hidden: int = 128
    log_every: int = 10
    val_every: int = 50
    val_samples: int = 32
    num_workers: int = 0  # Windows + SSH: stay single-process
    blank_penalty: float = 0.5  # subtract from blank logits before CTC + greedy


class CharCTCConfig(BaseModel):
    experiment: str
    data: DataSpec
    train: CharCTCTrainSpec = Field(default_factory=CharCTCTrainSpec)
    eval_datasets: list[str] = Field(default_factory=list)
    output_dir: str = "results/training/charctc-cpu"


def load_charctc_config(
    path: str | Path, *, datasets_cfg: str | Path = "configs/datasets.yaml"
) -> CharCTCConfig:
    cfg = CharCTCConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    for src in cfg.data.sources:
        hf_id = resolve_source_hf_id(src)
        if hf_id and src.split:
            check_source_not_scored(hf_id, src.split, cfg_path=datasets_cfg)
    return cfg
