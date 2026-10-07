"""Config loader reuses the training leak guard."""

from __future__ import annotations

from pathlib import Path

from arabic_ocr_bench.charctc.config import load_charctc_config

REPO = Path(__file__).resolve().parents[1]


def test_cpu_yaml_loads_and_keeps_muharaf_train_limit():
    cfg = load_charctc_config(REPO / "configs/training/charctc/cpu.yaml")
    assert cfg.experiment == "charctc-cpu"
    assert cfg.data.sources[0].split == "train"
    assert cfg.data.sources[0].limit == 8000
    assert cfg.train.batch_size == 4
    assert "nakba_lines" in cfg.eval_datasets
    assert cfg.train.num_workers == 0
    assert cfg.train.epochs >= 3
    assert cfg.train.blank_penalty > 0
