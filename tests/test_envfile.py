"""Suite-level experiment.env must fill in per-model dirs that have no copy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _envfile import load_env_file, load_experiment_env


def test_load_env_file_setdefault_does_not_override(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    p = tmp_path / "experiment.env"
    p.write_text("HF_TOKEN=from-file\nOTHER=1\n", encoding="utf-8")
    os.environ["HF_TOKEN"] = "already"
    load_env_file(p)
    assert os.environ["HF_TOKEN"] == "already"
    assert os.environ["OTHER"] == "1"


def test_load_experiment_env_falls_back_to_parent_suite_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)
    suite = tmp_path / "suite-v1"
    model = suite / "qwen3vl8b"
    model.mkdir(parents=True)
    (suite / "experiment.env").write_text("HF_TOKEN=suite-token\n", encoding="utf-8")
    load_experiment_env(model)
    assert os.environ["HF_TOKEN"] == "suite-token"


def test_model_dir_env_wins_over_parent(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    suite = tmp_path / "suite-v1"
    model = suite / "qwen3vl8b"
    model.mkdir(parents=True)
    (suite / "experiment.env").write_text("HF_TOKEN=suite-token\n", encoding="utf-8")
    (model / "experiment.env").write_text("HF_TOKEN=model-token\n", encoding="utf-8")
    load_experiment_env(model)
    assert os.environ["HF_TOKEN"] == "model-token"
