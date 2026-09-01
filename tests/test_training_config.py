"""Training configs load into a typed object and cannot reference a scored split."""

from __future__ import annotations

import pytest

from arabic_ocr_bench.training.config import load_training_config
from arabic_ocr_bench.training.data import ScoredSplitLeakError


def test_loads_the_bundled_smoke_config():
    cfg = load_training_config("configs/training/smoke.yaml")
    assert cfg.method == "qlora"
    assert cfg.base_model.arch == "qwen2_vl"
    assert cfg.data.sources[0].split == "train"
    assert cfg.eval_datasets == ["muharaf_lines"]


def test_rejects_a_config_that_names_a_scored_split(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "experiment: bad\n"
        "method: qlora\n"
        "base_model: {hf_id: Qwen/Qwen2-VL-2B-Instruct, arch: qwen2_vl}\n"
        "data:\n"
        "  sources:\n"
        "    - name: leak\n"
        "      adapter: arabic_ocr_bench.datasets.muharaf_lines:MuharafLines\n"
        "      hf_id: aamijar/muharaf-public\n"
        "      split: test\n"
        "output_dir: results/training/bad\n"
        "eval_datasets: [muharaf_lines]\n"
    )
    with pytest.raises(ScoredSplitLeakError):
        load_training_config(bad)
