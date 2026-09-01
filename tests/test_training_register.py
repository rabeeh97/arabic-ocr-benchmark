"""A trained model re-enters the benchmark as a plain models.yaml hf_local entry."""

from __future__ import annotations

import yaml

from arabic_ocr_bench.training.registry import build_model_entry, register_model


def test_merged_model_entry_points_hf_id_at_the_local_path():
    entry = build_model_entry(
        "smoke-ft", "results/training/smoke-merged", arch="qwen2_vl", card="x"
    )
    assert entry["client"] == "hf_local"
    assert entry["hf_id"] == "results/training/smoke-merged"
    assert entry["arch"] == "qwen2_vl"
    assert "adapter_id" not in entry


def test_adapter_model_entry_keeps_base_and_adapter():
    entry = build_model_entry(
        "smoke-ft-lora",
        "Qwen/Qwen2-VL-2B-Instruct",
        arch="qwen2_vl",
        adapter_id="results/training/smoke",
        card="x",
    )
    assert entry["hf_id"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert entry["adapter_id"] == "results/training/smoke"


def test_register_model_appends_without_touching_existing_entries(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("mock_echo:\n  client: mock\n  mode: echo_gt\n")
    register_model(
        "smoke-ft", build_model_entry("smoke-ft", "/m", arch="qwen2_vl", card="x"), path=path
    )
    doc = yaml.safe_load(path.read_text())
    assert doc["mock_echo"]["mode"] == "echo_gt"
    assert doc["smoke-ft"]["hf_id"] == "/m"
