"""A trained model re-enters the benchmark as a plain models.yaml hf_local entry."""

from __future__ import annotations

import yaml

from arabic_ocr_bench.training import cli as training_cli
from arabic_ocr_bench.training.registry import build_model_entry, register_model
from arabic_ocr_bench.training.sft_vlm import _PALIGEMMA_PREFIX


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


def test_load_in_4bit_flag_adds_the_key_only_when_set():
    plain = build_model_entry("ft", "/m", arch="qwen2_5_vl", card="x")
    assert "load_in_4bit" not in plain
    quant = build_model_entry("ft-4bit", "/m", arch="qwen2_5_vl", load_in_4bit=True, card="x")
    assert quant["load_in_4bit"] is True


def test_register_model_appends_without_touching_existing_entries(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("mock_echo:\n  client: mock\n  mode: echo_gt\n")
    register_model(
        "smoke-ft", build_model_entry("smoke-ft", "/m", arch="qwen2_vl", card="x"), path=path
    )
    doc = yaml.safe_load(path.read_text())
    assert doc["mock_echo"]["mode"] == "echo_gt"
    assert doc["smoke-ft"]["hf_id"] == "/m"


def test_prompt_is_omitted_by_default_but_included_when_given():
    assert "prompt" not in build_model_entry("ft", "/m", arch="qwen2_vl", card="x")
    entry = build_model_entry("ft", "/m", arch="paligemma", card="x", prompt="ocr\n")
    assert entry["prompt"] == "ocr\n"


def test_cli_register_auto_fills_the_training_prefix_for_paligemma(tmp_path):
    # A fine-tuned paligemma checkpoint must be evaluated with the exact
    # prefix it was trained on (sft_vlm._PALIGEMMA_PREFIX), or the benchmark
    # scores it on a prompt it never saw - see hf_local_client.HFLocalClient.
    models_path = tmp_path / "models.yaml"
    models_path.write_text("{}\n")
    training_cli.main([
        "register", "--name", "exp-ft", "--path", "/merged",
        "--arch", "paligemma", "--models", str(models_path),
    ])
    doc = yaml.safe_load(models_path.read_text())
    assert doc["exp-ft"]["prompt"] == _PALIGEMMA_PREFIX


def test_cli_register_leaves_prompt_unset_for_non_paligemma_arches(tmp_path):
    models_path = tmp_path / "models.yaml"
    models_path.write_text("{}\n")
    training_cli.main([
        "register", "--name", "exp-ft", "--path", "/merged",
        "--arch", "qwen2_vl", "--models", str(models_path),
    ])
    doc = yaml.safe_load(models_path.read_text())
    assert "prompt" not in doc["exp-ft"]
