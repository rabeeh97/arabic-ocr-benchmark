"""HFLocalClient's PaliGemma eval prefix: must match what a checkpoint was
fine-tuned with (see training/cli.py's _cmd_register), not a hardcoded default."""

from __future__ import annotations

from arabic_ocr_bench.models.hf_local_client import HFLocalClient


def test_paligemma_prompt_defaults_to_the_arabic_hw_adapter_prefix():
    client = HFLocalClient(name="paligemma-arabic-hw", hf_id="x", arch="paligemma")
    assert client.paligemma_prompt == "ocr\n"


def test_paligemma_prompt_uses_the_entry_override_when_given():
    client = HFLocalClient(
        name="exp3090-paligemma2-3b-ft",
        hf_id="x",
        arch="paligemma",
        paligemma_prompt="transcribe the arabic text in this image",
    )
    assert client.paligemma_prompt == "transcribe the arabic text in this image"
