"""PaliGemma SFT collator: max_seq_len guard + explicit image-token prefix."""

from __future__ import annotations

import pytest

from arabic_ocr_bench.training import sft_vlm


class _FakeProcessor:
    """Stands in for PaliGemmaProcessor: just records the call it received."""

    image_seq_length = 1024

    def __call__(self, **kwargs):
        self.last_call = kwargs
        return kwargs


def test_max_seq_len_guard_rejects_too_small_a_budget():
    with pytest.raises(ValueError, match="leaves no room for text"):
        sft_vlm._make_collator_paligemma(_FakeProcessor(), max_seq_len=640)


def test_collate_prepends_exact_image_token_count_to_avoid_processor_inference():
    # Passing the exact `image_seq_length` count of "<image>" tokens up front
    # is what stops PaliGemmaProcessor from inferring the count itself and
    # warning on every call.
    processor = _FakeProcessor()
    collate = sft_vlm._make_collator_paligemma(processor, max_seq_len=1280)
    row = {
        "images": [object()],
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "target text"}]},
        ],
    }
    collate([row])

    text = processor.last_call["text"][0]
    assert text.count("<image>") == processor.image_seq_length
    assert text.endswith(sft_vlm._PALIGEMMA_PREFIX)
    assert processor.last_call["suffix"][0] == "target text"


def test_visual_param_names():
    assert sft_vlm._is_visual_param("base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight")
    assert sft_vlm._is_visual_param("model.visual.patch_embed.proj.weight")
    assert not sft_vlm._is_visual_param("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight")
