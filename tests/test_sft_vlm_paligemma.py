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


def test_collate_prepends_exactly_one_image_token_not_image_seq_length_copies():
    # PaliGemmaProcessor.__call__ detects "<image>" already present in the text
    # and expands EVERY occurrence into `image_seq_length` copies itself
    # (processing_paligemma.py's `sample.replace(IMAGE_TOKEN, IMAGE_TOKEN *
    # self.image_seq_length)`). Passing image_seq_length copies here (the old,
    # buggy behaviour this test used to assert) made the processor expand each
    # of those into another image_seq_length -- e.g. 1024*1024 = 1,048,576
    # total "<image>" tokens for the -448 checkpoint, failing the processor's
    # own image-token-count check on every call. Exactly one "<image>" is the
    # fix: the processor's own expansion turns it into the correct count.
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
    assert text.count("<image>") == 1
    assert text.endswith(sft_vlm._PALIGEMMA_PREFIX)
    assert processor.last_call["suffix"][0] == "target text"


def test_visual_param_names():
    assert sft_vlm._is_visual_param("base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight")
    assert sft_vlm._is_visual_param("model.visual.patch_embed.proj.weight")
    assert not sft_vlm._is_visual_param("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight")
