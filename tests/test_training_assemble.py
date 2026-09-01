"""Assembling sources into an SFT dataset: conversation shape + train/val split."""

from __future__ import annotations

from PIL import Image

from arabic_ocr_bench.training.config import DataSpec, SourceSpec
from arabic_ocr_bench.training.data import assemble_dataset, build_conversation


def test_build_conversation_has_system_user_image_and_assistant_target():
    msgs = build_conversation("PROMPT TEXT", "الهدف")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant"]
    user_types = [c["type"] for c in msgs[1]["content"]]
    assert "image" in user_types and "text" in user_types
    assert msgs[2]["content"][0]["text"] == "الهدف"


def test_assemble_dataset_splits_train_and_val_deterministically():
    # `synthetic` is an offline in-repo dataset (10 lines) — no network needed.
    spec = DataSpec(
        sources=[
            SourceSpec(
                name="synthetic",
                adapter="arabic_ocr_bench.datasets.synthetic:Synthetic",
            )
        ],
        val_fraction=0.2,
        prompt_version="line-v2",
    )
    bundle = assemble_dataset(spec, seed=0)
    assert len(bundle["train"]) + len(bundle["val"]) == 10
    assert len(bundle["val"]) == 2
    # deterministic for a fixed seed
    again = assemble_dataset(spec, seed=0)
    assert bundle["val"][0]["messages"][2]["content"][0]["text"] == (
        again["val"][0]["messages"][2]["content"][0]["text"]
    )
    # every row carries its PIL image and a source tag
    row = bundle["train"][0]
    assert isinstance(row["images"][0], Image.Image)
    assert row["source"] == "synthetic"
