"""The training data assembler must never pull a split the benchmark scores on."""

from __future__ import annotations

import pytest

from arabic_ocr_bench.training.data import (
    ScoredSplitLeakError,
    check_source_not_scored,
    safe_eval_datasets,
    scored_splits,
)
from arabic_ocr_bench.training.config import DataSpec, SourceSpec


def test_scored_splits_reads_registry():
    pairs = scored_splits("configs/datasets.yaml")
    # muharaf_lines is scored on the `test` split
    assert ("aamijar/muharaf-public", "test") in pairs
    # KITAB-Bench subsets are scored on their (only) `train` split
    assert ("ahmedheakl/arocrbench_khatt", "train") in pairs


def test_rejects_the_muharaf_test_split():
    with pytest.raises(ScoredSplitLeakError):
        check_source_not_scored(
            "aamijar/muharaf-public", "test", cfg_path="configs/datasets.yaml"
        )


def test_allows_the_reserved_muharaf_train_split():
    # returns without raising
    check_source_not_scored(
        "aamijar/muharaf-public", "train", cfg_path="configs/datasets.yaml"
    )


def test_rejects_the_kitab_khatt_scored_split():
    with pytest.raises(ScoredSplitLeakError):
        check_source_not_scored(
            "ahmedheakl/arocrbench_khatt", "train", cfg_path="configs/datasets.yaml"
        )


def test_allows_the_original_khatt_corpus():
    # different hf_id than the scored KITAB slice -> not a scored split
    check_source_not_scored(
        "benhachem/KHATT", "train", cfg_path="configs/datasets.yaml"
    )


def _data_spec(*sources: SourceSpec) -> DataSpec:
    return DataSpec.model_construct(sources=list(sources), val_fraction=0.0, prompt_version="line-v2")


def test_safe_eval_datasets_drops_declared_overlap_when_muharaf_is_trained_on():
    data = _data_spec(
        SourceSpec(
            name="muharaf_train",
            adapter="arabic_ocr_bench.datasets.muharaf_lines:MuharafLines",
            hf_id="aamijar/muharaf-public",
            split="train",
        )
    )
    safe = safe_eval_datasets(data, cfg_path="configs/datasets.yaml")
    assert "kitab_muharaf" not in safe  # declared overlaps_source
    assert "muharaf_lines" in safe  # disjoint test split, still safe to score
    assert "synthetic" not in safe  # not a real benchmark
    assert "nakba_lines" in safe  # untouched domain, always safe


def test_safe_eval_datasets_keeps_everything_when_nothing_overlaps():
    data = _data_spec(
        SourceSpec(
            name="rasam1_train",
            adapter="arabic_ocr_bench.datasets.kitab_bench:KitabBench",
            params={"subset": "calfa-ai/RASAM-1"},
            split="train",
        )
    )
    safe = safe_eval_datasets(data, cfg_path="configs/datasets.yaml")
    assert "kitab_muharaf" in safe  # no Muharaf source in this mix
    assert "rasam1" in safe  # different split (test) than the trained-on one
