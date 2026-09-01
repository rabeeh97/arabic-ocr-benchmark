"""The training data assembler must never pull a split the benchmark scores on."""

from __future__ import annotations

import pytest

from arabic_ocr_bench.training.data import (
    ScoredSplitLeakError,
    check_source_not_scored,
    scored_splits,
)


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
