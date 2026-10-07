"""Launcher helpers for the 3090 held-out bench — stdlib, no torch import."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_3090_benchmark import intersect_listed, parse_datasets_block


def test_parse_datasets_block_skips_comments_and_other_keys():
    text = (
        "models:\n"
        "  - base\n"
        "datasets:                         # the sweep\n"
        "  - kitab_khatt                   # keep\n"
        "  # - kitab_hindawi               # dropped, page-scale\n"
        "  - nakba_lines\n"
        "concurrency: 1\n"
    )
    assert parse_datasets_block(text) == ["kitab_khatt", "nakba_lines"]


def test_intersect_listed_keeps_request_order_and_drops_unsafe():
    listed = ["kitab_khatt", "muharaf_lines", "nakba_lines", "kitab_hindawi"]
    safe = ["nakba_lines", "kitab_khatt", "rasam1"]
    assert intersect_listed(listed, safe) == ["kitab_khatt", "nakba_lines"]


def test_suite_v1_bench_yaml_is_the_line_level_core():
    text = (Path(__file__).resolve().parents[1]
            / "configs" / "training" / "suite_v1" / "bench.run.yaml").read_text(
        encoding="utf-8"
    )
    names = parse_datasets_block(text)
    assert names == ["kitab_khatt", "nakba_lines", "rasam1", "rasam2"]
    assert "kitab_hindawi" not in names
    assert "muharaf_lines" not in names
