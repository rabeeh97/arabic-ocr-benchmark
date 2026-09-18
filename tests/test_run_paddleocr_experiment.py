"""Launcher helpers for the PaddleOCR baseline — stdlib, no paddle import."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_paddleocr_experiment import (
    parse_matrix_datasets,
    pin_cpu_inference_env,
    print_log_tail,
    write_run_cfg,
)


def test_parse_matrix_datasets_skips_comments():
    text = (
        "models:\n"
        "  - arabic-paddleocr-v5\n"
        "datasets:\n"
        "  - muharaf_lines             # comment\n"
        "  - kitab_khatt\n"
        "concurrency: 1\n"
    )
    assert parse_matrix_datasets(text) == ["muharaf_lines", "kitab_khatt"]


def test_write_run_cfg_is_cpu_safe_single_model(tmp_path: Path):
    path = tmp_path / "smoke__kitab_khatt.run.yaml"
    write_run_cfg(
        path, dataset="kitab_khatt", limit=20, accumulate=False, results_dir=tmp_path / "results"
    )
    text = path.read_text(encoding="utf-8")
    assert "arabic-paddleocr-v5" in text
    assert "kitab_khatt" in text
    assert "accumulate: false" in text
    assert "concurrency: 1" in text


def test_pin_cpu_inference_env_hides_the_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("PADDLEOCR_DEVICE", raising=False)
    pin_cpu_inference_env()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["PADDLEOCR_DEVICE"] == "cpu"
    assert os.environ["FLAGS_use_gpu"] == "0"


class _ByteBuf:
    def __init__(self):
        self.data = bytearray()

    def write(self, b: bytes) -> int:
        self.data.extend(b)
        return len(b)


class _Cp1252Stream:
    encoding = "cp1252"

    def __init__(self):
        self.buffer = _ByteBuf()

    def write(self, s: str) -> int:
        s.encode("cp1252")  # raises on Arabic, like a real cp1252 console
        return len(s)

    def flush(self) -> None:
        pass


def test_print_log_tail_does_not_crash_on_cp1252_console():
    stream = _Cp1252Stream()
    print_log_tail("ok\nبسم الله\nend", n=10, stream=stream)
    dumped = bytes(stream.buffer.data)
    assert b"ok" in dumped
    assert b"end" in dumped
