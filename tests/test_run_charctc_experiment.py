"""Launcher helpers for the CPU character-CTC experiment — stdlib, no torch."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_charctc_experiment import (  # noqa: E402
    PHASES,
    pin_cpu_env,
    print_log_tail,
    smoke_still_collapsed,
    train_needs_more_epochs,
)


def test_phases_are_the_documented_pipeline():
    assert PHASES[0] == "install"
    assert PHASES[-1] == "report"
    assert "audit" in PHASES
    assert PHASES.index("charset") < PHASES.index("train")
    assert PHASES.index("train") < PHASES.index("smoke")


def test_pin_cpu_env_hides_the_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("AOCR_DEVICE", raising=False)
    pin_cpu_env()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["AOCR_DEVICE"] == "cpu"


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
        s.encode("cp1252")
        return len(s)

    def flush(self) -> None:
        pass


def test_train_needs_more_epochs_when_yaml_bumped():
    assert train_needs_more_epochs(completed=1, wanted=3) is True
    assert train_needs_more_epochs(completed=3, wanted=3) is False
    assert train_needs_more_epochs(completed=None, wanted=3) is True


def test_smoke_still_collapsed_requires_all_datasets_near_cer_one(tmp_path):
    from run_charctc_experiment import EVAL_DATASETS

    dirs = {"bench": tmp_path / "bench"}
    assert smoke_still_collapsed(dirs) is False
    for ds in EVAL_DATASETS:
        p = dirs["bench"] / "smoke" / ds
        p.mkdir(parents=True)
        (p / "summary.json").write_text('{"cer_micro_light": 1.0}', encoding="utf-8")
    assert smoke_still_collapsed(dirs) is True
    last = dirs["bench"] / "smoke" / EVAL_DATASETS[-1] / "summary.json"
    last.write_text('{"cer_micro_light": 0.4}', encoding="utf-8")
    assert smoke_still_collapsed(dirs) is False


def test_print_log_tail_does_not_crash_on_cp1252_console():
    stream = _Cp1252Stream()
    print_log_tail("ok\nبسم الله\nend", n=10, stream=stream)
    dumped = bytes(stream.buffer.data)
    assert b"ok" in dumped
    assert b"end" in dumped
