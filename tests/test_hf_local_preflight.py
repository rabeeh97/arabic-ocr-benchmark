"""The CPU RAM pre-flight: a model that won't fit is skipped with a normal
exception (the runner catches it) instead of OOM-killing the kernel."""

import pytest

from arabic_ocr_bench.models import hf_local_client as H


def test_preflight_raises_when_ram_below_budget(monkeypatch):
    monkeypatch.setattr(H, "_available_ram_gb", lambda: 4.0)
    client = H.HFLocalClient(name="qwen2.5-vl-3b", hf_id="x", min_ram_gb=11)
    with pytest.raises(RuntimeError, match=r"skip qwen2\.5-vl-3b on CPU"):
        client._ram_preflight()


def test_preflight_passes_when_ram_sufficient(monkeypatch):
    monkeypatch.setattr(H, "_available_ram_gb", lambda: 32.0)
    H.HFLocalClient(name="m", hf_id="x", min_ram_gb=11)._ram_preflight()  # no raise


def test_preflight_noop_without_a_budget(monkeypatch):
    monkeypatch.setattr(H, "_available_ram_gb", lambda: 0.5)
    H.HFLocalClient(name="m", hf_id="x")._ram_preflight()  # min_ram_gb defaults to 0


def test_available_ram_gb_returns_a_positive_number():
    assert H._available_ram_gb() > 0
