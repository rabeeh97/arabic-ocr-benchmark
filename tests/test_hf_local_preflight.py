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


# --- CUDA VRAM guard: same idea as the CPU one, but for a GPU sweep ---------- #

def test_vram_preflight_reclaims_then_skips_when_still_short(monkeypatch):
    # pretend we're on a CUDA GPU with only 3 GB free, even after a reclaim
    monkeypatch.setattr(H, "_cuda_free_total_gb", lambda: (3.0, 16.0))
    freed = []
    monkeypatch.setattr(H, "_hard_free", lambda: freed.append(True))
    client = H.HFLocalClient(name="qwen2.5-vl-3b", hf_id="x", min_ram_gb=11)
    with pytest.raises(RuntimeError, match=r"skip qwen2\.5-vl-3b: needs ~11 GB GPU"):
        client._mem_preflight()
    assert freed  # it tried to reclaim before giving up


def test_vram_preflight_passes_when_gpu_has_room(monkeypatch):
    monkeypatch.setattr(H, "_cuda_free_total_gb", lambda: (14.0, 16.0))
    H.HFLocalClient(name="m", hf_id="x", min_ram_gb=11)._mem_preflight()  # no raise


def test_load_oom_marks_model_skipped_for_the_rest_of_the_run(monkeypatch):
    client = H.HFLocalClient(name="qwen2-vl-2b", hf_id="x", min_ram_gb=0)
    monkeypatch.setattr(client, "_mem_preflight", lambda: None)

    def boom():
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(client, "_load_dispatch", boom)
    with pytest.raises(RuntimeError, match="CUDA out of memory during load"):
        client._ensure_loaded()
    assert client._oom is True
    # a later line doesn't retry the load — it fails fast with the skip message
    with pytest.raises(RuntimeError, match="hit CUDA OOM"):
        client._ensure_loaded()


def test_is_oom_recognises_common_shapes():
    assert H._is_oom(RuntimeError("CUDA out of memory"))
    assert H._is_oom(Exception("... torch OutOfMemoryError ..."))
    assert not H._is_oom(RuntimeError("connection reset by peer"))
