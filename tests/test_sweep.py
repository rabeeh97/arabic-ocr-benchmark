"""Sweep orchestration: partial+full phases, resume via a manifest, keep going
on per-model failure. Tests use mock models + the synthetic dataset (no network).
"""

import json

import pytest
from PIL import Image

from arabic_ocr_bench import runner as runner_mod
from arabic_ocr_bench import sweep
from arabic_ocr_bench.datasets.base import Sample


class _FakeDataset:
    name = "fake"
    track = "line"

    def __init__(self, limit=None):
        self.limit = 4 if limit is None else min(4, limit)

    def __iter__(self):
        img = Image.new("RGB", (8, 8))
        for i in range(self.limit):
            yield Sample(id=f"fake/{i:03d}", image=img, ground_truth="مرحبا", track="line")


class _BoomClient:
    name = "boom"

    def transcribe(self, image, prompt):
        raise RuntimeError("kaboom")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    from arabic_ocr_bench.models.registry import build_model as real_build

    monkeypatch.setattr(runner_mod, "load_dataset_configs", lambda *a, **k: {"fake": {}})
    monkeypatch.setattr(
        runner_mod, "build_dataset", lambda name, cfg, *, limit=None, offset=0: _FakeDataset(limit)
    )
    monkeypatch.setattr(
        runner_mod,
        "load_model_configs",
        lambda *a, **k: {
            "mock_echo": {"client": "mock", "mode": "echo_gt", "pricing": None},
            "mock_empty": {"client": "mock", "mode": "empty", "pricing": None},
            "boom": {"client": "mock", "mode": "empty", "pricing": None},
        },
    )
    monkeypatch.setattr(
        runner_mod,
        "build_model",
        lambda name, cfg: _BoomClient() if name == "boom" else real_build(name, cfg),
    )
    return tmp_path


def test_runs_partial_then_full_and_records_manifest(wired):
    manifest_path = wired / "sweep_manifest.json"
    entries = sweep.run_sweep(
        models=["mock_echo"], dataset="fake", partial_limit=2, run_full=True,
        manifest_path=manifest_path, results_dir=wired, profiles=("raw",),
    )
    phases = [(e["model"], e["phase"], e["limit"]) for e in entries]
    assert phases == [("mock_echo", "partial", 2), ("mock_echo", "full", None)]
    saved = json.loads(manifest_path.read_text())["entries"]
    assert {e["phase"] for e in saved} == {"partial", "full"}
    assert saved[0]["n"] == 2 and saved[1]["n"] == 4


def test_second_call_skips_completed_work(wired):
    manifest_path = wired / "sweep_manifest.json"
    calls = []
    sweep.run_sweep(models=["mock_echo"], dataset="fake", partial_limit=2, run_full=True,
                    manifest_path=manifest_path, results_dir=wired, profiles=("raw",))
    new = sweep.run_sweep(
        models=["mock_echo"], dataset="fake", partial_limit=2, run_full=True,
        manifest_path=manifest_path, results_dir=wired, profiles=("raw",),
        on_progress=calls.append,
    )
    assert new == []  # nothing new ran
    assert any("skip" in c.lower() for c in calls)


def test_on_done_callback_fires_per_completed_phase(wired):
    manifest_path = wired / "sweep_manifest.json"
    seen = []
    sweep.run_sweep(models=["mock_echo"], dataset="fake", partial_limit=2, run_full=False,
                    manifest_path=manifest_path, results_dir=wired, profiles=("raw",),
                    on_done=lambda entry, run_dir: seen.append(entry["phase"]))
    assert seen == ["partial"]


def test_a_failing_model_does_not_stop_the_sweep(wired):
    manifest_path = wired / "sweep_manifest.json"
    progress = []
    entries = sweep.run_sweep(
        models=["boom", "mock_echo"], dataset="fake", partial_limit=2, run_full=False,
        manifest_path=manifest_path, results_dir=wired, profiles=("raw",),
        on_progress=progress.append,
    )
    assert [e["model"] for e in entries] == ["mock_echo"]
    assert any("boom" in p and "fail" in p.lower() for p in progress)


def test_reconcile_drops_manifest_entries_whose_run_dir_is_missing(wired):
    manifest_path = wired / "sweep_manifest.json"
    manifest_path.write_text(json.dumps({"entries": [
        {"model": "ghost", "dataset": "fake", "phase": "partial", "run_id": "does-not-exist"},
    ]}))
    prog = []
    sweep.run_sweep(
        models=["mock_echo"], dataset="fake", partial_limit=2, run_full=False,
        manifest_path=manifest_path, results_dir=wired, profiles=("raw",),
        on_progress=prog.append,
    )
    saved = json.loads(manifest_path.read_text())["entries"]
    assert [e["model"] for e in saved] == ["mock_echo"]        # ghost dropped, real one kept
    assert any("reconcile" in p and "ghost" in p for p in prog)


class _IndexedDataset:
    """Fake with a fixed size that honours both `limit` and an explicit
    `indices` list (accumulate mode), clamped to range like the real adapters."""

    track = "line"
    SIZE = 6

    def __init__(self, limit=None, offset=0, indices=None):
        if indices is not None:
            self.rows = [i for i in indices if 0 <= i < self.SIZE]
        else:
            end = self.SIZE if limit is None else min(self.SIZE, offset + limit)
            self.rows = list(range(offset, end))

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        img = Image.new("RGB", (8, 8))
        for i in self.rows:
            yield Sample(id=f"fake/{i:03d}", image=img, ground_truth="مرحبا", track="line")


def test_accumulate_skips_a_model_once_the_split_is_exhausted(monkeypatch, tmp_path):
    from arabic_ocr_bench.models.registry import build_model as real_build

    def _bd(name, cfg, *, limit=None, offset=0, indices=None):
        return _IndexedDataset(limit=limit, offset=offset, indices=indices)

    for mod in (runner_mod, sweep):
        monkeypatch.setattr(mod, "load_dataset_configs", lambda *a, **k: {"fake": {}}, raising=False)
        monkeypatch.setattr(mod, "build_dataset", _bd, raising=False)
    monkeypatch.setattr(
        runner_mod, "load_model_configs",
        lambda *a, **k: {"mock_echo": {"client": "mock", "mode": "echo_gt", "pricing": None}},
    )
    monkeypatch.setattr(runner_mod, "build_model", real_build)

    manifest = tmp_path / "m.json"
    kw = dict(models=["mock_echo"], dataset="fake", partial_limit=4, run_full=False,
              manifest_path=manifest, results_dir=tmp_path, profiles=("raw",), accumulate=True)

    a = sweep.run_sweep(**kw)                      # scores rows 0-3
    b = sweep.run_sweep(**kw)                      # scores rows 4-5 (only 2 left)
    prog = []
    c = sweep.run_sweep(**kw, on_progress=prog.append)  # nothing left

    assert a[0]["n"] == 4
    assert b[0]["n"] == 2
    assert c == []                                # exhausted -> no empty run
    assert any("no un-scored lines" in p for p in prog)


def test_phase_not_marked_done_when_on_done_raises_and_re_runs_next_time(wired):
    manifest_path = wired / "sweep_manifest.json"

    # publish fails the first time
    calls = {"n": 0}

    def flaky_publish(entry, run_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("push failed")

    prog = []
    first = sweep.run_sweep(
        models=["mock_echo"], dataset="fake", partial_limit=2, run_full=False,
        manifest_path=manifest_path, results_dir=wired, profiles=("raw",),
        on_done=flaky_publish, on_progress=prog.append,
    )
    assert first == []                                   # nothing recorded
    assert json.loads(manifest_path.read_text())["entries"] == []
    assert any("not persisted" in p.lower() for p in prog)

    # second run retries the same phase; this time publish succeeds
    second = sweep.run_sweep(
        models=["mock_echo"], dataset="fake", partial_limit=2, run_full=False,
        manifest_path=manifest_path, results_dir=wired, profiles=("raw",),
        on_done=flaky_publish,
    )
    assert [e["phase"] for e in second] == ["partial"]
    assert len(json.loads(manifest_path.read_text())["entries"]) == 1
