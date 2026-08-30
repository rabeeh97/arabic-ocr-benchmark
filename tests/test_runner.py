"""Runner + report integration on a tiny synthetic dataset (no network).

This is the automated form of the plan's sanity checks:
- the empty-string model must score CER ~ 1.0
- a model that returns the ground truth must score CER 0.0
- the persistent cache must prevent a second call for the same key
"""

import json

import pytest
from PIL import Image

from arabic_ocr_bench import runner as runner_mod
from arabic_ocr_bench.datasets.base import Sample
from arabic_ocr_bench.report import score_run
from arabic_ocr_bench.runner import run


class _FakeDataset:
    name = "fake"
    track = "line"

    def __iter__(self):
        img = Image.new("RGB", (8, 8))
        yield Sample(id="fake/000", image=img, ground_truth="مرحبا", track="line")
        yield Sample(id="fake/001", image=img, ground_truth="كتاب", track="line")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_mod, "load_dataset_configs", lambda *a, **k: {"fake": {}})
    monkeypatch.setattr(runner_mod, "build_dataset", lambda *a, **k: _FakeDataset())
    monkeypatch.setattr(
        runner_mod,
        "load_model_configs",
        lambda *a, **k: {
            "mock_empty": {"client": "mock", "mode": "empty", "pricing": None},
            "mock_echo": {"client": "mock", "mode": "echo_gt", "pricing": None},
        },
    )
    return tmp_path


def test_run_writes_one_prediction_row_per_sample_and_model(wired):
    cfg = {"models": ["mock_empty", "mock_echo"], "datasets": ["fake"],
           "prompt_version": "line-v1", "concurrency": 2}
    run_dir = run(cfg, results_root=wired)
    rows = [json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {r["model"] for r in rows} == {"mock_empty", "mock_echo"}


def test_sanity_ceiling_and_floor(wired):
    cfg = {"models": ["mock_empty", "mock_echo"], "datasets": ["fake"],
           "prompt_version": "line-v1", "normalization_profiles": ["raw"]}
    run_dir = run(cfg, results_root=wired)
    score_run(run_dir, profiles=("raw",))
    summary = {
        (r["model"]): r
        for r in _read_csv(run_dir / "summary.csv")
    }
    assert float(summary["mock_echo"]["cer_micro_raw"]) == 0.0
    assert float(summary["mock_empty"]["cer_micro_raw"]) == 1.0


def test_cache_prevents_second_call(wired, monkeypatch):
    cfg = {"models": ["mock_echo"], "datasets": ["fake"], "prompt_version": "line-v1",
           "cache": True}
    run(cfg, results_root=wired)

    calls = {"n": 0}
    from arabic_ocr_bench.models.mock_client import MockClient

    orig = MockClient.transcribe

    def counting(self, image, prompt):
        calls["n"] += 1
        return orig(self, image, prompt)

    monkeypatch.setattr(MockClient, "transcribe", counting)
    run(cfg, results_root=wired)  # same keys -> should be served from cache
    assert calls["n"] == 0


def _read_csv(path):
    import csv

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
