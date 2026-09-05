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


def test_errored_predictions_are_not_scored_or_counted(wired):
    run_dir = wired / "run1"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [
                {"sample_id": "fake/000", "model": "m", "dataset": "fake",
                 "ground_truth": "مرحبا", "cleaned_text": "مرحبا", "error": None},
                {"sample_id": "fake/001", "model": "m", "dataset": "fake",
                 "ground_truth": "كتاب", "cleaned_text": "", "error": "429 rate limit"},
            ]
        )
    )
    score_run(run_dir, profiles=("raw",))
    summary = {r["model"]: r for r in _read_csv(run_dir / "summary.csv")}
    # only the one good pair is scored; the 429 row is quarantined, not CER 1.0
    assert int(summary["m"]["n"]) == 1
    assert int(summary["m"]["errors"]) == 1
    assert float(summary["m"]["cer_micro_raw"]) == 0.0


def test_errored_predictions_are_not_cached(wired, monkeypatch):
    from arabic_ocr_bench.models.mock_client import MockClient

    def boom(self, image, prompt):
        from arabic_ocr_bench.models.base import Prediction

        return Prediction(raw_text="", cleaned_text="", error="500 server error")

    monkeypatch.setattr(MockClient, "transcribe", boom)
    cfg = {"models": ["mock_echo"], "datasets": ["fake"], "prompt_version": "line-v1",
           "cache": True}
    run(cfg, results_root=wired)
    assert not list((wired / "_cache").glob("*.json"))


def test_accumulate_targets_fresh_indices_on_rerun(monkeypatch, tmp_path):
    seen_indices = []

    class _IdxDataset:
        name = "fake"
        track = "line"

        def __init__(self, indices=None, **_):
            self._idx = list(indices if indices is not None else range(2))
            seen_indices.append(list(self._idx))

        def __iter__(self):
            img = Image.new("RGB", (8, 8))
            for i in self._idx:
                yield Sample(id=f"fake/{i:03d}", image=img, ground_truth="ك", track="line")

    monkeypatch.setattr(runner_mod, "load_dataset_configs", lambda *a, **k: {"fake": {}})
    monkeypatch.setattr(runner_mod, "build_dataset",
                        lambda name, cfg, **kw: _IdxDataset(**kw))
    monkeypatch.setattr(
        runner_mod, "load_model_configs",
        lambda *a, **k: {"mock_echo": {"client": "mock", "mode": "echo_gt", "pricing": None}},
    )
    base = {"models": ["mock_echo"], "datasets": ["fake"], "prompt_version": "line-v1",
            "limit": 2, "accumulate": True}
    r1 = run(dict(base), results_root=tmp_path)
    score_run(r1, profiles=("raw",))
    run(dict(base), results_root=tmp_path)
    # first accumulate pass -> [0, 1]; second must move on to [2, 3]
    assert seen_indices[-2] == [0, 1]
    assert seen_indices[-1] == [2, 3]


class _RateLimitedClient:
    name = "ratelimited"

    def transcribe(self, image, prompt):
        from arabic_ocr_bench.models.base import Prediction

        return Prediction(error="RateLimitError: rate limit exceeded", latency_ms=1.0)


def test_out_of_quota_model_is_abandoned_not_retried_line_by_line(monkeypatch, tmp_path):
    img = Image.new("RGB", (8, 8))

    class _Big:
        track = "line"

        def __iter__(self):
            for i in range(20):
                yield Sample(id=f"fake/{i:03d}", image=img, ground_truth="x", track="line")

    monkeypatch.setattr(runner_mod, "load_dataset_configs", lambda *a, **k: {"a": {}, "b": {}})
    monkeypatch.setattr(runner_mod, "build_dataset", lambda *a, **k: _Big())
    monkeypatch.setattr(
        runner_mod, "load_model_configs",
        lambda *a, **k: {"rl": {"client": "mock", "mode": "empty", "pricing": None}},
    )
    monkeypatch.setattr(runner_mod, "build_model", lambda name, cfg: _RateLimitedClient())

    run_dir = run(
        {"models": ["rl"], "datasets": ["a", "b"], "prompt_version": "line-v1",
         "concurrency": 1, "max_retries": 0, "progress": False},
        results_root=tmp_path,
    )
    rows = [json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().splitlines()]
    # aborts dataset "a" after the consecutive-rate-limit threshold (well under 20),
    # and skips dataset "b" entirely
    assert 0 < len(rows) < 20
    assert {r["dataset"] for r in rows} == {"a"}
    assert all(r["error"] for r in rows)


def test_explicit_indices_and_runner_stamp(monkeypatch, tmp_path):
    """A distributed runner passes an exact index range + its identity; the runner
    forwards the range to build_dataset and stamps every row with runner/session."""
    from PIL import Image as _Image

    seen = {}

    class _Idx:
        name = "fake"
        track = "line"

        def __iter__(self):
            img = _Image.new("RGB", (8, 8))
            for i in seen["indices"]:
                yield Sample(id=f"fake/{i:03d}", image=img, ground_truth="x", track="line")

    def _spy_build(name, cfg, *, limit=None, offset=0, indices=None):
        seen["indices"] = indices
        return _Idx()

    monkeypatch.setattr(runner_mod, "load_dataset_configs", lambda *a, **k: {"fake": {}})
    monkeypatch.setattr(runner_mod, "build_dataset", _spy_build)
    monkeypatch.setattr(
        runner_mod, "load_model_configs",
        lambda *a, **k: {"m": {"client": "mock", "mode": "echo_gt", "pricing": None}},
    )

    run_dir = run(
        {"models": ["m"], "datasets": ["fake"], "prompt_version": "line-v1",
         "indices": [3, 1, 2], "cache": False, "progress": False,
         "runner_id": "alice@example.com", "session_id": "7f3a"},
        results_root=tmp_path,
    )
    assert seen["indices"] == [1, 2, 3]          # forwarded, sorted, accumulate implied
    rows = [json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(r["runner"] == "alice@example.com" and r["session"] == "7f3a" for r in rows)


def test_explicit_indices_rejects_multiple_pairs(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_mod, "load_dataset_configs", lambda *a, **k: {"a": {}, "b": {}})
    monkeypatch.setattr(runner_mod, "load_model_configs",
                        lambda *a, **k: {"m": {"client": "mock", "mode": "empty", "pricing": None}})
    with pytest.raises(ValueError, match="exactly one model and one dataset"):
        run({"models": ["m"], "datasets": ["a", "b"], "indices": [0, 1],
             "prompt_version": "line-v1"}, results_root=tmp_path)


def _read_csv(path):
    import csv

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
