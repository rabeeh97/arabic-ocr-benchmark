"""run_3090_experiment.py / run_3090_benchmark.py's --shared experiment.env
resolution — a real bug found deploying the suite: with --shared, a model's
own `dirs["root"]` is its per-model subfolder, not the suite root where
experiment.env (the HF_TOKEN) actually lives, so it was silently never loaded.

Imported via importlib rather than a normal package import: these are
top-level, `__main__`-oriented launcher scripts (not part of the
arabic_ocr_bench package), consistent with how the rest of this test suite
treats scripts/ generally.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clear(*keys):
    for k in keys:
        os.environ.pop(k, None)


def test_experiment_py_finds_env_at_the_shared_suite_root(tmp_path):
    m = _load("_test_run_3090_experiment", "run_3090_experiment.py")
    suite_dir = tmp_path
    shared = suite_dir / "_shared"
    model_dir = suite_dir / "some_model"
    (suite_dir / "experiment.env").write_text("HF_TOKEN=suite_token\n", encoding="utf-8")

    _clear("HF_TOKEN")
    dirs = m._dirs(model_dir, shared)
    assert dirs["shared_root"] == suite_dir
    m._pin_env(dirs)
    assert os.environ.get("HF_TOKEN") == "suite_token"
    _clear("HF_TOKEN")


def test_experiment_py_per_model_env_overrides_shared(tmp_path):
    m = _load("_test_run_3090_experiment_2", "run_3090_experiment.py")
    suite_dir = tmp_path
    shared = suite_dir / "_shared"
    model_dir = suite_dir / "some_model"
    model_dir.mkdir(parents=True)
    (suite_dir / "experiment.env").write_text("HF_TOKEN=suite_token\n", encoding="utf-8")
    (model_dir / "experiment.env").write_text("HF_TOKEN=override_token\n", encoding="utf-8")

    _clear("HF_TOKEN")
    dirs = m._dirs(model_dir, shared)
    m._pin_env(dirs)
    assert os.environ.get("HF_TOKEN") == "override_token"
    _clear("HF_TOKEN")


def test_experiment_py_without_shared_still_uses_its_own_root(tmp_path):
    m = _load("_test_run_3090_experiment_3", "run_3090_experiment.py")
    (tmp_path / "experiment.env").write_text("HF_TOKEN=solo_token\n", encoding="utf-8")

    _clear("HF_TOKEN")
    dirs = m._dirs(tmp_path)
    assert dirs["shared_root"] is None
    m._pin_env(dirs)
    assert os.environ.get("HF_TOKEN") == "solo_token"
    _clear("HF_TOKEN")


def test_benchmark_py_finds_env_at_the_shared_suite_root(tmp_path):
    m = _load("_test_run_3090_benchmark", "run_3090_benchmark.py")
    suite_dir = tmp_path
    shared = suite_dir / "_shared"
    model_dir = suite_dir / "some_model"
    (suite_dir / "experiment.env").write_text("HF_TOKEN=bench_suite_token\n", encoding="utf-8")

    _clear("HF_TOKEN")
    dirs = m._dirs(model_dir, shared)
    assert dirs["shared_root"] == suite_dir
    m._pin_env(dirs)
    assert os.environ.get("HF_TOKEN") == "bench_suite_token"
    _clear("HF_TOKEN")
