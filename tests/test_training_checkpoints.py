"""Resume logic: find the newest checkpoint so a killed Colab run continues."""

from __future__ import annotations

from arabic_ocr_bench.training.checkpoints import latest_checkpoint, resume_from


def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    for n in (50, 400, 200):
        (tmp_path / f"checkpoint-{n}").mkdir()
    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-400")


def test_latest_checkpoint_is_none_when_dir_missing_or_empty(tmp_path):
    assert latest_checkpoint(tmp_path / "nope") is None
    assert latest_checkpoint(tmp_path) is None


def test_resume_from_respects_the_resume_flag(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    assert resume_from(tmp_path, resume=True) == str(tmp_path / "checkpoint-10")
    assert resume_from(tmp_path, resume=False) is None
