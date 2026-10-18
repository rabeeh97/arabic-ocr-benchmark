"""Fill-in of the rec-finetune YAML — no PaddleOCR clone or GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finetune_paddleocr import write_filled_yaml


def test_write_filled_yaml_replaces_placeholders(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "train_list.txt").write_text("images/a.png\thi\n", encoding="utf-8")
    out = tmp_path / "ft"
    clone = tmp_path / "PaddleOCR"
    pretrained = tmp_path / "pretrained"
    dest = write_filled_yaml(labels=labels, out=out, pretrained=pretrained, clone=clone)
    text = dest.read_text(encoding="utf-8")
    assert "PLACEHOLDER_" not in text
    assert str(labels).replace("\\", "/") in text
    assert "train_list.txt" in text
    assert "val_list.txt" in text
