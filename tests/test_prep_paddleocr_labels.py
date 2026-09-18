"""prep_paddleocr_labels writes PaddleOCR + PaddleX label files (no HF needed)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prep_paddleocr_labels import _cap_sources, write_split


def test_write_split_emits_tab_files_and_pngs(tmp_path: Path):
    img = Image.new("RGB", (8, 4), "white")
    rows = [
        {
            "id": "src/line-1",
            "images": [img],
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "p"}]},
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "بسم"}]},
            ],
        },
        {
            "id": "empty",
            "images": [img],
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "   "}]},
            ],
        },
    ]
    n = write_split(rows, tmp_path, "train")
    assert n == 1
    listed = (tmp_path / "train_list.txt").read_text(encoding="utf-8")
    paddlex = (tmp_path / "train.txt").read_text(encoding="utf-8")
    assert listed == paddlex
    assert listed.startswith("images/src_line-1.png\tبسم\n")
    assert (tmp_path / "images" / "src_line-1.png").is_file()


def test_cap_sources_tightens_existing_limits():
    class Src:
        def __init__(self, limit):
            self.limit = limit

    class Data:
        def __init__(self):
            self.sources = [Src(6000), Src(None), Src(10)]

    class Cfg:
        def __init__(self):
            self.data = Data()

    cfg = Cfg()
    _cap_sources(cfg, 20)
    assert [s.limit for s in cfg.data.sources] == [20, 20, 10]
