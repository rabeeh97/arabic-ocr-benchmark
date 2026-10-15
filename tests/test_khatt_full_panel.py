"""Flags of the full KHATT v1.0 panel (scripts/khatt_full_panel.py, checklist 1.6)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
khatt = pytest.importorskip("khatt_full_panel")


def test_norm_text_drops_marks_digits_and_punctuation():
    assert khatt.norm_text("  قَالَ: 2005، نعم .  ") == "قال نعم"
    assert khatt.norm_text(None) == ""


def test_text_flags_repeat_cluster_fixed_and_kitab():
    texts = ["ا ب ج د", "ا ب ج د", "ب ج د ه", "س ص ط ظ", "ع غ"]
    splits = ["train", "test", "test", "test", "test"]
    f = khatt.text_flags(texts, splits, kitab_texts=["س ص ط ظ", "x"])
    assert [x["text_count"] for x in f] == [2, 2, 1, 1, 1]
    assert f[0]["text_cluster"] == f[1]["text_cluster"] >= 0 and f[2]["text_cluster"] == -1
    # line 2 is not an exact repeat but shares its 3-gram "ب ج د" with lines 0/1 (a different line break): fixed text
    assert f[2]["ngram_share"] == 0.5 and f[2]["fixed_text"] and not f[2]["text_repeat"]
    assert not f[3]["fixed_text"] and f[3]["kitab_text"] and f[3]["kitab_text_ids"] == ["kitab_khatt/train/000000"]
    assert not f[4]["fixed_text"]  # under 3 words and not repeated
    assert f[1]["text_in_train"] and not f[2]["text_in_train"]
    assert not f[0]["text_in_train"]  # a train line does not count itself


def test_same_image():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (40, 300), dtype=np.uint8)
    assert khatt.same_image(a, a.copy())
    assert not khatt.same_image(a, a[:, :299])
    assert not khatt.same_image(a, rng.integers(0, 255, (40, 300), dtype=np.uint8))


MANIFEST = ROOT / "results/panel/khatt_full_manifest.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="run scripts/khatt_full_panel.py first")
def test_built_manifest_counts():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {s: c["lines"] for s, c in m["counts"].items()} == {"train": 4672, "validation": 963, "test": 1038}
    assert m["panel_rows"] == 2 * 1038
    assert m["counts"]["test"]["kitab_image_id"] <= m["counts"]["test"]["kitab_text"]
