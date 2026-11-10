import json

from arabic_ocr_bench.cache import FILENAME, ResultCache


def test_put_then_get_roundtrip_and_persists(tmp_path):
    c = ResultCache(tmp_path)
    c.put("k1", {"cleaned_text": "مرحبا", "error": None})
    assert c.get("k1") == {"cleaned_text": "مرحبا", "error": None}
    assert c.get("missing") is None
    assert ResultCache(tmp_path).get("k1")["cleaned_text"] == "مرحبا"  # reloaded from disk


def test_last_write_wins(tmp_path):
    c = ResultCache(tmp_path)
    c.put("k", {"v": 1})
    c.put("k", {"v": 2})
    assert ResultCache(tmp_path).get("k") == {"v": 2}


def test_torn_last_line_is_ignored_and_next_append_is_clean(tmp_path):
    p = tmp_path / FILENAME
    p.write_text(json.dumps({"_k": "a", "v": 1}) + "\n" + '{"_k": "b", "v"', encoding="utf-8")
    c = ResultCache(tmp_path)
    assert c.get("a") == {"v": 1} and c.get("b") is None
    c.put("c", {"v": 3})
    again = ResultCache(tmp_path)
    assert again.get("a") == {"v": 1} and again.get("c") == {"v": 3}


def test_falls_back_to_legacy_per_file_cache(tmp_path):
    (tmp_path / "_cache").mkdir()
    (tmp_path / "_cache" / "old.json").write_text(json.dumps({"v": 9}), encoding="utf-8")
    assert ResultCache(tmp_path).get("old") == {"v": 9}
