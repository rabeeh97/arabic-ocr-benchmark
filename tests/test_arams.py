"""AraMS-28k-HTR adapter — split determinism (no network, synthetic fixtures)."""

from PIL import Image

import arabic_ocr_bench.datasets.arams as arams_mod
from arabic_ocr_bench.datasets.arams import TEST_SIZE, AraMSHTR, _split_ids


def _make_corpus(tmp_path, n=50):
    for i in range(n):
        Image.new("RGB", (40, 12), "white").save(tmp_path / f"line_{i:04d}.png")
        (tmp_path / f"line_{i:04d}.gt.txt").write_text(f"نص {i}", encoding="utf-8")
    return tmp_path


def test_split_is_deterministic_and_disjoint(tmp_path):
    _make_corpus(tmp_path, n=50)
    train_a = _split_ids(tmp_path, "train")
    train_b = _split_ids(tmp_path, "train")
    test_a = _split_ids(tmp_path, "test")
    assert train_a == train_b  # same call, same result, every time
    assert set(train_a).isdisjoint(test_a)
    assert set(train_a) | set(test_a) == {p.stem for p in tmp_path.glob("*.png")}


def test_split_rejects_unknown_split_name(tmp_path):
    _make_corpus(tmp_path, n=5)
    try:
        _split_ids(tmp_path, "val")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for an unknown split name")


def test_test_size_is_capped_by_corpus_size(tmp_path):
    # corpus smaller than TEST_SIZE: every id should still land somewhere,
    # and 'test' should just be the whole (small) corpus, not crash/duplicate.
    n = min(20, TEST_SIZE - 1)
    _make_corpus(tmp_path, n=n)
    test_ids = _split_ids(tmp_path, "test")
    train_ids = _split_ids(tmp_path, "train")
    assert len(test_ids) == n
    assert train_ids == []


def test_adapter_yields_stable_ids_and_reads_sidecar_gt(tmp_path, monkeypatch):
    monkeypatch.setattr(arams_mod, "TEST_SIZE", 4)  # small corpus needs a small test slice
    _make_corpus(tmp_path, n=10)
    ds = AraMSHTR(root=str(tmp_path), split="train", limit=3)
    xs = list(ds)
    assert [x.id for x in xs] == [f"arams_htr/train/{i:06d}" for i in range(3)]
    for x in xs:
        assert x.track == "line"
        assert x.ground_truth.startswith("نص")
        assert x.image.mode == "RGB"

    # re-instantiating (as a fresh eval/train run would) reproduces the same ids
    ds2 = AraMSHTR(root=str(tmp_path), split="train", limit=3)
    assert [x.id for x in ds2] == [x.id for x in xs]
