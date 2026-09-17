"""AraMS-28k-HTR adapter — manifest-driven split (no network, synthetic fixtures)."""

import pytest
from PIL import Image

from arabic_ocr_bench.datasets.arams import AraMSHTR, _manifest_rel_paths


def _make_corpus(tmp_path, n_train=4, n_val=2, n_test=3):
    """Mimics the real archive layout: <root>/AraMs-28k-HTR/{images/, *_manifest.txt}."""
    pkg = tmp_path / "AraMs-28k-HTR"
    images = pkg / "images"
    images.mkdir(parents=True)

    def _write(split, n, offset):
        lines = []
        for i in range(n):
            stem = f"book_{offset + i:02d}__line{i:04d}"
            Image.new("RGB", (40, 12), "white").save(images / f"{stem}.png")
            (images / f"{stem}.gt.txt").write_text(f"نص {stem}", encoding="utf-8")
            lines.append(f"./AraMs-28k-HTR/images/{stem}.png")
        (pkg / f"{split}_manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write("train", n_train, 0)
    _write("val", n_val, 100)
    _write("test", n_test, 200)
    return tmp_path


def test_manifest_rel_paths_reads_and_strips_leading_dot_slash(tmp_path):
    root = _make_corpus(tmp_path, n_train=3, n_val=0, n_test=0)
    rel = _manifest_rel_paths(root, "train")
    assert len(rel) == 3
    assert all(r.startswith("AraMs-28k-HTR/images/") for r in rel)
    assert not any(r.startswith("./") for r in rel)


def test_splits_are_disjoint_and_come_from_different_manifests(tmp_path):
    root = _make_corpus(tmp_path, n_train=4, n_val=2, n_test=3)
    train = set(_manifest_rel_paths(root, "train"))
    val = set(_manifest_rel_paths(root, "val"))
    test = set(_manifest_rel_paths(root, "test"))
    assert len(train) == 4 and len(val) == 2 and len(test) == 3
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        _manifest_rel_paths(__import__("pathlib").Path("/nonexistent"), "bogus")


def test_missing_manifest_raises_filenotfound(tmp_path):
    root = _make_corpus(tmp_path, n_train=1, n_val=0, n_test=0)
    (root / "AraMs-28k-HTR" / "val_manifest.txt").unlink()  # remove it entirely, not just empty
    with pytest.raises(FileNotFoundError):
        _manifest_rel_paths(root, "val")


def test_adapter_yields_stable_ids_and_reads_sidecar_gt(tmp_path):
    root = _make_corpus(tmp_path, n_train=4, n_val=2, n_test=3)
    ds = AraMSHTR(root=str(root), split="train", limit=3)
    xs = list(ds)
    assert [x.id for x in xs] == [f"arams_htr/train/{i:06d}" for i in range(3)]
    for x in xs:
        assert x.track == "line"
        assert x.ground_truth.startswith("نص")
        assert x.image.mode == "RGB"

    # re-instantiating (as a fresh eval/train run would) reproduces the same ids
    ds2 = AraMSHTR(root=str(root), split="train", limit=3)
    assert [x.id for x in ds2] == [x.id for x in xs]


def test_train_val_test_splits_yield_disjoint_item_ids(tmp_path):
    root = _make_corpus(tmp_path, n_train=4, n_val=2, n_test=3)
    train_ids = {x.meta["item_id"] for x in AraMSHTR(root=str(root), split="train")}
    val_ids = {x.meta["item_id"] for x in AraMSHTR(root=str(root), split="val")}
    test_ids = {x.meta["item_id"] for x in AraMSHTR(root=str(root), split="test")}
    assert len(train_ids) == 4 and len(val_ids) == 2 and len(test_ids) == 3
    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
