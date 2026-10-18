"""Offline dataset-adapter checks — no network."""

from arabic_ocr_bench.datasets.kitab_bench import KitabBench, resolve_hf_split
from arabic_ocr_bench.datasets.registry import build_dataset
from arabic_ocr_bench.datasets.synthetic import Synthetic


def test_synthetic_renders_legible_arabic():
    xs = list(Synthetic(limit=3))
    assert [x.id for x in xs] == [f"synthetic/test/{i:06d}" for i in range(3)]
    for x in xs:
        assert x.image.mode == "RGB"
        assert x.image.size[0] > 200 and x.image.size[1] > 20
        assert x.ground_truth and x.track == "line"
        # not a blank canvas — shaping actually drew ink
        assert min(x.image.convert("L").getextrema()) < 128


def test_kitab_bench_gt_key_resolution():
    assert KitabBench._gt({"image": 1, "text": "أ", "source": "s"}) == "أ"
    assert KitabBench._gt({"image": 1, "answer": "ب"}) == "ب"
    try:
        KitabBench._gt({"image": 1, "source": "s"})
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected KeyError when no GT column present")


def test_kitab_bench_id_and_hf_id_from_config():
    ds = build_dataset(
        "kitab_khatt",
        {
            "adapter": "arabic_ocr_bench.datasets.kitab_bench:KitabBench",
            "params": {"subset": "arocrbench_khatt"},
            "split": "train",
        },
        limit=5,
    )
    assert ds.hf_id == "ahmedheakl/arocrbench_khatt"
    assert ds.name == "kitab_khatt"  # registry passes `name` through
    assert ds.limit == 5 and ds.split == "train"


def test_build_dataset_still_works_without_params():
    ds = build_dataset(
        "muharaf_lines",
        {
            "adapter": "arabic_ocr_bench.datasets.muharaf_lines:MuharafLines",
            "hf_id": "aamijar/muharaf-public",
            "split": "test",
        },
        limit=3,
    )
    assert ds.hf_id == "aamijar/muharaf-public" and ds.limit == 3


def test_build_dataset_passes_offset_through():
    ds = build_dataset(
        "kitab_khatt",
        {
            "adapter": "arabic_ocr_bench.datasets.kitab_bench:KitabBench",
            "params": {"subset": "arocrbench_khatt"},
            "split": "train",
        },
        limit=5,
        offset=12,
    )
    assert ds.offset == 12


def test_kitab_bench_ids_reflect_absolute_row_when_offset(monkeypatch):
    class FakeImg:
        def convert(self, mode):
            return self

    rows = [{"image": FakeImg(), "text": f"row{i}"} for i in range(5)]
    ds = KitabBench(subset="arocrbench_khatt", offset=3, limit=2)
    monkeypatch.setattr(ds, "_load", lambda: rows[3:5])
    monkeypatch.setattr("arabic_ocr_bench.datasets.kitab_bench.cap_image", lambda img, m: img)
    ids = [s.id for s in ds]
    assert ids == ["arocrbench_khatt/train/000003", "arocrbench_khatt/train/000004"]
    assert [s.meta["row"] for s in ds] == [3, 4]


def test_muharaf_lines_ids_reflect_absolute_row_when_offset(monkeypatch):
    from arabic_ocr_bench.datasets.muharaf_lines import MuharafLines

    class FakeImg:
        def convert(self, mode):
            return self

    rows = [{"image": FakeImg(), "text": f"row{i}"} for i in range(5)]
    ds = MuharafLines(offset=3, limit=2)
    monkeypatch.setattr(ds, "_load", lambda: rows[3:5])
    monkeypatch.setattr("arabic_ocr_bench.datasets.muharaf_lines.cap_image", lambda img, m: img)
    ids = [s.id for s in ds]
    assert ids == ["muharaf_lines/test/000003", "muharaf_lines/test/000004"]
    assert [s.meta["row"] for s in ds] == [3, 4]


def test_resolve_hf_split_aliases_val_to_validation():
    assert resolve_hf_split("val", ["train", "validation", "test"]) == "validation"
    assert resolve_hf_split("validation", ["train", "validation", "test"]) == "validation"
    assert resolve_hf_split("train", ["train", "validation", "test"]) == "train"


def test_resolve_hf_split_keeps_val_when_the_dataset_actually_has_it():
    assert resolve_hf_split("val", ["train", "val", "test"]) == "val"


def test_splits_from_unknown_split_error_parses_datasets_message():
    from arabic_ocr_bench.datasets.kitab_bench import _splits_from_unknown_split_error

    err = ValueError("Unknown split \"val\". Should be one of ['train', 'validation', 'test'].")
    assert _splits_from_unknown_split_error(err) == ["train", "validation", "test"]
    assert _splits_from_unknown_split_error(ValueError("boom")) is None


def test_resolve_hf_split_unknown_raises():
    try:
        resolve_hf_split("holdout", ["train", "test"])
    except ValueError as e:
        assert "holdout" in str(e)
        assert "train" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
