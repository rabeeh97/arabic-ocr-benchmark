from pathlib import Path

import pytest

from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.difficulty import difficulty
from arabic_ocr_bench.analysis.loader import iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def test_bad_gt_line_is_flagged():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    ids = {d["sample_id"] for d in out["suspected_bad_gt"]}
    assert "dsx/test/000003" in ids  # planted: models agree, differ from GT


def test_perfect_line_is_consensus_easy():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    ids = {d["sample_id"] for d in out["consensus_easy"]}
    assert "dsx/test/000004" in ids


def test_cer_distribution_quantile_order():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    assert out["cer_distribution"]  # non-empty: the loop below must assert something
    for per_ds in out["cer_distribution"].values():
        for v in per_ds.values():
            assert v["min"] <= v["median"] <= v["max"]


def test_ecdf_monotonic():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    assert out["ecdf"]  # non-empty: the loop below must assert something
    for pts in out["ecdf"].values():
        ys = [y for _, y in pts]
        assert ys == sorted(ys)


def test_json_serialisable():
    import json

    rows, recs = _setup()
    json.dumps(difficulty(rows, recs))


def test_consensus_hard_entry_shape_and_errored_handling():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    entry = next(
        d for d in out["consensus_hard"] if d["sample_id"] == "dsy/test/000000"
    )
    assert set(entry) == {
        "sample_id", "dataset", "gt", "mean_cer", "n_models", "per_model",
    }
    # m1 & m3 rate-limit-errored -> null in per_model; m2 OK (~0.2).
    assert entry["per_model"]["m1"] is None
    assert entry["per_model"]["m3"] is None
    assert entry["n_models"] == 1
    m2_cer = entry["per_model"]["m2"]
    assert m2_cer is not None
    # errored models contribute cer 1.0 to the mean.
    assert entry["mean_cer"] == pytest.approx((1.0 + m2_cer + 1.0) / 3)


def test_suspected_bad_gt_entry_keys():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    assert out["suspected_bad_gt"]
    for entry in out["suspected_bad_gt"]:
        assert set(entry) == {
            "sample_id", "dataset", "gt", "score",
            "mean_pairwise_pred_cer", "mean_pred_vs_gt_cer", "n_models",
        }
