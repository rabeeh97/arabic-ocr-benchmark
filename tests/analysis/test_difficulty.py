from pathlib import Path

import pytest

from arabic_ocr_bench.analysis.align import AlignmentRecord, build_record
from arabic_ocr_bench.analysis.difficulty import difficulty
from arabic_ocr_bench.analysis.loader import PredRow, iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def _row(model, sid, cer, err=None):
    return PredRow(
        model=model, dataset="dsz", sample_id=sid, gt="جملة عربية طويلة جدا",
        pred="جملة عربية", error=err, attempts=1, latency_ms=1.0, cost_usd=0.0,
        input_tokens=0, output_tokens=0, cached=False,
        scores={} if err else {"cer_light": cer},
    )


def _rec(model, sid, fc, cer):
    return AlignmentRecord(
        model=model, dataset="dsz", sample_id=sid, failure_class=fc,
        char_ops=[], word_ops={"sub": 0, "ins": 0, "del": 0, "hits": 0},
        rasm_ops={"sub": 0, "ins": 0, "del": 0, "ref_len": 0},
        diac_seq_ref="", diac_seq_hyp="", spacing_del=0, spacing_ins=0,
        len_ref_chars=18, len_hyp_chars=10, len_ref_words=4, len_hyp_words=2,
        digit_swaps=0, presentation_form_chars=0, latin_run_len=0,
        has_code_fence=False, cer_light=cer, cer_raw=cer, cer_rasm=cer,
        wer_light=cer,
    )


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


def test_content_failure_row_is_not_a_scored_model():
    """A ``meta_commentary`` row (error is None, partial transcription, cer < 1)
    must NOT count toward a line's ``n_models`` and its cer must not enter
    ``mean_cer`` — it is gated on ``failure_class == 'ok'``, not ``error``."""
    sid = "dsz/test/000000"
    rows, recs = [], []
    for m in ("m1", "m2", "m3"):
        rows.append(_row(m, sid, 0.05))
        recs.append(_rec(m, sid, "ok", 0.05))
    rows.append(_row("m4", sid, 0.4))  # error is None, scores has cer_light
    recs.append(_rec("m4", sid, "meta_commentary", 0.4))

    out = difficulty(rows, recs)

    hard = next(d for d in out["consensus_hard"] if d["sample_id"] == sid)
    assert hard["n_models"] == 3
    assert hard["per_model"]["m4"] is None
    # m4 contributes 1.0 (full miss), not its 0.4, to the mean over all 4 models
    assert hard["mean_cer"] == pytest.approx((0.05 * 3 + 1.0) / 4)

    disc = next(d for d in out["discriminative"] if d["sample_id"] == sid)
    assert disc["n_models"] == 3
    assert disc["per_model"]["m4"] is None


def test_suspected_bad_gt_entry_keys():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    assert out["suspected_bad_gt"]
    for entry in out["suspected_bad_gt"]:
        assert set(entry) == {
            "sample_id", "dataset", "gt", "score",
            "mean_pairwise_pred_cer", "mean_pred_vs_gt_cer", "n_models",
        }
