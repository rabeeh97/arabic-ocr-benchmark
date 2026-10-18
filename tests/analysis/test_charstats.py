# tests/analysis/test_charstats.py
from pathlib import Path
from types import SimpleNamespace

import pytest

from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.charstats import (
    _decimate,
    error_anatomy,
    script_letters,
)
from arabic_ocr_bench.analysis.loader import iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _records():
    return [build_record(r) for r in iter_pred_rows(FIX)]


def _rbk():
    return {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}


def test_sid_split_has_pooled_and_per_model():
    out = error_anatomy(_records())
    assert "__pooled__" in out["sid_split"]
    assert "m1" in out["sid_split"]
    assert set(out["sid_split"]["m1"]) == {"sub", "ins", "del"}


def test_length_bias_quantiles_present():
    out = error_anatomy(_records())
    lb = out["length_bias"]["m1"]
    assert lb["delta_q1"] <= lb["delta_median"] <= lb["delta_q3"]


def test_rewrites_are_json_types():
    # Ruling 3: pass the source-row map so word rewrites are wired, and assert
    # the char rewrite list is populated (planted ب<->ت confusion: توت -> بوب)
    # and every entry is [str, str, int].
    rows_by_key = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    out = error_anatomy(_records(), rows_by_key)
    assert out["rewrites"]["char"], "char rewrites should be populated"
    for pair in out["rewrites"]["char"]:
        assert isinstance(pair[0], str)
        assert isinstance(pair[1], str)
        assert isinstance(pair[2], int)
    for pair in out["rewrites"]["word"]:
        assert isinstance(pair[0], str)
        assert isinstance(pair[1], str)
        assert isinstance(pair[2], int)


def test_rewrites_word_empty_without_rows_map():
    out = error_anatomy(_records())
    assert out["rewrites"]["word"] == []


def test_spacing_share_between_zero_and_one():
    out = error_anatomy(_records())
    for v in out["spacing_share"].values():
        assert 0.0 <= v["spacing_frac_of_ops"] <= 1.0


@pytest.mark.parametrize("n", [0, 1, 399, 400, 401, 799, 800, 1000, 40001])
def test_decimate_never_exceeds_cap(n):
    out = _decimate(list(range(n)), 400)
    assert len(out) <= 400
    if n:
        assert out[0] == 0  # strided slice starts at the head


def test_decimate_keeps_all_when_under_cap():
    assert _decimate([(1, 1), (2, 2), (3, 3)], 400) == [(1, 1), (2, 2), (3, 3)]


def _ok_stub(model, sample_id, *, ref_w=3, hyp_w=3):
    return SimpleNamespace(
        failure_class="ok", model=model, sample_id=sample_id, char_ops=[],
        wer_light=0.0, spacing_del=0, spacing_ins=0,
        len_ref_words=ref_w, len_hyp_words=hyp_w,
    )


def test_pred_vs_gt_capped_at_400_points():
    recs = [_ok_stub("mBig", f"dsx/test/{i:06d}") for i in range(1000)]
    out = error_anatomy(recs)
    assert len(out["length_bias"]["mBig"]["pred_vs_gt"]) <= 400


def test_sid_split_pooled_present_even_with_no_char_ops():
    out = error_anatomy([_ok_stub("mPerfect", "dsx/test/000000")])
    assert out["sid_split"]["__pooled__"] == {"sub": 0, "ins": 0, "del": 0}


def test_output_is_json_serialisable():
    import json

    rows_by_key = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    out = error_anatomy(_records(), rows_by_key)
    json.dumps(out)


# --------------------------------------------------------------------------- #
# Task 6: script_letters()
# --------------------------------------------------------------------------- #


def test_confusion_captures_planted_bt():
    out = script_letters(_records(), _rbk())
    pooled = {(a, b) for a, b, _ in out["confusion"]["pooled"]}
    assert ("ت", "ب") in pooled or ("ب", "ت") in pooled


def test_rasm_gap_nonnegative_and_present():
    out = script_letters(_records(), _rbk())
    for per_ds in out["rasm_gap"].values():
        for v in per_ds.values():
            assert v["gap"] >= -1e-9
            assert abs(v["gap"] - (v["cer_light"] - v["cer_rasm"])) < 1e-9


def test_presentation_leak_rate_bounds():
    out = script_letters(_records(), _rbk())
    for v in out["presentation_leak"].values():
        assert 0.0 <= v["rate"] <= 1.0


def test_all_json_types():
    import json

    json.dumps(script_letters(_records(), _rbk()))  # must not raise
