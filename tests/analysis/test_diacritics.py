# tests/analysis/test_diacritics.py
from pathlib import Path

from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.diacritics import diacritics
from arabic_ocr_bench.analysis.loader import iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    recs = [build_record(r) for r in rows]
    rbk = {(r.model, r.sample_id): r for r in rows}
    return recs, rbk


def test_attempt_rate_bounds():
    recs, rbk = _setup()
    out = diacritics(recs, rbk)
    for v in out["f1_vs_cer"].values():
        assert 0.0 <= v["attempt_rate"] <= 1.0


def test_per_mark_keys_are_names():
    recs, rbk = _setup()
    out = diacritics(recs, rbk)
    for per in out["per_mark"].values():
        for name in per:
            assert name in {"fatha", "damma", "kasra", "shadda", "sukun",
                            "tanwin_fath", "tanwin_damm", "tanwin_kasr",
                            "dagger_alef", "madda", "other"}


def test_json_serialisable():
    import json

    recs, rbk = _setup()
    json.dumps(diacritics(recs, rbk))


def test_planted_diacritic_line_scored():
    # dsx/test/000006 GT مَدْرَسَة: m1 reads it bare (4 missed fatha/sukun),
    # m2 keeps one fatha (3 missed), m3 exact (no diacritic ops).
    recs, rbk = _setup()
    out = diacritics(recs, rbk)

    # m1 dropped every mark -> misses bucketed as fatha / sukun
    m1 = out["per_mark"]["m1"]
    assert m1.get("fatha", {}).get("missed", 0) == 3
    assert m1.get("sukun", {}).get("missed", 0) == 1

    # m3 got it exactly right -> no diacritic errors at all
    assert out["per_mark"].get("m3", {}) == {}

    # attempt_rate = share of ALL OK records whose prediction carried a harakat.
    # Only dsx/test/000006 is diacritised, so the rate is small and bounded;
    # m1 read that one line bare -> 0 of its OK records attempted a mark.
    assert out["f1_vs_cer"]["m1"]["attempt_rate"] == 0.0
    assert 0.0 < out["f1_vs_cer"]["m3"]["attempt_rate"] <= 1.0

    # gt_density is a positive fraction for the diacritised dataset
    assert 0.0 < out["gt_density"]["dsx"] <= 1.0


def test_f1_vs_cer_uses_scores_diac_f1():
    recs, rbk = _setup()
    out = diacritics(recs, rbk)
    # m1's only diacritised line scores diac_f1 == 0.0; all its other OK
    # lines have diac_f1 == 1.0 -> mean strictly between 0 and 1.
    assert 0.0 < out["f1_vs_cer"]["m1"]["diac_f1"] < 1.0
    for v in out["f1_vs_cer"].values():
        assert isinstance(v["diac_f1"], float)
        assert isinstance(v["cer_light"], float)
