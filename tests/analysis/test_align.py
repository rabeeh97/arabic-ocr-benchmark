# tests/analysis/test_align.py
from arabic_ocr_bench.analysis.align import (
    LETTER_GROUPS,
    RASM_MAP,
    AlignmentRecord,
    CharOp,
    build_record,
    expand_char_chunks,
    letter_group,
    to_rasm,
)
from arabic_ocr_bench.analysis.loader import PredRow


def _row(gt, pred, **kw):
    d = dict(model="m1", dataset="dsx", sample_id="dsx/test/000000", gt=gt,
             pred=pred, error=None, attempts=1, latency_ms=1.0, cost_usd=0.0,
             input_tokens=0, output_tokens=0, cached=False, scores={})
    d.update(kw)
    return PredRow(**d)


def test_expand_equal_length_substitution():
    ops = expand_char_chunks("abc", "aXc")
    assert ops == [CharOp("sub", "b", "X", 1)]


def test_expand_pure_insertion():
    ops = expand_char_chunks("ab", "abc")
    assert ops == [CharOp("ins", "", "c", -1)]


def test_expand_pure_deletion():
    ops = expand_char_chunks("abc", "ab")
    assert ops == [CharOp("del", "c", "", 2)]


def test_expand_unequal_substitution_spills_to_insert():
    # ref "b" vs hyp "XY": the extra hyp char spills to a pure insert and the
    # aligned position stays a substitution. jiwer 4.0.0 pre-splits this region
    # as [insert "X", substitute b->Y] (see task-4-report: the brief predicted
    # the mirror split [insert "Y", substitute b->X]).
    ops = expand_char_chunks("abd", "aXYd")
    assert CharOp("sub", "b", "Y", 1) in ops
    assert CharOp("ins", "", "X", -1) in ops
    assert [o.op for o in ops] == ["ins", "sub"]


def test_build_record_stub_for_error():
    r = build_record(_row("مدرسة", "", error="RateLimitError: 429"))
    assert isinstance(r, AlignmentRecord)
    assert r.failure_class == "rate_limited"
    assert r.char_ops == []
    assert r.len_ref_chars == 0


def test_build_record_bt_confusion_is_substitution():
    r = build_record(_row("توت", "بوب"))
    subs = [(o.ref, o.hyp) for o in r.char_ops if o.op == "sub"]
    assert ("ت", "ب") in subs


def test_build_record_rasm_gap_smaller_than_light():
    # ت<->ب is a dot-only error: CER on the rasm skeleton is 0, light CER > 0
    r = build_record(_row("توت", "بوب"))
    assert r.cer_rasm == 0.0
    assert r.cer_light > 0.0


def test_build_record_diacritic_sequences():
    r = build_record(_row("مَدْرَسَة", "مدرسة"))
    assert r.diac_seq_ref != "" and r.diac_seq_hyp == ""


def test_build_record_spacing_op_counted():
    r = build_record(_row("بيت كبير", "بيتكبير"))
    assert r.spacing_del == 1


def test_rasm_collapses_dotted_family():
    # ب ت ث ن ي all share the same rasm -> to_rasm makes them equal
    skeletons = {to_rasm(c) for c in "بتثني"}
    assert len(skeletons) == 1


def test_rasm_passes_through_unmapped():
    assert to_rasm("ا") == "ا"
    assert to_rasm("م") == "م"


def test_rasm_string():
    assert to_rasm("توت") == to_rasm("بوب")  # both -> <skeleton>و<skeleton>


def test_letter_group_dot_confusable():
    assert letter_group("ت") == "dot_confusable"
    assert letter_group("ب") == "dot_confusable"


def test_letter_group_hamza_carrier():
    assert letter_group("ؤ") == "hamza_carrier"


def test_letter_group_unknown_is_none():
    assert letter_group("=") is None
    assert letter_group("x") is None


def test_groups_are_nonempty():
    assert all(v for v in LETTER_GROUPS.values())
    assert "dot_confusable" in LETTER_GROUPS
