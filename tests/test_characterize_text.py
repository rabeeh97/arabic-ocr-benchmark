from arabic_ocr_bench.characterize.text import (
    analyze_letters,
    diacritics,
    entropy,
    hygiene,
    lam_alef_count,
    line_features,
    rasm_ambiguous,
    rasm_of,
)


def _forms(s):
    return [(li.letter, li.position) for li in analyze_letters(s)]


def test_positional_forms_dual_joining_word():
    assert _forms("بيت") == [("ب", "initial"), ("ي", "medial"), ("ت", "final")]


def test_non_connectors_make_isolated_forms_and_paws():
    letters = analyze_letters("دار")
    assert [li.position for li in letters] == ["isolated"] * 3
    assert [li.paw for li in letters] == [0, 1, 2]


def test_paws_split_after_non_connector_inside_word():
    # مدرسة = مد | ر | سة  -> 3 PAWs, one word
    letters = analyze_letters("مدرسة")
    assert len({li.paw for li in letters}) == 3
    assert {li.word for li in letters} == {0}


def test_marks_are_transparent_for_joining():
    assert _forms("كَتَبَ") == _forms("كتب")
    assert analyze_letters("كَتَبَ")[0].marks == "َ"


def test_tatweel_joins_but_is_not_a_letter():
    assert _forms("بـا") == [("ب", "initial"), ("ا", "final")]


def test_presentation_forms_are_folded_before_analysis():
    # U+FEFB = lam-alef ligature presentation form
    assert _forms("ﻻ") == [("ل", "initial"), ("ا", "final")]


def test_rasm_is_position_aware():
    assert rasm_of("ن", "initial") == rasm_of("ب", "initial") == "ب"
    assert rasm_of("ن", "final") == "ں"
    assert rasm_of("ق", "medial") == "ف"
    assert rasm_of("ق", "final") == "ٯ"


def test_dot_ambiguity():
    assert rasm_ambiguous("ت", "initial")
    assert not rasm_ambiguous("ك", "initial")
    assert not rasm_ambiguous("ا", "isolated")  # hamza variants are not dots
    assert not rasm_ambiguous("ف", "final")      # final ق has its own skeleton


def test_lam_alef():
    assert lam_alef_count(analyze_letters("لا السلام")) == 2


def test_diacritics_coverage_and_invalid_marks():
    full = "كَتَبَ"
    d = diacritics(full, analyze_letters(full))
    assert d["coverage"] == 1.0 and d["types"]["fatha"] == 3 and d["invalid"] == 0
    bad = "ك َ"  # a fatha hosted by a space
    assert diacritics(bad, analyze_letters(bad))["invalid"] == 1


def test_hygiene_counts_raw_artifacts():
    h = hygiene("‏کتاب ١٢ 34 ab")
    assert h["bidi_zero_width"] == 1
    assert h["persian_lookalikes"] == 1
    assert h["digits_arabic_indic"] == 2 and h["digits_western"] == 2
    assert h["latin_letters"] == 2


def test_line_features_shape():
    f = line_features("السلام عليكم")
    assert f["words"] == 2 and f["letters"] == 11
    assert sum(f["positions"].values()) == f["letters"]


def test_entropy():
    assert entropy({"a": 1, "b": 1}) == 1.0
