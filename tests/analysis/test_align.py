# tests/analysis/test_align.py
from arabic_ocr_bench.analysis.align import (
    LETTER_GROUPS,
    RASM_MAP,
    letter_group,
    to_rasm,
)


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
