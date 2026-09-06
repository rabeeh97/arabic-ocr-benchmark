"""Tests for CER/WER, diacritic P/R/F1, and aggregation (written before impl)."""

import pytest

from arabic_ocr_bench.metrics import (
    aggregate,
    car,
    cer,
    char_edit_counts,
    diacritic_prf,
    mer,
    score_pair,
    wer,
    wil,
    wip,
)

KASRA = "ِ"
FATHA = "َ"


class TestCer:
    def test_identical_is_zero(self):
        assert cer("مرحبا", "مرحبا") == 0.0

    def test_one_substitution_in_four_chars(self):
        assert cer("abcd", "abXd") == 0.25

    def test_empty_hypothesis_is_one(self):
        assert cer("abcd", "") == 1.0

    def test_both_empty_is_zero(self):
        assert cer("", "") == 0.0

    def test_empty_reference_nonempty_hyp_is_one(self):
        assert cer("", "abc") == 1.0


class TestWer:
    def test_identical_is_zero(self):
        assert wer("the cat sat", "the cat sat") == 0.0

    def test_one_deletion_in_three_words(self):
        assert wer("the cat sat", "the cat") == pytest.approx(1 / 3)

    def test_both_empty_is_zero(self):
        assert wer("", "") == 0.0


class TestCar:
    def test_car_is_one_minus_cer(self):
        assert car("abcd", "abXd") == pytest.approx(0.75)

    def test_car_is_one_for_perfect(self):
        assert car("مرحبا", "مرحبا") == 1.0

    def test_car_clamped_at_zero_for_long_garbage(self):
        # 2 ref chars, a flood of insertions -> CER > 1 -> CAR floored at 0
        assert car("ab", "abXXXXXXXXXX") == 0.0


class TestWordFamily:
    def test_identical_is_perfect(self):
        assert wer("the cat sat", "the cat sat") == 0.0
        assert mer("the cat sat", "the cat sat") == 0.0
        assert wil("the cat sat", "the cat sat") == 0.0
        assert wip("the cat sat", "the cat sat") == 1.0

    def test_mer_bounded_when_wer_exceeds_one(self):
        # 1 ref word, 4 hyp words: WER = 4/1 = 4.0, MER = 4/(4+0)... hits=0 -> 1.0
        assert wer("hello", "a b c d") == pytest.approx(4.0)
        assert mer("hello", "a b c d") == 1.0
        assert 0.0 <= wil("hello", "a b c d") <= 1.0

    def test_wip_accounts_for_invented_words(self):
        ref = "the cat sat"
        hyp = "the cat sat on the warm mat"  # all 3 ref words kept, 4 invented
        # H=3, N_ref=3, N_hyp=7  -> WIP = (3/3)*(3/7) = 3/7
        assert wip(ref, hyp) == pytest.approx(3 / 7)
        assert wil(ref, hyp) == pytest.approx(1 - 3 / 7)
        # perfect recall (every ref word present) yet WIP is well under 1 because
        # precision is only 3/7 — plain recall-style scoring would miss that
        assert wil(ref, hyp) > 0.5

    def test_empty_reference_conventions(self):
        assert wer("", "") == 0.0 and mer("", "") == 0.0
        assert wip("", "") == 1.0 and wil("", "") == 0.0
        assert mer("", "a b") == 1.0 and wip("", "a b") == 0.0


class TestCharEditCounts:
    def test_substitution_counted(self):
        assert char_edit_counts("abcd", "abXd") == {
            "sub": 1,
            "del": 0,
            "ins": 0,
            "ref_len": 4,
        }

    def test_deletion_counted(self):
        assert char_edit_counts("abcd", "abd") == {
            "sub": 0,
            "del": 1,
            "ins": 0,
            "ref_len": 4,
        }


class TestDiacriticPrf:
    def test_perfect_when_neither_side_has_diacritics(self):
        r = diacritic_prf("كتاب", "كتاب")
        assert (r["precision"], r["recall"], r["f1"]) == (1.0, 1.0, 1.0)

    def test_all_matched_when_marks_identical(self):
        s = f"ك{KASRA}ت{FATHA}اب"
        r = diacritic_prf(s, s)
        assert (r["precision"], r["recall"], r["f1"]) == (1.0, 1.0, 1.0)

    def test_recall_zero_when_hypothesis_drops_all_marks(self):
        ref = f"ك{KASRA}ت{FATHA}اب"
        r = diacritic_prf(ref, "كتاب")
        assert r["recall"] == 0.0
        assert r["precision"] == 1.0
        assert r["f1"] == 0.0

    def test_precision_zero_when_hypothesis_invents_marks(self):
        hyp = f"ك{KASRA}تاب"
        r = diacritic_prf("كتاب", hyp)
        assert r["precision"] == 0.0
        assert r["recall"] == 1.0
        assert r["f1"] == 0.0


class TestScorePair:
    def test_normalization_profile_changes_cer_and_exact_match(self):
        # أحمد vs احمد : differ only by the alef form
        out = score_pair("أحمد", "احمد", profiles=("raw", "light"))
        assert out["cer"]["raw"] == 0.25
        assert out["cer"]["light"] == 0.0
        assert out["exact_match"]["raw"] is False
        assert out["exact_match"]["light"] is True

    def test_diacritic_score_is_computed_on_raw_text_not_normalized(self):
        # diac_insensitive CER would be 0, but the diacritic metric must still
        # see that the hypothesis dropped the harakat.
        out = score_pair(f"م{FATHA}درسة", "مدرسة", profiles=("diac_insensitive",))
        assert out["cer"]["diac_insensitive"] == 0.0
        assert out["diacritic"]["recall"] == 0.0


class TestAggregate:
    def test_macro_cer_is_mean_of_per_pair_cer(self):
        p1 = score_pair("ab", "ab", profiles=("raw",))          # cer 0.0, ref_len 2
        p2 = score_pair("abcdefgh", "abcdXXXX", profiles=("raw",))  # cer 0.5, ref_len 8
        agg = aggregate([p1, p2], profiles=("raw",))
        assert agg["cer_macro"]["raw"] == pytest.approx(0.25)

    def test_micro_cer_weights_by_reference_length(self):
        p1 = score_pair("ab", "ab", profiles=("raw",))              # 0 errors / 2
        p2 = score_pair("abcdefgh", "abcdXXXX", profiles=("raw",))  # 4 errors / 8
        agg = aggregate([p1, p2], profiles=("raw",))
        assert agg["micro_cer"]["raw"] == pytest.approx(4 / 10)

    def test_exact_match_rate(self):
        p1 = score_pair("ab", "ab", profiles=("raw",))
        p2 = score_pair("ab", "aX", profiles=("raw",))
        agg = aggregate([p1, p2], profiles=("raw",))
        assert agg["exact_match_rate"]["raw"] == 0.5

    def test_micro_car_is_one_minus_micro_cer(self):
        p1 = score_pair("ab", "ab", profiles=("raw",))              # 0 / 2
        p2 = score_pair("abcdefgh", "abcdXXXX", profiles=("raw",))  # 4 / 8
        agg = aggregate([p1, p2], profiles=("raw",))
        assert agg["micro_car"]["raw"] == pytest.approx(1 - 4 / 10)

    def test_micro_mer_and_wil_are_populated_and_bounded(self):
        p1 = score_pair("the cat sat", "the cat sat", profiles=("raw",))
        p2 = score_pair("a b c", "x y z w v", profiles=("raw",))
        agg = aggregate([p1, p2], profiles=("raw",))
        for key in ("micro_mer", "micro_wil", "micro_wip", "mer_macro", "wil_macro"):
            assert 0.0 <= agg[key]["raw"] <= 1.0
        assert agg["micro_wil"]["raw"] == pytest.approx(1 - agg["micro_wip"]["raw"])

    def test_score_pair_exposes_new_metric_keys(self):
        out = score_pair("مرحبا", "مرحبا", profiles=("light",))
        for k in ("car", "mer", "wil", "wip"):
            assert k in out and "light" in out[k]
