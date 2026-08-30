"""Tests for CER/WER, diacritic P/R/F1, and aggregation (written before impl)."""

import pytest

from arabic_ocr_bench.metrics import (
    aggregate,
    cer,
    char_edit_counts,
    diacritic_prf,
    score_pair,
    wer,
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
