"""Tests for Arabic text normalization (written before implementation).

Uses explicit \\u escapes for the characters under test so expectations are
unambiguous. Arabic in comments is for the reader.
"""

import pytest

from arabic_ocr_bench.normalize import (
    collapse_whitespace,
    normalize,
    normalize_alef,
    normalize_digits,
    normalize_hamza,
    normalize_taa_marbuta,
    normalize_yaa,
    strip_harakat,
    strip_tatweel,
)

ALEF = "ا"  # ا
ALEF_HAMZA_ABOVE = "أ"  # أ
ALEF_HAMZA_BELOW = "إ"  # إ
ALEF_MADDA = "آ"  # آ
ALEF_WASLA = "ٱ"  # ٱ
YAA = "ي"  # ي
ALEF_MAQSURA = "ى"  # ى
TAA_MARBUTA = "ة"  # ة
HAA = "ه"  # ه
TATWEEL = "ـ"  # ـ
WAW = "و"  # و
WAW_HAMZA = "ؤ"  # ؤ
YAA_HAMZA = "ئ"  # ئ


class TestSingleSteps:
    def test_strip_tatweel_removes_kashida(self):
        assert strip_tatweel(f"k{TATWEEL}t{TATWEEL}b") == "ktb"

    def test_normalize_alef_unifies_all_forms(self):
        src = ALEF_HAMZA_ABOVE + ALEF_HAMZA_BELOW + ALEF_MADDA + ALEF_WASLA
        assert normalize_alef(src) == ALEF * 4

    def test_normalize_yaa_maps_alef_maqsura(self):
        assert normalize_yaa(f"عل{ALEF_MAQSURA}") == f"عل{YAA}"  # على -> علي

    def test_normalize_taa_marbuta_maps_to_haa(self):
        assert normalize_taa_marbuta(f"مدرس{TAA_MARBUTA}") == f"مدرس{HAA}"

    def test_normalize_hamza_maps_carriers(self):
        assert normalize_hamza(f"م{WAW_HAMZA}من") == f"م{WAW}من"  # مؤمن -> مومن
        assert normalize_hamza(f"ق{YAA_HAMZA}م") == f"ق{YAA}م"  # قئم -> قيم

    def test_strip_harakat_removes_short_vowels_and_shadda(self):
        # مُدَرِّسٌ -> مدرس
        src = "مُدَرِّسٌ"
        assert strip_harakat(src) == "مدرس"

    def test_strip_harakat_removes_dagger_alef(self):
        # هَٰذَا -> هذا  (contains U+0670 superscript/dagger alef)
        src = "هَٰذَا"
        assert strip_harakat(src) == "هذا"

    def test_normalize_digits_arabic_indic_to_ascii(self):
        assert normalize_digits("١٢٣٤٥٦٧٨٩٠") == "1234567890"

    def test_normalize_digits_extended_arabic_indic_to_ascii(self):
        assert normalize_digits("۰۱۲۹") == "0129"

    def test_collapse_whitespace_folds_runs_and_trims(self):
        assert collapse_whitespace("  a  b\n\tc  ") == "a b c"


class TestProfiles:
    def test_raw_profile_leaves_text_untouched(self):
        s = f"كِتَابٌ  x{TATWEEL}y ١٢"
        assert normalize(s, "raw") == s

    def test_light_profile_normalizes_alef_digits_tatweel_whitespace_but_keeps_harakat(self):
        # أحمــد  ١٢  ->  احمد 12
        src = f"{ALEF_HAMZA_ABOVE}حم{TATWEEL}{TATWEEL}د   ١٢"
        assert normalize(src, "light") == f"{ALEF}حمد 12"
        # harakat survive light
        withharaka = "مَدْر"
        assert normalize(withharaka, "light") == withharaka

    def test_diac_insensitive_strips_harakat_and_unifies_letters(self):
        # مَدْرَسَةٌ -> مدرسه
        src = "مَدْرَسَةٌ"
        assert normalize(src, "diac_insensitive") == "مدرسه"
        # إلى -> الي
        assert normalize(f"{ALEF_HAMZA_BELOW}ل{ALEF_MAQSURA}", "diac_insensitive") == f"{ALEF}ل{YAA}"

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            normalize("x", "bogus")
