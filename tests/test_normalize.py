"""Tests for Arabic text normalization (written before implementation).

Uses explicit \\u escapes for the characters under test so expectations are
unambiguous. Arabic in comments is for the reader.
"""

import unicodedata

import pytest

from arabic_ocr_bench.normalize import (
    canonicalize_encoding,
    collapse_whitespace,
    fold_presentation_forms,
    normalize,
    normalize_alef,
    normalize_digits,
    normalize_hamza,
    normalize_lookalikes,
    normalize_taa_marbuta,
    normalize_yaa,
    strip_bidi_zw,
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


class TestEncodingCanonicalisation:
    def test_strip_bidi_and_zero_width(self):
        s = "a" + chr(0x200F) + "b" + chr(0x200D) + "c" + chr(0xFEFF) + "d"  # RLM, ZWJ, BOM
        assert strip_bidi_zw(s) == "abcd"

    def test_fold_presentation_form_letter_to_nominal(self):
        # U+FE9F ARABIC LETTER JEEM INITIAL FORM -> U+062C ARABIC LETTER JEEM
        assert fold_presentation_forms(chr(0xFE9F)) == "ج"

    def test_fold_lam_alef_ligature_to_two_letters(self):
        # U+FEFB ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM -> lam + alef
        assert fold_presentation_forms(chr(0xFEFB)) == "لا"

    def test_canonicalize_reorders_combining_marks_to_nfc(self):
        non_nfc = "شَّ"      # sheen, SHADDA, FATHA
        nfc = unicodedata.normalize("NFC", non_nfc)
        assert non_nfc != nfc               # they really differ as byte strings
        assert canonicalize_encoding(non_nfc) == nfc


class TestProfiles:
    def test_raw_profile_keeps_orthography_but_canonicalises_encoding(self):
        # tatweel, Arabic digits and harakat all SURVIVE raw
        s = f"كِتَابٌ  x{TATWEEL}y ١٢"
        assert normalize(s, "raw") == s
        # ...but a presentation form and a stray bidi mark are cleaned up
        assert normalize(chr(0x200E) + chr(0xFE9F) + "ميل", "raw") == "جميل"
        # ...and combining marks are put in canonical (NFC) order
        non_nfc = "ش" + chr(0x0651) + chr(0x064E)  # sheen, SHADDA, FATHA
        assert normalize(non_nfc, "raw") == unicodedata.normalize("NFC", non_nfc)

    def test_light_folds_persian_lookalikes_but_keeps_alef_maqsura(self):
        assert normalize_lookalikes("کی") == "كي"   # keheh, farsi-yaa
        assert normalize("کتاب", "light") == "كتاب"  # ک -> ك
        assert normalize(f"عل{ALEF_MAQSURA}", "light") == f"عل{ALEF_MAQSURA}"  # ى kept
        assert normalize(f"عل{ALEF_MAQSURA}", "diac_insensitive") == f"عل{YAA}"  # ى -> ي

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
