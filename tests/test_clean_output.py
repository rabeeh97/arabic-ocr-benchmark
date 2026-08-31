"""clean_output must strip the markup / fences / preambles that OCR models emit,
leaving bare transcription text for scoring."""

from arabic_ocr_bench.models.litellm_client import clean_output


def test_strips_html_style_markup():
    # Qari-OCR v0.3 wraps words in <b>/<i>
    assert clean_output("مرحبا <b>بك</b> <i>هنا</i>") == "مرحبا بك هنا"


def test_strips_code_fences():
    assert clean_output("```\nالنص العربي\n```") == "النص العربي"


def test_strips_leading_markdown_marker():
    assert clean_output("## العنوان") == "العنوان"
    assert clean_output("- بند") == "بند"


def test_strips_english_preamble_and_wrapping_quotes():
    assert clean_output('Transcription: "مدرسة"') == "مدرسة"


def test_leaves_clean_arabic_untouched():
    s = "بسم الله الرحمن الرحيم"
    assert clean_output(s) == s


def test_does_not_eat_arabic_that_looks_nothing_like_a_tag():
    s = "قيمة x < y و a > b"  # stray comparison chars, not markup
    assert clean_output(s) == s
