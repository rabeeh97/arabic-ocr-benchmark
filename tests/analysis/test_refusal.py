from arabic_ocr_bench.analysis.refusal import FAILURE_CLASSES, classify_failure


def test_rate_limited_from_error():
    assert classify_failure("", "RateLimitError: 429 Rate limit exceeded", 4) == "rate_limited"


def test_other_error():
    assert classify_failure("", "TimeoutError: deadline", 1) == "other_error"


def test_empty_output_no_error():
    assert classify_failure("", None, 1) == "empty"
    assert classify_failure(" \n", None, 1) == "empty"


def test_wrong_script_long_latin():
    assert classify_failure("bismillah alrahman alrahim wa bihi nastaeen", None, 1) == "wrong_script"


def test_refusal_via_error_flag():
    assert classify_failure("I'm sorry, I can't read this image.", "refusal", 1) == "refusal"


def test_refusal_via_content_no_error():
    assert classify_failure("I'm sorry, I cannot read this image.", None, 1) == "refusal"


def test_short_arabic_with_one_latin_token_is_ok():
    assert classify_failure("رقم PDF هنا", None, 1) == "ok"


def test_meta_commentary_arabic():
    assert classify_failure("عذراً، لا يمكنني قراءة هذه الصورة بوضوح", None, 1) == "meta_commentary"


def test_code_fence_is_meta():
    assert classify_failure("```\nبسم الله\n```", None, 1) == "meta_commentary"


def test_prompt_echo():
    from arabic_ocr_bench.prompting import get_prompt
    assert classify_failure(get_prompt("line-v2"), None, 1) == "prompt_echo"


def test_plain_transcription_is_ok():
    assert classify_failure("بسم الله الرحمن الرحيم", None, 1) == "ok"


def test_classes_constant_has_eight():
    assert len(FAILURE_CLASSES) == 8 and "ok" in FAILURE_CLASSES
