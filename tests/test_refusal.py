"""looks_like_refusal flags a model that declined instead of transcribing."""

from arabic_ocr_bench.models.litellm_client import looks_like_refusal


def test_english_refusal():
    assert looks_like_refusal("I'm sorry, I can't help with that.")
    assert looks_like_refusal("As an AI, I am unable to read this image.")


def test_arabic_refusal():
    # "Sorry, I cannot determine whether these characters are Arabic..."
    assert looks_like_refusal("عذراً، لا يمكنني تحديد ما إذا كانت هذه الأحرف عربية")
    assert looks_like_refusal("يرجى تقديم المزيد من المعلومات")


def test_a_real_transcription_is_not_a_refusal():
    assert not looks_like_refusal("بسم الله الرحمن الرحيم")
    assert not looks_like_refusal("اهدي سلامي الي جاركم الخواجه اسعد")
