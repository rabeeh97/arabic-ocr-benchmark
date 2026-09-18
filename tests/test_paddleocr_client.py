"""PaddleOCR client — offline logic only (no paddle install needed/available here)."""

from __future__ import annotations

from arabic_ocr_bench.models import paddleocr_client
from arabic_ocr_bench.models.paddleocr_client import PaddleOCRClient, _extract_text
from arabic_ocr_bench.models.registry import build_model


class _FakeTextRecResult(dict):
    """Stands in for paddlex's TextRecResult, which behaves dict-like."""


def test_extract_text_from_dict_like_result():
    result = [_FakeTextRecResult(rec_text="بسم الله", rec_score=0.9)]
    assert _extract_text(result) == "بسم الله"


def test_extract_text_from_attribute_object():
    class Obj:
        rec_text = "الحمد لله"

    assert _extract_text([Obj()]) == "الحمد لله"


def test_extract_text_handles_empty_result():
    assert _extract_text([]) == ""
    assert _extract_text(None) == ""


def test_reverse_output_undoes_the_confirmed_ltr_ordering_bug(monkeypatch):
    """arabic_PP-OCRv5_mobile_rec (tested 2026-09-17) emits RTL text in raw LTR
    pixel order; reverse_output=True is the documented fix."""
    client = PaddleOCRClient(name="test", reverse_output=True)

    class FakeModel:
        def predict(self, input, batch_size):
            return [_FakeTextRecResult(rec_text="مي رل انمحدلا ملامسب")]

    monkeypatch.setattr(client, "_ensure_loaded", lambda: FakeModel())
    monkeypatch.setattr(paddleocr_client, "_image_array", lambda image: None)

    from PIL import Image

    pred = client.transcribe(Image.new("RGB", (100, 30), "white"), prompt="")
    assert pred.error is None, pred.error
    assert pred.raw_text == "بسمالم الدحمنا لر يم"  # == "مي رل انمحدلا ملامسب"[::-1]


def test_without_reverse_output_text_stays_as_returned(monkeypatch):
    client = PaddleOCRClient(name="test", reverse_output=False)

    class FakeModel:
        def predict(self, input, batch_size):
            return [_FakeTextRecResult(rec_text="abc")]

    monkeypatch.setattr(client, "_ensure_loaded", lambda: FakeModel())
    monkeypatch.setattr(paddleocr_client, "_image_array", lambda image: None)

    from PIL import Image

    pred = client.transcribe(Image.new("RGB", (100, 30), "white"), prompt="")
    assert pred.error is None, pred.error
    assert pred.raw_text == "abc"


def test_transcribe_reports_errors_instead_of_raising(monkeypatch):
    client = PaddleOCRClient(name="test")

    def _boom():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(client, "_ensure_loaded", _boom)

    from PIL import Image

    pred = client.transcribe(Image.new("RGB", (100, 30), "white"), prompt="")
    assert pred.error is not None
    assert "RuntimeError" in pred.error


def test_build_model_forwards_device_and_does_not_import_paddle(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_DEVICE", "cpu")
    client = build_model(
        "arabic-paddleocr-v5",
        {
            "client": "paddleocr",
            "model_name": "arabic_PP-OCRv5_mobile_rec",
            "reverse_output": True,
        },
    )
    assert isinstance(client, PaddleOCRClient)
    assert client.reverse_output is True
    assert client.device == "cpu"
    assert client._model is None


def test_yaml_device_overrides_env(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_DEVICE", "cpu")
    client = build_model(
        "arabic-paddleocr-v5",
        {"client": "paddleocr", "reverse_output": True, "device": "gpu:0"},
    )
    assert client.device == "gpu:0"
