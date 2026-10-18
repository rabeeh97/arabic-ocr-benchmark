"""TesseractClient and OCRCorrectedClient — the Tier-5 anchor and the
OCR-conditioned correction wrapper (docs/models.md Tier 5)."""

from PIL import Image

from arabic_ocr_bench.models.base import Prediction
from arabic_ocr_bench.models.registry import build_model
from arabic_ocr_bench.models.tesseract_client import OCRCorrectedClient, TesseractClient

_IMG = Image.new("RGB", (32, 32), "white")


class _RecordingClient:
    """Fake inner client that just records what prompt it was called with."""

    def __init__(self, name: str = "inner"):
        self.name = name
        self.seen_prompt: str | None = None

    def transcribe(self, image, prompt, max_tokens=None):
        del image, max_tokens
        self.seen_prompt = prompt
        return Prediction(raw_text=prompt, cleaned_text=prompt)


def test_tesseract_client_ignores_prompt_and_surfaces_missing_binary():
    # pytesseract/tesseract-ocr aren't installed in the test env — assert the
    # graceful failure mode (a tagged error), not a specific transcription.
    client = TesseractClient(name="tesseract-ara")
    pred = client.transcribe(_IMG, "this prompt should be ignored")
    assert pred.error is not None
    assert pred.error.startswith("tesseract_error:")


def test_ocr_corrected_falls_back_to_wrapped_client_when_tesseract_unavailable():
    inner = _RecordingClient()
    client = OCRCorrectedClient(name="wrapped", inner=inner)
    pred = client.transcribe(_IMG, "base prompt")
    # No Tesseract prior available (pytesseract not installed in the test
    # env) -> the wrapped client still runs, with the prompt unaugmented.
    assert pred.error is None
    assert inner.seen_prompt == "base prompt"


def test_registry_builds_ocr_corrected_client_from_config(monkeypatch):
    import arabic_ocr_bench.models.registry as registry

    monkeypatch.setattr(
        registry,
        "load_model_configs",
        lambda *a, **k: {"inner-mock": {"client": "mock", "mode": "empty"}},
    )
    client = build_model(
        "wrapped", {"client": "ocr_corrected", "inner": "inner-mock", "lang": "ara"}
    )
    assert isinstance(client, OCRCorrectedClient)
    assert client.inner.name == "inner-mock"


def test_registry_builds_tesseract_client_from_config():
    client = build_model("tesseract-ara", {"client": "tesseract", "lang": "ara"})
    assert isinstance(client, TesseractClient)
    assert client.lang == "ara"
