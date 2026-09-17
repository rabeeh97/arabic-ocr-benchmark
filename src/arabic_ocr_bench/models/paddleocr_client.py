"""PaddleOCR client — a fundamentally different stack from the rest of this
package: PaddlePaddle, not PyTorch, and a dedicated recognition model, not a
VLM. Deliberately self-contained (no shared code with hf_local_client.py or
training/) since PaddleOCR's own training tooling is a separate wrapped
process too — see scripts/finetune_paddleocr.py.

Requires the `paddleocr` extra, NOT installed by default (see
docs/experiments/paddleocr-runbook.md for the exact install command and a
real, box-specific DLL-loading gotcha to check for first):

    uv pip install paddleocr
    uv pip install paddlepaddle-gpu==<version> -i <paddle-wheel-index-for-your-CUDA>

VERIFIED ON HARDWARE (2026-09-17, paddleocr 3.7.0 / paddlepaddle-gpu 3.2.2,
CUDA 12.6, RTX 3090): confirmed the real return type is a
`paddlex.inference.models.text_recognition.result.TextRecResult`, whose
`rec_text` key holds the string — the `_TEXT_KEYS`/attribute fallbacks below
are defensive against future schema drift, not a guess.

**Confirmed bug, not a guess**: `arabic_PP-OCRv5_mobile_rec` returns Arabic
text in raw left-to-right pixel/reading order, uncorrected for RTL script —
tested against our own synthetic renderer's ground truth
("بسم الله الرحمن الرحيم" -> reversed-order garbled output, character-level
comparison against the char-reversed ground truth confirms the ordering, not
just noise). `reverse_output=True` (the default `arabic-paddleocr-v5` config
row sets) undoes this. Character-level accuracy was still poor even after
reversal on a clean, legible synthetic line and much worse on a
fully-diacritized (harakat) line — see
docs/experiments/paddleocr-runbook.md for the actual side-by-side transcript.
Treat the untuned baseline as weak, consistent with the pre-integration
research's caveat that this is a small shared 8-language model, not a deep
Arabic specialist.
"""

from __future__ import annotations

import time

from PIL import Image

from ..imaging import cap_image
from .base import Prediction

DEFAULT_MODEL_NAME = "arabic_PP-OCRv5_mobile_rec"

# PaddleOCR's own docs use "rec_text"; other schema names have appeared across
# PaddleOCR/PaddleX API versions. Try each in order rather than hardcoding one
# guess — mirrors KitabBench's `_GT_KEYS` multi-column fallback pattern used
# elsewhere in this repo for the same reason (an upstream schema that varies).
_TEXT_KEYS = ("rec_text", "text", "transcription")


def _extract_text(result) -> str:
    if not result:
        return ""
    item = result[0]
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in _TEXT_KEYS:
            if k in item and isinstance(item[k], str):
                return item[k]
        return ""
    for k in _TEXT_KEYS:
        v = getattr(item, k, None)
        if isinstance(v, str):
            return v
    return ""


class PaddleOCRClient:
    """No chat/prompt concept — PaddleOCR's recognition model takes an image
    only, so `prompt` is accepted (to satisfy the `ModelClient` protocol) and
    ignored."""

    def __init__(
        self, name: str, model_name: str = DEFAULT_MODEL_NAME,
        reverse_output: bool = False, **_ignored,
    ):
        self.name = name
        self.model_name = model_name
        # See this module's docstring: arabic_PP-OCRv5_mobile_rec (as of the
        # 2026-09-17 hardware test) emits characters in raw LTR pixel order,
        # uncorrected for RTL script. Confirmed empirically, not assumed —
        # don't set this blindly for a different/future model without re-testing.
        self.reverse_output = reverse_output
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from paddleocr import TextRecognition

            self._model = TextRecognition(model_name=self.model_name)
        return self._model

    def transcribe(self, image: Image.Image, prompt: str) -> Prediction:
        del prompt
        t0 = time.monotonic()
        try:
            model = self._ensure_loaded()
            import numpy as np

            arr = np.array(cap_image(image.convert("RGB"), 1400))
            result = model.predict(input=arr, batch_size=1)
            text = _extract_text(result)
            if self.reverse_output:
                text = text[::-1]
        except Exception as e:  # noqa: BLE001
            return Prediction(
                error=f"paddleocr_error:{type(e).__name__}: {e}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        return Prediction(
            raw_text=text, cleaned_text=text.strip(), latency_ms=(time.monotonic() - t0) * 1000
        )
