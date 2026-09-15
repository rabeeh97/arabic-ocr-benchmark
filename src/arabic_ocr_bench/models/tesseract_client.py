"""Classical-OCR clients: a plain Tesseract baseline, and an OCR-conditioned
correction wrapper that feeds Tesseract's raw pass to a VLM as a prior.

Both ignore the harness's usual `max_tokens` never-lowers plumbing where it
doesn't apply to them: `TesseractClient` takes no tokens at all (it's not an
LLM), and `OCRCorrectedClient` simply forwards whatever it's given to the
wrapped client, which already implements that logic itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image

from .base import ModelClient, Prediction

# arXiv 2608.22366 ("When Do VLMs Help Arabic Manuscript OCR?") finds
# OCR-conditioned correction helps most on clean printed text — up to 53%
# relative CER improvement over Tesseract alone on their aged-print analog —
# but actively HURTS on cursive/Maghrebi manuscripts, where Tesseract's own
# output is unreliable enough that the "prior" misleads the corrector more
# than it helps. This client is deliberately meant for kitab_hindawi /
# kitab_historicalbooks (printed book subsets), not RASAM/Muharaf/NAKBA.
CORRECTION_INSTRUCTION = (
    "\n\nA classical OCR engine (Tesseract) already produced a rough, "
    "error-prone transcription of this same image, included below as a "
    "starting point. It may contain wrong characters, dropped words, or "
    "garbled segments — do not trust it blindly. Look at the image yourself "
    "and use the OCR text only where it helps; correct or ignore it wherever "
    "it disagrees with what you actually see.\n\n"
    "--- Tesseract OCR (unreliable, for reference only) ---\n"
    "{prior}\n"
    "--- end OCR ---"
)


def _run_tesseract(image: Image.Image, *, lang: str = "ara") -> str:
    import pytesseract

    text = pytesseract.image_to_string(image.convert("RGB"), lang=lang)
    return text.strip()


@dataclass
class TesseractClient:
    """Tier-5 classical-OCR anchor. Ignores the prompt entirely."""

    name: str
    lang: str = "ara"

    def transcribe(
        self, image: Image.Image, prompt: str, max_tokens: int | None = None
    ) -> Prediction:
        del prompt, max_tokens
        t0 = time.perf_counter()
        try:
            text = _run_tesseract(image, lang=self.lang)
            return Prediction(
                raw_text=text,
                cleaned_text=text,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:  # noqa: BLE001
            return Prediction(
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"tesseract_error:{type(e).__name__}",
            )


@dataclass
class OCRCorrectedClient:
    """Two-stage client: Tesseract's raw pass, fed to a wrapped VLM client as
    a correction prior. `inner` is any other `ModelClient` (litellm or
    hf_local) — this class only ever adds the prior text to the prompt and
    delegates, so it reuses the wrapped client's own decoding/token/error
    handling unchanged.

    A Tesseract failure (missing binary/traineddata, decode error) is not
    fatal — falls back to the wrapped client's plain-prompt behavior instead
    of surfacing an error, since the underlying transcription attempt should
    still proceed.
    """

    name: str
    inner: ModelClient
    lang: str = "ara"

    def transcribe(
        self, image: Image.Image, prompt: str, max_tokens: int | None = None
    ) -> Prediction:
        try:
            prior = _run_tesseract(image, lang=self.lang)
        except Exception:  # noqa: BLE001
            prior = ""
        augmented = prompt
        if prior:
            augmented = prompt + CORRECTION_INSTRUCTION.format(prior=prior)
        return self.inner.transcribe(image, augmented, max_tokens=max_tokens)
