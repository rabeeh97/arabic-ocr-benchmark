"""Zero-cost deterministic clients for proving the pipeline.

- mode="empty"   -> always returns "".  Sanity floor: CER should be ~1.0.
- mode="echo_gt" -> returns the ground truth (passed via prompt injection by the
  runner in mock mode).  Sanity ceiling: CER should be 0.0.
"""

from __future__ import annotations

from PIL import Image

from .base import Prediction

_ECHO_MARKER = "\n<<GROUND_TRUTH>>\n"


def build_mock_prompt(base_prompt: str, ground_truth: str, mode: str) -> str:
    """The runner calls this for mock models so echo_gt has something to echo.
    Real clients never see this."""
    if mode == "echo_gt":
        return base_prompt + _ECHO_MARKER + ground_truth
    return base_prompt


class MockClient:
    def __init__(self, name: str, mode: str = "empty"):
        if mode not in ("empty", "echo_gt"):
            raise ValueError(f"unknown mock mode {mode!r}")
        self.name = name
        self.mode = mode

    def transcribe(self, image: Image.Image, prompt: str) -> Prediction:
        if self.mode == "empty":
            text = ""
        else:  # echo_gt
            text = prompt.split(_ECHO_MARKER, 1)[1] if _ECHO_MARKER in prompt else ""
        return Prediction(raw_text=text, cleaned_text=text, latency_ms=0.0)
