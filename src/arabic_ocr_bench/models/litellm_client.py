"""Hosted VLMs via LiteLLM (Gemini, OpenAI, Claude, or any OpenAI-compatible
server via api_base). One call path for every provider."""

from __future__ import annotations

import base64
import io
import time

from PIL import Image

from .base import Prediction

_REFUSAL_HINTS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "cannot assist", "can't help with",
)


def _image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def clean_output(raw: str) -> str:
    """Strip code fences / preambles / wrapping quotes. raw is preserved elsewhere."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = s.strip()
    for prefix in ("Transcription:", "Text:", "Here is the transcription:", "Output:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    if len(s) >= 2 and s[0] in "\"'«" and s[-1] in "\"'»":
        s = s[1:-1].strip()
    return s


class LiteLLMClient:
    def __init__(self, name: str, litellm_id: str, params: dict | None = None,
                 api_base: str | None = None):
        self.name = name
        self.litellm_id = litellm_id
        self.params = params or {}
        self.api_base = api_base

    def transcribe(self, image: Image.Image, prompt: str) -> Prediction:
        import litellm

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_data_uri(image)}},
            ],
        }]
        kwargs = dict(model=self.litellm_id, messages=messages, **self.params)
        if self.api_base:
            kwargs["api_base"] = self.api_base

        t0 = time.perf_counter()
        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            return Prediction(error=f"{type(e).__name__}: {e}",
                              latency_ms=(time.perf_counter() - t0) * 1000)
        latency_ms = (time.perf_counter() - t0) * 1000

        raw = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        err = "refusal" if any(h in raw.lower() for h in _REFUSAL_HINTS) else None
        return Prediction(
            raw_text=raw,
            cleaned_text=clean_output(raw),
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            error=err,
        )
