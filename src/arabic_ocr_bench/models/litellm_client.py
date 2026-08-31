"""Hosted VLMs via LiteLLM (Gemini, OpenAI, Claude, or any OpenAI-compatible
server via api_base). One call path for every provider."""

from __future__ import annotations

import base64
import io
import re
import time

from PIL import Image

from ..imaging import cap_image
from .base import Prediction

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")  # <b> <i> <p> </b> ... (markup, not Arabic)
_MD_PREFIX_RE = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|>\s+|\|)")

_REFUSAL_HINTS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "cannot assist", "can't help with", "i'm sorry", "i am sorry",
    # Arabic: "sorry", "I can't", "unable", "please provide more"
    "عذرا", "آسف", "أعتذر", "لا يمكنني", "لا أستطيع", "لا استطيع",
    "يرجى تقديم", "لا يمكن قراءة", "غير واضح",
)


def looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()[:200]
    return any(h in t for h in _REFUSAL_HINTS)


def _image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    cap_image(image.convert("RGB"), 1400).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def clean_output(raw: str) -> str:
    """Normalize a model response to bare transcription text. The raw response is
    kept verbatim elsewhere; this is only what scoring sees."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = _TAG_RE.sub("", s)                       # drop <b>/<i>/... markup (Qari v0.3 emits it)
    s = _MD_PREFIX_RE.sub("", s.strip())         # drop a leading #, -, >, | markdown marker
    s = s.strip()
    for prefix in ("Transcription:", "Text:", "Here is the transcription:", "Output:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    if len(s) >= 2 and s[0] in "\"'«" and s[-1] in "\"'»":
        s = s[1:-1].strip()
    return s.strip()


class LiteLLMClient:
    def __init__(self, name: str, litellm_id: str, params: dict | None = None,
                 api_base: str | None = None, max_tokens: int = 256,
                 api_key: str | None = None):
        self.name = name
        self.litellm_id = litellm_id
        self.params = params or {}
        self.api_base = api_base
        # only needed for OpenAI-compatible endpoints (e.g. Ollama cloud) where
        # litellm can't infer the key from a provider-specific env var
        self.api_key = api_key
        # a line is <=181 chars; cap output so a model that rambles doesn't run
        # up cost / CER. Mirrors hf_local_client's 160-token cap.
        self.max_tokens = max_tokens

    def transcribe(self, image: Image.Image, prompt: str) -> Prediction:
        import litellm

        litellm.drop_params = True  # silently drop params a given model rejects

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_data_uri(image)}},
            ],
        }]
        kwargs = dict(model=self.litellm_id, messages=messages,
                      max_tokens=self.max_tokens, **self.params)
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

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
        err = "refusal" if looks_like_refusal(raw) else None
        return Prediction(
            raw_text=raw,
            cleaned_text=clean_output(raw),
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            error=err,
        )
