"""Hosted VLMs via LiteLLM (Gemini, OpenAI, Claude, or any OpenAI-compatible
server via api_base). One call path for every provider."""

from __future__ import annotations

import base64
import io
import re
import threading
import time
from collections import deque

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


_RATE_LIMIT_HINTS = (
    "rate limit", "ratelimit", "rate_limit", "429", "too many requests",
    "resource_exhausted", "resourceexhausted", "quota", "insufficient_quota",
)


def _is_rate_limit(err: str | None) -> bool:
    return bool(err) and any(h in err.lower() for h in _RATE_LIMIT_HINTS)


def _is_daily_quota(err: str | None) -> bool:
    """Tell a per-day free-tier exhaustion apart from a transient per-minute
    burst. Gemini names the offending metric in the 429 body (e.g.
    ``...GenerateContentRequestsPerDayPerProjectPerModel``, or plain "per day" /
    "daily"); a bare per-minute 429 carries none of these tokens."""
    if not err:
        return False
    s = err.lower()
    return "perday" in s.replace(" ", "") or "daily" in s or "per day" in s


class _KeySlot:
    """One API key plus its own sliding requests/min window and 429 state."""

    __slots__ = ("key", "label", "rpm", "_win", "_lock", "cool_until", "capped")

    def __init__(self, key: str, label: str, rpm: float):
        self.key = key
        self.label = label
        self.rpm = float(rpm or 0)
        self._win: deque[float] = deque()
        self._lock = threading.Lock()
        self.cool_until = 0.0  # time.monotonic() until which a per-minute 429 rests it
        self.capped = False    # a per-day 429 — dropped for the rest of the process

    def slot_wait(self) -> float:
        """Seconds until this key has room in its 60 s window (0.0 = free now)."""
        if self.rpm <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            while self._win and now - self._win[0] >= 60.0:
                self._win.popleft()
            if len(self._win) < self.rpm:
                return 0.0
            return 60.0 - (now - self._win[0]) + 0.01

    def take_slot(self) -> None:
        if self.rpm > 0:
            with self._lock:
                self._win.append(time.monotonic())


class _AllKeysCapped(Exception):
    """Every key in the pool is out of daily quota (or unavailable long enough
    that the caller should back off as if it were)."""


class _KeyPool:
    """A rotating pool of API keys for one model. Hands out the key that can
    serve soonest; a key that returns a per-minute 429 is rested briefly, one
    that returns a per-day 429 is dropped until the process restarts. A given
    request is only ever in flight on one key — a 429 moves it to another key,
    it is never fanned out to several at once."""

    def __init__(self, keys: list[tuple[str, str]], rpm: float,
                 cooldown: float = 60.0, max_wait: float = 150.0):
        self._slots = [_KeySlot(k, label, rpm) for label, k in keys]
        self._cooldown = float(cooldown)
        self._max_wait = float(max_wait)
        self._rr = 0
        self._lock = threading.Lock()
        self._announced: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._slots)

    def acquire(self) -> _KeySlot:
        deadline = time.monotonic() + self._max_wait
        while True:
            with self._lock:
                live = [s for s in self._slots if not s.capped]
                if not live:
                    raise _AllKeysCapped()
                now = time.monotonic()
                ready = [s for s in live if s.cool_until <= now and s.slot_wait() <= 0.0]
                if ready:
                    self._rr = (self._rr + 1) % len(ready)
                    slot = ready[self._rr]
                    slot.take_slot()
                    return slot
                nap = min(max(s.cool_until - now, 0.0) + s.slot_wait() for s in live)
            if time.monotonic() >= deadline:
                raise _AllKeysCapped()  # a long cool-out — treat as capped for now
            time.sleep(min(max(nap, 0.2), 3.0))

    def penalize(self, slot: _KeySlot, *, capped: bool) -> None:
        with self._lock:
            if capped:
                slot.capped = True
                live = sum(1 for s in self._slots if not s.capped)
                msg = (f"[gemini] key {slot.label} hit its daily quota — dropping it "
                       f"for this run ({live} key(s) still live)")
            else:
                slot.cool_until = time.monotonic() + self._cooldown
                msg = (f"[gemini] key {slot.label} rate-limited — "
                       f"resting it {self._cooldown:.0f}s")
            if capped or msg not in self._announced:
                self._announced.add(msg)
                print(msg, flush=True)


class LiteLLMClient:
    def __init__(self, name: str, litellm_id: str, params: dict | None = None,
                 api_base: str | None = None, max_tokens: int = 256,
                 api_key: str | None = None,
                 api_keys: list[tuple[str, str]] | None = None,
                 per_key_rpm: float = 0):
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
        # Optional multi-key pool (e.g. three Gemini free-tier keys). Each
        # (label, value) pair gets its own per-minute window; requests rotate
        # across keys and route around any that is 429-ing / daily-capped.
        self._pool: _KeyPool | None = None
        pairs = [(lbl, k) for lbl, k in (api_keys or []) if k]
        if len(pairs) > 1:
            self._pool = _KeyPool(pairs, rpm=per_key_rpm)
        elif pairs and not self.api_key:
            self.api_key = pairs[0][1]

    def transcribe(self, image: Image.Image, prompt: str) -> Prediction:
        if self._pool is None:
            return self._complete(image, prompt, self.api_key)
        last_err: str | None = None
        for _ in range(self._pool.size + 2):
            try:
                slot = self._pool.acquire()
            except _AllKeysCapped:
                return Prediction(
                    error=last_err
                    or "RateLimitError: all Gemini API keys are out of quota",
                    latency_ms=0.0,
                )
            pred = self._complete(image, prompt, slot.key)
            if pred.error and _is_rate_limit(pred.error):
                self._pool.penalize(slot, capped=_is_daily_quota(pred.error))
                last_err = pred.error
                continue
            return pred
        return Prediction(
            error=last_err or "RateLimitError: Gemini key pool exhausted",
            latency_ms=0.0,
        )

    def _complete(self, image: Image.Image, prompt: str,
                  api_key: str | None) -> Prediction:
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
        if api_key:
            kwargs["api_key"] = api_key

        t0 = time.perf_counter()
        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            # Some newer models (e.g. gpt-6-astra) reject `max_tokens` outright and
            # demand `max_completion_tokens` instead — litellm only auto-translates
            # this for models its own internal registry already recognises, so a
            # brand-new model id can slip through and forward `max_tokens` as-is.
            # Retry once with the swap rather than failing every line on day one.
            if "max_tokens" in kwargs and "max_completion_tokens" in str(e) and "max_tokens" in str(e):
                retry_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
                retry_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
                try:
                    resp = litellm.completion(**retry_kwargs)
                except Exception as e2:  # noqa: BLE001
                    return Prediction(error=f"{type(e2).__name__}: {e2}",
                                      latency_ms=(time.perf_counter() - t0) * 1000)
            else:
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
