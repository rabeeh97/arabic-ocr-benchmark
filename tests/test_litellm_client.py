"""LiteLLMClient: the max_tokens -> max_completion_tokens fallback for models
(e.g. gpt-6-astra) that reject max_tokens outright."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from arabic_ocr_bench.models.litellm_client import LiteLLMClient


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content, in_tok=10, out_tok=5):
        self.choices = [SimpleNamespace(message=_FakeMessage(content))]
        self.usage = SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok)


def _img():
    return Image.new("RGB", (20, 10), "white")


def test_retries_with_max_completion_tokens_when_max_tokens_is_rejected(monkeypatch):
    import litellm

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise Exception(
                "litellm.BadRequestError: OpenAIException - Unsupported parameter: "
                "'max_tokens' is not supported with this model. Use "
                "'max_completion_tokens' instead."
            )
        assert kwargs.get("max_completion_tokens") == 4000
        return _FakeResponse("hello")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient("gpt-6-astra-paid", "openai/gpt-6-astra", max_tokens=4000)
    pred = client.transcribe(_img(), "prompt")

    assert pred.error is None
    assert pred.cleaned_text == "hello"
    assert len(calls) == 2  # first attempt (rejected), then the fallback
    assert "max_tokens" not in calls[1]


def test_other_errors_are_not_retried(monkeypatch):
    import litellm

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise Exception("RateLimitError: 429 Too Many Requests")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient("some-model", "openai/some-model", max_tokens=256)
    pred = client.transcribe(_img(), "prompt")

    assert pred.error is not None
    assert "RateLimitError" in pred.error
    assert len(calls) == 1  # no retry for an unrelated error


# --------------------------------------------------------------------------- #
# multi-key pool (e.g. three Gemini free-tier keys)
# --------------------------------------------------------------------------- #
def _pool_client(monkeypatch, responder):
    """A LiteLLMClient with a 3-key pool (no per-minute throttle) whose
    litellm.completion is `responder(api_key) -> _FakeResponse | raises`."""
    import litellm

    seen: list[str] = []

    def fake_completion(**kwargs):
        key = kwargs.get("api_key")
        seen.append(key)
        return responder(key)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient(
        "gemini-flash", "gemini/gemini-3.6-flash", max_tokens=256,
        api_keys=[("GEMINI1_API_KEY", "k1"), ("GEMINI2_API_KEY", "k2"),
                  ("GEMINI3_API_KEY", "k3")],
        per_key_rpm=0,
    )
    return client, seen


def test_pool_rotates_across_keys(monkeypatch):
    client, seen = _pool_client(monkeypatch, lambda key: _FakeResponse("ok"))
    for _ in range(6):
        assert client.transcribe(_img(), "p").error is None
    # every key gets used; load is spread, not pinned to one key
    assert set(seen) == {"k1", "k2", "k3"}
    assert max(seen.count(k) for k in ("k1", "k2", "k3")) <= 3


def test_per_minute_429_reroutes_without_dropping_the_key(monkeypatch):
    def responder(key):
        if key == "k2":
            raise Exception("RateLimitError: 429 quota exceeded for "
                            "GenerateRequestsPerMinutePerProjectPerModel")
        return _FakeResponse("ok")

    client, seen = _pool_client(monkeypatch, responder)
    pred = client.transcribe(_img(), "p")
    assert pred.error is None and pred.cleaned_text == "ok"
    # k2 was tried, hit a per-minute cap, and the call moved to another key
    assert "k2" in seen and seen[-1] in {"k1", "k3"}
    k2 = next(s for s in client._pool._slots if s.label == "GEMINI2_API_KEY")
    assert k2.capped is False  # only rested, not dropped for the run
    import time as _t
    assert k2.cool_until > _t.monotonic()  # currently resting


def test_per_day_429_caps_the_key_for_the_rest_of_the_run(monkeypatch):
    def responder(key):
        if key == "k2":
            raise Exception("RateLimitError: 429 Resource has been exhausted — "
                            "GenerateRequestsPerDayPerProjectPerModel (per day)")
        return _FakeResponse("ok")

    client, seen = _pool_client(monkeypatch, responder)
    for _ in range(8):
        assert client.transcribe(_img(), "p").error is None
    k2 = next(s for s in client._pool._slots if s.label == "GEMINI2_API_KEY")
    assert k2.capped is True
    # k2 is never selected again after it is capped
    assert seen.count("k2") == 1


def test_all_keys_daily_capped_yields_a_rate_limit_error(monkeypatch):
    def responder(key):
        raise Exception("RateLimitError: 429 exhausted — RequestsPerDay (per day)")

    client, _ = _pool_client(monkeypatch, responder)
    pred = client.transcribe(_img(), "p")
    assert pred.error is not None
    assert "RateLimitError" in pred.error or "out of quota" in pred.error
