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
