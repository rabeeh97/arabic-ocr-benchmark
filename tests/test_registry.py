"""available_hosted_models() selects litellm models whose key/base is present."""

from arabic_ocr_bench.models.registry import available_hosted_models, build_model

_CFGS = {
    "mock_x": {"client": "mock"},
    "qwen": {"client": "hf_local", "hf_id": "x"},
    "claude": {"client": "litellm", "env_key": "ANTHROPIC_API_KEY"},
    "gem": {"client": "litellm", "env_key": "GEMINI_API_KEY"},
    "ollama": {"client": "litellm", "env_key": "OLLAMA_API_KEY",
               "api_base_env": "OLLAMA_API_BASE"},
    "keyless": {"client": "litellm"},
}


def test_only_litellm_models_with_key_present():
    got = available_hosted_models(_CFGS, env={"GEMINI_API_KEY": "k"})
    assert got == ["gem", "keyless"]        # claude/ollama lack keys; mock/qwen not hosted


def test_api_base_env_must_also_be_set():
    got = available_hosted_models(_CFGS, env={"OLLAMA_API_KEY": "k"})
    assert "ollama" not in got             # OLLAMA_API_BASE missing
    got2 = available_hosted_models(_CFGS, env={"OLLAMA_API_KEY": "k", "OLLAMA_API_BASE": "b"})
    assert "ollama" in got2


def test_empty_env_still_returns_keyless_litellm_models():
    assert available_hosted_models(_CFGS, env={}) == ["keyless"]


def test_build_model_forwards_key_from_env_key(monkeypatch):
    """A litellm model pinned to a non-default env_key (e.g. *_PAID) must have
    that key passed to the client explicitly, even with no api_base — otherwise
    litellm falls back to the provider-default var (ANTHROPIC_API_KEY / ...)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY_PAID", "sk-paid-xyz")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-free-should-not-be-used")
    client = build_model(
        "claude-opus-paid",
        {"client": "litellm", "litellm_id": "anthropic/claude-opus-5",
         "env_key": "ANTHROPIC_API_KEY_PAID", "params": {}},
    )
    assert client.api_key == "sk-paid-xyz"
    assert client.api_base is None


def test_build_model_no_env_key_leaves_key_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_model(
        "gpt", {"client": "litellm", "litellm_id": "openai/gpt-4o"},
    )
    assert client.api_key is None
