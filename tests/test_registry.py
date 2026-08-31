"""available_hosted_models() selects litellm models whose key/base is present."""

from arabic_ocr_bench.models.registry import available_hosted_models

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
