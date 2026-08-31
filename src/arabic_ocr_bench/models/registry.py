"""Instantiate a model client from configs/models.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .base import ModelClient

_DEFAULT_CFG = Path("configs/models.yaml")


def load_model_configs(path: str | Path = _DEFAULT_CFG) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text()) or {}


def available_hosted_models(
    configs: dict[str, dict] | None = None, env: dict | None = None
) -> list[str]:
    """Every `client: litellm` model whose `env_key` is present (or which needs
    no key). Used so run_hosted.py / notebook 02 pick up new models in
    models.yaml automatically — no hardcoded list."""
    cfgs = configs if configs is not None else load_model_configs()
    e = env if env is not None else os.environ
    out = []
    for name, cfg in cfgs.items():
        if cfg.get("client") != "litellm":
            continue
        key = cfg.get("env_key")
        if key and not e.get(key):
            continue
        if cfg.get("api_base_env") and not e.get(cfg["api_base_env"]):
            continue
        out.append(name)
    return out


def build_model(name: str, cfg: dict[str, Any]) -> ModelClient:
    client = cfg.get("client")
    if client == "mock":
        from .mock_client import MockClient

        return MockClient(name=name, mode=cfg.get("mode", "empty"))
    if client == "litellm":
        from .litellm_client import LiteLLMClient

        api_base = cfg.get("api_base")
        if cfg.get("api_base_env"):
            api_base = os.environ.get(cfg["api_base_env"], api_base)
        # for OpenAI-compatible custom endpoints (api_base set) litellm won't
        # find a provider env var, so pass the key from env_key explicitly
        api_key = None
        if api_base and cfg.get("env_key"):
            api_key = os.environ.get(cfg["env_key"])
        return LiteLLMClient(
            name=name,
            litellm_id=cfg["litellm_id"],
            params=cfg.get("params") or {},
            api_base=api_base,
            max_tokens=int(cfg.get("max_tokens", 256)),
            api_key=api_key,
        )
    if client == "hf_local":
        from .hf_local_client import HFLocalClient

        return HFLocalClient(
            name=name,
            hf_id=cfg["hf_id"],
            arch=cfg.get("arch", "qwen2_5_vl"),
            dtype=cfg.get("dtype", "auto"),
            adapter_id=cfg.get("adapter_id"),
            load_in_4bit=bool(cfg.get("load_in_4bit", False)),
            max_new_tokens=int(cfg.get("max_new_tokens", 160)),
        )
    raise ValueError(f"model {name!r}: unknown client {client!r}")


def is_mock(cfg: dict[str, Any]) -> bool:
    return cfg.get("client") == "mock"
