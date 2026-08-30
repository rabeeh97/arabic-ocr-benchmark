"""Instantiate a model client from configs/models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import ModelClient

_DEFAULT_CFG = Path("configs/models.yaml")


def load_model_configs(path: str | Path = _DEFAULT_CFG) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text()) or {}


def build_model(name: str, cfg: dict[str, Any]) -> ModelClient:
    client = cfg.get("client")
    if client == "mock":
        from .mock_client import MockClient

        return MockClient(name=name, mode=cfg.get("mode", "empty"))
    if client == "litellm":
        from .litellm_client import LiteLLMClient

        return LiteLLMClient(
            name=name,
            litellm_id=cfg["litellm_id"],
            params=cfg.get("params") or {},
            api_base=cfg.get("api_base"),
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
            max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        )
    raise ValueError(f"model {name!r}: unknown client {client!r}")


def is_mock(cfg: dict[str, Any]) -> bool:
    return cfg.get("client") == "mock"
