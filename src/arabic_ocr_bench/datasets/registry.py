"""Instantiate a dataset adapter from configs/datasets.yaml."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CFG = Path("configs/datasets.yaml")


def load_dataset_configs(path: str | Path = _DEFAULT_CFG) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _resolve(target: str):
    """'pkg.mod:ClassName' -> the class object."""
    mod_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(mod_name), attr)


def build_dataset(name: str, cfg: dict[str, Any], *, limit: int | None = None):
    adapter_cls = _resolve(cfg["adapter"])
    kwargs: dict[str, Any] = {}
    if "split" in cfg:
        kwargs["split"] = cfg["split"]
    if "hf_id" in cfg:
        kwargs["hf_id"] = cfg["hf_id"]
    if limit is not None:
        kwargs["limit"] = limit
    return adapter_cls(**kwargs)
