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


def build_dataset(name: str, cfg: dict[str, Any], *, limit: int | None = None, offset: int = 0):
    adapter_cls = _resolve(cfg["adapter"])
    kwargs: dict[str, Any] = dict(cfg.get("params") or {})
    for k in ("split", "hf_id", "subset"):
        if k in cfg:
            kwargs[k] = cfg[k]
    if limit is not None:
        kwargs["limit"] = limit
    if offset:
        kwargs["offset"] = offset
    # let an adapter that accepts `name` stamp stable sample ids from the config
    try:
        import inspect

        if "name" in inspect.signature(adapter_cls).parameters:
            kwargs.setdefault("name", name)
    except (TypeError, ValueError):
        pass
    return adapter_cls(**kwargs)
