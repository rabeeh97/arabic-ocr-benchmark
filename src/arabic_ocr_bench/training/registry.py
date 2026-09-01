"""Register a fine-tuned model into `configs/models.yaml`.

Keeps the round-trip trivial: after `sft` (+ optional `merge`), one call adds an
`hf_local` row the existing runner scores with no new eval code.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_MODELS = Path("configs/models.yaml")


def build_model_entry(
    name: str,
    hf_id_or_path: str,
    *,
    arch: str,
    adapter_id: str | None = None,
    dtype: str = "bfloat16",
    min_ram_gb: int | None = None,
    card: str | None = None,
) -> dict:
    """One `hf_local` registry entry for a trained model.

    Pass a merged directory as `hf_id_or_path` (no `adapter_id`), or the base
    `hf_id` plus an `adapter_id` pointing at the LoRA output dir.
    """
    entry: dict = {"client": "hf_local", "hf_id": hf_id_or_path, "arch": arch}
    if adapter_id:
        entry["adapter_id"] = adapter_id
    if min_ram_gb is not None:
        entry["min_ram_gb"] = min_ram_gb
    entry["dtype"] = dtype
    entry["pricing"] = None
    entry["card"] = card or f"Fine-tuned model {name} (arabic_ocr_bench training subsystem)."
    return entry


def register_model(
    name: str, entry: dict, *, path: str | Path = _DEFAULT_MODELS
) -> None:
    """Add / overwrite one entry in models.yaml, leaving the rest untouched."""
    p = Path(path)
    doc = yaml.safe_load(p.read_text()) or {}
    doc[name] = entry
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
