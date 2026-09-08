"""Merge a trained LoRA adapter into its base weights for plain inference.

The merged directory loads through the existing `HFLocalClient` as an ordinary
`hf_id` (a local path) with no `adapter_id` — so a fine-tuned model re-enters the
benchmark as a normal `configs/models.yaml` entry.
"""

from __future__ import annotations

import json
from pathlib import Path

_ARCH_MODEL_CLS = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",   # needs transformers>=4.57
    "paligemma": "PaliGemmaForConditionalGeneration",  # PaliGemma 1 & 2
}


def _adapter_meta(adapter_dir: Path) -> dict:
    cfg = adapter_dir / "training_config.yaml"
    if cfg.exists():
        return json.loads(cfg.read_text())
    return {}


def merge_adapter(
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    base_hf_id: str | None = None,
    arch: str | None = None,
    dtype: str = "bfloat16",
) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    adapter_dir = Path(adapter_dir)
    meta = _adapter_meta(adapter_dir)
    base_hf_id = base_hf_id or (meta.get("base_model") or {}).get("hf_id")
    arch = arch or (meta.get("base_model") or {}).get("arch") or "qwen2_vl"
    if not base_hf_id:
        raise ValueError(
            "base_hf_id not given and not recorded in the adapter's training_config.yaml"
        )

    cls_name = _ARCH_MODEL_CLS.get(arch, _ARCH_MODEL_CLS["qwen2_vl"])
    model_cls = getattr(__import__("transformers", fromlist=[cls_name]), cls_name)

    base = model_cls.from_pretrained(
        base_hf_id, torch_dtype=getattr(torch, dtype), device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    try:
        AutoProcessor.from_pretrained(str(adapter_dir)).save_pretrained(str(out))
    except Exception:  # noqa: BLE001 - fall back to the base processor
        AutoProcessor.from_pretrained(base_hf_id).save_pretrained(str(out))
    (out / "merge_info.json").write_text(
        json.dumps({"base_hf_id": base_hf_id, "arch": arch, "adapter": str(adapter_dir)}, indent=2)
    )
    return out
