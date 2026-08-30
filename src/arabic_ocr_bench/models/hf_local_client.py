"""Local open-weight / Arabic-specific VLMs (Qwen2.5-VL, Qwen2-VL / Qari-OCR,
PaliGemma). Requires the `local` extra:  uv sync --extra local

Kept deliberately small: it covers the Qwen2(.5)-VL family (which includes
Qari-OCR) and PaliGemma. Other architectures get their own branch when added.
"""

from __future__ import annotations

import time

from PIL import Image

from .base import Prediction
from .litellm_client import clean_output


def _load_model(hf_id: str, load_kwargs: dict):
    """Try the vision-text auto classes in order — which one carries a given
    architecture shifts between transformers releases."""
    import transformers

    errors = []
    for cls_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(hf_id, **load_kwargs)
        except (ValueError, KeyError, OSError) as e:
            errors.append(f"{cls_name}: {type(e).__name__}: {e}")
    raise RuntimeError("no auto class could load " + hf_id + "\n" + "\n".join(errors))


class HFLocalClient:
    def __init__(self, name: str, hf_id: str, arch: str = "qwen2_5_vl",
                 dtype: str = "auto", adapter_id: str | None = None,
                 max_new_tokens: int = 512, load_in_4bit: bool = False):
        self.name = name
        self.hf_id = hf_id
        self.arch = arch
        self.dtype = dtype
        self.adapter_id = adapter_id
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    # -- lazy load ---------------------------------------------------------- #
    def _ensure_loaded(self):
        if self._model is not None:
            return

        import sys

        if "bitsandbytes" in sys.modules and not self.load_in_4bit:
            # importing bitsandbytes patches torch's linear layers to route
            # through 4-bit matmul; an unquantized model then fails forward with
            # "'Parameter' object has no attribute 'compress_statistics'".
            print(
                "WARNING: bitsandbytes is imported but this model is not 4-bit. "
                "If inference fails with 'compress_statistics', restart the "
                "runtime without importing bitsandbytes (or set load_in_4bit)."
            )

        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "local models need the 'local' extra: uv sync --extra local"
            ) from e

        import torch
        from transformers import AutoConfig

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch_dtype = (
            torch.float16 if self.dtype in ("float16", "fp16")
            else torch.bfloat16 if self.dtype in ("bfloat16", "bf16")
            else "auto"
        )
        self._processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)

        # Some checkpoints (unsloth-derived, e.g. Qari-OCR) carry a bitsandbytes
        # 4-bit `quantization_config` in config.json. That path is fragile on
        # Colab ("'Parameter' object has no attribute 'compress_statistics'") and
        # unnecessary for a 2-3B model on a 16 GB GPU. Unless the caller opted
        # into quantization, drop it and load in plain fp16/bf16.
        cfg = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True)
        if not self.load_in_4bit and hasattr(cfg, "quantization_config"):
            delattr(cfg, "quantization_config")

        load_kwargs = {
            "config": cfg,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"

        model = _load_model(self.hf_id, load_kwargs)
        if self.adapter_id:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_id)
        # with device_map="auto" (4-bit) accelerate already placed the weights
        model = model if self.load_in_4bit else model.to(device)
        self._model = model.eval()
        self._device = "cuda" if self.load_in_4bit else device

    def close(self) -> None:
        """Free the model from (GPU) memory — the runner calls this between
        models in a sweep so a single GPU can host them one at a time."""
        self._model = None
        self._processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"close(): cache clear skipped: {e}")

    # -- inference -------------------------------------------------------- #
    def transcribe(self, image: Image.Image, prompt: str) -> Prediction:
        t0 = time.perf_counter()
        try:
            self._ensure_loaded()
            text = self._generate(image.convert("RGB"), prompt)
        except Exception as e:  # noqa: BLE001
            return Prediction(error=f"{type(e).__name__}: {e}",
                              latency_ms=(time.perf_counter() - t0) * 1000)
        latency_ms = (time.perf_counter() - t0) * 1000
        return Prediction(raw_text=text, cleaned_text=clean_output(text),
                          latency_ms=latency_ms)

    def _generate(self, image: Image.Image, prompt: str) -> str:
        import torch

        if self.arch == "paligemma":
            return self._generate_paligemma(image, prompt)

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        chat = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=[chat], images=[image], return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(gen, skip_special_tokens=True).strip()

    def _generate_paligemma(self, image: Image.Image, prompt: str) -> str:
        import torch

        # PaliGemma uses a task-prefix, not a chat template. The Arabic-HW adapter
        # was trained with a short OCR instruction; keep it terse.
        pg_prompt = "ocr\n"
        inputs = self._processor(text=pg_prompt, images=image, return_tensors="pt").to(
            self._device
        )
        in_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        return self._processor.decode(out[0][in_len:], skip_special_tokens=True).strip()
