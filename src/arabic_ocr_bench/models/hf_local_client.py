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


class HFLocalClient:
    def __init__(self, name: str, hf_id: str, arch: str = "qwen2_5_vl",
                 dtype: str = "auto", adapter_id: str | None = None,
                 max_new_tokens: int = 256):
        self.name = name
        self.hf_id = hf_id
        self.arch = arch
        self.dtype = dtype
        self.adapter_id = adapter_id
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    # -- lazy load ---------------------------------------------------------- #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "local models need the 'local' extra: uv sync --extra local"
            ) from e

        import torch
        from transformers import AutoModelForImageTextToText

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch_dtype = (
            torch.float16 if self.dtype in ("float16", "fp16")
            else torch.bfloat16 if self.dtype in ("bfloat16", "bf16")
            else "auto"
        )
        self._processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        if self.adapter_id:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_id)
        self._model = model.to(device).eval()
        self._device = device

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
