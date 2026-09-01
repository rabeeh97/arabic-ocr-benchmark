"""Local open-weight / Arabic-specific VLMs (Qwen2.5-VL, Qwen2-VL / Qari-OCR,
PaliGemma). Requires the `local` extra:  uv sync --extra local

Kept deliberately small: it covers the Qwen2(.5)-VL family (which includes
Qari-OCR) and PaliGemma. Other architectures get their own branch when added.
"""

from __future__ import annotations

import time

from PIL import Image

from ..imaging import cap_image
from .base import Prediction
from .litellm_client import clean_output, looks_like_refusal

# greedy decoding on a hard manuscript line collapses into "الشبق الشبق ..." loops;
# these keep it honest and let EOS fire naturally.
_GEN_KW = {"do_sample": False, "repetition_penalty": 1.15, "no_repeat_ngram_size": 3}


def _available_ram_gb() -> float:
    """Best-effort free system RAM in GB. Returns a huge number if it can't be
    determined, so a missing psutil never wrongly blocks a load."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:  # noqa: BLE001
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1e6  # kB -> GB
        except OSError:
            pass
    return 1e6


def _cpu_dtype():
    """CPU inference dtype. bfloat16 halves fp32's footprint and its
    matmul/layernorm are supported on modern torch CPU (float16 is not). Override
    with env ARABIC_OCR_CPU_DTYPE=float32 if a model misbehaves in bf16."""
    import os

    import torch

    return {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
    }.get(os.environ.get("ARABIC_OCR_CPU_DTYPE", "bfloat16").lower(), torch.bfloat16)


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
                 max_new_tokens: int = 160, load_in_4bit: bool = False,
                 min_ram_gb: float = 0.0):
        self.name = name
        self.hf_id = hf_id
        self.arch = arch
        self.dtype = dtype
        self.adapter_id = adapter_id
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        # skip (rather than OOM-kill the kernel) if less than this much free RAM
        # when running on CPU. From `min_ram_gb` in configs/models.yaml.
        self.min_ram_gb = float(min_ram_gb or 0.0)
        self._model = None
        self._processor = None
        self._tokenizer = None

    def _ram_preflight(self) -> None:
        """On CPU, a model bigger than free RAM doesn't raise — the Linux OOM
        killer takes the whole kernel. Check first and raise a normal exception
        instead (the runner catches it and moves to the next model)."""
        if self.min_ram_gb <= 0:
            return
        try:
            import torch

            on_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        except Exception:  # noqa: BLE001 - no torch yet; treat as CPU
            on_gpu = False
        if on_gpu:
            return  # this budget is a CPU-only guard
        avail = _available_ram_gb()
        if avail < self.min_ram_gb:
            raise RuntimeError(
                f"skip {self.name} on CPU: needs ~{self.min_ram_gb:.0f} GB free "
                f"RAM, only {avail:.1f} GB available (loading it would OOM-kill "
                f"the kernel). Use a GPU runtime, or drop this model."
            )

    # -- lazy load ---------------------------------------------------------- #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._ram_preflight()
        if self.arch == "trocr":
            self._ensure_loaded_trocr()
            return
        if self.arch == "minicpmv":
            self._ensure_loaded_minicpmv()
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
        # 4-bit (bitsandbytes) needs CUDA; fp16 matmul is unsupported / slow on CPU.
        use_4bit = self.load_in_4bit and device == "cuda"
        if device == "cpu":
            torch_dtype = _cpu_dtype()
            print(
                f"WARNING: no GPU — loading {self.hf_id} on CPU in "
                f"{str(torch_dtype).replace('torch.', '')} "
                f"({'4-bit disabled; ' if self.load_in_4bit else ''}"
                f"{_available_ram_gb():.1f} GB RAM free; expect ~30-90 s/line)."
            )
        else:
            torch_dtype = (
                torch.float16 if self.dtype in ("float16", "fp16")
                else torch.bfloat16 if self.dtype in ("bfloat16", "bf16")
                else "auto"
            )
        _t0 = time.perf_counter()
        adapter = f" + adapter {self.adapter_id}" if self.adapter_id else ""
        print(f"[load] {self.hf_id}{adapter} "
              f"({'4-bit' if use_4bit else str(torch_dtype).replace('torch.', '')} on {device}) ...",
              flush=True)
        self._processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)

        # Some checkpoints (unsloth-derived, e.g. Qari-OCR) carry a bitsandbytes
        # 4-bit `quantization_config` in config.json. That path is fragile on
        # Colab ("'Parameter' object has no attribute 'compress_statistics'") and
        # unnecessary for a 2-3B model on a 16 GB GPU. Unless the caller opted
        # into quantization, drop it and load in plain fp16/bf16.
        cfg = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True)
        if not use_4bit and hasattr(cfg, "quantization_config"):
            delattr(cfg, "quantization_config")

        load_kwargs = {
            "config": cfg,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if use_4bit:
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
        model = model if use_4bit else model.to(device)
        self._model = model.eval()
        self._device = "cuda" if use_4bit else device
        print(f"[load] {self.name} ready in {time.perf_counter() - _t0:.1f}s", flush=True)

    def _ensure_loaded_trocr(self) -> None:
        """TrOCR (VisionEncoderDecoder) — no chat template, no quantization; a
        small encoder-decoder that runs fine on CPU/MPS. Used for
        David-Magdy/TR_OCR_LARGE (TrOCR-Large finetuned on KHATT + IAM)."""
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "local models need the 'local' extra: uv sync --extra local"
            ) from e

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _t0 = time.perf_counter()
        print(f"[load] {self.hf_id} (trocr on {device}) ...", flush=True)
        self._processor = TrOCRProcessor.from_pretrained(self.hf_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.hf_id).to(device).eval()
        self._device = device
        print(f"[load] {self.name} ready in {time.perf_counter() - _t0:.1f}s", flush=True)

    def _ensure_loaded_minicpmv(self) -> None:
        """MiniCPM-V (openbmb/MiniCPM-V-2_6) — multilingual doc OCR, ~8B. Uses a
        custom `model.chat(msgs, tokenizer)` API (trust_remote_code), not the
        standard processor+generate path. 4-bit on CUDA, fp32 on CPU (slow)."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "local models need the 'local' extra: uv sync --extra local"
            ) from e

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        use_4bit = self.load_in_4bit and device == "cuda"
        dtype = _cpu_dtype() if device == "cpu" else torch.bfloat16
        load_kwargs: dict = {"trust_remote_code": True, "torch_dtype": dtype,
                             "attn_implementation": "sdpa"}
        if use_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        _t0 = time.perf_counter()
        print(f"[load] {self.hf_id} (minicpmv {'4-bit' if use_4bit else dtype} on {device}) ...",
              flush=True)
        model = AutoModel.from_pretrained(self.hf_id, **load_kwargs)
        self._model = (model if use_4bit else model.to(device)).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        self._device = "cuda" if use_4bit else device
        print(f"[load] {self.name} ready in {time.perf_counter() - _t0:.1f}s", flush=True)

    def close(self) -> None:
        """Free the model from (GPU) memory — the runner calls this between
        models in a sweep so a single GPU can host them one at a time."""
        self._model = None
        self._processor = None
        self._tokenizer = None
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
        if looks_like_refusal(text):
            return Prediction(raw_text=text, cleaned_text="", error="refusal",
                              latency_ms=latency_ms)
        return Prediction(raw_text=text, cleaned_text=clean_output(text),
                          latency_ms=latency_ms)

    def _generate(self, image: Image.Image, prompt: str) -> str:
        import torch

        image = cap_image(image, 1024)  # bound vision tokens / activation memory
        if self.arch == "paligemma":
            return self._generate_paligemma(image, prompt)
        if self.arch == "trocr":
            return self._generate_trocr(image)
        if self.arch == "minicpmv":
            return self._generate_minicpmv(image, prompt)

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
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, **_GEN_KW
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(gen, skip_special_tokens=True).strip()

    def _generate_minicpmv(self, image: Image.Image, prompt: str) -> str:
        msgs = [{"role": "user", "content": [image, prompt]}]
        out = self._model.chat(
            image=None, msgs=msgs, tokenizer=self._tokenizer,
            sampling=False, max_new_tokens=self.max_new_tokens,
        )
        return (out if isinstance(out, str) else str(out)).strip()

    def _generate_trocr(self, image: Image.Image) -> str:
        import torch

        pixel_values = self._processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self._device)
        with torch.no_grad():
            ids = self._model.generate(
                pixel_values, max_new_tokens=self.max_new_tokens, **_GEN_KW
            )
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

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
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, **_GEN_KW
            )
        return self._processor.decode(out[0][in_len:], skip_special_tokens=True).strip()
