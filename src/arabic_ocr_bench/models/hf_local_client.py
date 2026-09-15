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

_DEFAULT_PALIGEMMA_PROMPT = "ocr\n"  # matches the paligemma-arabic-hw adapter's training


def _cuda_free_total_gb():
    """(free, total) GPU memory in GB for the current CUDA device, or None when
    there is no usable CUDA GPU (CPU / MPS / no torch)."""
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            return free_b / 1e9, total_b / 1e9
    except Exception:  # noqa: BLE001
        pass
    return None


def _trim_cpu_ram() -> None:
    """Ask glibc to return free heap pages to the OS. PyTorch / HF free CPU
    tensors back into glibc's malloc arenas, but glibc keeps the pages mapped;
    over a long sweep (a model reloaded every batch) that shows up as system RSS
    that only ever grows and eventually OOM-kills the Colab kernel. No-op off
    glibc/Linux, never raises."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _hard_free() -> None:
    """Aggressively hand memory back — Python garbage first, then the CUDA
    caching allocator's free blocks, then glibc's heap pages. A no-op (never
    raises) without torch/GPU. Called after every model in a sweep so the next
    one starts with a clean GPU *and* a clean RSS."""
    try:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        _trim_cpu_ram()
    except Exception:  # noqa: BLE001
        pass


def _is_oom(err: BaseException) -> bool:
    """Does this exception look like an out-of-memory (CUDA or host)?"""
    s = str(err).lower()
    if "out of memory" in s or "outofmemory" in s or "oom" in s:
        return True
    try:
        import torch

        if isinstance(err, torch.cuda.OutOfMemoryError):  # torch >= 1.13
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


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
                 min_ram_gb: float = 0.0, subfolder: str | None = None,
                 paligemma_prompt: str | None = None):
        self.name = name
        self.hf_id = hf_id
        self.arch = arch
        self.dtype = dtype
        self.adapter_id = adapter_id
        # some checkpoints live in a sub-directory of the repo (e.g. a repo that
        # ships `final_model/` alongside training checkpoints)
        self.subfolder = subfolder or None
        self.max_new_tokens = max_new_tokens
        # PaliGemma has no chat template - the eval prefix must match whatever
        # prefix the checkpoint was fine-tuned with (`prompt:` in the model's
        # models.yaml entry), or eval scores it on a prefix it never saw.
        # Falls back to the paligemma-arabic-hw adapter's prefix for entries
        # that don't set one.
        self.paligemma_prompt = paligemma_prompt or _DEFAULT_PALIGEMMA_PROMPT
        self.load_in_4bit = load_in_4bit
        # How much memory this model needs to load safely, in GB — enforced
        # against free system RAM on CPU and against free VRAM on a CUDA GPU.
        # From `min_ram_gb` in configs/models.yaml. Loading a model that doesn't
        # fit either OOM-kills the kernel (CPU) or leaves the GPU fragmented
        # enough to need a runtime restart (CUDA); we'd rather skip it cleanly.
        self.min_ram_gb = float(min_ram_gb or 0.0)
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = None
        self._oom = False  # set once a load OOMs — the rest of the run skips it

    def _mem_preflight(self) -> None:
        """Refuse to start loading unless there's enough free memory for it.

        On a CUDA GPU the usual reason a small model won't load is that an
        *earlier* model in the sweep didn't hand its VRAM back — so try to
        reclaim the caching allocator first, and only skip if it's still short.
        On CPU a model bigger than free RAM doesn't raise, the Linux OOM killer
        just takes the kernel, so the check is the only guard. MPS has no
        queryable budget, so it's left unchecked."""
        need = self.min_ram_gb
        if need <= 0:
            return

        ft = _cuda_free_total_gb()
        if ft is not None:  # CUDA GPU
            free_gb, total_gb = ft
            # `min_ram_gb` is sized for the fp16 model in CPU RAM. When it will
            # actually be loaded 4-bit onto the GPU the real VRAM need is ~1/4 of
            # that (nf4 weights) plus room for the vision tower, KV cache and
            # activations — call it a third, with a 3.5 GB floor.
            need_vram = max(need / 3.0 if self.load_in_4bit else need, 3.5)
            if free_gb < need_vram * 1.10:
                _hard_free()  # an earlier model probably still holds VRAM
                ft = _cuda_free_total_gb() or ft
                free_gb = ft[0]
            if free_gb < need_vram:
                raise RuntimeError(
                    f"skip {self.name}: needs ~{need_vram:.0f} GB free GPU RAM "
                    f"({'4-bit' if self.load_in_4bit else 'fp16'}), only "
                    f"{free_gb:.1f} of {total_gb:.0f} GB free after reclaiming "
                    f"cache. A previous model likely didn't release its VRAM — "
                    f"skipping this one rather than risk an OOM that needs a "
                    f"runtime restart."
                )
            # weights are staged in CPU RAM during from_pretrained even when the
            # target is the GPU; guard the kernel against a host OOM too.
            ram = _available_ram_gb()
            if ram < 2.0:
                raise RuntimeError(
                    f"skip {self.name}: only {ram:.1f} GB system RAM free — "
                    f"loading weights stages them in RAM first and this would "
                    f"OOM-kill the kernel. Restart the runtime (RAM creeps up "
                    f"across many model loads)."
                )
            return

        try:
            import torch

            if torch.backends.mps.is_available():
                return  # no queryable budget on MPS
        except Exception:  # noqa: BLE001 - no torch yet; treat as CPU
            pass
        avail = _available_ram_gb()
        if avail < need:
            raise RuntimeError(
                f"skip {self.name} on CPU: needs ~{need:.0f} GB free RAM, only "
                f"{avail:.1f} GB available (loading it would OOM-kill the "
                f"kernel). Use a GPU runtime, or drop this model."
            )

    # back-compat alias
    _ram_preflight = _mem_preflight

    # -- lazy load ---------------------------------------------------------- #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self._oom:
            raise RuntimeError(
                f"skip {self.name}: hit CUDA OOM while loading earlier this run "
                f"— not retrying it for the remaining lines."
            )
        self._mem_preflight()
        try:
            self._load_dispatch()
        except Exception as e:  # noqa: BLE001
            if _is_oom(e):
                self.close()          # hand back whatever was allocated
                self._oom = True      # short-circuit the rest of this model's lines
                raise RuntimeError(
                    f"skip {self.name}: CUDA out of memory during load. Freed "
                    f"what could be freed; the rest of this run skips this "
                    f"model. Original error: {e}"
                ) from e
            raise

    def _load_dispatch(self):
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
        _ft = _cuda_free_total_gb()
        _vram = f", {_ft[0]:.1f}/{_ft[1]:.0f} GB VRAM free" if _ft else ""
        print(f"[load] {self.hf_id}{adapter} "
              f"({'4-bit' if use_4bit else str(torch_dtype).replace('torch.', '')} on {device}"
              f"{_vram}) ...",
              flush=True)
        _sub = {"subfolder": self.subfolder} if self.subfolder else {}
        self._processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=True, **_sub)

        # Some checkpoints (unsloth-derived, e.g. Qari-OCR) carry a bitsandbytes
        # 4-bit `quantization_config` in config.json. That path is fragile on
        # Colab ("'Parameter' object has no attribute 'compress_statistics'") and
        # unnecessary for a 2-3B model on a 16 GB GPU. Unless the caller opted
        # into quantization, drop it and load in plain fp16/bf16.
        cfg = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True, **_sub)
        if not use_4bit and hasattr(cfg, "quantization_config"):
            delattr(cfg, "quantization_config")

        load_kwargs = {
            "config": cfg,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            **_sub,
        }
        if use_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            # Force EVERY layer onto GPU 0. With device_map="auto", accelerate
            # silently offloads layers it isn't sure will fit into *system RAM*
            # — which then bleeds host memory across a sweep and keeps VRAM
            # deceptively low. Pinning to {"": 0} makes a too-big model raise a
            # clean CUDA OOM at load (caught below → model skipped) instead.
            load_kwargs["device_map"] = {"": 0}

        model = _load_model(self.hf_id, load_kwargs)
        if self.adapter_id:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_id)
        # 4-bit: device_map={"": 0} already placed every weight on the GPU.
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
        models in a sweep so a single GPU can host them one at a time. Drops
        every reference, then reclaims Python garbage and the CUDA caching
        allocator so the next model starts with a clean GPU."""
        had_model = self._model is not None
        self._model = None
        self._processor = None
        self._tokenizer = None
        _hard_free()
        if had_model:
            ft = _cuda_free_total_gb()
            if ft is not None:
                print(f"[unload] {self.name}: {ft[0]:.1f}/{ft[1]:.0f} GB GPU free",
                      flush=True)

    # -- inference -------------------------------------------------------- #
    def transcribe(self, image: Image.Image, prompt: str,
                    max_tokens: int | None = None) -> Prediction:
        # max_tokens: an optional per-call override (the runner passes the
        # calling dataset's `max_tokens_hint` from configs/datasets.yaml, if
        # any). Only ever RAISES the effective cap above this client's own
        # registered self.max_new_tokens, mirroring the existing run-level
        # max_tokens_override in runner.py -- a page-scale dataset can ask for
        # more room without a tighter per-model default ever being clobbered.
        eff_max_new_tokens = (
            max(self.max_new_tokens, max_tokens) if max_tokens is not None
            else self.max_new_tokens
        )
        t0 = time.perf_counter()
        try:
            self._ensure_loaded()
            text = self._generate(image.convert("RGB"), prompt, eff_max_new_tokens)
        except Exception as e:  # noqa: BLE001
            # An OOM *during generate* (vs. during load) is usually one
            # pathological image, not the whole model — reclaim and let the next
            # line try. `_ensure_loaded` already handles a load-time OOM by
            # marking the model skipped for the rest of the run.
            if _is_oom(e) and not self._oom:
                _hard_free()
            return Prediction(error=f"{type(e).__name__}: {e}",
                              latency_ms=(time.perf_counter() - t0) * 1000)
        latency_ms = (time.perf_counter() - t0) * 1000
        if looks_like_refusal(text):
            return Prediction(raw_text=text, cleaned_text="", error="refusal",
                              latency_ms=latency_ms)
        return Prediction(raw_text=text, cleaned_text=clean_output(text),
                          latency_ms=latency_ms)

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        import torch

        image = cap_image(image, 1024)  # bound vision tokens / activation memory
        if self.arch == "paligemma":
            return self._generate_paligemma(image, prompt, max_new_tokens)
        if self.arch == "trocr":
            return self._generate_trocr(image, max_new_tokens)
        if self.arch == "minicpmv":
            return self._generate_minicpmv(image, prompt, max_new_tokens)

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
                **inputs, max_new_tokens=max_new_tokens, **_GEN_KW
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(gen, skip_special_tokens=True).strip()

    def _generate_minicpmv(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        msgs = [{"role": "user", "content": [image, prompt]}]
        out = self._model.chat(
            image=None, msgs=msgs, tokenizer=self._tokenizer,
            sampling=False, max_new_tokens=max_new_tokens,
        )
        return (out if isinstance(out, str) else str(out)).strip()

    def _generate_trocr(self, image: Image.Image, max_new_tokens: int) -> str:
        import torch

        pixel_values = self._processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self._device)
        with torch.no_grad():
            ids = self._model.generate(
                pixel_values, max_new_tokens=max_new_tokens, **_GEN_KW
            )
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def _generate_paligemma(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        import torch

        # PaliGemma uses a task-prefix, not a chat template (see self.paligemma_prompt).
        #
        # Exactly ONE "<image>" placeholder -- NOT `image_seq_length` copies.
        # PaliGemmaProcessor.__call__ (transformers/models/paligemma/
        # processing_paligemma.py) detects that the text already contains
        # "<image>" and expands EVERY occurrence into `self.image_seq_length`
        # copies itself (`sample.replace(IMAGE_TOKEN, IMAGE_TOKEN *
        # self.image_seq_length)`). Passing image_seq_length copies here made
        # it expand each of those into another image_seq_length -- e.g.
        # 1024*1024 = 1,048,576 total "<image>" tokens for the -448 checkpoint
        # -- failing the processor's own image-token-count check on every
        # single call (100% error rate). See the identical bug/fix in
        # training/sft_vlm.py's _make_collator_paligemma for the full story;
        # this is the same mistake in the separate inference code path.
        pg_prompt = "<image>" + self.paligemma_prompt
        inputs = self._processor(text=pg_prompt, images=image, return_tensors="pt").to(
            self._device
        )
        in_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, **_GEN_KW
            )
        return self._processor.decode(out[0][in_len:], skip_special_tokens=True).strip()
