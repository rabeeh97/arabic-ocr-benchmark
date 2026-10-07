"""Small image helpers shared across the harness."""

from __future__ import annotations

import math

from PIL import Image, ImageOps

# qwen_vl_utils.smart_resize raises if max(h, w) / min(h, w) > 200. Line-level
# OCR crops (a long thin manuscript line) routinely exceed that; letterbox the
# short side instead of dropping the example or crashing the SFT collator.
_QWEN_MAX_RATIO = 200


def cap_image(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Downscale so the longest edge is <= max_dim (keeps aspect), then pad the
    short side if the ratio would trip Qwen's ``smart_resize`` MAX_RATIO=200.

    A text line stays perfectly legible at 1024 px, and this keeps memory /
    vision-token counts bounded.
    """
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        w, h = img.size
    return _clamp_aspect_ratio(img, max_ratio=_QWEN_MAX_RATIO)


def _clamp_aspect_ratio(img: Image.Image, max_ratio: float = _QWEN_MAX_RATIO) -> Image.Image:
    w, h = img.size
    shortest = min(w, h)
    if shortest <= 0 or max(w, h) / shortest <= max_ratio:
        return img
    if w >= h:
        extra = math.ceil(w / max_ratio) - h
        top, bottom = extra // 2, extra - extra // 2
        return ImageOps.expand(img, border=(0, top, 0, bottom), fill="white")
    extra = math.ceil(h / max_ratio) - w
    left, right = extra // 2, extra - extra // 2
    return ImageOps.expand(img, border=(left, 0, right, 0), fill="white")
