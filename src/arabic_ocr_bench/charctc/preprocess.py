"""Line-crop -> (1, H, W) float32 array. Height fixed, width capped, aspect kept."""

from __future__ import annotations

import numpy as np
from PIL import Image

TARGET_H = 32
MAX_W = 512
# Two (2, 2) pools on width in the CNN -> time steps = width // WIDTH_STRIDE.
WIDTH_STRIDE = 4


def _to_multiple(n: int, stride: int) -> int:
    return max(stride, ((n + stride - 1) // stride) * stride)


def line_to_array(
    image: Image.Image,
    *,
    height: int = TARGET_H,
    max_width: int = MAX_W,
    width_stride: int = WIDTH_STRIDE,
) -> np.ndarray:
    """Grayscale, height-normalized, width padded to a multiple of ``width_stride``.

    Returns float32 in ``[-1, 1]`` with shape ``(1, height, width)``.
    """
    gray = image.convert("L")
    w, h = gray.size
    if h <= 0 or w <= 0:
        arr = np.zeros((1, height, width_stride), dtype=np.float32)
        return arr
    scale = height / h
    new_w = max(1, int(round(w * scale)))
    new_w = min(new_w, max_width)
    new_w = _to_multiple(new_w, width_stride)
    resized = gray.resize((new_w, height), Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr[np.newaxis, :, :]


def time_steps(width: int, *, width_stride: int = WIDTH_STRIDE) -> int:
    return width // width_stride
