"""Small image helpers shared across the harness."""

from __future__ import annotations

from PIL import Image


def cap_image(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Downscale so the longest edge is <= max_dim (keeps aspect). A text line
    stays perfectly legible at 1024 px, and this keeps memory / vision-token
    counts bounded."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
