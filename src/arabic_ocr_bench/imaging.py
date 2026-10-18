"""Small image helpers shared across the harness."""

from __future__ import annotations

import math

from PIL import Image, ImageOps

# qwen_vl_utils.smart_resize raises if max(h, w) / min(h, w) > 200. Line-level
# OCR crops (a long thin manuscript line) routinely exceed that; letterbox the
# short side instead of dropping the example or crashing the SFT collator.
_QWEN_MAX_RATIO = 200


def cap_image(img: Image.Image, max_dim: int = 1024, min_dim: int = 8) -> Image.Image:
    """Downscale so the longest edge is <= max_dim (keeps aspect); floor so
    neither edge is < min_dim. A text line stays perfectly legible at
    1024 px, and capping keeps memory / vision-token counts bounded.

    The min_dim floor exists for a different reason: a corrupt or near-empty
    bounding polygon in a source dataset can produce a crop just a few pixels
    (sometimes 1 px) tall. That's unreadable to any model regardless, but it
    also makes the array shape genuinely ambiguous to HF image processors'
    channel-dimension heuristic -- a dimension of 1, 3 or 4 can be mistaken
    for a channel count ("The channel dimension is ambiguous... Assuming
    channels are the first dimension"), which silently transposes H/W and
    garbles the whole image rather than just scoring one bad line as a normal
    error. min_dim=8 clears that ambiguity zone by a comfortable margin while
    staying far below any real line crop's height (normal crops resize to
    tens of pixels tall even at max_dim=1024).

    The floor is applied per-axis *after* the aspect-preserving max_dim cap,
    not as a second uniform scale -- a uniform scale-up-then-down for an
    extreme aspect ratio (e.g. 434x1) would just undo itself (scale up to
    clear min_dim, then the max_dim cap scales the now-huge long edge back
    down, dragging the short edge back under min_dim with it). Flooring each
    axis independently breaks aspect ratio only for that rare pathological
    case -- readability wins over exact proportions on an already-corrupt
    crop -- and never touches an ordinary image, since real line crops are
    always well above 8 px on both edges.

    After capping and flooring, a ratio beyond Qwen's ``smart_resize`` MAX_RATIO=200 is
    letterboxed on the short side (``_clamp_aspect_ratio``).
    """
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / longest
        w, h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((w, h), Image.LANCZOS)
    nw, nh = max(w, min_dim), max(h, min_dim)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
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
