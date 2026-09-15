"""Small image helpers shared across the harness."""

from __future__ import annotations

from PIL import Image


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
    return img
