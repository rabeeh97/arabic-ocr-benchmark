"""cap_image: longest-edge bound + Qwen VL aspect-ratio clamp."""

from PIL import Image

from arabic_ocr_bench.imaging import cap_image

# qwen_vl_utils.smart_resize raises if max(h, w) / min(h, w) > 200
_QWEN_MAX_RATIO = 200


def _ratio(img: Image.Image) -> float:
    w, h = img.size
    return max(w, h) / min(w, h)


def test_cap_image_leaves_modest_image_unchanged():
    img = Image.new("RGB", (100, 40), "black")
    out = cap_image(img)
    assert out.size == (100, 40)
    assert out is img


def test_cap_image_downscales_longest_edge():
    img = Image.new("RGB", (2048, 512), "black")
    out = cap_image(img)
    assert out.size == (1024, 256)


def test_cap_image_letterboxes_ratio_over_qwen_max():
    # The crash on spring-win-server-3090: a line crop with ratio 389.0
    # (e.g. 389×1) trips qwen_vl_utils.smart_resize's MAX_RATIO=200 check
    # inside the SFT collator. Padding the short side keeps the example.
    img = Image.new("RGB", (389, 1), "black")
    out = cap_image(img)
    assert out.size[0] == 389
    assert _ratio(out) <= _QWEN_MAX_RATIO
    # original ink is still there, not stretched (pad may sit above/below it)
    assert any(out.getpixel((0, y)) == (0, 0, 0) for y in range(out.size[1]))


def test_cap_image_letterboxes_tall_ratio_over_qwen_max():
    img = Image.new("RGB", (1, 389), "black")
    out = cap_image(img)
    assert out.size[1] == 389
    assert _ratio(out) <= _QWEN_MAX_RATIO


def test_cap_image_downscale_then_letterbox_when_still_too_skinny():
    # Longest-edge cap alone is not enough: 3890×10 → 1024×3, ratio ~341.
    img = Image.new("RGB", (3890, 10), "black")
    out = cap_image(img)
    assert max(out.size) <= 1024
    assert _ratio(out) <= _QWEN_MAX_RATIO
