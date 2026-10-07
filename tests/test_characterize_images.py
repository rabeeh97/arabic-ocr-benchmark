"""Image-layer features (characterize/images.py) on synthetic images."""
import numpy as np
from PIL import Image, ImageDraw

from arabic_ocr_bench.characterize.images import image_features, otsu


def _line(w=600, h=60, bar=(100, 20, 500, 40), left_ink=False):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    d.rectangle(bar, fill=0)
    if left_ink:
        d.rectangle((0, 25, 5, 35), fill=0)
    return img.convert("RGB")


def test_otsu_splits_two_levels():
    g = np.array([[10] * 50 + [240] * 50], dtype=np.uint8)
    assert 10 <= otsu(g) < 240


def test_geometry_ink_and_scale():
    f = image_features(_line(), n_chars=40)
    assert f["width"] == 600 and f["height"] == 60 and f["aspect"] == 10
    assert abs(f["ink_density"] - (401 * 21) / (600 * 60)) < 0.01
    assert abs(f["ink_width_frac"] - 401 / 600) < 0.01
    assert f["core_height"] == 21 and f["stroke_width"] == 21
    assert abs(f["ink_px_per_char"] - 401 / 40) < 0.1
    assert f["contrast"] > 200 and f["ink_left_edge"] == 0


def test_left_edge_ink_is_detected():
    assert image_features(_line(left_ink=True))["ink_left_edge"] > 0


def test_light_on_dark_is_inverted():
    dark = Image.eval(_line().convert("L"), lambda v: 255 - v).convert("RGB")
    assert abs(image_features(dark)["ink_density"] - image_features(_line())["ink_density"]) < 0.01


def test_flat_line_has_no_skew():
    assert image_features(_line())["skew_deg"] == 0
