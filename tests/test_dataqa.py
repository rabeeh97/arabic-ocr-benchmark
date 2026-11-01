"""Unit tests for arabic_ocr_bench.dataqa on synthetic images and strings."""

from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from arabic_ocr_bench import dataqa as q

GT = "وقال الشيخ رحمه الله تعالى في كتابه"


def line_img(w=600, h=60, lines=1, gap=20, line_h=None, pad=0, seed=0):
    """Light page with ``lines`` rows of dark 'words' (blobs with gaps)."""
    rng = np.random.default_rng(seed)
    line_h = line_h or (h - gap * (lines - 1)) // lines // 2
    H = h
    im = Image.new("L", (w, H), 225)
    d = ImageDraw.Draw(im)
    total = lines * line_h + (lines - 1) * gap
    y0 = (H - total) // 2
    for k in range(lines):
        y = y0 + k * (line_h + gap)
        x = 10
        while x < w - 30:
            ww = int(rng.integers(15, 60))
            d.rectangle([x, y, min(x + ww, w - 10), y + line_h], fill=40)
            x += ww + int(rng.integers(8, 20))
    if pad:
        a = np.asarray(im).copy()
        a[:pad] = 0
        a[-pad:] = 0
        im = Image.fromarray(a)
    return im.convert("RGB")


def checks(fs):
    return {f.check for f in fs}


# ---- text ------------------------------------------------------------------

def test_empty_gt():
    assert checks(q.check_text("  ..  ")) == {"empty_gt"}


def test_repeated_gt_paste():
    half = "فيه مدخل الا بعتق او جبر يعني ان ميراث الولاء"
    assert "repeated_gt" in checks(q.check_text(half + " " + half))
    assert "repeated_gt" not in checks(q.check_text(GT))


def test_natural_repeat_not_flagged():
    assert "repeated_gt" not in checks(q.check_text("ثم تضرب الاثنين في الاثنين باربعة"))


def test_char_run_and_off_script():
    assert "char_run" in checks(q.check_text("قال ـــــــ ههههههههه"))
    assert "char_run" not in checks(q.check_text("قال ــــــــــ"))  # tatweel runs are fine
    assert "off_script" in checks(q.check_text("this is latin text"))
    assert "off_script" not in checks(q.check_text(GT))


def test_longest_repeat():
    s = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    i, j, L = q.longest_repeat(s[:12] + "نا" + s[:12] + s[12:])
    assert L >= 12 and j - (i + L) <= 2


# ---- image -----------------------------------------------------------------

def test_degenerate_and_vertical():
    assert checks(q.check_image(q.image_stats(line_img(400, 5)), GT)) == {"degenerate"}
    tall = line_img(60, 300)
    assert "vertical" in checks(q.check_image(q.image_stats(tall), GT))
    assert "vertical" not in checks(q.check_image(q.image_stats(tall), "في"))  # short GT may be narrow


def test_blank():
    im = Image.new("RGB", (500, 60), (230, 225, 210))
    assert "blank" in checks(q.check_image(q.image_stats(im), GT))
    assert "blank" not in checks(q.check_image(q.image_stats(line_img()), GT))


def test_multi_line():
    one = q.image_stats(line_img(800, 80, lines=1, line_h=24))
    two = q.image_stats(line_img(800, 140, lines=2, line_h=24, gap=40))
    assert one["n_lines"] == 1 and "multi_line" not in checks(q.check_image(one, GT))
    assert two["n_lines"] == 2 and "multi_line" in checks(q.check_image(two, GT))


def test_padding_is_not_ink():
    st = q.image_stats(line_img(800, 100, lines=1, line_h=24, pad=12))
    assert st["n_lines"] == 1


def test_fill_mask_keeps_strokes():
    g = np.asarray(line_img().convert("L"))
    assert not q.fill_mask(g).any()


# ---- joint -----------------------------------------------------------------

def test_length_mismatch_sign():
    rng = np.random.default_rng(0)
    ratios = {f"d/t/{i}": float(rng.normal(0, 0.1)) for i in range(100)}
    ratios["d/t/long"] = 2.0
    ratios["d/t/short"] = -2.0
    fl = q.check_length(ratios)
    assert set(fl) == {"d/t/long", "d/t/short"}
    assert fl["d/t/long"].extra["z"] > 0 > fl["d/t/short"].extra["z"]


# ---- reader ----------------------------------------------------------------

def test_gt_truncated_and_image_truncated():
    full = GT + " وهو كتاب جليل"
    assert "gt_truncated" in checks(q.check_reader(GT[:15], GT))
    assert "image_truncated" in checks(q.check_reader(full, GT[:15]))
    assert not q.check_reader(GT, GT)


def test_doubled_gt_needs_reader_confirmation():
    half = "فيه مدخل الا بعتق او جبر يعني ان ميراث الولاء"
    assert "doubled_gt" in checks(q.check_reader(half + " " + half, half))
    nat = "ثم تضرب الاثنين في الاثنين باربعة"
    assert "doubled_gt" not in checks(q.check_reader(nat, nat))


def test_shifted():
    a, b = "بدا اخذت منه ما تقدما فما بقي فهو الذي به اتبع", "اوتجمع التركتين واقسما على فريض فما قد انتما"
    items = {"p/t/1": (a, b), "p/t/2": (b, a)}
    fl = q.check_shifted(items, {"pg": ["p/t/1", "p/t/2"]})
    assert set(fl) == {"p/t/1", "p/t/2"} and fl["p/t/1"].extra["match"] == "p/t/2"


# ---- dataset + driver ------------------------------------------------------

def test_duplicates_and_scan():
    im = line_img()
    s = [SimpleNamespace(id="d/train/0", image=im, ground_truth=GT),
         SimpleNamespace(id="d/test/1", image=im.copy(), ground_truth=GT + " ما"),
         SimpleNamespace(id="d/test/2", image=line_img(seed=3), ground_truth="")]
    res = q.scan(s)
    assert res["n"] == 3
    f0 = {f["check"]: f for f in res["flags"]["d/train/0"]}
    assert "DIFFERS" in f0["duplicate_image"]["detail"] and "crosses splits" in f0["duplicate_image"]["detail"]
    assert {f["check"] for f in res["flags"]["d/test/2"]} == {"empty_gt"}


def test_every_check_is_defined():
    for name in ["empty_gt", "repeated_gt", "char_run", "off_script", "degenerate", "vertical", "blank",
                 "multi_line", "neighbour_clip", "length_mismatch", "gt_truncated", "image_truncated",
                 "doubled_gt", "shifted", "unreadable", "duplicate_image", "duplicate_gt"]:
        assert name in q.CHECKS
