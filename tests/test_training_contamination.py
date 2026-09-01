"""The contamination checker flags training items that collide with a test set."""

from __future__ import annotations

from PIL import Image, ImageDraw

from arabic_ocr_bench.training.contamination import (
    dhash,
    hamming,
    scan_overlap,
)


def _img(seed: int, size=(64, 32)) -> Image.Image:
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    d.rectangle([seed % 20, 4, seed % 20 + 30, 24], fill="black")
    return im


def test_dhash_is_stable_for_the_same_image():
    assert dhash(_img(3)) == dhash(_img(3))


def test_dhash_differs_for_clearly_different_images():
    a = Image.new("RGB", (64, 32), "white")
    ImageDraw.Draw(a).rectangle([2, 2, 20, 28], fill="black")
    b = Image.new("RGB", (64, 32), "white")
    ImageDraw.Draw(b).ellipse([30, 4, 60, 26], fill="black")
    assert hamming(dhash(a), dhash(b)) > 8


def test_scan_overlap_finds_exact_text_collision():
    train = [("t/0", _img(1), "بسم الله الرحمن الرحيم")]
    test = [("x/0", _img(9), "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ")]  # same words, diacritized
    report = scan_overlap(train, test)
    assert report["n_text_collisions"] == 1
    assert report["text_collisions"][0]["train_id"] == "t/0"


def test_scan_overlap_finds_near_duplicate_image():
    shared = _img(5)
    train = [("t/0", shared, "abc")]
    test = [("x/0", shared.copy(), "totally different text")]
    report = scan_overlap(train, test, near_dup_threshold=4)
    assert report["n_near_dup_images"] == 1


def test_scan_overlap_clean_sets_report_nothing():
    train = [("t/0", _img(1), "الكتاب على الطاولة")]
    test = [("x/0", _img(40), "الطائرة تحلق فوق الجبال")]
    report = scan_overlap(train, test, near_dup_threshold=2)
    assert report["n_text_collisions"] == 0
    assert report["n_near_dup_images"] == 0
