"""Synthetic printed-line generation: filtering, rendering, degradation, adapter."""

from __future__ import annotations

import numpy as np
from PIL import Image

from arabic_ocr_bench.training.synth import (
    SyntheticPrinted,
    arabic_ratio,
    degrade,
    drop_overlap,
    fonts,
    render_line,
    sentences,
)

_BLOBS = [
    "المدرسة قريبة من البيت. ذهب الطالب إلى الصف مبكرا وقرأ الدرس بتأنٍ شديد.",
    "القراءة غذاء العقل والروح، وهي عادة يومية ينبغي أن يحرص عليها كل إنسان مهذب.",
    "hello world this is english only and should be dropped by the arabic ratio filter",
    "الشمس تشرق من الشرق وتغرب في الغرب، وهذه حقيقة كونية يعرفها الجميع منذ القدم.",
]


def test_arabic_ratio_scores_script_share():
    assert arabic_ratio("مرحبا بالعالم") > 0.9
    assert arabic_ratio("hello world") == 0.0


def test_sentences_filters_and_is_deterministic():
    a = sentences(_BLOBS, n=3, seed=0, min_len=15, max_len=200)
    b = sentences(_BLOBS, n=3, seed=0, min_len=15, max_len=200)
    assert a == b
    assert len(a) == 3
    assert all(15 <= len(s) <= 200 for s in a)
    assert all(arabic_ratio(s) >= 0.6 for s in a)  # the english blob is gone


def test_drop_overlap_removes_matching_ground_truth():
    from arabic_ocr_bench.training.contamination import _match_key

    keep = "الكتاب على الطاولة الخشبية"
    drop = "بسم الله الرحمن الرحيم"
    out = drop_overlap([keep, drop], {_match_key(drop)})
    assert out == [keep]


def test_fonts_always_offers_the_bundled_face_and_real_paths():
    fs = fonts()
    assert fs and all(p.exists() for p in fs)
    assert any(p.name == "NotoNaskhArabic-Regular.ttf" for p in fs)


def test_render_line_draws_dark_ink_on_a_wide_image():
    im = render_line("السلام عليكم ورحمة الله", None, size=40)
    assert im.mode == "RGB"
    assert im.width > im.height
    assert np.asarray(im.convert("L")).min() < 80  # actual glyphs, not blank


def test_degrade_is_deterministic_and_keeps_size():
    base = render_line("القراءة غذاء العقل", None, size=40)
    d1 = degrade(base, np.random.default_rng(0), "medium")
    d2 = degrade(base, np.random.default_rng(0), "medium")
    assert d1.size == base.size
    assert np.array_equal(np.asarray(d1), np.asarray(d2))
    assert not np.array_equal(np.asarray(d1), np.asarray(base))


def test_synthetic_printed_adapter_yields_n_deterministic_samples():
    kw = {"corpora": [], "texts": _BLOBS, "n": 5, "seed": 1, "min_len": 15, "max_len": 200}
    s1 = list(SyntheticPrinted(**kw))
    s2 = list(SyntheticPrinted(**kw))
    assert len(s1) == 5
    assert [s.id for s in s1] == [s.id for s in s2]
    assert all(s.ground_truth.strip() for s in s1)
    assert all(isinstance(s.image, Image.Image) and s.track == "line" for s in s1)
