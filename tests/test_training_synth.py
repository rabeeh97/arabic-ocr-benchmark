"""Synthetic printed-line generation: filtering, rendering, degradation, adapter."""

from __future__ import annotations

import numpy as np
from PIL import Image

from arabic_ocr_bench.training.synth import (
    LazyImage,
    SyntheticPrinted,
    arabic_ratio,
    degrade,
    drop_overlap,
    fonts,
    render_line,
    sentences,
    synth_render_key,
)


def test_synth_render_key_ignores_n_and_limit_but_tracks_pixel_params():
    base = {"corpora": ["wikipedia", "classical"], "seed": 0,
            "degrade_level": "medium", "diacritics": "mixed",
            "drop_overlap_with": ["kitab_hindawi"], "corpus_docs": 4000}
    k = synth_render_key(base)
    # a 5k and a 15k config that agree on everything else share one render cache
    assert synth_render_key({**base, "n": 5000, "limit": 5000}) == k
    assert synth_render_key({**base, "n": 15000}) == k
    # anything that changes the rendered pixels changes the key
    assert synth_render_key({**base, "seed": 1}) != k
    assert synth_render_key({**base, "degrade_level": "heavy"}) != k
    assert synth_render_key({**base, "diacritics": "none"}) != k
    assert synth_render_key({**base, "corpora": ["wikipedia"]}) != k
    assert len(k) == 12

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


def test_cache_dir_yields_lazy_images_that_decode_on_demand(tmp_path):
    kw = {
        "corpora": [],
        "texts": _BLOBS,
        "n": 4,
        "seed": 1,
        "min_len": 15,
        "max_len": 200,
        "cache_dir": tmp_path,
    }
    samples = list(SyntheticPrinted(**kw))
    assert len(samples) == 4
    assert len(list(tmp_path.glob("*.png"))) == 4  # rendered once, to disk

    img = samples[0].image
    assert isinstance(img, LazyImage)
    assert img._lazy_loaded is False  # nothing decoded yet
    assert img.size[0] > img.size[1]  # touching a pixel-op pages it in
    assert img._lazy_loaded is True

    # a second adapter reuses the cache without re-rendering and stays lazy
    again = list(SyntheticPrinted(**kw))
    assert isinstance(again[0].image, LazyImage)
    assert again[0].image._lazy_loaded is False
    assert again[0].image.convert("L").size == img.size


def test_work_dir_checkpoints_to_a_single_tar_and_resumes(tmp_path):
    drive = tmp_path / "drive"      # durable store (stand-in for Google Drive)
    local1 = tmp_path / "local1"    # fast scratch, wiped between "sessions"
    kw = {
        "corpora": [], "texts": _BLOBS, "n": 6, "seed": 2,
        "min_len": 15, "max_len": 200,
        "cache_dir": drive, "work_dir": local1, "flush_every": 2,
    }
    s1 = list(SyntheticPrinted(**kw))
    assert len(s1) == 6
    # images rendered to the local dir, checkpoint is one tar on the durable store
    assert len(list(local1.glob("*.png"))) == 6
    assert (drive / "images.tar").exists()
    assert not list(drive.glob("*.png"))          # no loose files on "Drive"
    assert (drive / f"pool_{SyntheticPrinted(**kw)._pool_key()}.json").exists()

    # new "session": fresh local dir, only the tar survives -> restored, no re-render
    local2 = tmp_path / "local2"
    adapter = SyntheticPrinted(**{**kw, "work_dir": local2})
    import arabic_ocr_bench.training.synth as synth_mod

    calls = []
    orig = synth_mod.SyntheticPrinted._render
    synth_mod.SyntheticPrinted._render = lambda self, *a, **k: calls.append(1) or orig(self, *a, **k)
    try:
        s2 = list(adapter)
    finally:
        synth_mod.SyntheticPrinted._render = orig
    assert len(s2) == 6
    assert calls == []                              # everything came from the tar
    assert len(list(local2.glob("*.png"))) == 6
    assert [x.id for x in s1] == [x.id for x in s2]
