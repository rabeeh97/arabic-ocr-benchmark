"""Per-sample characteristics (characterize/samples.py) on hand-built inputs."""
import random
from types import SimpleNamespace as NS

import pytest

from arabic_ocr_bench.characterize import samples as S


def _rd(ids, err=()):
    return {i: NS(error="x" if i in err else None, gt="", pred="") for i in ids}


def test_complete_block_keeps_shared_successful_samples():
    by_model = {
        "a": _rd(range(100)), "b": _rd(range(100)), "c": _rd(range(100), err={5}),
        "d": _rd(range(50)), "e": _rd(range(100)), "f": _rd(range(100)),
    }
    models, sids = S.complete_block(by_model, min_models=5)
    # 5 models x 99 (drop 'd', drop errored sample 5) beats 6 x 49
    assert sorted(models) == ["a", "b", "c", "e", "f"]
    assert len(sids) == 99 and 5 not in sids


def test_line_cer_is_capped_and_diacritic_insensitive():
    assert S.line_cer("كَتَبَ", "كتب") == 0
    assert S.line_cer("باب", "the image shows a page of text") == S.CER_CAP


def test_letter_outcomes_mark_the_wrong_letter_with_its_form():
    out = S.letter_outcomes("باب", "تاب")
    assert [(l, p, ok) for l, p, _, _, _, ok in out] == [
        ("ب", "initial", False), ("ا", "final", True), ("ب", "isolated", True)]
    assert out[0][3] is True          # initial ب is dot-ambiguous
    assert 0 <= out[-1][4] < 1        # relative position


def test_gt_features_basic_shares():
    f = S.gt_features("باب 12", word_freq={"باب": 3})
    assert f["words"] == 2 and f["digit_share"] > 0 and f["oov_share"] == 0.5
    assert 0 < f["ambiguous_share"] <= 1


def test_spearman_and_partial_spearman():
    x = list(range(50))
    assert S.spearman(x, [v * 2 for v in x]) == pytest.approx(1)
    rng = random.Random(0)
    length = [rng.random() for _ in x]
    a = [l + 0.01 * rng.random() for l in length]   # a and b both just track length
    b = [l + 0.01 * rng.random() for l in length]
    assert S.spearman(a, b) > 0.9
    assert abs(S.partial_spearman(a, b, length)) < 0.5
