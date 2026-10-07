"""Optional torch smoke for the CPU CRNN — skipped when torch isn't installed."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from arabic_ocr_bench.charctc.charset import build_charset
from arabic_ocr_bench.charctc.decode import ctc_greedy, ids_to_text
from arabic_ocr_bench.charctc.model import CharCRNN, time_steps_for
from arabic_ocr_bench.charctc.preprocess import line_to_array


def test_penalize_blank_flips_greedy_away_from_all_blank():
    from arabic_ocr_bench.charctc.decode import (
        blank_fraction,
        greedy_from_logits,
        penalize_blank_logits,
    )

    cs = build_charset(["اب"])
    a_id = cs.stoi("ا")
    # T=4, B=1: blank slightly ahead of alef on every step → greedy is empty.
    logits = torch.full((4, 1, cs.n_classes), -10.0)
    logits[..., 0] = 0.2
    logits[..., a_id] = 0.0
    assert blank_fraction(logits) == 1.0
    assert greedy_from_logits(logits, [4], cs) == [""]

    flipped = penalize_blank_logits(logits, 0.5)
    assert blank_fraction(flipped) == 0.0
    assert greedy_from_logits(flipped, [4], cs) == ["ا"]
    # original tensor is not mutated
    assert blank_fraction(logits) == 1.0


def test_crnn_logits_match_width_stride():
    cs = build_charset(["بسم الله الرحمن الرحيم"])
    model = CharCRNN(cs.n_classes, hidden=32)
    model.eval()
    img = Image.new("L", (128, 32), 255)
    arr = line_to_array(img, height=32, max_width=128)
    x = torch.from_numpy(np.array(arr)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    t, b, c = logits.shape
    assert b == 1
    assert c == cs.n_classes
    assert t == time_steps_for(arr.shape[-1])
    ids = ctc_greedy(logits.argmax(-1).squeeze(1).tolist())
    assert isinstance(ids_to_text(ids, cs), str)
