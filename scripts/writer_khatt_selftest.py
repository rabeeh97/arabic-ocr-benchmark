"""Unit tests for the pure helpers of writer_khatt_pages.py and writer_khatt_fixed_text.py (checklist 2.16 / 2.17).

    .geomvenv/bin/python scripts/writer_khatt_selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from writer_khatt_fixed_text import additive_fit, icc1, residuals, word_errors  # noqa: E402
from writer_khatt_pages import CANON, align, chains, link, span  # noqa: E402


def test_align():
    assert len(CANON) == 77
    s = align("ذهب نوح مطفر ضرغام".split())
    assert s == [0, 1, 2, 3], s
    s = align("لهذا النص مشمش دراق غيظ ناء بث نسر".split())
    assert span(s) == (69, 76), s
    s = align("أننا في الحج هل".split())  # the second "في" (slot 62), not "في الخيمة" (slot 28)
    assert s[1] == 62, s
    assert align([]) == []


def test_chains_and_link():
    assert chains({0: 1, 1: 2, 5: 6}) == [[0, 1, 2], [5, 6]]
    # two writers (style 0 vs 5), each wrote slots 0-3, 4-7: links must stay within a writer
    start = np.array([0, 4, 0, 4])
    end = np.array([3, 7, 3, 7])
    Z = np.array([[0.0], [0.1], [5.0], [5.1]])
    succ, cost, margin, _ = link(start, end, Z, unlinked=3.0)
    assert succ == {0: 1, 2: 3}, succ
    assert margin[0] > 4


def test_stats():
    assert word_errors("ذهب نوح مظفر", "ذهب نوخ مظفر") == [0, 1, 0]
    assert word_errors("ذهب نوح", "") == [1, 1]
    # purely column (text) structure -> partial writer ~ 0
    Y = np.tile(np.array([1, 0, 1, 0, 0, 1], float), (8, 1))  # adjusted R^2 of an exact fit is still 1
    M = np.ones_like(Y, bool)
    f = additive_fit(Y, M)
    assert abs(f["r2_text"] - 1) < 1e-9 and abs(f["partial_writer"]) < 1e-9
    f = additive_fit(Y.T.copy(), M.T.copy())
    assert abs(f["r2_writer"] - 1) < 1e-9 and abs(f["partial_text"]) < 1e-9
    assert icc1([1, 1, 0, 0], ["a", "a", "b", "b"]) == 1.0
    assert icc1([1, 0, 1, 0], ["a", "a", "b", "b"]) < 0
    rng = np.random.default_rng(0)
    f = additive_fit((rng.random((60, 40)) < 0.3).astype(float), np.ones((60, 40), bool))
    assert abs(f["partial_writer"]) < 0.05 and abs(f["partial_text"]) < 0.05, f  # pure noise: adjusted shares near 0
    r = residuals(Y, M)
    assert np.allclose(r, 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
