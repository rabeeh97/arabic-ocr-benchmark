"""Unit checks for scripts/eval_stats.py (run: .geomvenv/bin/python -m pytest tests/test_eval_stats.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_stats import bh, boot_p, fixed_broken, wer_compare, wer_compare_seeds, word_edits  # noqa: E402


def test_bh_textbook_example():
    # Benjamini & Hochberg (1995) style: m = 10, q = 0.1 -> thresholds k * 0.01
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    rej, adj = bh(p, q=0.1)
    # largest k with p_(k) <= k q / m: k = 7 (0.074 <= 0.07? no), k = 6 (0.06 <= 0.06 yes)
    assert rej.tolist() == [True] * 6 + [False] * 4
    assert adj[0] == pytest.approx(0.01)
    assert adj[5] == pytest.approx(0.1)  # 0.06 * 10 / 6
    assert np.all(np.diff(adj) >= -1e-12)  # monotone in sorted order


def test_bh_step_up_rescues_earlier_failures():
    # p_(1) = 0.03 > 0.1 / 3 * 1 would fail alone, but p_(3) = 0.09 <= 0.1 lets all three through
    rej, adj = bh([0.09, 0.03, 0.06], q=0.1)
    assert rej.all()
    assert adj.tolist() == pytest.approx([0.09, 0.09, 0.09])


def test_bh_order_nan_and_matches_statsmodels_formula():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=50) ** 3
    p[[4, 9]] = np.nan
    rej, adj = bh(p, q=0.1)
    assert not rej[4] and not rej[9] and np.isnan(adj[4])
    ok = ~np.isnan(p)
    po = p[ok]
    m = len(po)
    srt = np.sort(po)
    k = max([i + 1 for i in range(m) if srt[i] <= (i + 1) * 0.1 / m], default=0)
    expected = po <= (srt[k - 1] if k else -1)
    assert rej[ok].tolist() == expected.tolist()
    assert bh([])[0].size == 0
    assert not bh([0.5, 0.9])[0].any()


def test_boot_p():
    rng = np.random.default_rng(0)
    boots = rng.normal(0.05, 0.01, 4000)
    assert boot_p(0.05, boots) < 1e-4
    assert boot_p(0.0, boots) == pytest.approx(1.0)
    assert boot_p(0.0196 * 1.0, rng.normal(0, 0.01, 200000)) == pytest.approx(0.05, abs=0.003)
    assert boot_p(0.0, np.zeros(10)) == 1.0


def test_word_edits_and_view():
    assert word_edits("a b c", "a b c") == (0, 3)
    assert word_edits("a b c", "a x c") == (1, 3)
    assert word_edits("a b c", "a bc") == (2, 3)
    assert word_edits("a b", "") == (2, 2)
    assert word_edits("A b", "a b", view=str.lower) == (0, 2)


def test_fixed_broken_single_and_seeds():
    ea = np.array([2, 0, 0, 3, 1])
    eb = np.array([0, 0, 4, 1, 0])
    fb = fixed_broken(ea, eb)
    assert (fb["fixed"], fb["broken"], fb["exact_A"], fb["exact_B"]) == (2, 1, 2, 3)
    EA = np.array([[2, 0, 1], [1, 0, 0]])  # seeds × lines
    EB = np.array([[0, 1, 0], [0, 2, 0]])
    fb = fixed_broken(EA, EB)
    assert fb["fixed"] == 1  # line 0: all A wrong, all B exact; line 2: one A seed exact -> not fixed
    assert fb["broken"] == 1  # line 1


def test_wer_compare():
    wa, wb, nw = np.array([1, 0, 2, 0]), np.array([0, 0, 2, 1]), np.array([5, 5, 5, 5])
    r = wer_compare(wa, wb, nw, reps=500)
    assert r["wer_A"] == pytest.approx(3 / 20) and r["wer_B"] == pytest.approx(3 / 20) and r["delta"] == 0
    assert (r["lines_fixed_words"], r["lines_broken_words"]) == (1, 1)
    assert r["ci95"][0] <= 0 <= r["ci95"][1]
    big = wer_compare(np.full(300, 3), np.full(300, 1), np.full(300, 10), reps=500)
    assert big["delta"] == pytest.approx(-0.2) and big["p"] < 1e-6  # constant lines: zero bootstrap spread


def test_wer_compare_seeds():
    WA = np.array([[2, 0, 1, 1], [2, 0, 1, 1]])
    WB = np.array([[0, 0, 1, 0], [0, 0, 1, 1]])
    r = wer_compare_seeds(WA, WB, np.array([4, 4, 4, 4]), reps=300)
    assert r["wer_A_by_seed"] == pytest.approx([4 / 16, 4 / 16])
    assert r["wer_B_by_seed"] == pytest.approx([1 / 16, 2 / 16])
    assert r["delta"] == pytest.approx((1.5 - 4) / 16)
    assert (r["lines_fixed_words"], r["lines_broken_words"]) == (1, 0)  # line 3 is exact on only one B seed
