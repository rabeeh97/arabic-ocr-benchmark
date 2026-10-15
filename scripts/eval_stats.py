"""Shared statistics for the evaluation contract (docs/research/designs/evaluation-standards.md, items 5 and 6).

    from eval_stats import bh, boot_p, word_edits, fixed_broken, wer_compare, wer_compare_seeds

fdr (item 5)        Benjamini–Hochberg step-up at q = 0.1 over one table's family of tests: which cells survive, with
                    adjusted p-values (q-values). p-values for bootstrap deltas come from the bootstrap standard error
                    (normal approximation, so they are not floored at 1 / reps).
wer-lines (item 6)  WER on the letters view (canonical, harakat and tatweel stripped, as analyze_letters B6), and lines
                    fixed (A wrong, B exact) / broken (A exact, B wrong). With seeds, a line is fixed only when every A seed
                    is wrong and every B seed exact, and broken the other way round (same rule as "better on all seeds").
Pure functions over arrays and strings; the text view is passed in so the module has no package dependency.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from rapidfuzz.distance import Levenshtein

Q_FDR = 0.1


def bh(pvals: Sequence[float], q: float = Q_FDR) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg: (reject mask, adjusted p-values), in the input order. NaN p-values never reject and do not count
    towards the family size."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    rej, adj = np.zeros(p.shape, bool), np.full(p.shape, np.nan)
    m = int(ok.sum())
    if m == 0:
        return rej, adj
    po = p[ok]
    order = np.argsort(po, kind="stable")
    ranked = po[order] * m / np.arange(1, m + 1)
    a = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])  # monotone step-up adjustment
    adj_ok = np.empty(m)
    adj_ok[order] = a
    adj[ok] = adj_ok
    rej[ok] = adj_ok <= q * (1 + 1e-9)  # p_(k) == k q / m exactly must survive despite float rounding
    return rej, adj


def boot_p(delta: float, boots: np.ndarray) -> float:
    """Two-sided p-value for H0: delta = 0 from the bootstrap standard error (normal approximation)."""
    se = float(np.std(boots, ddof=1)) if len(boots) > 1 else 0.0
    if se == 0.0:
        return 1.0 if delta == 0 else 0.0
    return math.erfc(abs(delta) / se / math.sqrt(2))


def word_edits(gt: str, hyp: str, view: Callable[[str], str] = lambda s: s) -> tuple[int, int]:
    """(word-level edit distance, GT word count) after `view`; WER = Σ edits / Σ words."""
    g, h = view(gt).split(), view(hyp).split()
    return Levenshtein.distance(g, h), len(g)


def fixed_broken(ea: np.ndarray, eb: np.ndarray) -> dict:
    """Lines fixed / broken from A to B given per-line edit counts. 1-D arrays: one reading per side. 2-D arrays (seeds ×
    lines): fixed when every A seed has edits and every B seed none; broken the other way round."""
    EA, EB = np.atleast_2d(np.asarray(ea, float)), np.atleast_2d(np.asarray(eb, float))
    fixed = (EA.min(0) > 0) & (EB.max(0) == 0)
    broken = (EA.max(0) == 0) & (EB.min(0) > 0)
    return {"fixed": int(fixed.sum()), "broken": int(broken.sum()), "exact_A": int((EA.max(0) == 0).sum()),
            "exact_B": int((EB.max(0) == 0).sum()), "fixed_mask": fixed, "broken_mask": broken}


def wer_compare(wa: np.ndarray, wb: np.ndarray, nw: np.ndarray, reps: int = 4000, seed: int = 0) -> dict:
    """Paired A vs B word error rate from per-line word edits (wa, wb) and GT word counts (nw), with a line bootstrap
    interval of ΔWER; also lines fixed / broken at the word level."""
    wa, wb, nw = (np.asarray(x, float) for x in (wa, wb, nw))
    rng = np.random.default_rng(seed)
    s = rng.integers(0, len(nw), (reps, len(nw)))
    boots = (wb[s] - wa[s]).sum(1) / np.maximum(1, nw[s].sum(1))
    d = float((wb.sum() - wa.sum()) / max(1, nw.sum()))
    fb = fixed_broken(wa, wb)
    return {"wer_A": float(wa.sum() / max(1, nw.sum())), "wer_B": float(wb.sum() / max(1, nw.sum())), "delta": d,
            "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "p": boot_p(d, boots),
            "words": int(nw.sum()), "lines_fixed_words": fb["fixed"], "lines_broken_words": fb["broken"]}


def wer_compare_seeds(WA: np.ndarray, WB: np.ndarray, nw: np.ndarray, reps: int = 4000, seed: int = 0) -> dict:
    """Seed-pooled WER (seeds × lines word edits per arm): per-seed WER, Δ of the seed-averaged WER with a hierarchical
    bootstrap (seeds within each arm, then lines), lines fixed / broken on all seeds at the word level."""
    WA, WB, nw = np.atleast_2d(np.asarray(WA, float)), np.atleast_2d(np.asarray(WB, float)), np.asarray(nw, float)
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    for i in range(reps):
        ia = rng.integers(0, len(WA), len(WA))
        ib = rng.integers(0, len(WB), len(WB))
        s = rng.integers(0, len(nw), len(nw))
        boots[i] = (WB[ib][:, s].mean(0).sum() - WA[ia][:, s].mean(0).sum()) / max(1, nw[s].sum())
    d = float((WB.mean(0).sum() - WA.mean(0).sum()) / max(1, nw.sum()))
    fb = fixed_broken(WA, WB)
    return {"wer_A_by_seed": [float(w.sum() / max(1, nw.sum())) for w in WA], "wer_B_by_seed": [float(w.sum() / max(1, nw.sum())) for w in WB],
            "delta": d, "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "p": boot_p(d, boots), "words": int(nw.sum()),
            "lines_fixed_words": fb["fixed"], "lines_broken_words": fb["broken"]}
