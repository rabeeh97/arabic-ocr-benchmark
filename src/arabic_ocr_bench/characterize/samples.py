"""Per-sample characteristics measured against many models at once
(characterisation P3 / R3, R4.1; first pass of the R7 validity question).

Everything here works on a **complete block**: a set of models that all have a
successful canonical reading for the same set of samples of one dataset, so
every comparison is over identical lines.

- :func:`complete_block` — greedy largest (models × samples) block.
- :func:`line_cer` — per-line CER at a ladder rung, capped (a hallucinated
  page cannot outweigh a line).
- :func:`gt_features` — script and language features of one GT line.
- :func:`letter_outcomes` — for every GT letter, whether the model got it
  (aligned *equal*), with its positional form, rasm class, ambiguity and its
  relative position in the line.
- :func:`spearman` / :func:`partial_spearman` — rank statistics, the partial
  one removing a control variable (line length) from both sides.

Pure functions; ``scripts/sample_characteristics.py`` runs them.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rapidfuzz.distance import Levenshtein

from arabic_ocr_bench.characterize.errors import RUNGS, ladder_strings
from arabic_ocr_bench.characterize.text import analyze_letters, line_features
from arabic_ocr_bench.normalize import (
    canonicalize_encoding,
    collapse_whitespace,
    normalize,
    strip_harakat,
    strip_tatweel,
)

CER_CAP = 1.0


# ---------------------------------------------------------------------------
# blocks


def complete_block(readings_by_model: dict[str, dict], min_models: int = 5,
                   min_samples: int = 20) -> tuple[list[str], list[str]]:
    """Greedy (models, sample ids) maximising models × samples, with at least
    ``min_models`` models. ``readings_by_model``: model -> {sample_id: Reading}."""
    ok = {m: {s for s, r in rd.items() if not r.error} for m, rd in readings_by_model.items()}
    order = sorted(ok, key=lambda m: -len(ok[m]))
    best: tuple[int, list[str], set[str]] = (0, [], set())
    chosen: list[str] = []
    inter: set[str] | None = None
    for m in order:
        nxt = ok[m] if inter is None else inter & ok[m]
        if len(nxt) < min_samples:
            continue
        chosen.append(m)
        inter = nxt
        if len(chosen) >= min_models and len(chosen) * len(inter) > best[0]:
            best = (len(chosen) * len(inter), list(chosen), set(inter))
    return best[1], sorted(best[2])


# ---------------------------------------------------------------------------
# outcomes


def line_cer(gt: str, pred: str, rung: str = "no_diacritics", cap: float = CER_CAP) -> float:
    i = RUNGS.index(rung)
    g, p = ladder_strings(gt)[i], ladder_strings(pred)[i]
    if not g:
        return 0.0 if not p else cap
    return min(Levenshtein.distance(g, p) / len(g), cap)


def _letters_view(s: str) -> str:
    """Canonical, harakat and tatweel stripped: letter identity incl. hamza and
    taa-marbuta, without the diacritic and kashida policy."""
    return collapse_whitespace(strip_tatweel(strip_harakat(canonicalize_encoding(s))))


def letter_outcomes(gt: str, pred: str) -> list[tuple]:
    """[(letter, position, rasm, ambiguous, rel_pos, correct)] for every GT
    letter. ``rel_pos`` is the letter's index / line length in [0, 1)."""
    g, p = _letters_view(gt), _letters_view(pred)
    good = [False] * len(g)
    for op in Levenshtein.opcodes(g, p):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                good[i] = True
    n = max(len(g), 1)
    return [(li.letter, li.position, li.rasm, li.ambiguous, li.index / n, good[li.index])
            for li in analyze_letters(g)]


# ---------------------------------------------------------------------------
# GT features

_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")
_LATIN = re.compile(r"[A-Za-z]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def words_of(gt: str) -> list[str]:
    return normalize(gt, "diac_insensitive").split()


def gt_features(gt: str, word_freq: Counter | None = None) -> dict[str, float]:
    f = line_features(gt)
    letters = max(f["letters"], 1)
    words = words_of(gt)
    nw = max(len(words), 1)
    s = _letters_view(gt)
    out = {
        "chars": f["chars"],
        "log_chars": math.log(max(f["chars"], 1)),
        "words": len(words),
        "word_len": sum(len(w) for w in words) / nw,
        "diac_coverage": f["diac_coverage"],
        "dotted_share": f["dotted"] / letters,
        "ambiguous_share": f["ambiguous"] / letters,
        "hamza_share": f["hamza"] / letters,
        "paws_per_word": (sum(f["paws_per_word"]) / len(f["paws_per_word"])) if f["paws_per_word"] else 0.0,
        "lam_alef_per_word": f["lam_alef"] / nw,
        "isolated_final_share": (f["positions"].get("isolated", 0) + f["positions"].get("final", 0)) / letters,
        "digit_share": len(_DIGIT.findall(s)) / max(len(s), 1),
        "latin_share": len(_LATIN.findall(s)) / max(len(s), 1),
        "punct_share": len(_PUNCT.findall(s)) / max(len(s), 1),
        "non_arabic_share": 1 - f["letters"] / max(len(s.replace(" ", "")), 1),
    }
    if word_freq is not None:
        total = sum(word_freq.values()) or 1
        lf = [math.log10((word_freq.get(w, 0) + 1) / total) for w in words]
        out["oov_share"] = sum(1 for w in words if w not in word_freq) / nw
        out["mean_log_word_freq"] = sum(lf) / nw if lf else 0.0
    return out


# ---------------------------------------------------------------------------
# rank statistics


def _ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if not sa or not sb:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_ranks(a), _ranks(b))


def _residual(y: list[float], x: list[float]) -> list[float]:
    n = len(y)
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    beta = sum((u - mx) * (v - my) for u, v in zip(x, y)) / vx if vx else 0.0
    return [v - my - beta * (u - mx) for u, v in zip(x, y)]


def partial_spearman(a: list[float], b: list[float], control: list[float]) -> float:
    """Spearman of ``a`` and ``b`` with the rank of ``control`` regressed out of both."""
    ra, rb, rc = _ranks(a), _ranks(b), _ranks(control)
    return _pearson(_residual(ra, rc), _residual(rb, rc))


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if n else float("nan")


__all__ = [
    "CER_CAP", "complete_block", "gt_features", "letter_outcomes", "line_cer",
    "median", "partial_spearman", "spearman", "words_of",
]
