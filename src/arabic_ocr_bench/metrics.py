"""CER / WER, diacritic precision-recall-F1, and corpus aggregation.

Pure functions over (reference, hypothesis) strings — no I/O. `report.py` calls
`score_pair` per sample and `aggregate` over the list.

Conventions for degenerate inputs (documented so results are reproducible):
- reference empty, hypothesis empty      -> rate 0.0
- reference empty, hypothesis non-empty  -> rate 1.0
- diacritic P/R/F1 when neither side has any harakat -> 1.0 / 1.0 / 1.0
"""

from __future__ import annotations

import re

import jiwer

from .normalize import PROFILES, normalize

# same class as normalize._HARAKAT_RE; duplicated here so metrics has no reliance
# on a private name. Exercised by the diacritic_prf tests.
_HARAKAT_RE = re.compile("[\u064B-\u065F\u0670]")


# --------------------------------------------------------------------------- #
# rates
# --------------------------------------------------------------------------- #
def _char_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    o = jiwer.process_characters(ref, hyp)
    return o.substitutions, o.deletions, o.insertions, o.hits


def cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    s, d, i, h = _char_counts(ref, hyp)
    denom = s + d + h  # == len(ref)
    return (s + d + i) / denom if denom else 0.0


def wer(ref: str, hyp: str) -> float:
    if len(ref.split()) == 0:
        return 0.0 if len(hyp.split()) == 0 else 1.0
    return jiwer.process_words(ref, hyp).wer


def char_edit_counts(ref: str, hyp: str) -> dict[str, int]:
    if len(ref) == 0:
        return {"sub": 0, "del": 0, "ins": len(hyp), "ref_len": 0}
    s, d, i, h = _char_counts(ref, hyp)
    return {"sub": s, "del": d, "ins": i, "ref_len": s + d + h}


def word_edit_counts(ref: str, hyp: str) -> dict[str, int]:
    if len(ref.split()) == 0:
        return {"sub": 0, "del": 0, "ins": len(hyp.split()), "ref_len": 0}
    o = jiwer.process_words(ref, hyp)
    return {
        "sub": o.substitutions,
        "del": o.deletions,
        "ins": o.insertions,
        "ref_len": o.substitutions + o.deletions + o.hits,
    }


# --------------------------------------------------------------------------- #
# diacritics
# --------------------------------------------------------------------------- #
def _harakat_only(s: str) -> str:
    return "".join(ch for ch in s if _HARAKAT_RE.match(ch))


def diacritic_prf(ref: str, hyp: str) -> dict[str, float]:
    """Precision / recall / F1 over the ordered sequence of harakat marks.
    Computed on RAW text (never normalized), so it still sees dropped or
    invented diacritics even when a diacritics-insensitive CER is 0."""
    r = _harakat_only(ref)
    h = _harakat_only(hyp)
    if not r and not h:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not r:  # hypothesis invented marks
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not h:  # hypothesis dropped every mark
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    o = jiwer.process_characters(r, h)
    tp = o.hits
    fp = o.insertions + o.substitutions
    fn = o.deletions + o.substitutions
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# per-sample and corpus scoring
# --------------------------------------------------------------------------- #
def score_pair(ref: str, hyp: str, profiles=PROFILES) -> dict:
    cer_by, wer_by, cc_by, wc_by, exact_by = {}, {}, {}, {}, {}
    for p in profiles:
        r, h = normalize(ref, p), normalize(hyp, p)
        cer_by[p] = cer(r, h)
        wer_by[p] = wer(r, h)
        cc_by[p] = char_edit_counts(r, h)
        wc_by[p] = word_edit_counts(r, h)
        exact_by[p] = r == h
    return {
        "cer": cer_by,
        "wer": wer_by,
        "char_counts": cc_by,
        "word_counts": wc_by,
        "exact_match": exact_by,
        "diacritic": diacritic_prf(ref, hyp),
        "profiles": list(profiles),
    }


def _sum_errs(counts: dict[str, int]) -> int:
    return counts["sub"] + counts["del"] + counts["ins"]


def aggregate(pair_scores: list[dict], profiles=PROFILES) -> dict:
    n = len(pair_scores)
    out: dict = {
        "n": n,
        "cer_macro": {},
        "wer_macro": {},
        "micro_cer": {},
        "micro_wer": {},
        "exact_match_rate": {},
    }
    for p in profiles:
        cers = [ps["cer"][p] for ps in pair_scores]
        wers = [ps["wer"][p] for ps in pair_scores]
        out["cer_macro"][p] = sum(cers) / n if n else 0.0
        out["wer_macro"][p] = sum(wers) / n if n else 0.0

        c_err = sum(_sum_errs(ps["char_counts"][p]) for ps in pair_scores)
        c_ref = sum(ps["char_counts"][p]["ref_len"] for ps in pair_scores)
        out["micro_cer"][p] = c_err / c_ref if c_ref else 0.0

        w_err = sum(_sum_errs(ps["word_counts"][p]) for ps in pair_scores)
        w_ref = sum(ps["word_counts"][p]["ref_len"] for ps in pair_scores)
        out["micro_wer"][p] = w_err / w_ref if w_ref else 0.0

        out["exact_match_rate"][p] = (
            sum(1 for ps in pair_scores if ps["exact_match"][p]) / n if n else 0.0
        )

    dp = [ps["diacritic"] for ps in pair_scores]
    out["diacritic"] = {
        k: (sum(d[k] for d in dp) / n if n else 0.0)
        for k in ("precision", "recall", "f1")
    }
    return out
