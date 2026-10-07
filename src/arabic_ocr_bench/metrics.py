"""Edit-rate and word-information metrics, diacritic precision-recall-F1, and
corpus aggregation.

Pure functions over (reference, hypothesis) strings — no I/O. `report.py` calls
`score_pair` per sample and `aggregate` over the list.

Metrics computed (each under every normalization profile — see `normalize.py`
and `docs/metrics.md`):

- **CER**  — character error rate, `(S + D + I) / N_ref_chars`. Lower is better;
  can exceed 1 when the hypothesis is much longer than the reference.
- **CAR**  — character accuracy rate, `max(0, 1 - CER)`. Higher is better.
- **WER**  — word error rate, `(S + D + I) / N_ref_words`. Lower is better; can
  exceed 1.
- **MER**  — match error rate, `(S + D + I) / (S + D + I + H)`. Like WER but the
  denominator also counts hits, so it is bounded in [0, 1].
- **WIL / WIP** — word information lost / preserved. `WIP = (H/N_ref)·(H/N_hyp)`
  rewards getting words right *and* not inventing extras; `WIL = 1 - WIP`.
  Lower WIL / higher WIP is better.

`micro` aggregates weight by reference length (sum the counts, then divide);
`macro` is the plain mean of the per-sample values.

Conventions for degenerate inputs (documented so results are reproducible):
- reference empty, hypothesis empty      -> error rate 0.0 (CAR/WIP 1.0)
- reference empty, hypothesis non-empty  -> error rate 1.0 (CAR/WIP 0.0)
- diacritic P/R/F1 when neither side has any harakat -> 1.0 / 1.0 / 1.0
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

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


def car(ref: str, hyp: str) -> float:
    """Character Accuracy Rate = ``max(0, 1 - CER)``. Higher is better."""
    return max(0.0, 1.0 - cer(ref, hyp))


def _word_metrics(ref: str, hyp: str) -> dict[str, float]:
    """WER / MER / WIL / WIP plus the underlying S/D/I/H word counts, from a
    single jiwer alignment. Applies the same empty-reference conventions as the
    character metrics."""
    if len(ref.split()) == 0:
        bad = len(hyp.split()) > 0
        return {
            "wer": 1.0 if bad else 0.0, "mer": 1.0 if bad else 0.0,
            "wil": 1.0 if bad else 0.0, "wip": 0.0 if bad else 1.0,
            "sub": 0, "del": 0, "ins": len(hyp.split()), "hits": 0,
        }
    o = jiwer.process_words(ref, hyp)
    return {
        "wer": o.wer, "mer": o.mer, "wil": o.wil, "wip": o.wip,
        "sub": o.substitutions, "del": o.deletions,
        "ins": o.insertions, "hits": o.hits,
    }


def wer(ref: str, hyp: str) -> float:
    return _word_metrics(ref, hyp)["wer"]


def mer(ref: str, hyp: str) -> float:
    """Match Error Rate — WER with hits in the denominator, so it stays in
    [0, 1]."""
    return _word_metrics(ref, hyp)["mer"]


def wil(ref: str, hyp: str) -> float:
    """Word Information Lost — ``1 - (H/N_ref)·(H/N_hyp)``. Lower is better."""
    return _word_metrics(ref, hyp)["wil"]


def wip(ref: str, hyp: str) -> float:
    """Word Information Preserved — ``(H/N_ref)·(H/N_hyp)``. Higher is better."""
    return _word_metrics(ref, hyp)["wip"]


def char_edit_counts(ref: str, hyp: str) -> dict[str, int]:
    if len(ref) == 0:
        return {"sub": 0, "del": 0, "ins": len(hyp), "ref_len": 0}
    s, d, i, h = _char_counts(ref, hyp)
    return {"sub": s, "del": d, "ins": i, "ref_len": s + d + h}


# --------------------------------------------------------------------------- #
# NAKBA-NLP 2026 contest CER / WER (no Arabic normalization)
# --------------------------------------------------------------------------- #
# Two recipes exist in the public materials:
#
# * U4RASD/ar-ms-baseline ``evaluate.py``: Unicode-codepoint Levenshtein, each
#   line clipped to 1.0, then the *mean* of those per-line scores (macro).
#   Empty hypothesis -> 1.0. This is what their baseline script prints as CER.
# * The shared-task paper: corpus-level CERc = sum(dchar) / sum(|R|), then
#   clip the *ratio* to [0, 1]. That is the official ranking metric.
#
# Neither applies our ``raw``/``light`` profiles. ``contest_aggregate`` reports
# both so a stored ``predictions.jsonl`` can be compared to the paper table
# without re-running the model.


def unicode_levenshtein(ref: Sequence[str] | str, hyp: Sequence[str] | str) -> int:
    """Codepoint (or token) Levenshtein; insert / delete / substitute = 1."""
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, ca in enumerate(ref, 1):
        cur = [i]
        for j, cb in enumerate(hyp, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def contest_cer_line(ref: str, hyp: str) -> float:
    """Per-line CER from ``U4RASD/ar-ms-baseline/evaluate.py:get_cer``."""
    if not ref:
        return 1.0 if hyp else 0.0
    if not hyp:
        return 1.0
    return min(unicode_levenshtein(ref, hyp) / len(ref), 1.0)


def contest_wer_line(ref: str, hyp: str) -> float:
    """Per-line WER from ``U4RASD/ar-ms-baseline/evaluate.py:get_wer``."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    if not hyp_words:
        return 1.0
    return min(unicode_levenshtein(ref_words, hyp_words) / len(ref_words), 1.0)


def contest_aggregate(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Paper corpus (micro, then clip) + baseline-script macro (mean of clipped)."""
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "cer_macro": 0.0,
            "wer_macro": 0.0,
            "cer_micro": 0.0,
            "wer_micro": 0.0,
        }
    cer_lines = [contest_cer_line(r, h) for r, h in pairs]
    wer_lines = [contest_wer_line(r, h) for r, h in pairs]
    char_edits = sum(unicode_levenshtein(r, h) for r, h in pairs)
    char_ref = sum(len(r) for r, h in pairs)
    word_edits = sum(unicode_levenshtein(r.split(), h.split()) for r, h in pairs)
    word_ref = sum(len(r.split()) for r, h in pairs)
    return {
        "n": float(n),
        "cer_macro": sum(cer_lines) / n,
        "wer_macro": sum(wer_lines) / n,
        "cer_micro": min(char_edits / char_ref, 1.0) if char_ref else 0.0,
        "wer_micro": min(word_edits / word_ref, 1.0) if word_ref else 0.0,
    }


def word_edit_counts(ref: str, hyp: str) -> dict[str, int]:
    m = _word_metrics(ref, hyp)
    return {
        "sub": m["sub"],
        "del": m["del"],
        "ins": m["ins"],
        "hits": m["hits"],
        "ref_len": m["sub"] + m["del"] + m["hits"],
    }


# --------------------------------------------------------------------------- #
# diacritics
# --------------------------------------------------------------------------- #
def _harakat_only(s: str) -> str:
    return "".join(ch for ch in s if _HARAKAT_RE.match(ch))


def diacritic_prf(ref: str, hyp: str) -> dict[str, float]:
    """Precision / recall / F1 over the ordered sequence of harakat marks.
    Computed on otherwise-unnormalized text — so it still sees dropped or
    invented diacritics even when a diacritics-insensitive CER is 0 — but each
    side is NFC-normalized first, so a model that emits shadda+fatha instead of
    the canonical fatha+shadda is not scored as two diacritic errors."""
    ref = unicodedata.normalize("NFC", ref)
    hyp = unicodedata.normalize("NFC", hyp)
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
    cer_by, car_by, wer_by, mer_by, wil_by, wip_by = {}, {}, {}, {}, {}, {}
    cc_by, wc_by, exact_by = {}, {}, {}
    for p in profiles:
        r, h = normalize(ref, p), normalize(hyp, p)
        c = cer(r, h)
        cer_by[p] = c
        car_by[p] = max(0.0, 1.0 - c)
        wm = _word_metrics(r, h)
        wer_by[p], mer_by[p] = wm["wer"], wm["mer"]
        wil_by[p], wip_by[p] = wm["wil"], wm["wip"]
        cc_by[p] = char_edit_counts(r, h)
        wc_by[p] = {
            "sub": wm["sub"], "del": wm["del"], "ins": wm["ins"],
            "hits": wm["hits"], "ref_len": wm["sub"] + wm["del"] + wm["hits"],
        }
        exact_by[p] = r == h
    return {
        "cer": cer_by,
        "car": car_by,
        "wer": wer_by,
        "mer": mer_by,
        "wil": wil_by,
        "wip": wip_by,
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
        "cer_macro": {}, "car_macro": {}, "wer_macro": {},
        "mer_macro": {}, "wil_macro": {}, "wip_macro": {},
        "micro_cer": {}, "micro_car": {}, "micro_wer": {},
        "micro_mer": {}, "micro_wil": {}, "micro_wip": {},
        "exact_match_rate": {},
    }

    def _mean(key: str, p: str) -> float:
        return sum(ps[key][p] for ps in pair_scores) / n if n else 0.0

    for p in profiles:
        for m in ("cer", "car", "wer", "mer", "wil", "wip"):
            out[f"{m}_macro"][p] = _mean(m, p)

        c_err = sum(_sum_errs(ps["char_counts"][p]) for ps in pair_scores)
        c_ref = sum(ps["char_counts"][p]["ref_len"] for ps in pair_scores)
        micro_cer = c_err / c_ref if c_ref else 0.0
        out["micro_cer"][p] = micro_cer
        out["micro_car"][p] = max(0.0, 1.0 - micro_cer)

        wc = [ps["word_counts"][p] for ps in pair_scores]
        s = sum(x["sub"] for x in wc)
        d = sum(x["del"] for x in wc)
        i = sum(x["ins"] for x in wc)
        h = sum(x["hits"] for x in wc)
        n_ref = s + d + h          # reference word count
        n_hyp = s + i + h          # hypothesis word count
        edits = s + d + i
        out["micro_wer"][p] = edits / n_ref if n_ref else 0.0
        out["micro_mer"][p] = edits / (edits + h) if (edits + h) else 0.0
        micro_wip = (h / n_ref) * (h / n_hyp) if (n_ref and n_hyp) else 0.0
        out["micro_wip"][p] = micro_wip
        out["micro_wil"][p] = 1.0 - micro_wip

        out["exact_match_rate"][p] = (
            sum(1 for ps in pair_scores if ps["exact_match"][p]) / n if n else 0.0
        )

    dp = [ps["diacritic"] for ps in pair_scores]
    out["diacritic"] = {
        k: (sum(d[k] for d in dp) / n if n else 0.0)
        for k in ("precision", "recall", "f1")
    }
    return out
