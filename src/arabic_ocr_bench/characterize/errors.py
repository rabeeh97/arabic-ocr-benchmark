"""Arabic-specific error decomposition (characterisation P2, tasks R2.1–R2.4).

- :data:`LADDER` / :func:`ladder_edits` — the **normalisation ladder** (R2.1):
  the same (GT, prediction) pair scored at cumulative rungs
  raw → canonical → +tatweel/digits → +orthographic folds → −diacritics →
  rasm → −spaces. The edits that disappear between two rungs are the share of
  error owed to that rung; what is left at the last rung is letter identity.
- **rasm-CER and the dot tax** (R2.2) fall out of the ladder: rasm-CER is the
  ``rasm`` rung, the dot tax is CER(``-diacritics``) − rasm-CER, i.e. error
  that only the dots distinguish.
- :func:`diacritic_mismatch` (R2.3) — on aligned letters, marks the model
  *inserted* where the GT has none, GT marks it *dropped*, and *wrong* marks.
- :func:`taxonomy` (R2.4) — Levenshtein with backtrace at the canonical rung,
  each edit classified as whole-line / diacritic / ligature / space /
  orthographic / dot / shape-similar / other (first match wins, in that order).

Pure functions over strings; ``scripts/decompose_errors.py`` runs them over the
canonical predictions (``characterize/runs.py``).
"""

from __future__ import annotations

import re
from collections import Counter

from rapidfuzz.distance import Levenshtein

from arabic_ocr_bench.characterize.text import (
    analyze_letters,
    is_mark,
    joining_type,
    rasm_of,
)
from arabic_ocr_bench.normalize import (
    canonicalize_encoding,
    collapse_whitespace,
    normalize_alef,
    normalize_digits,
    normalize_hamza,
    normalize_lookalikes,
    normalize_taa_marbuta,
    normalize_yaa,
    strip_harakat,
    strip_tatweel,
)

# ---------------------------------------------------------------------------
# R2.1 the ladder


def to_rasm(s: str) -> str:
    """Replace every Arabic letter by its position-aware dotless skeleton
    (``text.rasm_of``); everything else is kept."""
    chars = list(s)
    for li in analyze_letters(s):
        if li.rasm:
            chars[li.index] = li.rasm
    return "".join(chars)


def _orthographic(s: str) -> str:
    for f in (normalize_alef, normalize_lookalikes, normalize_yaa,
              normalize_taa_marbuta, normalize_hamza):
        s = f(s)
    return s


# (rung name, step applied on top of the previous rung). Rung 0 is the string
# exactly as stored.
LADDER: tuple[tuple[str, object], ...] = (
    ("raw", lambda s: s),
    ("canonical", lambda s: collapse_whitespace(canonicalize_encoding(s))),
    ("tatweel_digits", lambda s: normalize_digits(strip_tatweel(s))),
    ("orthographic", _orthographic),
    ("no_diacritics", lambda s: collapse_whitespace(strip_harakat(s))),
    ("rasm", to_rasm),
    ("no_spaces", lambda s: re.sub(r"\s+", "", s)),
)
RUNGS = tuple(name for name, _ in LADDER)


def ladder_strings(s: str) -> list[str]:
    out = []
    for _, step in LADDER:
        s = step(s)
        out.append(s)
    return out


def ladder_edits(gt: str, pred: str) -> list[tuple[int, int]]:
    """[(edits, ref_len)] per rung."""
    return [(Levenshtein.distance(g, p), len(g))
            for g, p in zip(ladder_strings(gt), ladder_strings(pred))]


def summarize_ladder(per_line: list[list[tuple[int, int]]]) -> dict:
    """Corpus (micro) CER per rung, and the share of the raw-rung edits that
    each rung removes. ``residual`` is the share still there at the last rung."""
    k = len(RUNGS)
    edits = [sum(line[i][0] for line in per_line) for i in range(k)]
    refs = [sum(line[i][1] for line in per_line) for i in range(k)]
    cer = {RUNGS[i]: (edits[i] / refs[i] if refs[i] else 0.0) for i in range(k)}
    e0 = edits[0] or 1
    share = {RUNGS[i]: (edits[i - 1] - edits[i]) / e0 for i in range(1, k)}
    share["residual"] = edits[-1] / e0
    return {
        "cer": cer, "share": share, "edits": dict(zip(RUNGS, edits)),
        "rasm_cer": cer["rasm"],
        "dot_tax": cer["no_diacritics"] - cer["rasm"],
        "diacritic_tax": cer["orthographic"] - cer["no_diacritics"],
    }


# ---------------------------------------------------------------------------
# R2.3 diacritic mismatch


def _units(s: str) -> list[tuple[str, str]]:
    """(base character, its combining marks) units."""
    out: list[list[str]] = []
    for ch in s:
        if is_mark(ch) and out:
            out[-1][1] += ch
        elif not is_mark(ch):
            out.append([ch, ""])
    return [(b, m) for b, m in out]


def diacritic_mismatch(gt: str, pred: str) -> dict[str, int]:
    """Align the base letters (marks stripped) and compare marks on every
    aligned letter. Strings should already be at the ``orthographic`` rung.

    - ``inserted``: marks on a predicted letter whose GT letter carries none
      (a policy mismatch when the GT is undiacritised);
    - ``dropped``: GT marks on a letter the model left bare;
    - ``wrong``: both carry marks but they differ (counted once per letter);
    - ``gt_marks`` / ``pred_marks`` / ``aligned_letters``: denominators.
    """
    gu, pu = _units(gt), _units(pred)
    gb, pb = [b for b, _ in gu], [b for b, _ in pu]
    pairs = []
    for op in Levenshtein.opcodes(gb, pb):
        if op.tag in ("equal", "replace"):
            n = min(op.src_end - op.src_start, op.dest_end - op.dest_start)
            pairs += [(op.src_start + j, op.dest_start + j) for j in range(n)]
    out = Counter(inserted=0, dropped=0, wrong=0,
                  gt_marks=sum(len(m) for _, m in gu),
                  pred_marks=sum(len(m) for _, m in pu),
                  aligned_letters=len(pairs))
    for gi, pi in pairs:
        gm, pm = gu[gi][1], pu[pi][1]
        if pm and not gm:
            out["inserted"] += len(pm)
        elif gm and not pm:
            out["dropped"] += len(gm)
        elif gm and pm and sorted(gm) != sorted(pm):
            out["wrong"] += 1
    return dict(out)


# ---------------------------------------------------------------------------
# R2.4 aligned taxonomy

CATEGORIES = ("whole_line", "diacritic", "ligature", "space", "orthographic",
              "dot", "shape", "other")

# pairs the orthographic rungs fold together
_ORTHO_CLASSES = [set("اأإآٱ"), set("يىیئ"), set("هةہھە"), set("وؤ"), set("كک"),
                  set("ءأإؤئ")]
for _i in range(10):
    _ORTHO_CLASSES.append({str(_i), chr(0x0660 + _i), chr(0x06F0 + _i)})

# heuristic visually-similar letter groups whose members do NOT share a rasm
# skeleton (same-skeleton pairs are "dot"): د/ر, ر/و, ح/ع, final ف/ق, final
# ن/ي, ه/م, ك/ل, ط/ك
_SHAPE_GROUPS = [set("دذرز"), set("رزو"), set("حجخعغ"), set("فق"), set("نيىبتث"),
                 set("هةم"), set("كل"), set("طظك"), set("سشصض")]

WHOLE_LINE_CER = 0.8  # a line this bad is a failure of the line, not of letters


def _same_class(a: str, b: str, classes) -> bool:
    return any(a in c and b in c for c in classes)


def _lam_alef_spans(s: str) -> set[int]:
    """Indices of the lam and alef of every lam-alef ligature in ``s``."""
    letters = analyze_letters(s)
    idx: set[int] = set()
    for a, b in zip(letters, letters[1:]):
        if (a.letter == "ل" and a.position in ("initial", "medial")
                and b.letter in "اأإآٱ" and a.word == b.word):
            idx |= {a.index, b.index}
    return idx


def classify_ops(gt: str, pred: str) -> Counter:
    """Category counts for every edit aligning ``pred`` to ``gt``. Both strings
    should be at the ``canonical`` rung (marks, tatweel, alef forms intact)."""
    ops = Levenshtein.editops(gt, pred)
    out: Counter = Counter()
    if not ops:
        return out
    if pred.strip() and len(ops) / max(len(gt), 1) > WHOLE_LINE_CER:
        out["whole_line"] = len(ops)
        return out
    pos = {li.index: li for li in analyze_letters(gt)}
    lam_alef = _lam_alef_spans(gt)
    for op in ops:
        r = gt[op.src_pos] if op.tag != "insert" and op.src_pos < len(gt) else ""
        h = pred[op.dest_pos] if op.tag != "delete" and op.dest_pos < len(pred) else ""
        out[_classify(op.tag, op.src_pos, r, h, gt, pos, lam_alef)] += 1
    return out


def _classify(tag, i, r, h, gt, pos, lam_alef) -> str:
    if (r and is_mark(r)) or (h and is_mark(h)):
        return "diacritic"
    # an insertion sits before gt[i]: it is a ligature error when it inserts a
    # lam or alef next to a lam-alef (e.g. the pair read in visual order)
    if tag != "insert":
        if i in lam_alef and h != " ":
            return "ligature"
    elif h in "لاأإآٱ" and lam_alef & {i - 1, i}:
        return "ligature"
    if r == " " or h == " ":
        return "space"
    if r == "ـ" or h == "ـ":
        return "orthographic"
    if tag == "replace":
        if _same_class(r, h, _ORTHO_CLASSES):
            return "orthographic"
        li = pos.get(i)
        if li is not None and li.rasm and rasm_of(h, li.position) == li.rasm:
            return "dot"
        if _same_class(r, h, _SHAPE_GROUPS):
            return "shape"
    return "other"


def is_paw_gap_space(gt: str, pred: str) -> tuple[int, int]:
    """(spaces inserted inside a GT word, of which at a PAW boundary). A PAW
    boundary is after a letter that cannot join forward (joining type R/U)."""
    inside = paw = 0
    for op in Levenshtein.editops(gt, pred):
        if op.tag != "insert" or pred[op.dest_pos] != " ":
            continue
        i = op.src_pos
        if 0 < i < len(gt) and gt[i - 1] != " " and gt[i] != " ":
            inside += 1
            j = i - 1
            while j > 0 and is_mark(gt[j]):
                j -= 1
            if joining_type(gt[j]) in ("R", "U"):
                paw += 1
    return inside, paw


def paw_gap_base_rate(gt: str) -> tuple[int, int]:
    """(word-internal letter boundaries, of which PAW gaps) in ``gt``: the
    chance level for :func:`is_paw_gap_space`'s ``at_paw_gap`` share."""
    letters = analyze_letters(gt)
    inside = paw = 0
    for a, b in zip(letters, letters[1:]):
        if a.word == b.word:
            inside += 1
            paw += a.paw != b.paw
    return inside, paw


__all__ = [
    "CATEGORIES", "LADDER", "RUNGS", "classify_ops",
    "diacritic_mismatch", "is_paw_gap_space", "ladder_edits", "paw_gap_base_rate", "ladder_strings",
    "summarize_ladder", "to_rasm",
]
