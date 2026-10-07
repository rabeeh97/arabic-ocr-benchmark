"""Script-level text features of an Arabic transcription (characterisation layer L1).

The central piece is :func:`analyze_letters`: it walks a line in logical order
and assigns every Arabic letter

- its **positional form** ("locus"): isolated / initial / medial / final, decided
  the way a renderer decides it -- from the Unicode *joining type* of the letter
  and of its nearest non-mark neighbours (``ArabicShaping.txt``: D = joins on
  both sides, R = joins only to the preceding letter, U = never joins,
  C = join-causing tatweel, T = transparent combining marks, skipped);
- its **rasm class**: the dotless skeleton it shares with other letters at that
  position (19 classes, after Al-Shaibani & Ahmad 2025). ن/ي/ق only share a
  skeleton with ب/ف in initial/medial position; their final forms have their
  own dotless shapes (ں / ى / ٯ), so the map is position-aware;
- the **PAW** (piece of Arabic word) it belongs to: a new PAW starts after any
  letter that does not join the next one.

These are the *intended* forms implied by the transcription, not what the
writer actually drew -- handwriting sometimes breaks or adds joins.

Arabic letters below are written as literals for readability; they are all
plain base-block code points (U+0621..U+064A, U+0671..U+06D3).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from arabic_ocr_bench.normalize import fold_presentation_forms, strip_bidi_zw

# ---------------------------------------------------------------------------
# joining types (subset of Unicode ArabicShaping.txt covering what we see)

_R = set("اأإآٱدذرزوؤة") | set("ۀۃۄۅۆۇۈۉۊۋۍۏڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙۯ")
_U = set("ء")
_D = set("بتثجحخسشصضطظعغفقكلمنهيىئ") | set("ٮٯںڡپچژکگیہھڤڨ")
_C = {"ـ"}  # tatweel / kashida: join-causing, not a letter

LETTERS = _R | _U | _D
# the 28 base letters in alphabetical (hijā'ī) order, for display
ALPHABET = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

POSITIONS = ("isolated", "initial", "medial", "final")

# harakat / tanwin / shadda / sukun / small marks / dagger alef / Quranic marks
_MARK_RE = re.compile("[ً-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]")
HARAKAT_NAMES = {
    "ً": "fathatan", "ٌ": "dammatan", "ٍ": "kasratan",
    "َ": "fatha", "ُ": "damma", "ِ": "kasra",
    "ّ": "shadda", "ْ": "sukun", "ٰ": "dagger alef",
}


def is_mark(ch: str) -> bool:
    return bool(_MARK_RE.match(ch))


def joining_type(ch: str) -> str:
    if ch in _D:
        return "D"
    if ch in _R:
        return "R"
    if ch in _U:
        return "U"
    if ch in _C:
        return "C"
    if is_mark(ch):
        return "T"
    return "X"  # non-joining non-Arabic: space, digit, Latin, punctuation


# ---------------------------------------------------------------------------
# rasm (dotless skeleton) classes, position-aware

_RASM_ANY = {
    **dict.fromkeys("اأإآٱ", "ا"),
    **dict.fromkeys("بتث", "ب"),
    **dict.fromkeys("جحخ", "ح"),
    **dict.fromkeys("دذ", "د"),
    **dict.fromkeys("رز", "ر"),
    **dict.fromkeys("سش", "س"),
    **dict.fromkeys("صض", "ص"),
    **dict.fromkeys("طظ", "ط"),
    **dict.fromkeys("عغ", "ع"),
    "ف": "ف", "ك": "ك", "ل": "ل", "م": "م",
    **dict.fromkeys("هة", "ه"),
    **dict.fromkeys("وؤ", "و"),
    "ى": "ى", "ء": "ء",
}
# letters whose skeleton depends on position
_RASM_JOINED = {"ن": "ب", "ي": "ب", "ئ": "ب", "ق": "ف"}      # initial / medial
_RASM_FREE = {"ن": "ں", "ي": "ى", "ئ": "ى", "ق": "ٯ"}        # final / isolated

RASM_CLASSES = list("ابحدرسصطعفٯكلمںهوىء")  # 19


def rasm_of(letter: str, position: str) -> str | None:
    if letter in _RASM_JOINED:
        return (_RASM_JOINED if position in ("initial", "medial") else _RASM_FREE)[letter]
    return _RASM_ANY.get(letter)


def _rasm_members() -> dict[tuple[str, str], set[str]]:
    members: dict[tuple[str, str], set[str]] = {}
    for letter in set(_RASM_ANY) | set(_RASM_JOINED):
        for pos in POSITIONS:
            r = rasm_of(letter, pos)
            if r is None:
                continue
            if letter in _R | _U and pos in ("initial", "medial"):
                continue  # these letters never take a joined-forward form
            members.setdefault((r, pos), set()).add(letter)
    return members


RASM_MEMBERS = _rasm_members()

# letters carrying i'jam (dots) as part of their identity
DOTTED = set("بتثجخذزشضظغفقنية")
HAMZA_FORMS = set("أإآؤئء")


def rasm_ambiguous(letter: str, position: str) -> bool:
    """True if another letter shares this letter's skeleton at this position and
    only **dots** tell them apart (e.g. initial ب/ت/ث/ن/ي). Hamza variants
    (ا/أ/إ/آ, و/ؤ) are not counted here -- see HAMZA_FORMS."""
    if letter in HAMZA_FORMS or letter == "ٱ":
        return False
    r = rasm_of(letter, position)
    others = RASM_MEMBERS.get((r, position), set()) - HAMZA_FORMS - {"ٱ"}
    return r is not None and len(others) > 1


# ---------------------------------------------------------------------------
# per-letter analysis

@dataclass(frozen=True)
class LetterInfo:
    index: int          # index into the analysed string
    letter: str
    position: str       # isolated / initial / medial / final
    rasm: str | None
    ambiguous: bool     # shares its skeleton with another letter here
    word: int           # word number in the line
    paw: int            # PAW number within the line
    marks: str          # combining marks attached to this letter


def prepare(text: str) -> str:
    """Encoding clean-up before analysis: drop invisible controls, de-shape
    presentation forms, NFC. Same as the 'raw' scoring profile."""
    return unicodedata.normalize("NFC", fold_presentation_forms(strip_bidi_zw(text)))


def analyze_letters(text: str) -> list[LetterInfo]:
    s = prepare(text)
    # indices of non-transparent characters, so neighbours skip over marks
    solid = [i for i, ch in enumerate(s) if not is_mark(ch)]
    jt = {i: joining_type(s[i]) for i in solid}
    out: list[LetterInfo] = []
    word = -1
    paw = -1
    in_word = False
    for k, i in enumerate(solid):
        ch = s[i]
        t = jt[i]
        if ch not in LETTERS:
            if t != "C":
                in_word = False
            continue
        p = jt[solid[k - 1]] if k > 0 else "X"
        n = jt[solid[k + 1]] if k + 1 < len(solid) else "X"
        joins_prev = p in ("D", "C") and t in ("D", "R")
        joins_next = t == "D" and n in ("D", "R", "C")
        if not in_word:
            word += 1
            in_word = True
        if not joins_prev:
            paw += 1
        position = (
            "medial" if joins_prev and joins_next
            else "initial" if joins_next
            else "final" if joins_prev
            else "isolated"
        )
        j = i + 1
        while j < len(s) and is_mark(s[j]):
            j += 1
        marks = s[i + 1:j]
        out.append(LetterInfo(
            index=i, letter=ch, position=position, rasm=rasm_of(ch, position),
            ambiguous=rasm_ambiguous(ch, position), word=word, paw=paw, marks=marks,
        ))
    return out


def lam_alef_count(letters: list[LetterInfo]) -> int:
    """Mandatory lam-alef ligatures: a lam that joins forward into an alef."""
    n = 0
    for a, b in zip(letters, letters[1:]):
        if a.letter == "ل" and a.position in ("initial", "medial") and b.letter in "اأإآٱ" and a.word == b.word:
            n += 1
    return n


# ---------------------------------------------------------------------------
# encoding hygiene & non-Arabic content (counted on the RAW string)

_PRESENTATION_RE = re.compile("[ﭐ-﷿ﹰ-﻿]")
_BIDI_ZW_RE = re.compile("[​-‏‪-‮⁦-⁩؜﻿]")
_LOOKALIKE = set("کیہھە")
_AR_INDIC = re.compile("[٠-٩]")
_EXT_INDIC = re.compile("[۰-۹]")
_WESTERN = re.compile("[0-9]")
_LATIN = re.compile("[A-Za-zÀ-ɏ]")


def hygiene(raw: str) -> dict[str, int]:
    return {
        "presentation_forms": len(_PRESENTATION_RE.findall(raw)),
        "bidi_zero_width": len(_BIDI_ZW_RE.findall(raw)),
        "tatweel": raw.count("ـ"),
        "persian_lookalikes": sum(ch in _LOOKALIKE for ch in raw),
        "not_nfc": int(unicodedata.normalize("NFC", raw) != raw),
        "digits_arabic_indic": len(_AR_INDIC.findall(raw)) + len(_EXT_INDIC.findall(raw)),
        "digits_western": len(_WESTERN.findall(raw)),
        "latin_letters": len(_LATIN.findall(raw)),
    }


# ---------------------------------------------------------------------------
# diacritics

def diacritics(text: str, letters: list[LetterInfo]) -> dict:
    s = prepare(text)
    marks = [ch for ch in s if is_mark(ch)]
    n_letters = len(letters)
    with_marks = sum(1 for li in letters if li.marks)
    # invalid placement: a mark whose host (nearest preceding non-mark) is not
    # an Arabic letter, or the same mark repeated on one host
    invalid = 0
    for i, ch in enumerate(s):
        if not is_mark(ch):
            continue
        j = i - 1
        while j >= 0 and is_mark(s[j]):
            if s[j] == ch:
                invalid += 1
                break
            j -= 1
        else:
            if j < 0 or s[j] not in LETTERS:
                invalid += 1
    return {
        "marks": len(marks),
        "letters_with_marks": with_marks,
        "coverage": with_marks / n_letters if n_letters else 0.0,
        "types": Counter(HARAKAT_NAMES.get(m, "other") for m in marks),
        "invalid": invalid,
    }


def diacritic_level(coverage: float) -> str:
    if coverage == 0:
        return "none"
    if coverage < 0.15:
        return "sparse"
    if coverage < 0.6:
        return "partial"
    return "full"


# ---------------------------------------------------------------------------
# per-line feature row

_WORD_SPLIT = re.compile(r"\s+")


def line_features(raw: str) -> dict:
    letters = analyze_letters(raw)
    s = prepare(raw)
    words = [w for w in _WORD_SPLIT.split(s.strip()) if w]
    arabic_words = {li.word for li in letters}
    paw_ids_per_word: dict[int, set[int]] = {}
    for li in letters:
        paw_ids_per_word.setdefault(li.word, set()).add(li.paw)
    diac = diacritics(raw, letters)
    pos = Counter(li.position for li in letters)
    return {
        "chars": len(s),
        "words": len(words),
        "arabic_words": len(arabic_words),
        "letters": len(letters),
        "positions": dict(pos),
        "dotted": sum(li.letter in DOTTED for li in letters),
        "ambiguous": sum(li.ambiguous for li in letters),
        "hamza": sum(li.letter in HAMZA_FORMS for li in letters),
        "paws": len({li.paw for li in letters}),
        "paws_per_word": [len(v) for v in paw_ids_per_word.values()],
        "lam_alef": lam_alef_count(letters),
        "diac_marks": diac["marks"],
        "diac_coverage": diac["coverage"],
        "diac_level": diacritic_level(diac["coverage"]),
        "diac_types": dict(diac["types"]),
        "diac_invalid": diac["invalid"],
        "hygiene": hygiene(raw),
    }


# ---------------------------------------------------------------------------
# corpus-level statistics

def entropy(counts: Counter | dict) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counts.values() if c)


def shape_counts(lines: Iterable[str]) -> tuple[Counter, Counter, Counter]:
    """(letter, position) counts, letter counts and (rasm, position) counts."""
    shapes: Counter = Counter()
    letters: Counter = Counter()
    rasm: Counter = Counter()
    for line in lines:
        for li in analyze_letters(line):
            shapes[(li.letter, li.position)] += 1
            letters[li.letter] += 1
            if li.rasm:
                rasm[(li.rasm, li.position)] += 1
    return shapes, letters, rasm


def gini(values: list[int]) -> float:
    v = sorted(values)
    n = len(v)
    if n == 0 or sum(v) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2 * cum) / (n * sum(v)) - (n + 1) / n
