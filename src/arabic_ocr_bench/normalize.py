"""Arabic text normalization for benchmark scoring.

Three profiles, applied to both reference and hypothesis before CER/WER:

- "raw"             : no changes. Strictest; penalizes trivial orthographic variants.
- "light"           : NFC, strip tatweel, unify alef forms, unify digits, fold
                      whitespace. Keeps harakat, taa-marbuta, alef-maqsura.
- "diac_insensitive": light + unify yaa/alef-maqsura, taa-marbuta -> haa,
                      hamza carriers, and strip all harakat.

Individual steps are exported so tests and the diacritic metric can compose them.
All Arabic code points are written as \\u escapes so the source stays
copy-safe (no bidi surprises); Arabic in comments is for the reader.
"""

from __future__ import annotations

import re
import unicodedata

_TATWEEL = "ـ"  # kashida

_ALEF_MAP = str.maketrans({
    "أ": "ا",  # alef hamza above -> alef
    "إ": "ا",  # alef hamza below -> alef
    "آ": "ا",  # alef madda      -> alef
    "ٱ": "ا",  # alef wasla      -> alef
})
_YAA_MAP = str.maketrans({"ى": "ي"})          # alef maqsura -> yaa
_TAA_MARBUTA_MAP = str.maketrans({"ة": "ه"})  # taa marbuta -> haa
_HAMZA_CARRIER_MAP = str.maketrans({
    "ؤ": "و",  # waw hamza -> waw
    "ئ": "ي",  # yaa hamza -> yaa
})

# harakat / tanwin / shadda / sukun (U+064B..U+0652), assorted combining marks
# through U+065F, plus the superscript (dagger) alef U+0670.
_HARAKAT_RE = re.compile("[\u064B-\u065F\u0670]")

# Arabic-Indic (U+0660..U+0669) and Extended Arabic-Indic (U+06F0..U+06F9).
_DIGIT_MAP = {0x0660 + i: str(i) for i in range(10)}
_DIGIT_MAP.update({0x06F0 + i: str(i) for i in range(10)})

_WS_RE = re.compile(r"\s+")


def strip_tatweel(s: str) -> str:
    return s.replace(_TATWEEL, "")


def normalize_alef(s: str) -> str:
    return s.translate(_ALEF_MAP)


def normalize_yaa(s: str) -> str:
    return s.translate(_YAA_MAP)


def normalize_taa_marbuta(s: str) -> str:
    return s.translate(_TAA_MARBUTA_MAP)


def normalize_hamza(s: str) -> str:
    return s.translate(_HAMZA_CARRIER_MAP)


def strip_harakat(s: str) -> str:
    return _HARAKAT_RE.sub("", s)


def normalize_digits(s: str) -> str:
    return s.translate(_DIGIT_MAP)


def collapse_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


_LIGHT = (
    lambda s: unicodedata.normalize("NFC", s),
    strip_tatweel,
    normalize_alef,
    normalize_digits,
)
_DIAC_EXTRA = (
    normalize_yaa,
    normalize_taa_marbuta,
    normalize_hamza,
    strip_harakat,
)


def _apply(s: str, steps) -> str:
    for step in steps:
        s = step(s)
    return s


def normalize(s: str, profile: str) -> str:
    if profile == "raw":
        return s
    if profile == "light":
        return collapse_whitespace(_apply(s, _LIGHT))
    if profile == "diac_insensitive":
        return collapse_whitespace(_apply(s, (*_LIGHT, *_DIAC_EXTRA)))
    raise ValueError(f"unknown normalization profile {profile!r}")


PROFILES = ("raw", "light", "diac_insensitive")
