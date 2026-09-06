"""Arabic text normalization for benchmark scoring.

Three profiles, applied to both reference and hypothesis before CER/WER:

- "raw"             : encoding-level canonicalisation ONLY -- strip bidi /
                      zero-width controls, fold Arabic Presentation Forms
                      (isolated / initial / medial / final glyph codepoints and
                      ligatures) to their nominal letters, then NFC. No
                      orthographic change: alef forms, digits, tatweel, harakat
                      and spacing are all still scored. Strictest.
- "light"           : raw, plus strip tatweel, unify alef forms, fold Persian /
                      Urdu look-alike letters (keheh -> kaf, farsi-yaa -> yaa,
                      heh-goal / doachashmee / ae -> haa), map Arabic-Indic
                      digits to ASCII, collapse whitespace. Keeps harakat,
                      taa-marbuta, alef-maqsura. **The headline profile.**
- "diac_insensitive": light + unify yaa / alef-maqsura, taa-marbuta -> haa,
                      hamza carriers, and strip all harakat.

The "raw" canonicalisation is deliberately lossless: NFC only reorders combining
marks into canonical order (a model that emits shadda+fatha instead of
fatha+shadda is not making an error), and presentation forms / bidi controls are
rendering artifacts that also occur in some reference transcriptions, so scoring
them as differences is noise, not signal.

Individual steps are exported so tests and the diacritic metric can compose them.
All Arabic / control code points are written as \\u escapes so the source stays
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

# Persian / Urdu keyboard look-alikes: the SAME Arabic letter at a different code
# point. Folding them makes scoring keyboard-agnostic (CAMeL Tools does the same
# in normalize_unicode). Alef-maqsura is intentionally NOT here -- it is a real
# Arabic orthographic distinction and lives in the diac_insensitive fold.
_LOOKALIKE_MAP = str.maketrans({
    "ک": "ك",  # keheh           -> kaf
    "ی": "ي",  # farsi yaa       -> yaa
    "ہ": "ه",  # heh goal        -> haa
    "ھ": "ه",  # heh doachashmee -> haa
    "ە": "ه",  # ae              -> haa
})

_HAMZA_CARRIER_MAP = str.maketrans({
    "ؤ": "و",  # waw hamza -> waw
    "ئ": "ي",  # yaa hamza -> yaa
})

# bidi controls (LRM/RLM/ALM, the LRE..RLO embeddings/overrides, PDF, and the
# LRI..PDI isolates) plus the zero-width space / joiner / non-joiner and the BOM.
# Invisible formatting, never text content.
_BIDI_ZW = "".join(chr(c) for c in (
    *range(0x200B, 0x2010), *range(0x202A, 0x202F), *range(0x2066, 0x206A),
    0x061C, 0xFEFF,
))
_BIDI_ZW_RE = re.compile("[" + re.escape(_BIDI_ZW) + "]")

# Arabic Presentation Forms-A (U+FB50..U+FDFF: ligatures + Farsi letter forms)
# and -B (U+FE70..U+FEFF: isolated / initial / medial / final glyph codepoints).
_PRESENTATION_RE = re.compile(
    "[" + chr(0xFB50) + "-" + chr(0xFDFF) + chr(0xFE70) + "-" + chr(0xFEFF) + "]"
)

# harakat / tanwin / shadda / sukun (U+064B..U+0652), assorted combining marks
# through U+065F, plus the superscript (dagger) alef U+0670.
_HARAKAT_RE = re.compile(
    "[" + chr(0x064B) + "-" + chr(0x065F) + chr(0x0670) + "]"
)

# Arabic-Indic (U+0660..U+0669) and Extended Arabic-Indic (U+06F0..U+06F9).
_DIGIT_MAP = {0x0660 + i: str(i) for i in range(10)}
_DIGIT_MAP.update({0x06F0 + i: str(i) for i in range(10)})

_WS_RE = re.compile(r"\s+")


def strip_bidi_zw(s: str) -> str:
    """Drop bidi-control and zero-width characters (LRM/RLM, embeddings,
    isolates, ZWSP/ZWJ/ZWNJ, BOM). Invisible; never part of the transcription."""
    return _BIDI_ZW_RE.sub("", s)


def fold_presentation_forms(s: str) -> str:
    """Map Arabic Presentation Forms (contextual glyph codepoints and ligatures,
    e.g. an "initial jeem" codepoint, or the lam-alef ligature) to their nominal
    letters via a per-character compatibility decomposition. A PDF-extraction /
    rendering artifact -- and it turns up in some reference transcriptions too,
    so it must be folded on both sides."""
    if not _PRESENTATION_RE.search(s):
        return s
    return "".join(
        unicodedata.normalize("NFKD", ch) if _PRESENTATION_RE.match(ch) else ch
        for ch in s
    )


def canonicalize_encoding(s: str) -> str:
    """The lossless, non-orthographic clean-up applied at every profile including
    "raw": strip invisible controls, de-shape presentation forms, then NFC so
    combining marks sit in canonical order."""
    return unicodedata.normalize("NFC", fold_presentation_forms(strip_bidi_zw(s)))


def strip_tatweel(s: str) -> str:
    return s.replace(_TATWEEL, "")


def normalize_alef(s: str) -> str:
    return s.translate(_ALEF_MAP)


def normalize_lookalikes(s: str) -> str:
    return s.translate(_LOOKALIKE_MAP)


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


# "raw" is exactly this and nothing else.
_RAW = (canonicalize_encoding,)
# "light" adds orthographic folds that do not change meaning.
_LIGHT = (
    canonicalize_encoding,
    strip_tatweel,
    normalize_alef,
    normalize_lookalikes,
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
        return _apply(s, _RAW)
    if profile == "light":
        return collapse_whitespace(_apply(s, _LIGHT))
    if profile == "diac_insensitive":
        return collapse_whitespace(_apply(s, (*_LIGHT, *_DIAC_EXTRA)))
    raise ValueError(f"unknown normalization profile {profile!r}")


PROFILES = ("raw", "light", "diac_insensitive")
