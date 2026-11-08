"""Automatic tags attached to every gold line at export time.

Tags describe the line as it was exported (original label + image). They are for slicing results ("diacritised vs
not", "short fragments", "Maghrebi"...). Difficulty is deliberately NOT here: it is computed after the labels are
final, from a model panel kept separate from the models being compared (see gold/README.md, step 6).
"""
from __future__ import annotations

import re
import unicodedata

from . import images

TAGS_VERSION = 1
_AR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_AR_LETTER = re.compile(r"[ؠ-يٮ-ۓۺ-ۿݐ-ݿࢠ-ࣿ]")
_LATIN = re.compile(r"[A-Za-z]")
_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}،؛؟«»…\-–—/\\]")


def length_band(chars: int) -> str:
    return "tiny_le10" if chars <= 10 else ("short_11_25" if chars <= 25 else ("medium_26_60" if chars <= 60 else "long_gt60"))


def text_tags(text: str, language: str) -> dict:
    s = unicodedata.normalize("NFC", text or "")
    letters_ar = len(_AR_LETTER.findall(s))
    latin = len(_LATIN.findall(s))
    alpha = letters_ar + latin
    digits_w = bool(re.search(r"[0-9]", s))
    digits_a = bool(re.search(r"[٠-٩]", s))
    digits_p = bool(re.search(r"[۰-۹]", s))
    chars = len(s.strip())
    t = {
        "chars": chars, "words": len(s.split()), "length_band": length_band(chars), "is_fragment": chars <= 20,
        "script_main": "arabic" if letters_ar >= latin and letters_ar else ("latin" if latin else "none"),
        "arabic_letter_share": round(letters_ar / alpha, 3) if alpha else 0.0,
        "has_latin": latin > 0, "has_digits": digits_w or digits_a or digits_p,
        "digit_kinds": [k for k, on in (("western", digits_w), ("arabic_indic", digits_a), ("persian", digits_p)) if on],
        "has_punctuation": bool(_PUNCT.search(s)),
        "has_tatweel": "ـ" in s,
    }
    if language == "ar" and letters_ar:
        from ..characterize.text import line_features
        f = line_features(s)
        n = max(f["letters"], 1)
        t.update({
            "has_diacritics": f["diac_marks"] > 0, "diac_level": f["diac_level"], "diac_marks": f["diac_marks"],
            "dot_ambiguous_share": round(f["ambiguous"] / n, 3),
            "medial_share": round(f["positions"].get("medial", 0) / n, 3),
            "lam_alef": f["lam_alef"],
        })
    else:
        t.update({"has_diacritics": None, "diac_level": None, "diac_marks": None,
                  "dot_ambiguous_share": None, "medial_share": None, "lam_alef": None})
    return t


def auto_tags(spec: dict, text: str, raw: bytes | None) -> dict:
    t = {
        "tags_version": TAGS_VERSION,
        "dataset": spec["key"], "tier": spec["tier"], "language": spec["language"], "medium": spec["medium"],
        "era": spec["era"], "script_style": spec["script_style"], "split": spec["split"],
    }
    t.update(text_tags(text, spec["language"]))
    if raw:
        t.update(images.image_tags(raw))
    return t
