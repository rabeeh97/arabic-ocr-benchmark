# src/arabic_ocr_bench/analysis/align.py
from __future__ import annotations

import re

ARABIC_LETTER_RE = re.compile(r"[ء-ي]")

# dot-bearing letter -> undotted skeleton (rasm). Skeleton code points are the
# real "dotless" Arabic letters where Unicode has them (ٮ ٯ ٰ …), else a
# representative member of the family used purely as a bucket key.
RASM_MAP = {
    "ب": "ٮ", "ت": "ٮ", "ث": "ٮ", "ن": "ٮ", "ي": "ٮ", "ى": "ٮ",
    "ج": "ح", "خ": "ح",
    "ذ": "د",
    "ز": "ر",
    "ش": "س",
    "ض": "ص",
    "ظ": "ط",
    "غ": "ع",
    "ف": "ٯ", "ق": "ٯ",
    "ة": "ه",
    "ئ": "ٮ", "ؤ": "و", "أ": "ا", "إ": "ا", "آ": "ا",
}

LETTER_GROUPS = {
    "dot_confusable": tuple("بتثنيىقفجحخزرشسضصطظعغ"),
    "hamza_carrier": tuple("ءأإآؤئ"),
    "alef_form": tuple("اأإآٱ"),
    "taa_haa": tuple("ةه"),
    "yaa_alefmaqsura": tuple("يى"),
    "sun_letter": tuple("تثدذرزسشصضطظلن"),
    "moon_letter": tuple("ابجحخعغفقكمهوي"),
}


def to_rasm(s: str) -> str:
    return "".join(RASM_MAP.get(c, c) for c in s)


def letter_group(ch: str) -> str | None:
    for name, members in LETTER_GROUPS.items():
        if ch in members:
            return name
    return None
