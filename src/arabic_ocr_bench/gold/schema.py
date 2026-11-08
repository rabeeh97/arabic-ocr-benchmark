"""Label record schema, controlled vocabularies and text hygiene for the labeling tool.

The vocabularies are the single source of truth: the tool, the guidelines and the agreement report all read them.
"""
from __future__ import annotations

import re
import unicodedata

from ..normalize import canonicalize_encoding, strip_tatweel
from . import GUIDELINE_VERSION

LABELER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,23}$")

VERDICTS = {
    "correct": "The original label is already right (apart from encoding hygiene).",
    "corrected": "The original label was wrong; text_verified holds the fixed transcription.",
    "unusable": "Cannot be verified or has no valid text (blank, non-text, hopeless crop). Give the reason.",
    "unsure": "I cannot decide; needs a second opinion. Say why in the notes.",
}
CORRECTION_TYPES = {
    "letters": "wrong, missing or extra letters (not only dots)",
    "dots": "dots wrong or missing (ب/ت/ث, ج/ح/خ, ق/ف ...)",
    "missing_text": "text visible in the image but absent from the label",
    "extra_text": "label has text that is not in the image",
    "diacritics": "harakat added, removed or changed",
    "spacing": "word spaces wrong (merged or split words)",
    "punct_digits": "punctuation or digits wrong",
    "order": "words or segments in the wrong reading order",
    "convention": "only a transcription-convention difference (see guidelines), not a misreading",
    "other": "anything else (explain in notes)",
}
IMAGE_ISSUES = {
    "cropped_start": "a word is cut off at the start (right edge) of the line",
    "cropped_end": "a word is cut off at the end (left edge) of the line",
    "neighbour_text": "ink of the line above or below intrudes into the crop",
    "multi_line": "the crop contains more than one line of text",
    "scratched": "crossed-out, erased or overwritten text",
    "marginal_or_interlinear": "marginal note or interlinear addition inside the crop",
    "faded_or_bleed": "faded ink, show-through or ink bleed",
    "stain_or_hole": "stain, hole, tear or other damage over the text",
    "low_resolution": "too small or blurry to read some letters",
    "skewed_or_rotated": "the line is rotated or strongly skewed",
    "ruled_or_underlined": "ruling, underline or border lines touch the text",
    "non_text": "drawing, signature, stamp, ornament or blank",
    "mixed_script": "more than one script in the line",
}
LINE_KINDS = {
    "main_text": "ordinary running text", "heading": "title or heading", "marginal": "marginal or side note",
    "catchword": "catchword at a page bottom", "number_only": "page or folio number, date or other digits only",
    "signature_or_stamp": "signature, seal or stamp", "poetry": "verse line", "other": "none of the above",
}
LEGIBILITY = {
    "clear": "every letter can be read", "hard": "readable but needs effort or context",
    "partly_illegible": "some words or letters cannot be read (marked #)", "illegible": "nothing can be read",
}
DIACRITICS_VISIBLE = {
    "none": "no marks visible", "sparse": "a few marks", "partial": "marks on many letters",
    "full": "marks on almost every letter", "na": "not applicable (not Arabic script)",
}



def clean_text(s: str, language: str = "ar") -> tuple[str, list[str]]:
    """Mechanical hygiene applied on save. Returns (clean, notes). Never changes letters or marks."""
    notes: list[str] = []
    out = canonicalize_encoding(s)  # strips bidi/zero-width controls, folds presentation forms, NFC
    if out != unicodedata.normalize("NFC", s):
        notes.append("removed invisible direction/zero-width marks or folded presentation forms")
    if language == "ar" and "ـ" in out:
        out = strip_tatweel(out)
        notes.append("removed tatweel (kashida)")
    out2 = re.sub(r"\s+", " ", out).strip()
    if out2 != out:
        notes.append("collapsed whitespace / joined line breaks into spaces")
    return out2, notes


def validate_label(rec: dict) -> list[str]:
    """Return a list of problems (empty = valid)."""
    errs: list[str] = []
    if not LABELER_RE.match(rec.get("labeler", "")):
        errs.append("labeler id must be 2-24 chars of a-z 0-9 _ - (start with a letter or digit)")
    if not rec.get("line_id"):
        errs.append("line_id missing")
    v = rec.get("verdict")
    if v not in VERDICTS:
        errs.append(f"verdict must be one of {sorted(VERDICTS)}")
        return errs
    if v == "corrected":
        if not (rec.get("text_verified") or "").strip():
            errs.append("corrected needs text_verified")
        elif rec["text_verified"] == rec.get("text_original_clean"):
            errs.append("verdict is 'corrected' but the text equals the original; use 'correct'")
        if not rec.get("correction_types"):
            errs.append("pick at least one correction type")
    if v == "correct" and rec.get("text_verified") != rec.get("text_original_clean"):
        errs.append("verdict is 'correct' but the text differs from the original; use 'corrected'")
    if v in ("unusable", "unsure") and not ((rec.get("notes") or "").strip() or rec.get("image_issues")):
        errs.append(f"'{v}' needs a reason: notes or at least one image issue")
    for field, vocab in (("correction_types", CORRECTION_TYPES), ("image_issues", IMAGE_ISSUES)):
        bad = [x for x in rec.get(field, []) if x not in vocab]
        if bad:
            errs.append(f"unknown {field}: {bad}")
    for field, vocab in (("line_kind", LINE_KINDS), ("legibility", LEGIBILITY), ("diacritics_visible", DIACRITICS_VISIBLE)):
        if rec.get(field) not in vocab:
            errs.append(f"{field} must be one of {sorted(vocab)}")
    if v in ("correct", "corrected") and rec.get("legibility") == "illegible":
        errs.append("an illegible line cannot be 'correct'/'corrected'; mark it unusable")
    return errs


def new_record(line: dict, labeler: str, form: dict, language: str = "ar") -> tuple[dict, list[str], list[str]]:
    """Build a label record from the tool's form. Returns (record, errors, hygiene_notes)."""
    orig_clean, _ = clean_text(line["text_original"], language)
    verdict = form.get("verdict")
    notes: list[str] = []
    if verdict == "correct":
        text_verified = orig_clean
    elif verdict == "corrected":
        text_verified, notes = clean_text(form.get("text_verified", ""), language)
    else:
        text_verified = None
    rec = {
        "line_id": line["line_id"], "dataset": line["dataset"], "tranche": line["tranche"], "labeler": labeler,
        "guideline_version": GUIDELINE_VERSION, "verdict": verdict,
        "text_original": line["text_original"], "text_original_clean": orig_clean, "text_verified": text_verified,
        "correction_types": list(form.get("correction_types", [])) if verdict == "corrected" else [],
        "image_issues": list(form.get("image_issues", [])),
        "line_kind": form.get("line_kind"), "legibility": form.get("legibility"),
        "diacritics_visible": form.get("diacritics_visible"), "script_style_seen": (form.get("script_style_seen") or "").strip() or None,
        "notes": (form.get("notes") or "").strip(), "seconds": int(form.get("seconds") or 0),
        "saved_utc": form.get("saved_utc"),
    }
    return rec, validate_label(rec), notes
