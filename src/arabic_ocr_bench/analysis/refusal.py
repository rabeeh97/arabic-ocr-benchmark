"""Failure classification for benchmark predictions.

`classify_failure` maps a (prediction, error, attempts) triple to one of eight
labels. Rules are checked in a fixed precedence (first match wins):

    rate_limited -> refusal (error == "refusal") -> other_error -> empty
    -> prompt_echo -> refusal / meta_commentary (from content) -> wrong_script
    -> ok

Notes on precedence:

- The runner's own `error="refusal"` flag is honored before `other_error`, so a
  declared decline is not demoted to a generic error.
- `prompt_echo` is checked *before* the content refusal/meta block and before
  `wrong_script`: the `line-v2` prompt template is English prose, so a verbatim
  echo would otherwise be swallowed by the Latin-script rule.
- Declared declines (recognized refusal phrases / meta-commentary / code
  fences) are checked *before* the coarse Latin-ratio `wrong_script` rule, so a
  model that declines in English lands in `refusal` while a model that
  romanizes the Arabic lands in `wrong_script` — distinct failure modes.

`FAILURE_CLASSES` below is a separate canonical ordering used only for stable
chart ordering and does not reflect rule precedence.
"""

from __future__ import annotations

import re

from ..models.litellm_client import looks_like_refusal
from ..prompting import get_prompt

FAILURE_CLASSES = (
    "ok", "empty", "refusal", "meta_commentary",
    "wrong_script", "prompt_echo", "rate_limited", "other_error",
)

_RATE_RE = re.compile(r"\b429\b|rate.?limit", re.I)
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_META_RE = re.compile(
    r"(i (cannot|can'?t|am unable)|as an ai|the (text|image) (says|reads|shows)"
    r"|here is the|لا (أستطيع|يمكنني)|عذرا|يرجى تقديم)",
    re.I,
)


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta) if ta else 0.0


def classify_failure(
    pred: str, error: str | None, attempts: int, *, prompt_version: str = "line-v2"
) -> str:
    if error and _RATE_RE.search(error):
        return "rate_limited"
    if error == "refusal":
        return "refusal"
    if error:
        return "other_error"
    s = (pred or "").strip()
    if len(s) < 2:
        return "empty"
    try:
        tmpl = get_prompt(prompt_version)
    except ValueError:
        tmpl = ""
    if tmpl and _token_overlap(tmpl, s) >= 0.8:
        return "prompt_echo"
    if "```" in pred or _META_RE.search(s) or looks_like_refusal(s):
        # a genuine refusal (no Arabic content) vs. meta-commentary that still
        # contains a transcription attempt
        return "refusal" if not _ARABIC_RE.search(s) else "meta_commentary"
    nonspace = [c for c in s if not c.isspace()]
    if len(s) >= 8 and nonspace:
        latin = sum(bool(_LATIN_RE.match(c)) for c in nonspace)
        if latin / len(nonspace) > 0.5:
            return "wrong_script"
    return "ok"
