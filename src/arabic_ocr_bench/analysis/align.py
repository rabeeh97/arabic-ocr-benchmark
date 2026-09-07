# src/arabic_ocr_bench/analysis/align.py
from __future__ import annotations

import re
from dataclasses import dataclass

import jiwer

from ..normalize import normalize, strip_harakat
from .refusal import classify_failure

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


# --------------------------------------------------------------------------- #
# Task 4: AlignmentRecord + single-pass build_record()
# --------------------------------------------------------------------------- #

# harakat / tanwin / shadda / sukun (U+064B..U+065F) + superscript alef (U+0670)
_HARAKAT_RE = re.compile("[" + chr(0x064B) + "-" + chr(0x065F) + chr(0x0670) + "]")
# Arabic Presentation Forms-A (U+FB50..U+FDFF) and -B (U+FE70..U+FEFF)
_PRESENTATION_RE = re.compile(
    "[" + chr(0xFB50) + "-" + chr(0xFDFF) + chr(0xFE70) + "-" + chr(0xFEFF) + "]"
)
# ASCII, Arabic-Indic (U+0660..U+0669) and Extended Arabic-Indic (U+06F0..U+06F9)
_DIGIT_RE = re.compile(
    "[0-9" + chr(0x0660) + "-" + chr(0x0669) + chr(0x06F0) + "-" + chr(0x06F9) + "]"
)


@dataclass(frozen=True)
class CharOp:
    op: str  # "sub" | "ins" | "del"
    ref: str  # "" for ins
    hyp: str  # "" for del
    ref_i: int  # index into the light-normalised reference; -1 for pure ins


def expand_char_chunks(ref: str, hyp: str) -> list[CharOp]:
    """Expand jiwer's char-level alignment chunks into per-character CharOps."""
    out: list[CharOp] = []
    res = jiwer.process_characters(ref, hyp)
    # jiwer 4.0.0: `.alignments` is a list-of-lists; [0] is this single pair.
    chunks = res.alignments[0] if res.alignments else []
    for ch in chunks:
        if ch.type == "equal":
            continue
        rs, re_, hs, he = (
            ch.ref_start_idx, ch.ref_end_idx, ch.hyp_start_idx, ch.hyp_end_idx
        )
        rseg, hseg = ref[rs:re_], hyp[hs:he]
        if ch.type == "substitute":
            k = min(len(rseg), len(hseg))
            for j in range(k):
                out.append(CharOp("sub", rseg[j], hseg[j], rs + j))
            for j in range(k, len(rseg)):
                out.append(CharOp("del", rseg[j], "", rs + j))
            for j in range(k, len(hseg)):
                out.append(CharOp("ins", "", hseg[j], -1))
        elif ch.type == "delete":
            for j, c in enumerate(rseg):
                out.append(CharOp("del", c, "", rs + j))
        elif ch.type == "insert":
            for c in hseg:
                out.append(CharOp("ins", "", c, -1))
    return out


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    o = jiwer.process_characters(ref, hyp)
    return (o.substitutions + o.deletions + o.insertions) / (
        o.substitutions + o.deletions + o.hits
    )


def _longest_latin_run(s: str) -> int:
    best = cur = 0
    for c in s:
        cur = cur + 1 if ("a" <= c.lower() <= "z") else 0
        best = max(best, cur)
    return best


@dataclass(frozen=True)
class AlignmentRecord:
    model: str
    dataset: str
    sample_id: str
    failure_class: str
    char_ops: list
    word_ops: dict
    rasm_ops: dict
    diac_seq_ref: str
    diac_seq_hyp: str
    spacing_del: int
    spacing_ins: int
    len_ref_chars: int
    len_hyp_chars: int
    len_ref_words: int
    len_hyp_words: int
    digit_swaps: int
    presentation_form_chars: int
    latin_run_len: int
    has_code_fence: bool
    cer_light: float
    cer_raw: float
    cer_rasm: float
    wer_light: float


def _stub(row, fc: str) -> AlignmentRecord:
    return AlignmentRecord(
        model=row.model, dataset=row.dataset, sample_id=row.sample_id,
        failure_class=fc, char_ops=[],
        word_ops={"sub": 0, "ins": 0, "del": 0, "hits": 0},
        rasm_ops={"sub": 0, "ins": 0, "del": 0, "ref_len": 0},
        diac_seq_ref="", diac_seq_hyp="", spacing_del=0, spacing_ins=0,
        len_ref_chars=0, len_hyp_chars=0, len_ref_words=0, len_hyp_words=0,
        digit_swaps=0, presentation_form_chars=0, latin_run_len=0,
        has_code_fence=False, cer_light=0.0, cer_raw=0.0, cer_rasm=0.0,
        wer_light=0.0,
    )


def build_record(row, *, prompt_version: str = "line-v2") -> AlignmentRecord:
    fc = classify_failure(
        row.pred, row.error, row.attempts, prompt_version=prompt_version
    )
    if fc != "ok":
        return _stub(row, fc)

    g_light, p_light = normalize(row.gt, "light"), normalize(row.pred, "light")
    g_raw, p_raw = normalize(row.gt, "raw"), normalize(row.pred, "raw")
    char_ops = expand_char_chunks(g_light, p_light)

    w = jiwer.process_words(g_light, p_light)
    word_ops = {
        "sub": w.substitutions, "ins": w.insertions,
        "del": w.deletions, "hits": w.hits,
    }

    g_rasm = to_rasm(strip_harakat(g_light))
    p_rasm = to_rasm(strip_harakat(p_light))
    ro = jiwer.process_characters(g_rasm, p_rasm)
    rasm_ops = {
        "sub": ro.substitutions, "ins": ro.insertions, "del": ro.deletions,
        "ref_len": ro.substitutions + ro.deletions + ro.hits,
    }

    spacing_del = sum(1 for o in char_ops if o.op == "del" and o.ref == " ")
    spacing_ins = sum(1 for o in char_ops if o.op == "ins" and o.hyp == " ")
    digit_swaps = sum(
        1 for o in char_ops
        if o.op == "sub" and _DIGIT_RE.match(o.ref) and _DIGIT_RE.match(o.hyp)
    )
    return AlignmentRecord(
        model=row.model, dataset=row.dataset, sample_id=row.sample_id,
        failure_class="ok", char_ops=char_ops, word_ops=word_ops,
        rasm_ops=rasm_ops,
        diac_seq_ref="".join(_HARAKAT_RE.findall(g_raw)),
        diac_seq_hyp="".join(_HARAKAT_RE.findall(p_raw)),
        spacing_del=spacing_del, spacing_ins=spacing_ins,
        len_ref_chars=len(g_light), len_hyp_chars=len(p_light),
        len_ref_words=len(g_light.split()), len_hyp_words=len(p_light.split()),
        digit_swaps=digit_swaps,
        presentation_form_chars=len(_PRESENTATION_RE.findall(row.pred)),
        latin_run_len=_longest_latin_run(row.pred),
        has_code_fence="```" in row.pred,
        cer_light=_cer(g_light, p_light), cer_raw=_cer(g_raw, p_raw),
        cer_rasm=_cer(g_rasm, p_rasm), wer_light=w.wer,
    )
