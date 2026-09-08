"""Task 7: ``diacritics()`` — per-mark diacritic error reduction.

Reduces the OK :class:`~arabic_ocr_bench.analysis.align.AlignmentRecord`s into a
JSON-serialisable summary of *where harakat go wrong*:

- ``per_mark``  : ``{model: {mark_name: {"missed", "inserted", "substituted"}}}``
                  — the ordered ``diac_seq_ref`` vs ``diac_seq_hyp`` strings are
                  aligned per record with :func:`jiwer.process_characters`; a
                  ``delete`` chunk char is *missed*, an ``insert`` char is
                  *inserted*, a ``substitute`` char is *substituted*. Deletes and
                  substitutes bucket by the **reference** mark's name; inserts
                  (no reference mark) bucket by the **hypothesis** mark's name.
- ``f1_vs_cer`` : ``{model: {"diac_f1", "cer_light", "attempt_rate"}}`` —
                  ``diac_f1`` is the mean of the per-sample ``scores["diac_f1"]``
                  over OK records (samples with no score row are skipped),
                  ``cer_light`` the mean of ``AlignmentRecord.cer_light``, and
                  ``attempt_rate`` the fraction of OK records whose prediction
                  carried at least one harakat (``diac_seq_hyp != ""``).
- ``gt_density``: ``{dataset: float}`` — mean of
                  ``len(diac_seq_ref) / max(1, len_ref_chars)`` over OK records.

Every scalar in the returned dict is a plain Python ``int`` / ``float`` so the
whole structure survives ``json.dumps``.
"""

from __future__ import annotations

from collections import defaultdict

import jiwer

from .charstats import _ok

# Each harakat code point (U+064B..U+0652 range, plus U+0670 dagger-alef and
# U+0653 madda) mapped to a stable bucket name. Anything else that slips through
# ``_HARAKAT_RE`` (U+0653..U+065F combining marks) buckets as ``"other"``.
MARK_NAMES: dict[str, str] = {
    "ً": "tanwin_fath",   # ARABIC FATHATAN
    "ٌ": "tanwin_damm",   # ARABIC DAMMATAN
    "ٍ": "tanwin_kasr",   # ARABIC KASRATAN
    "َ": "fatha",         # ARABIC FATHA
    "ُ": "damma",         # ARABIC DAMMA
    "ِ": "kasra",         # ARABIC KASRA
    "ّ": "shadda",        # ARABIC SHADDA
    "ْ": "sukun",         # ARABIC SUKUN
    "ٰ": "dagger_alef",   # ARABIC LETTER SUPERSCRIPT ALEF
    "ٓ": "madda",         # ARABIC MADDAH ABOVE
}


def _mark_name(ch: str) -> str:
    return MARK_NAMES.get(ch, "other")


def _empty_counts() -> dict[str, int]:
    return {"missed": 0, "inserted": 0, "substituted": 0}


def _bump(marks: dict[str, dict[str, int]], name: str, field: str) -> None:
    marks.setdefault(name, _empty_counts())[field] += 1


def diacritics(records, rows_by_key) -> dict:
    """Pool OK ``AlignmentRecord``s into a per-mark diacritic error summary.

    Parameters
    ----------
    records:
        Iterable of ``AlignmentRecord``. Only ``failure_class == "ok"`` rows are
        pooled; stubs (errors / refusals) are ignored.
    rows_by_key:
        ``{(model, sample_id): PredRow}`` map. Used to read the per-sample
        ``scores["diac_f1"]``; a missing key or missing ``diac_f1`` is skipped.
    """
    ok = _ok(records)

    per_mark: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    f1: dict[str, list[float]] = defaultdict(list)
    cer: dict[str, list[float]] = defaultdict(list)
    attempts: dict[str, list[int]] = defaultdict(list)
    density: dict[str, list[float]] = defaultdict(list)

    for r in ok:
        # touch per-model buckets so an all-correct model still appears
        marks = per_mark[r.model]
        _ = f1[r.model]

        ref, hyp = r.diac_seq_ref, r.diac_seq_hyp
        res = jiwer.process_characters(ref, hyp)
        chunks = res.alignments[0] if res.alignments else []
        for ch in chunks:
            if ch.type == "equal":
                continue
            if ch.type == "delete":
                for c in ref[ch.ref_start_idx:ch.ref_end_idx]:
                    _bump(marks, _mark_name(c), "missed")
            elif ch.type == "insert":
                for c in hyp[ch.hyp_start_idx:ch.hyp_end_idx]:
                    _bump(marks, _mark_name(c), "inserted")
            elif ch.type == "substitute":
                rseg = ref[ch.ref_start_idx:ch.ref_end_idx]
                hseg = hyp[ch.hyp_start_idx:ch.hyp_end_idx]
                k = min(len(rseg), len(hseg))
                for c in rseg[:k]:
                    _bump(marks, _mark_name(c), "substituted")
                # ragged tails (jiwer substitute chunks are normally square)
                for c in rseg[k:]:
                    _bump(marks, _mark_name(c), "missed")
                for c in hseg[k:]:
                    _bump(marks, _mark_name(c), "inserted")

        cer[r.model].append(float(r.cer_light))
        attempts[r.model].append(1 if r.diac_seq_hyp != "" else 0)
        density[r.dataset].append(
            len(r.diac_seq_ref) / max(1, r.len_ref_chars)
        )

        row = rows_by_key.get((r.model, r.sample_id))
        if row is not None and "diac_f1" in getattr(row, "scores", {}):
            f1[r.model].append(float(row.scores["diac_f1"]))

    def _mean(xs) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    f1_vs_cer = {
        m: {
            "diac_f1": _mean(f1.get(m, [])),
            "cer_light": _mean(cer.get(m, [])),
            "attempt_rate": _mean(attempts.get(m, [])),
        }
        for m in per_mark
    }

    gt_density = {ds: _mean(xs) for ds, xs in density.items()}

    return {
        "per_mark": {
            m: {
                name: {k: int(v) for k, v in counts.items()}
                for name, counts in marks.items()
            }
            for m, marks in per_mark.items()
        },
        "f1_vs_cer": f1_vs_cer,
        "gt_density": gt_density,
    }
