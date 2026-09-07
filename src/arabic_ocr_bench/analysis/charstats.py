# src/arabic_ocr_bench/analysis/charstats.py
"""Task 5: ``error_anatomy()`` — pooled character/word error anatomy.

Reduces a list of :class:`~arabic_ocr_bench.analysis.align.AlignmentRecord` (the
OK ones only) into JSON-serialisable summaries:

- ``sid_split``      : substitution / insertion / deletion split of char ops,
                       per model plus a ``"__pooled__"`` total.
- ``spacing_share``  : how much of each model's char-op budget is spacing.
- ``length_bias``    : word-count delta quantiles + a capped scatter of
                       (gt_words, pred_words) points.
- ``rewrites``       : the most common char-run and word substitutions
                       (count >= 2, top 25 each).

Every number in the returned dict is a plain Python ``int`` / ``float`` so the
whole structure survives ``json.dumps``.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

_POINT_CAP = 400


def _ok(records):
    return [r for r in records if r.failure_class == "ok"]


def _decimate(pairs, cap):
    """Return at most ``cap`` items from ``pairs``, evenly strided.

    Stride decimation alone (``pairs[::step]`` with ``step = len // cap``) can
    still overshoot — e.g. 401..799 items keep step 1, and 1000 items yield 500 —
    so the strided slice is hard-capped at ``cap``.
    """
    step = max(1, len(pairs) // cap)
    return pairs[::step][:cap]


def _char_sub_runs(char_ops):
    """Yield ``(ref_run, hyp_run)`` for every maximal run of substitutions that
    are *adjacent in the reference* — a run breaks on any non-sub op and on any
    gap in ``ref_i`` (so ``توت`` -> ``بوب``, whose middle char is unchanged,
    yields the ت->ب rewrite twice rather than one merged ``تت`` -> ``بب``)."""
    run_ref: list[str] = []
    run_hyp: list[str] = []
    prev_i: int | None = None
    for o in list(char_ops) + [None]:
        cont = (
            o is not None
            and o.op == "sub"
            and (prev_i is None or o.ref_i == prev_i + 1)
        )
        if cont:
            run_ref.append(o.ref)
            run_hyp.append(o.hyp)
            prev_i = o.ref_i
        else:
            if run_ref:
                yield "".join(run_ref), "".join(run_hyp)
            run_ref, run_hyp = [], []
            prev_i = None
            if o is not None and o.op == "sub":
                run_ref.append(o.ref)
                run_hyp.append(o.hyp)
                prev_i = o.ref_i


def _word_subs(rows_by_key, ok):
    """Re-align light-normalised gt/pred per OK record; yield word sub pairs."""
    import jiwer

    from ..normalize import normalize

    for r in ok:
        row = rows_by_key.get((r.model, r.sample_id))
        if row is None:
            continue
        g_light = normalize(row.gt, "light")
        p_light = normalize(row.pred, "light")
        ref_words = g_light.split()
        hyp_words = p_light.split()
        res = jiwer.process_words(g_light, p_light)
        # jiwer 4.0.0: `.alignments` (plural) is a list-of-lists; [0] is this pair.
        chunks = res.alignments[0] if res.alignments else []
        for ch in chunks:
            if ch.type != "substitute":
                continue
            ref_ngram = " ".join(ref_words[ch.ref_start_idx:ch.ref_end_idx])
            hyp_ngram = " ".join(hyp_words[ch.hyp_start_idx:ch.hyp_end_idx])
            yield ref_ngram, hyp_ngram


def error_anatomy(records, rows_by_key=None) -> dict:
    """Pool OK ``AlignmentRecord``s into an error-anatomy summary dict.

    Parameters
    ----------
    records:
        Iterable of ``AlignmentRecord``. Only ``failure_class == "ok"`` rows are
        pooled; stubs (errors / refusals) are ignored.
    rows_by_key:
        Optional ``{(model, sample_id): PredRow}`` map. When supplied, the word
        rewrite list is computed by re-aligning ``normalize(gt, "light")`` vs
        ``normalize(pred, "light")`` for each OK record. When ``None`` the word
        list is left empty (the record itself does not carry the source text).
    """
    ok = _ok(records)

    sid: dict[str, Counter] = defaultdict(Counter)
    length: dict[str, list] = defaultdict(list)   # model -> [(sid, gt_w, pred_w)]
    delta: dict[str, list] = defaultdict(list)    # model -> [pred_w - gt_w]
    spacing: dict[str, dict] = defaultdict(lambda: {"wer": [], "sp": 0, "tot": 0})
    char_rw: Counter = Counter()

    _ = sid["__pooled__"]  # contract: always present, even with zero char ops

    for r in ok:
        # touch per-model buckets so a model with only perfect predictions
        # still shows up with zeros.
        _ = sid[r.model]
        for o in r.char_ops:
            sid[r.model][o.op] += 1
            sid["__pooled__"][o.op] += 1

        length[r.model].append((r.sample_id, r.len_ref_words, r.len_hyp_words))
        delta[r.model].append(r.len_hyp_words - r.len_ref_words)

        spacing[r.model]["wer"].append(r.wer_light)
        spacing[r.model]["sp"] += r.spacing_del + r.spacing_ins
        spacing[r.model]["tot"] += len(r.char_ops)

        for ref_run, hyp_run in _char_sub_runs(r.char_ops):
            char_rw[(ref_run, hyp_run)] += 1

    word_rw: Counter = Counter()
    if rows_by_key is not None:
        for ref_ngram, hyp_ngram in _word_subs(rows_by_key, ok):
            word_rw[(ref_ngram, hyp_ngram)] += 1

    def q(xs, p):
        return float(np.percentile(xs, p)) if xs else 0.0

    out = {
        "sid_split": {
            m: {
                "sub": int(c["sub"]),
                "ins": int(c["ins"]),
                "del": int(c["del"]),
            }
            for m, c in sid.items()
        },
        "spacing_share": {
            m: {
                "wer_light": float(np.mean(v["wer"])) if v["wer"] else 0.0,
                "spacing_ops": int(v["sp"]),
                "total_char_ops": int(v["tot"]),
                "spacing_frac_of_ops": float(v["sp"]) / float(max(1, v["tot"])),
            }
            for m, v in spacing.items()
        },
        "length_bias": {},
        "rewrites": {
            "char": [
                [a, b, int(n)]
                for (a, b), n in char_rw.most_common(25)
                if n >= 2
            ],
            "word": [
                [a, b, int(n)]
                for (a, b), n in word_rw.most_common(25)
                if n >= 2
            ],
        },
    }

    for m, triples in length.items():
        by_sid = sorted(triples, key=lambda t: t[0])
        sampled = _decimate(by_sid, _POINT_CAP)
        out["length_bias"][m] = {
            "delta_median": q(delta[m], 50),
            "delta_q1": q(delta[m], 25),
            "delta_q3": q(delta[m], 75),
            "pred_vs_gt": [[int(gt_w), int(pred_w)] for _sid, gt_w, pred_w in sampled],
        }

    return out
