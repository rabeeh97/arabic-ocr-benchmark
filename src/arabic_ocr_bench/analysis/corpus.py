"""Task 13: ``corpus_health()`` — ground-truth corpus statistics, leaderboard
churn and a contamination cross-reference.

Reduces the raw ``PredRow`` list (for GT text + per-sample ``diac_f1``) and the
index-aligned ``AlignmentRecord`` list (for the leaderboard-style micro
aggregates) into one JSON-serialisable summary:

- ``gt_stats``          : per dataset — line count, charset size, mean line
                          length, diacritic density and a top-60 Zipf table,
                          all over that dataset's GT lines deduped by
                          ``sample_id``.
- ``charset_coverage``  : per dataset — characters unique to it, and the
                          fraction of its charset shared with another dataset.
- ``leaderboard_churn`` : a strict 1..N rank per model for each of
                          ``cer_light`` / ``wer_light`` / ``diac_f1`` /
                          ``cost_adjusted`` (Ruling 5: contiguous permutation,
                          ties broken by model name).
- ``contamination``     : when a contamination report is supplied, the subset
                          of its sample ids that this run also finds
                          "consensus-easy + exact" (every model that scored the
                          line got ``cer_light == 0.0``).

Every number in the returned structure is a plain Python ``int`` / ``float`` so
the whole dict survives ``json.dumps``; iteration is sorted for determinism.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ..normalize import strip_harakat
from .operational import operational

_ZIPF_TOP = 60

METRICS = ("cer_light", "wer_light", "diac_f1", "cost_adjusted")


def _by_count(counter) -> list:
    """Deterministic ``[(key, count), ...]`` sorted ``(-count, key)``."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))


def _gt_lines_by_dataset(rows) -> dict[str, dict[str, str]]:
    """``{dataset: {sample_id: gt}}`` — one GT per sample id (deduped)."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for row in sorted(rows, key=lambda r: (r.dataset, r.sample_id, r.model)):
        out[row.dataset].setdefault(row.sample_id, row.gt)
    return {ds: dict(m) for ds, m in out.items()}


def _gt_stats(gt_by_ds: dict[str, dict[str, str]]) -> dict:
    out: dict[str, dict] = {}
    for ds in sorted(gt_by_ds):
        lines = [gt_by_ds[ds][sid] for sid in sorted(gt_by_ds[ds])]
        total_chars = sum(len(g) for g in lines)
        diac = sum(len(g) - len(strip_harakat(g)) for g in lines)
        freq: Counter = Counter()
        for g in lines:
            freq.update(g)
        zipf = [
            [int(rank), int(count)]
            for rank, (_ch, count) in enumerate(_by_count(freq)[:_ZIPF_TOP], 1)
        ]
        out[ds] = {
            "n_lines": len(lines),
            "charset_size": len(freq),
            "mean_line_chars": (
                float(total_chars / len(lines)) if lines else 0.0
            ),
            "diac_density": (
                float(diac / total_chars) if total_chars else 0.0
            ),
            "zipf": zipf,
        }
    return out


def _charset_coverage(gt_by_ds: dict[str, dict[str, str]]) -> dict:
    charsets = {
        ds: {c for g in gt_by_ds[ds].values() for c in g}
        for ds in gt_by_ds
    }
    out: dict[str, dict] = {}
    for ds in sorted(charsets):
        mine = charsets[ds]
        others: set[str] = set()
        for o, cs in charsets.items():
            if o != ds:
                others |= cs
        only_here = sorted(mine - others)
        shared = sum(1 for c in mine if c in others)
        out[ds] = {
            "only_here": only_here,
            "shared_frac": float(shared / len(mine)) if mine else 0.0,
        }
    return out


def _model_metrics(rows, records) -> tuple[list[str], dict[str, dict[str, float]]]:
    models = sorted({rec.model for rec in records if rec.failure_class == "ok"})

    cer_num: dict[str, float] = defaultdict(float)
    cer_den: dict[str, float] = defaultdict(float)
    werr: dict[str, float] = defaultdict(float)   # word substitutions+ins+del
    wref: dict[str, float] = defaultdict(float)   # word substitutions+del+hits
    diac: dict[str, list[float]] = defaultdict(list)

    for row, rec in zip(rows, records):
        if rec.failure_class != "ok":
            continue
        cer_num[rec.model] += round(rec.cer_light * rec.len_ref_chars)
        cer_den[rec.model] += rec.len_ref_chars
        wo = rec.word_ops
        werr[rec.model] += wo["sub"] + wo["ins"] + wo["del"]
        wref[rec.model] += wo["sub"] + wo["del"] + wo["hits"]
        f1 = row.scores.get("diac_f1")
        if f1 is not None:
            diac[rec.model].append(float(f1))

    pareto = operational(rows, records)["pareto"]

    vals: dict[str, dict[str, float]] = {}
    for m in models:
        cer = float(cer_num[m] / cer_den[m]) if cer_den[m] else 0.0
        wer = float(werr[m] / wref[m]) if wref[m] else 0.0
        f1 = float(sum(diac[m]) / len(diac[m])) if diac[m] else 0.0
        usd_per_1k = float(pareto.get(m, {}).get("usd_per_1k", 0.0))
        cost_adjusted = float(cer * (1.0 + math.log10(1.0 + usd_per_1k)))
        vals[m] = {
            "cer_light": cer,
            "wer_light": wer,
            "diac_f1": f1,
            "cost_adjusted": cost_adjusted,
        }
    return models, vals


def _leaderboard_churn(rows, records) -> dict:
    models, vals = _model_metrics(rows, records)
    ranks: dict[str, dict[str, int]] = {m: {} for m in models}
    for metric in METRICS:
        higher_better = metric == "diac_f1"
        ordered = sorted(
            models,
            key=lambda m: (
                -vals[m][metric] if higher_better else vals[m][metric],
                m,
            ),
        )
        for i, m in enumerate(ordered, 1):
            ranks[m][metric] = int(i)
    return {"metrics": list(METRICS), "ranks": ranks}


def _consensus_easy_exact(records) -> set[str]:
    """Sample ids where every model that scored the line got ``cer_light``
    exactly 0.0 (and at least one model scored it)."""
    seen: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if rec.failure_class == "ok":
            seen[rec.sample_id].append(float(rec.cer_light))
    return {
        sid
        for sid, cers in seen.items()
        if cers and all(c == 0.0 for c in cers)
    }


def _load_contamination_ids(path: str | Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k) for k in data}
    if isinstance(data, list):
        return {str(x) for x in data}
    return set()


def _contamination(records, contamination_path) -> dict:
    if contamination_path is None or not Path(contamination_path).exists():
        return {"available": False, "flagged": []}
    flagged_ids = _load_contamination_ids(contamination_path)
    hits = flagged_ids & _consensus_easy_exact(records)
    return {"available": True, "flagged": sorted(hits)}


def corpus_health(rows, records, *, contamination_path: str | Path | None = None) -> dict:
    """Ground-truth corpus stats, leaderboard churn and a contamination xref.

    ``rows`` is the full :func:`iter_pred_rows` list (errored rows included);
    ``records`` is ``[build_record(r) for r in rows]``, index-aligned.
    """
    if len(rows) != len(records):
        raise ValueError("rows and records must be index-aligned")
    gt_by_ds = _gt_lines_by_dataset(rows)
    return {
        "gt_stats": _gt_stats(gt_by_ds),
        "charset_coverage": _charset_coverage(gt_by_ds),
        "leaderboard_churn": _leaderboard_churn(rows, records),
        "contamination": _contamination(records, contamination_path),
    }
