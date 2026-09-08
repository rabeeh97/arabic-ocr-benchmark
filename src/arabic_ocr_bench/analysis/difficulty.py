"""Task 8: ``difficulty()`` — sample & dataset difficulty analysis.

Reduces the per-sample prediction rows (:class:`~arabic_ocr_bench.analysis.
loader.PredRow`) and their alignment records
(:class:`~arabic_ocr_bench.analysis.align.AlignmentRecord`) into a single
JSON-serialisable dict:

- ``consensus_hard``    : lines every model struggles with (top by mean CER).
- ``consensus_easy``    : lines every model nails (bottom by mean CER, ``>= 3``
                          models scored).
- ``discriminative``    : lines that split the field (top by population variance
                          of per-model CER).
- ``suspected_bad_gt``  : lines where the models agree with *each other* but not
                          with the ground truth — a likely mis-transcribed
                          reference. Ranked by
                          ``(1 - mean_pairwise_pred_cer) * mean_pred_vs_gt_cer``.
- ``cer_vs_length``     : per-model CER binned over the pooled reference-length
                          range.
- ``cer_distribution``  : per-(model, dataset) CER quantiles.
- ``ecdf``              : per-model empirical CDF of per-sample CER (<= 50
                          points, monotonic, last ``y == 1.0``).

Per-sample CER for the consensus / discriminative views is the row's
``scores["cer_light"]`` when it scored, else ``1.0`` (an errored or unscored
model still counts as a full miss). Every scalar in the returned structure is a
plain Python ``int`` / ``float`` so the whole thing survives ``json.dumps``.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from ..metrics import cer as _pair_cer
from ..normalize import normalize
from .charstats import _ok

_ECDF_MAX_POINTS = 50


def _row_scored(row) -> bool:
    """True when the row carries a usable ``cer_light`` (no error, score present)."""
    return row.error is None and row.scores.get("cer_light") is not None


def _row_cer(row) -> float:
    """Per-sample CER for the consensus / discriminative views: the row's
    ``cer_light`` when it scored, else ``1.0``."""
    if _row_scored(row):
        return float(row.scores["cer_light"])
    return 1.0


def _consensus_entry(sid, ds, gt, per_model_out, mean_cer, n_models) -> dict:
    return {
        "sample_id": sid,
        "dataset": ds,
        "gt": gt,
        "mean_cer": float(mean_cer),
        "n_models": int(n_models),
        "per_model": dict(per_model_out),
    }


def difficulty(rows, records, *, cap=30, bins=12) -> dict:
    # --------------------------------------------------------------------- #
    # group rows by sample
    # --------------------------------------------------------------------- #
    by_sample: dict[str, dict[str, object]] = defaultdict(dict)
    ds_of: dict[str, str] = {}
    gt_of: dict[str, str] = {}
    for r in rows:
        by_sample[r.sample_id][r.model] = r
        ds_of.setdefault(r.sample_id, r.dataset)
        gt_of.setdefault(r.sample_id, r.gt)

    ok_by_sample: dict[str, dict[str, object]] = defaultdict(dict)
    for rec in _ok(records):
        ok_by_sample[rec.sample_id][rec.model] = rec

    consensus_hard: list[dict] = []
    consensus_easy: list[dict] = []
    discriminative: list[dict] = []
    suspected_bad_gt: list[dict] = []

    for sid, per_model in by_sample.items():
        ds, gt = ds_of[sid], gt_of[sid]
        models = sorted(per_model)

        per_model_cer = {m: _row_cer(per_model[m]) for m in models}
        n_models = sum(1 for m in models if _row_scored(per_model[m]))
        per_model_out = {
            m: (float(per_model[m].scores["cer_light"]) if _row_scored(per_model[m]) else None)
            for m in models
        }
        mean_cer = float(np.mean([per_model_cer[m] for m in models])) if models else 0.0

        consensus_hard.append(
            _consensus_entry(sid, ds, gt, per_model_out, mean_cer, n_models)
        )

        if n_models >= 3:
            vals = [per_model_cer[m] for m in models]
            consensus_easy.append(
                _consensus_entry(sid, ds, gt, per_model_out, mean_cer, n_models)
            )
            entry = _consensus_entry(sid, ds, gt, per_model_out, mean_cer, n_models)
            entry["variance"] = float(np.var(vals))
            discriminative.append(entry)

        # suspected bad GT: every model that attempted the line produced an OK
        # transcription, at least 3 models scored it, and they agree with each
        # other more than with the reference. The ">= 3" check uses the same
        # "distinct models that scored the line" count as consensus_easy /
        # discriminative, so a model with an OK record but no cer_light is
        # treated consistently across all three lists.
        ok_models = ok_by_sample.get(sid, {})
        if n_models >= 3 and len(ok_models) == len(models):
            oms = sorted(ok_models)
            preds = {m: normalize(per_model[m].pred, "light") for m in oms}
            pair_cers = [_pair_cer(preds[a], preds[b]) for a, b in combinations(oms, 2)]
            mean_pairwise = float(np.mean(pair_cers)) if pair_cers else 0.0
            gt_cers = []
            for m in oms:
                v = per_model[m].scores.get("cer_light")
                gt_cers.append(float(v) if v is not None else float(ok_models[m].cer_light))
            mean_pred_vs_gt = float(np.mean(gt_cers)) if gt_cers else 0.0
            score = (1.0 - mean_pairwise) * mean_pred_vs_gt
            suspected_bad_gt.append({
                "sample_id": sid,
                "dataset": ds,
                "gt": gt,
                "score": float(score),
                "mean_pairwise_pred_cer": mean_pairwise,
                "mean_pred_vs_gt_cer": mean_pred_vs_gt,
                "n_models": len(oms),
            })

    consensus_hard.sort(key=lambda d: (-d["mean_cer"], d["sample_id"]))
    consensus_easy.sort(key=lambda d: (d["mean_cer"], d["sample_id"]))
    discriminative.sort(key=lambda d: (-d["variance"], d["sample_id"]))
    suspected_bad_gt.sort(key=lambda d: (-d["score"], d["sample_id"]))

    # --------------------------------------------------------------------- #
    # CER vs reference length — pooled equal-width bins
    # --------------------------------------------------------------------- #
    ok_recs = _ok(records)
    cer_vs_length: dict[str, list] = {}
    if ok_recs:
        lengths = [int(r.len_ref_chars) for r in ok_recs]
        lo, hi = min(lengths), max(lengths)
        width = (hi - lo) / bins if hi > lo else 0.0
        per_model_recs: dict[str, list] = defaultdict(list)
        for r in ok_recs:
            per_model_recs[r.model].append(r)
        for m in sorted(per_model_recs):
            recs_m = per_model_recs[m]
            if width == 0.0:
                cer_vs_length[m] = [[
                    float(lo),
                    float(np.mean([r.cer_light for r in recs_m])),
                    len(recs_m),
                ]]
                continue
            buckets: list[list[float]] = [[] for _ in range(bins)]
            for r in recs_m:
                idx = int((r.len_ref_chars - lo) / width)
                idx = min(max(idx, 0), bins - 1)
                buckets[idx].append(float(r.cer_light))
            series = []
            for i, b in enumerate(buckets):
                mid = float(lo + (i + 0.5) * width)
                if b:
                    series.append([mid, float(np.mean(b)), len(b)])
                else:
                    series.append([mid, 0.0, 0])
            cer_vs_length[m] = series

    # --------------------------------------------------------------------- #
    # per-(model, dataset) CER quantiles
    # --------------------------------------------------------------------- #
    cer_distribution: dict[str, dict] = {}
    md: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in ok_recs:
        md[r.model][r.dataset].append(float(r.cer_light))
    for m in sorted(md):
        cer_distribution[m] = {}
        for dsname in sorted(md[m]):
            arr = np.asarray(md[m][dsname], dtype=float)
            cer_distribution[m][dsname] = {
                "min": float(arr.min()),
                "q1": float(np.percentile(arr, 25)),
                "median": float(np.percentile(arr, 50)),
                "q3": float(np.percentile(arr, 75)),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "n": int(arr.size),
            }

    # --------------------------------------------------------------------- #
    # per-model empirical CDF
    # --------------------------------------------------------------------- #
    ecdf: dict[str, list] = {}
    pm: dict[str, list] = defaultdict(list)
    for r in ok_recs:
        pm[r.model].append(float(r.cer_light))
    for m in sorted(pm):
        xs = sorted(pm[m])
        n = len(xs)
        pts: list[list[float]] = []
        i = 0
        while i < n:
            v = xs[i]
            j = i
            while j < n and xs[j] == v:
                j += 1
            pts.append([float(v), float(j) / float(n)])
            i = j
        if len(pts) > _ECDF_MAX_POINTS:
            last = len(pts) - 1
            idxs = sorted({
                round(k * last / (_ECDF_MAX_POINTS - 1))
                for k in range(_ECDF_MAX_POINTS)
            })
            pts = [pts[k] for k in idxs]
        ecdf[m] = pts

    return {
        "consensus_hard": consensus_hard[:cap],
        "consensus_easy": consensus_easy[:cap],
        "discriminative": discriminative[:cap],
        "suspected_bad_gt": suspected_bad_gt[:cap],
        "cer_vs_length": cer_vs_length,
        "cer_distribution": cer_distribution,
        "ecdf": ecdf,
    }
