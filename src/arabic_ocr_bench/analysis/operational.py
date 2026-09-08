"""Task 12: ``operational()`` — cost / latency / failure / coverage analysis.

Reduces the raw ``PredRow`` list (INCLUDING errored / refused rows) and the
index-aligned ``AlignmentRecord`` list into one JSON-serialisable operational
summary:

- ``pareto``          : per-model micro ``cer_light`` vs. ``usd_per_1k`` with a
                        free flag and an O(n^2) domination check for the
                        accuracy/cost frontier.
- ``latency``         : p50 / p95 / mean / n over each model's OK-row latencies.
- ``failure_causes``  : ``{model: {dataset: {class: count}}}`` over every
                        record's ``failure_class`` — all eight
                        :data:`FAILURE_CLASSES` present, zero-filled.
- ``retries``         : per-provider call / retry / rate-limit counts plus a
                        histogram of the ``attempts`` field over every row.
- ``empty_rate``      : per-model share of rows that produced empty output.
- ``coverage``        : a ``(model, dataset)`` grid of ``[scored, attempted]``
                        distinct sample ids.

``rows`` carries latency / cost / attempts / coverage; ``records[i]`` carries
the failure taxonomy for ``rows[i]`` (a stub record with the right
``failure_class`` stands in for each errored / refused row).

Every number in the returned structure is a plain Python ``int`` / ``float`` so
the whole dict survives ``json.dumps``; iteration is sorted for determinism.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .refusal import FAILURE_CLASSES

_REPO = Path(__file__).resolve().parents[3]


def _priced_models() -> set[str]:
    """Model names with a real pricing block in ``configs/models.yaml`` — the
    same "priced" rule as ``scripts/make_leaderboard.collect_costs``: the model
    is defined in models.yaml and ``pricing.input_usd_per_1m`` is not None."""
    try:
        cfg = yaml.safe_load((_REPO / "configs" / "models.yaml").read_text()) or {}
    except OSError:
        return set()
    return {
        name
        for name, c in cfg.items()
        if isinstance(c, dict)
        and (c.get("pricing") or {}).get("input_usd_per_1m") is not None
    }


def _provider(model: str) -> str:
    if model.startswith("openrouter"):
        return "openrouter"
    if model.startswith("nvidia"):
        return "nvidia"
    if model.startswith("gemini"):
        return "gemini"
    return model.split("-")[0]


def _pareto(rows, records, priced: set[str]) -> dict:
    # Only models with >= 1 OK record appear on the frontier (mirrors _latency):
    # a model whose every row errored would otherwise get cer_light = 0.0 /
    # usd_per_1k = 0.0 from the divide-by-zero guards and spuriously dominate.
    models = sorted({rec.model for rec in records if rec.failure_class == "ok"})

    # micro cer_light over that model's OK records
    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    for rec in records:
        if rec.failure_class != "ok":
            continue
        num[rec.model] += round(rec.cer_light * rec.len_ref_chars)
        den[rec.model] += rec.len_ref_chars

    # mean cost over that model's BILLABLE OK rows
    costs: dict[str, list[float]] = defaultdict(list)
    for row, rec in zip(rows, records):
        if rec.failure_class != "ok":
            continue
        if row.model not in priced:
            continue
        costs[row.model].append(float(row.cost_usd))

    stats: dict[str, dict] = {}
    for m in models:
        cer = float(num[m] / den[m]) if den[m] else 0.0
        cs = costs.get(m, [])
        usd_per_1k = 1000.0 * (sum(cs) / len(cs)) if cs else 0.0
        stats[m] = {
            "cer_light": cer,
            "usd_per_1k": float(usd_per_1k),
            "free": bool(usd_per_1k == 0.0 or m not in priced),
        }

    for m in models:
        me = stats[m]
        dominated = any(
            o != m
            and stats[o]["cer_light"] < me["cer_light"]
            and stats[o]["usd_per_1k"] <= me["usd_per_1k"]
            for o in models
        )
        me["on_frontier"] = bool(not dominated)
    return {m: stats[m] for m in models}


def _latency(rows, records) -> dict:
    by_model: dict[str, list[float]] = defaultdict(list)
    for row, rec in zip(rows, records):
        if rec.failure_class == "ok":
            by_model[row.model].append(float(row.latency_ms))
    out: dict[str, dict] = {}
    for m in sorted(by_model):
        arr = np.asarray(by_model[m], dtype=float)
        out[m] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()),
            "n": int(arr.size),
        }
    return out


def _failure_causes(rows, records) -> dict:
    cells: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row, rec in zip(rows, records):
        cells[row.model][row.dataset][rec.failure_class] += 1
    out: dict[str, dict] = {}
    for m in sorted(cells):
        out[m] = {}
        for ds in sorted(cells[m]):
            c = cells[m][ds]
            out[m][ds] = {cls: int(c.get(cls, 0)) for cls in FAILURE_CLASSES}
    return out


def _retries(rows, records) -> dict:
    by_provider: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "retried": 0, "rate_limited": 0}
    )
    attempts_hist: Counter = Counter()
    for row, rec in zip(rows, records):
        p = _provider(row.model)
        b = by_provider[p]
        b["calls"] += 1
        if row.attempts > 1:
            b["retried"] += 1
        if rec.failure_class == "rate_limited":
            b["rate_limited"] += 1
        attempts_hist[int(row.attempts)] += 1
    return {
        "by_provider": {
            p: {k: int(v) for k, v in by_provider[p].items()}
            for p in sorted(by_provider)
        },
        "attempts_hist": {
            str(k): int(attempts_hist[k]) for k in sorted(attempts_hist)
        },
    }


def _empty_rate(rows, records) -> dict:
    total: Counter = Counter()
    empty: Counter = Counter()
    for row, rec in zip(rows, records):
        total[row.model] += 1
        if rec.failure_class == "empty":
            empty[row.model] += 1
    out: dict[str, dict] = {}
    for m in sorted(total):
        n = int(total[m])
        e = int(empty[m])
        out[m] = {"empty": e, "n": n, "rate": float(e / n) if n else 0.0}
    return out


def _coverage(rows, records) -> dict:
    models = sorted({r.model for r in rows})
    datasets = sorted({r.dataset for r in rows})
    attempted: dict[tuple[str, str], set] = defaultdict(set)
    scored: dict[tuple[str, str], set] = defaultdict(set)
    for row, rec in zip(rows, records):
        key = (row.model, row.dataset)
        attempted[key].add(row.sample_id)
        if rec.failure_class == "ok":
            scored[key].add(row.sample_id)
    grid = [
        [
            [len(scored[(m, ds)]), len(attempted[(m, ds)])]
            for ds in datasets
        ]
        for m in models
    ]
    return {"models": models, "datasets": datasets, "grid": grid}


def operational(rows, records) -> dict:
    """Cost / latency / failure / coverage operational summary.

    ``rows`` is the full :func:`iter_pred_rows` list (errored rows included);
    ``records`` is ``[build_record(r) for r in rows]``, index-aligned.
    """
    if len(rows) != len(records):
        raise ValueError("rows and records must be index-aligned")
    priced = _priced_models()
    return {
        "pareto": _pareto(rows, records, priced),
        "latency": _latency(rows, records),
        "failure_causes": _failure_causes(rows, records),
        "retries": _retries(rows, records),
        "empty_rate": _empty_rate(rows, records),
        "coverage": _coverage(rows, records),
    }
