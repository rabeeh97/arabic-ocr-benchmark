"""Task 9: ``model_landscape()`` — part 1 of the model-comparison reduction.

This module owns the ``model_landscape(rows, records, *, seed, b)`` signature
that Tasks 10 and 11 extend (they append more keys to the same returned dict).

Part 1 populates three keys:

- ``agreement``: pairwise line-identity and per-line CER correlation between
  models, with the model axis ordered by a pure-numpy average-linkage
  agglomerative clustering on ``1 - identity``.
- ``metric_correlation``: Pearson correlation across all OK ``(model, sample)``
  score rows for the five headline metrics.
- ``profile_decomposition``: each model's micro CER split into a real /
  orthographic / diacritic budget using the raw, light and diac-insensitive
  normalization profiles.

Every one of the four keys is driven from the SAME OK set — ``_ok(records)``
joined back to a ``{(model, sample_id): PredRow}`` map — so a row that
``classify_failure`` rejected (``empty`` / ``prompt_echo`` / content refusal /
``meta_commentary`` / ``wrong_script``) never leaks into any matrix even though
the harness still attached a ``scores`` dict to it.

Every scalar in the returned structure is a plain Python ``float`` / ``int`` so
the whole dict survives ``json.dumps``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..normalize import normalize
from .charstats import _ok

_METRICS = ["cer_light", "wer_light", "mer_light", "wil_light", "diac_f1"]


def _avg_linkage_order(dist: np.ndarray) -> list[int]:
    """Return the leaf order of an average-linkage agglomerative clustering of a
    square distance matrix — pure numpy, no scipy.

    Clusters start as singletons; on each step the closest pair of clusters is
    merged and inter-cluster distance is recomputed as the mean over all member
    pairs. The leaf list of the final single cluster is the returned order.
    """
    n = dist.shape[0]
    if n <= 1:
        return list(range(n))
    clusters = {i: [i] for i in range(n)}
    D = dist.astype(float).copy()
    np.fill_diagonal(D, np.inf)
    while len(clusters) > 1:
        ks = list(clusters)
        a, b = min(
            ((x, y) for i, x in enumerate(ks) for y in ks[i + 1:]),
            key=lambda ab: D[ab[0], ab[1]],
        )
        merged = clusters[a] + clusters[b]
        for c in list(clusters):
            if c in (a, b):
                continue
            m = np.mean([dist[x, y] for x in merged for y in clusters[c]])
            D[a, c] = D[c, a] = m
        clusters[a] = merged
        del clusters[b]
        for c in list(clusters):
            D[b, c] = D[c, b] = np.inf
    return next(iter(clusters.values()))


def _pearson(x, y) -> float:
    """Pearson r as a plain float; 0.0 when undefined (< 2 points or a
    zero-variance input)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def model_landscape(
    rows, records, *, seed: int = 1234, b: int | None = None
) -> dict:
    rng = np.random.default_rng(seed)  # noqa: F841  -- Task 10 uses this.
    _ = b  # bootstrap-size override, unused until Task 10.

    # Single OK gate + single row lookup, shared by all four keys.
    ok = _ok(records)
    rows_by_key = {(row.model, row.sample_id): row for row in rows}
    ok_pairs = [
        (r.model, r.sample_id)
        for r in ok
        if (r.model, r.sample_id) in rows_by_key
    ]

    models = sorted({m for m, _sid in ok_pairs})
    n = len(models)

    # ---- per-model line data (OK rows only) --------------------------------- #
    pred_light: dict[str, dict[str, str]] = defaultdict(dict)
    cer_by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for m, sid in ok_pairs:
        row = rows_by_key[(m, sid)]
        sc = row.scores or {}
        pred_light[m][sid] = normalize(row.pred, "light")
        if "cer_light" in sc:
            cer_by_model[m][sid] = float(sc["cer_light"])

    identity = [[0.0] * n for _ in range(n)]
    cer_corr = [[0.0] * n for _ in range(n)]
    for i, mi in enumerate(models):
        identity[i][i] = 1.0
        cer_corr[i][i] = 1.0
        for j in range(i + 1, n):
            mj = models[j]

            shared = set(pred_light[mi]) & set(pred_light[mj])
            if shared:
                eq = sum(
                    1 for s in shared if pred_light[mi][s] == pred_light[mj][s]
                )
                val = eq / len(shared)
            else:
                val = 0.0
            identity[i][j] = identity[j][i] = float(val)

            sc = sorted(set(cer_by_model[mi]) & set(cer_by_model[mj]))
            r = _pearson(
                [cer_by_model[mi][s] for s in sc],
                [cer_by_model[mj][s] for s in sc],
            )
            cer_corr[i][j] = cer_corr[j][i] = r

    dist = 1.0 - np.array(identity, dtype=float) if n else np.zeros((0, 0))
    order = _avg_linkage_order(dist)
    ordered_models = [models[k] for k in order]

    def _reorder(mat):
        return [[float(mat[a][b]) for b in order] for a in order]

    agreement = {
        "models": ordered_models,
        "identity": _reorder(identity),
        "cer_corr": _reorder(cer_corr),
    }

    # ---- metric_correlation (OK rows only) -------------------------------- #
    obs: list[list[float]] = []
    for m, sid in ok_pairs:
        sc = rows_by_key[(m, sid)].scores or {}
        if any(k not in sc for k in _METRICS):
            continue
        obs.append([float(sc[k]) for k in _METRICS])

    m_n = len(_METRICS)
    matrix = [[0.0] * m_n for _ in range(m_n)]
    for a in range(m_n):
        matrix[a][a] = 1.0
    if len(obs) >= 2:
        arr = np.asarray(obs, dtype=float)
        for a in range(m_n):
            for c in range(a + 1, m_n):
                r = _pearson(arr[:, a], arr[:, c])
                matrix[a][c] = matrix[c][a] = r
    metric_correlation = {"metrics": list(_METRICS), "matrix": matrix}

    # ---- profile_decomposition (OK rows only) --------------------------- #
    acc: dict[str, dict[str, float]] = defaultdict(
        lambda: {"er": 0.0, "rr": 0.0, "el": 0.0, "rl": 0.0, "ed": 0.0, "rd": 0.0}
    )
    for m, sid in ok_pairs:
        row = rows_by_key[(m, sid)]
        sc = row.scores or {}
        lr_raw = len(normalize(row.gt, "raw"))
        lr_light = len(normalize(row.gt, "light"))
        lr_di = len(normalize(row.gt, "diac_insensitive"))
        a = acc[m]
        a["er"] += round(float(sc.get("cer_raw", 0.0)) * lr_raw)
        a["rr"] += lr_raw
        a["el"] += round(float(sc.get("cer_light", 0.0)) * lr_light)
        a["rl"] += lr_light
        a["ed"] += round(float(sc.get("cer_diac_insensitive", 0.0)) * lr_di)
        a["rd"] += lr_di

    profile_decomposition: dict[str, dict[str, float]] = {}
    for m in sorted(acc):
        a = acc[m]
        cer_raw = a["er"] / a["rr"] if a["rr"] else 0.0
        cer_light = a["el"] / a["rl"] if a["rl"] else 0.0
        cer_di = a["ed"] / a["rd"] if a["rd"] else 0.0
        profile_decomposition[m] = {
            "real": float(max(0.0, cer_di)),
            "orthographic": float(max(0.0, cer_raw - cer_light)),
            "diacritic": float(max(0.0, cer_light - cer_di)),
        }

    return {
        "agreement": agreement,
        "metric_correlation": metric_correlation,
        "profile_decomposition": profile_decomposition,
    }
