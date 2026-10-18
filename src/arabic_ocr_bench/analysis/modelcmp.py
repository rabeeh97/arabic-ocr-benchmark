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

from collections import Counter, defaultdict
from itertools import zip_longest

import numpy as np

from ..metrics import cer as _cer
from ..normalize import normalize
from .charstats import _ok

_METRICS = ["cer_light", "wer_light", "mer_light", "wil_light", "diac_f1"]

_BOOTSTRAP_B = 2000


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
    rng = np.random.default_rng(seed)
    bootstrap_b = _BOOTSTRAP_B if b is None else max(1, int(b))

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

    # ---- bootstrap_cer (OK records only) ------------------------------- #
    # Percentile bootstrap of each (model, dataset)'s micro-CER over its OK
    # records. RNG is drawn here FIRST, iterating sorted models then, within
    # each, sorted datasets -- one ``rng.integers`` call per non-degenerate
    # cell (a cell whose reference-char total is 0 draws nothing).
    B = bootstrap_b
    md_pairs: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in ok:
        err = round(r.cer_light * r.len_ref_chars)
        md_pairs[r.model][r.dataset].append((int(err), int(r.len_ref_chars)))

    bootstrap_cer: dict[str, dict[str, dict]] = {}
    for m in sorted(md_pairs):
        per_ds: dict[str, dict] = {}
        for ds in sorted(md_pairs[m]):
            pairs = md_pairs[m][ds]
            cnt = len(pairs)
            err_arr = np.array([p[0] for p in pairs], dtype=float)
            ref_arr = np.array([p[1] for p in pairs], dtype=float)
            ref_total = float(ref_arr.sum())
            if ref_total == 0.0:
                per_ds[ds] = {
                    "point": 0.0, "lo": 0.0, "hi": 0.0, "n": int(cnt)
                }
                continue
            point = float(err_arr.sum() / ref_total)
            idx = rng.integers(0, cnt, size=(B, cnt))
            err_s = err_arr[idx].sum(axis=1)
            ref_s = ref_arr[idx].sum(axis=1)
            safe = np.where(ref_s == 0.0, 1.0, ref_s)
            micro = np.where(ref_s == 0.0, 0.0, err_s / safe)
            per_ds[ds] = {
                "point": point,
                "lo": float(np.percentile(micro, 2.5)),
                "hi": float(np.percentile(micro, 97.5)),
                "n": int(cnt),
            }
        bootstrap_cer[m] = per_ds

    # ---- rank_stability (OK records only) ----------------------------- #
    # Per dataset: one joint resample of that dataset's OK sample set per
    # replicate; each model scored on the resampled samples it actually has,
    # then models ranked 1..k by ascending replicate micro-CER (ties -> stable
    # by model name). RNG is drawn AFTER every bootstrap_cer cell, iterating
    # datasets in sorted order -- one ``rng.integers`` call per dataset.
    datasets_sorted = sorted({r.dataset for r in ok})
    rank_stability: dict = {"datasets": list(datasets_sorted)}
    rs_by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for ds in datasets_sorted:
        recs_ds = [r for r in ok if r.dataset == ds]
        samples = sorted({r.sample_id for r in recs_ds})
        s_ix = {s: i for i, s in enumerate(samples)}
        ns = len(samples)
        ds_models = sorted({r.model for r in recs_ds})
        k = len(ds_models)
        m_ix = {m: i for i, m in enumerate(ds_models)}

        err_mat = np.zeros((k, ns), dtype=float)
        ref_mat = np.zeros((k, ns), dtype=float)
        for r in recs_ds:
            i = m_ix[r.model]
            j = s_ix[r.sample_id]
            err_mat[i, j] += round(r.cer_light * r.len_ref_chars)
            ref_mat[i, j] += r.len_ref_chars

        idx = rng.integers(0, ns, size=(B, ns))
        err_s = err_mat[:, idx].sum(axis=2)  # (k, B)
        ref_s = ref_mat[:, idx].sum(axis=2)  # (k, B)
        # Ruling 12: a model with zero resampled records in a replicate is
        # excluded from that replicate's ranking (and from its own tally),
        # never handed rank 1 off a no-data draw. Forcing its replicate micro
        # to +inf parks it behind every model that actually drew data, so the
        # data-having models occupy ranks 1..d contiguously; the per-model
        # accumulator then keeps only the replicates where that model itself
        # had data. A k-model dataset thus legitimately yields a <k-model
        # ranking in replicates that miss a sparse model's samples.
        safe = np.where(ref_s == 0.0, 1.0, ref_s)
        micro = np.where(ref_s > 0.0, err_s / safe, np.inf)
        order = np.argsort(micro, axis=0, kind="stable")  # (k, B)
        ranks = np.argsort(order, axis=0, kind="stable") + 1  # ranks[model, rep]

        for m, i in m_ix.items():
            drew = ref_s[i] > 0.0  # replicates where this model had >=1 record
            rr = ranks[i][drew]
            if rr.size == 0:
                # Ranked in zero replicates (astronomically unlikely at
                # B >= 2000, n >= 1): omit this (model, dataset) entry.
                continue
            counts = np.bincount(rr, minlength=k + 1)
            rs_by_model[m][ds] = {
                "best": int(rr.min()),
                "worst": int(rr.max()),
                "mode": int(np.argmax(counts)),
            }
    for m in sorted(rs_by_model):
        rank_stability[m] = rs_by_model[m]

    # ---- ensemble: oracle / ROVER ceiling + greedy marginal --------------- #
    # Deterministic, no RNG. All aggregation is micro CER-light in the module's
    # house style: ``sum(round(cer_i * len_ref_i)) / sum(len_ref_i)``.
    ok_sids_by_model: dict[str, set] = defaultdict(set)
    rec_by_key: dict[tuple[str, str], object] = {}
    for r in ok:
        ok_sids_by_model[r.model].add(r.sample_id)
        rec_by_key[(r.model, r.sample_id)] = r
    ens_models = sorted(ok_sids_by_model)

    common = (
        set.intersection(*(ok_sids_by_model[m] for m in ens_models))
        if ens_models
        else set()
    )
    common_sorted = sorted(common)

    def _micro(pairs) -> float:
        num = sum(round(c * length) for c, length in pairs)
        den = sum(length for _c, length in pairs)
        return float(num / den) if den else 0.0

    if common_sorted:
        # Per-line reference length: taken from the first model (identical GT
        # across models for a shared line), used as the common denominator so
        # ``oracle`` and each ``single`` divide by the same total.
        ref_len = {
            sid: int(rec_by_key[(ens_models[0], sid)].len_ref_chars)
            for sid in common_sorted
        }

        single = {
            m: _micro(
                [
                    (
                        rec_by_key[(m, sid)].cer_light,
                        rec_by_key[(m, sid)].len_ref_chars,
                    )
                    for sid in common_sorted
                ]
            )
            for m in ens_models
        }

        def _oracle_cer(subset) -> float:
            num = 0
            den = 0
            for sid in common_sorted:
                best_c = min(
                    rec_by_key[(m, sid)].cer_light for m in subset
                )
                length = ref_len[sid]
                num += round(best_c * length)
                den += length
            return float(num / den) if den else 0.0

        oracle = _oracle_cer(ens_models)

        # ROVER: positional character-level majority vote over the models'
        # light-normalized predictions (zip_longest columns, modal char per
        # column, drop a column whose winner is the fill sentinel). Modal
        # char ties break by codepoint so the vote is deterministic.
        _FILL = "\x00"

        def _vote(preds) -> str:
            out_chars: list[str] = []
            for col in zip_longest(*preds, fillvalue=_FILL):
                counts = Counter(col)
                ch = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                if ch != _FILL:
                    out_chars.append(ch)
            return "".join(out_chars)

        r_num = 0
        r_den = 0
        for sid in common_sorted:
            avail = [
                rows_by_key[(m, sid)]
                for m in ens_models
                if (m, sid) in rows_by_key
            ]
            if not avail:
                continue  # no prediction text on hand for this line
            gt_light = normalize(avail[0].gt, "light")
            preds = [normalize(row.pred, "light") for row in avail]
            voted = _vote(preds)
            length = len(gt_light)
            r_num += round(_cer(gt_light, voted) * length)
            r_den += length
        rover = float(r_num / r_den) if r_den else 0.0

        # Greedy marginal: start at the best single model, then repeatedly add
        # whichever remaining model most lowers the subset's oracle CER.
        best = min(ens_models, key=lambda m: (single[m], m))
        selected = [best]
        marginal: list[list] = [[best, single[best]]]
        remaining = [m for m in ens_models if m != best]
        while remaining:
            cand = sorted(
                ((_oracle_cer(selected + [m]), m) for m in remaining),
                key=lambda t: (t[0], t[1]),
            )
            oc, pick = cand[0]
            selected.append(pick)
            remaining.remove(pick)
            marginal.append([pick, oc])

        ensemble = {
            "single": {m: float(single[m]) for m in ens_models},
            "oracle": float(oracle),
            "rover": float(rover),
            "marginal": marginal,
        }
    else:
        # No line is OK for every model -> no shared basis for an ensemble
        # ceiling. Fall back to each model's overall micro CER-light.
        overall = {}
        for m in ens_models:
            recs_m = [rec_by_key[(m, sid)] for sid in ok_sids_by_model[m]]
            overall[m] = _micro(
                [(r.cer_light, r.len_ref_chars) for r in recs_m]
            )
        ensemble = {
            "single": {m: float(overall[m]) for m in ens_models},
            "oracle": 0.0,
            "rover": 0.0,
            "marginal": [],
        }

    # ---- win_rate: pairwise per-line CER-light win fraction --------------- #
    wr_models = sorted(ok_sids_by_model)
    wn = len(wr_models)
    win_matrix = [[0.5] * wn for _ in range(wn)]
    # wins[a][b] = wins of a vs b (tie -> 0.5 each); games[a][b] = shared lines.
    bt_wins = np.zeros((wn, wn), dtype=float)
    bt_games = np.zeros((wn, wn), dtype=float)
    for a in range(wn):
        for c in range(a + 1, wn):
            ma, mc = wr_models[a], wr_models[c]
            shared = sorted(ok_sids_by_model[ma] & ok_sids_by_model[mc])
            bt_games[a, c] = bt_games[c, a] = float(len(shared))
            if not shared:
                win_matrix[a][c] = win_matrix[c][a] = 0.5
                continue
            wa = 0.0
            wc = 0.0
            for sid in shared:
                ca = rec_by_key[(ma, sid)].cer_light
                cc = rec_by_key[(mc, sid)].cer_light
                if ca < cc:
                    wa += 1.0
                elif ca > cc:
                    wc += 1.0
                else:
                    wa += 0.5
                    wc += 0.5
            bt_wins[a, c] = wa
            bt_wins[c, a] = wc
            win_matrix[a][c] = float(wa / len(shared))
            win_matrix[c][a] = float(wc / len(shared))
    win_rate = {"models": list(wr_models), "matrix": win_matrix}

    # ---- bradley_terry: MM-algorithm strengths from pairwise wins -------- #
    bradley_terry: dict[str, float] = {}
    if wn:
        w_i = bt_wins.sum(axis=1)
        p = np.ones(wn, dtype=float)
        for _ in range(50):
            new_p = np.empty(wn, dtype=float)
            for i in range(wn):
                denom = 0.0
                for j in range(wn):
                    if j == i:
                        continue
                    denom += bt_games[i, j] / (p[i] + p[j] + 1e-12)
                new_p[i] = w_i[i] / (denom + 1e-12)
            new_p = np.where(new_p <= 0.0, 1e-12, new_p)
            new_p = new_p / np.exp(np.mean(np.log(new_p)))
            p = new_p
        bradley_terry = {
            m: float(p[i]) for i, m in enumerate(wr_models)
        }

    return {
        "agreement": agreement,
        "metric_correlation": metric_correlation,
        "profile_decomposition": profile_decomposition,
        "bootstrap_cer": bootstrap_cer,
        "rank_stability": rank_stability,
        "ensemble": ensemble,
        "win_rate": win_rate,
        "bradley_terry": bradley_terry,
    }
