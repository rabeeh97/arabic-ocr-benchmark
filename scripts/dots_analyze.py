"""Image-side studies on the span features (checklist 2.1 PAW gaps, 2.4 dot styles, 2.5 vertical stacking, 2.18 dot
detection and a second baseline estimator). Rules were written first: results/characterize/dots/prereg.json.

    .geomvenv/bin/python scripts/dots_extract.py --native     # features.npz
    .geomvenv/bin/python scripts/dots_analyze.py              # image_studies.json, gallery_picks.json

Every reading of every family (analyze_dots.fam) on the panel lines with spans; intervals are line-cluster bootstraps
(lines resampled, all readings of a line move together); BH q = 0.1 via eval_stats.bh over the cells of one table.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

import numpy as np

from dots_common import OUT, ci, load_features, load_store, rasm_of, readings
from eval_stats import bh, boot_p

REPS = 1000
BINS = np.linspace(0, 4, 201)
STYLE_CELLS = [(1, "displaced"), (1, "touching"), (2, "dash"), (2, "blob"), (2, "displaced"), (2, "touching"), (3, "caret"),
               (3, "partial"), (3, "dash"), (3, "displaced"), (3, "touching")]
STACK_T = (1.0, 0.5)  # x-heights (spec; sensitivity)


def weights(n: int, reps: int = REPS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Wt = np.zeros((reps + 1, n), np.float32)
    Wt[0] = 1.0  # row 0 = the point estimate
    for r in range(reps):
        Wt[r + 1] = np.bincount(rng.integers(0, n, n), minlength=n)
    return Wt


def mh(line: np.ndarray, stratum: np.ndarray, expo: np.ndarray, event: np.ndarray, Wt: np.ndarray, nl: int):
    """Mantel-Haenszel rate ratio (exposed vs not) stratified by `stratum`, rows of Wt = line-cluster bootstrap weights
    (row 0 = point estimate). Returns (ratio per rep, rate exposed, rate unexposed, n exposed)."""
    _, st = np.unique(stratum, return_inverse=True)
    ns = int(st.max()) + 1
    idx = line * ns + st
    Ms = []
    for flt in (expo, ~expo):
        Ms.append(np.bincount(idx[flt], weights=event[flt].astype(float), minlength=nl * ns).reshape(nl, ns))
        Ms.append(np.bincount(idx[flt], minlength=nl * ns).reshape(nl, ns).astype(float))
    a_, n1_, c_, n0_ = (Wt @ m for m in Ms)
    Nn = n1_ + n0_
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.where(Nn > 0, a_ * n0_ / Nn, 0).sum(1)
        den = np.where(Nn > 0, c_ * n1_ / Nn, 0).sum(1)
    return num / np.maximum(1e-12, den), float(a_[0].sum() / max(1, n1_[0].sum())), float(c_[0].sum() / max(1, n0_[0].sum())), int(n1_[0].sum())


def hist_stats(Hs: np.ndarray, Hu: np.ndarray, Hw: np.ndarray):
    """Rows = bootstrap reps. Medians (bin centres) and AUC(word > split), AUC(split > unsplit)."""
    cen = (BINS[:-1] + BINS[1:]) / 2
    cen = np.append(cen, 4.0)  # overflow bin

    def med(H):
        c = np.cumsum(H, 1)
        return cen[np.minimum(len(cen) - 1, (c < c[:, -1:] / 2).sum(1))]

    def auc(Ha, Hb):  # P(a > b) + 0.5 P(a = b)
        cb = np.cumsum(Hb, 1) - Hb
        return (Ha * (cb + 0.5 * Hb)).sum(1) / np.maximum(1e-9, Ha.sum(1) * Hb.sum(1))
    return med(Hs), med(Hu), med(Hw), auc(Hw, Hs), auc(Hs, Hu), auc(Hw, Hu)


def main() -> None:
    S, fams = load_store()
    T = load_features()
    L, B, N = T["L"], T["B"], T["N"]
    keys = list(N["sample_key"])
    lix = {k: i for i, k in enumerate(keys)}
    nl = len(keys)
    xh, sw, dsn = N["xh"].astype(float), N["sw"].astype(float), N["dataset"]
    # per-line structures
    bnd = defaultdict(list)
    for k, a, b, kind, gp in zip(B["sample_key"], B["gi_a"], B["gi_b"], B["kind"], B["gap_px"]):
        bnd[k].append((int(a), int(b), str(kind), float(gp)))
    let = {}
    for j, (k, gi) in enumerate(zip(L["sample_key"], L["gi"])):
        let[(k, int(gi))] = j
    dotted = defaultdict(list)
    for j in np.flatnonzero(L["n_exp"] > 0):
        if L["style"][j]:
            dotted[L["sample_key"][j]].append(j)
    cy = L["cy_body"].astype(float)
    res = {"prereg": "results/characterize/dots/prereg.json", "families": {}}
    picks = {"gaps": [], "styles": [], "stack": []}
    rnd = random.Random(0)
    Wt = weights(nl)
    pair_code: dict[tuple, int] = {}

    # ---------- 2.18 validation of dot detection on spans, and baseline estimators (GT only) ----------
    val = {}
    for d in sorted(set(dsn)):
        m_d = np.isin(L["sample_key"], [k for k in keys if dsn[lix[k]] == d])
        dm = m_d & (L["n_exp"] > 0)
        um = m_d & (L["n_exp"] == 0) & (L["side"] == "above")
        hit, fp = float(np.mean(L["n_small"][dm] > 0)), float(np.mean(L["n_small"][um] > 0))
        near = float(np.mean(np.isin(L["style"][dm], ["separate", "dash", "blob", "caret", "partial", "displaced"])))
        val[d] = {"dotted_letters": int(dm.sum()), "hit_in_span": hit, "hit_within_one_letter": near, "undotted_letters": int(um.sum()),
                  "false_dot_rate": fp, "usable": bool(hit >= 0.8 and fp <= 0.2)}
    # baselines: dot side accuracy for each estimator (band bottom vs Abu-Ain-style sub-word)
    corr = np.zeros((2, 2, nl))  # estimator x (correct, n) x line
    li_of = np.array([lix[k] for k in L["sample_key"]])
    for j in np.flatnonzero((L["n_exp"] > 0) & np.isfinite(L["dot_y"].astype(float))):
        i = li_of[j]
        y = float(L["dot_y"][j])
        below = L["side"][j] == "below"
        for e, base in enumerate((float(N["band_bot"][i]), float(N["base_sub"][i]))):
            if not np.isfinite(base):
                continue
            corr[e, 0, i] += (y > base) == below
            corr[e, 1, i] += 1
    acc = (Wt @ corr[:, 0].T) / np.maximum(1, Wt @ corr[:, 1].T)  # reps x estimators
    dlt = acc[:, 1] - acc[:, 0]
    res["baseline_2_18"] = {"band_bottom_acc": float(acc[0, 0]), "band_ci": ci(acc[1:, 0]), "subword_acc": float(acc[0, 1]),
                            "subword_ci": ci(acc[1:, 1]), "delta_sub_minus_band": float(dlt[0]), "delta_ci": ci(dlt[1:]),
                            "dots_scored": int(corr[0, 1].sum()),
                            "outcome": "sub-word estimator preferred" if np.percentile(dlt[1:], 2.5) > 0 else
                            "band estimator stays" if np.percentile(dlt[1:], 97.5) < 0 else "no difference: band estimator stays"}
    res["dot_detection_2_18"] = val
    print("2.18 baseline", json.dumps(res["baseline_2_18"]))
    for d, v in val.items():
        print(f"  {d:14s} hit {v['hit_in_span']:.2f} (within one letter {v['hit_within_one_letter']:.2f}) false {v['false_dot_rate']:.2f} usable {v['usable']}")

    p_style, p_cells = [], []
    for F, rows in fams.items():
        rows = [r for r in rows if r["sample_key"] in lix]
        if len({r["sample_key"] for r in rows}) < 150:
            continue
        nb = len(BINS)
        Hgap = np.zeros((4, nl, nb), np.float32)  # unsplit paw, split paw, word kept, word merged
        sty = defaultdict(lambda: np.zeros((2, nl), np.float32))
        sty_rows = []  # (line, stratum = dataset x letter x form, n_exp, style, error)
        n_stk = {t: 0 for t in STACK_T}
        stk_rows = []  # (line, pair code, stacked@1.0, stacked@0.5, event, transposition)
        n_read = 0
        for key, run, g, read, ins in readings(S, rows):
            i = lix[key]
            n_read += 1
            x, s_ = xh[i], sw[i]
            for a, b, kind, gp in bnd[key]:
                gx = gp / x
                bi = min(nb - 1, int(np.searchsorted(BINS, gx, side="right")) - 1) if gx < 4 else nb - 1
                if kind == "paw":
                    split = " " in ins.get(b, "")
                    Hgap[int(split), i, bi] += 1
                    if split and rnd.random() < 0.02:
                        picks["gaps"].append({"key": key, "run": run, "family": F, "gi_a": a, "gi_b": b, "gap_xh": gx, "gap_sw": gp / s_})
                elif kind == "word":
                    merged = read[a + 1] != " "
                    Hgap[2 + int(merged), i, bi] += 1
                else:  # joined: stacking
                    ja, jb = let.get((key, a)), let.get((key, b))
                    if ja is None or jb is None or not (np.isfinite(cy[ja]) and np.isfinite(cy[jb])):
                        continue
                    dy = (cy[ja] - cy[jb]) / x
                    trans = read[a] == g[b] and read[b] == g[a] and g[a] != g[b]
                    trans |= read[a] is None and g[a] in ins.get(b + 1, "")  # A moved after B
                    trans |= read[b] is None and g[b] in ins.get(a, "")  # B moved before A
                    ev = trans or read[a] is None or read[b] is None
                    pc = pair_code.setdefault((dsn[i], g[a], g[b]), len(pair_code))
                    stk_rows.append((i, pc, dy <= -STACK_T[0], dy <= -STACK_T[1], ev, trans))
                    if dy <= -STACK_T[0] and rnd.random() < 0.05:
                        picks["stack"].append({"key": key, "run": run, "family": F, "gi_a": a, "gi_b": b, "dy_xh": dy, "event": bool(ev),
                                               "transposition": bool(trans)})
            for j in dotted.get(key, ()):
                gi = int(L["gi"][j])
                if gi >= len(g):
                    continue
                h, form = read[gi], str(L["form"][j])
                err = h is not None and h != g[gi] and h != " " and rasm_of(h, form) is not None and rasm_of(h, form) == rasm_of(g[gi], form)
                c = sty[(int(L["n_exp"][j]), str(L["style"][j]))]
                c[0, i] += err
                c[1, i] += 1
                sty_rows.append((i, pair_code.setdefault((dsn[i], g[gi], form), len(pair_code)), int(L["n_exp"][j]), str(L["style"][j]), bool(err)))
                if str(L["style"][j]) != "separate" and rnd.random() < 0.01:
                    picks["styles"].append({"key": key, "run": run, "family": F, "gi": gi, "style": str(L["style"][j]), "n_exp": int(L["n_exp"][j]),
                                            "dot_error": bool(err), "read": h})
        fr = {"readings": n_read, "lines": len({r["sample_key"] for r in rows})}
        # ---- 2.1 gaps ----
        gaps = {}
        for scope in ["all", "no_rasam"] + sorted(set(dsn)):
            msk = (np.ones(nl, bool) if scope == "all" else ~np.isin(dsn, ["rasam1", "rasam2"]) if scope == "no_rasam" else (dsn == scope))
            Wm = Wt * msk[None, :]
            Hs, Hu, Hk, Hm = (Wm @ Hgap[c] for c in (1, 0, 2, 3))
            if Hs[0].sum() < 30:
                continue
            ms, mu, mw, auc_ws, auc_su, auc_wu = hist_stats(Hs, Hu, Hk + Hm)
            mk, mm = hist_stats(Hm, Hk, Hk)[0:2]
            d_ = ms - mu
            auc_hi, auc_lo = np.percentile(auc_ws[1:], 97.5), np.percentile(auc_ws[1:], 2.5)
            d_lo, d_hi = np.percentile(d_[1:], [2.5, 97.5])
            out = ("overlap: spacing visually ambiguous at the split gaps" if auc_hi < 0.75 else
                   "models over-split" if auc_lo >= 0.75 and d_lo <= 0 else
                   "split gaps wider than other PAW gaps yet separable: partial ambiguity" if auc_lo >= 0.75 and d_lo > 0 else "undecided")
            gaps[scope] = {"split_paw_gaps": float(Hs[0].sum()), "unsplit_paw_gaps": float(Hu[0].sum()), "word_gaps": float((Hk + Hm)[0].sum()),
                           "merged_word_gaps": float(Hm[0].sum()),
                           "median_xh": {"split": float(ms[0]), "unsplit": float(mu[0]), "word": float(mw[0]), "merged_word": float(mk[0]), "kept_word": float(mm[0])},
                           "delta_median_split_minus_unsplit": float(d_[0]), "delta_ci": [float(d_lo), float(d_hi)],
                           "auc_word_gt_split": float(auc_ws[0]), "auc_word_gt_split_ci": ci(auc_ws[1:]),
                           "auc_split_gt_unsplit": float(auc_su[0]), "auc_split_gt_unsplit_ci": ci(auc_su[1:]),
                           "auc_word_gt_unsplit": float(auc_wu[0]),
                           "split_rate_per_paw_gap": float(Hs[0].sum() / max(1, Hs[0].sum() + Hu[0].sum())), "outcome": out}
        fr["paw_gaps_2_1"] = gaps
        # ---- 2.4 styles ----
        st = {}
        SR_i = np.array([r[0] for r in sty_rows])
        SR_s = np.array([r[1] for r in sty_rows])
        SR_n = np.array([r[2] for r in sty_rows])
        SR_y = np.array([r[3] for r in sty_rows])
        SR_e = np.array([r[4] for r in sty_rows])
        for (n, sname) in STYLE_CELLS:
            m = (SR_n == n) & np.isin(SR_y, [sname, "separate"])
            ex = SR_y[m] == sname
            if ex.sum() < 100 or (~ex).sum() < 100:
                continue
            ratio, ra, rr, n1 = mh(SR_i[m], SR_s[m], ex, SR_e[m], Wt[:501], nl)
            lr = np.log(np.clip(ratio[1:], 1e-3, None))
            p = boot_p(float(np.log(max(ratio[0], 1e-3))), lr)
            st[f"{n}:{sname}"] = {"letters": n1, "rate": ra, "separate_rate": rr, "ratio_mh_dataset_letter_form": float(ratio[0]),
                                  "ratio": float(ratio[0]), "ratio_ci": ci(ratio[1:]), "p": p}
            p_style.append(p)
            p_cells.append((F, f"{n}:{sname}"))
        st["prevalence"] = {f"{n}:{s}": float(v[1].sum()) for (n, s), v in sty.items()}
        fr["dot_styles_2_4"] = st
        # ---- 2.5 stacking ----
        if stk_rows:
            A = np.array(stk_rows, dtype=np.int64)
            npc = len(pair_code)
            stk = {}
            for ti, t in enumerate(STACK_T):
                col = 2 + ti
                n1 = int(A[:, col].sum())
                n_stk[t] = n1
                if n1 < 30:
                    stk[str(t)] = {"stacked_pairs": n1}
                    continue
                out = {}
                for evcol, name in ((4, "del_or_transposition"), (5, "transposition")):
                    r_, ra, rn, _ = mh(A[:, 0], A[:, 1], A[:, col] == 1, A[:, evcol] == 1, Wt[:301], nl)
                    out[name] = {"rate_stacked": ra, "rate_not_stacked": rn, "mh_ratio": float(r_[0]), "mh_ci": ci(r_[1:])}
                stk[str(t)] = {"stacked_pairs": n1, "joined_pairs": int(len(A)), **out}
            fr["stacking_2_5"] = stk
        res["families"][F] = fr
        g_ = gaps.get("all", {})
        print(f"\n{F}: readings {n_read}, lines {fr['lines']}")
        if g_:
            print(f"  2.1 med xh split {g_['median_xh']['split']:.2f} unsplit {g_['median_xh']['unsplit']:.2f} word {g_['median_xh']['word']:.2f} | "
                  f"AUC word>split {g_['auc_word_gt_split']:.2f} {np.round(g_['auc_word_gt_split_ci'], 3)} | AUC split>unsplit {g_['auc_split_gt_unsplit']:.2f} "
                  f"{np.round(g_['auc_split_gt_unsplit_ci'], 3)} | delta {g_['delta_median_split_minus_unsplit']:.2f} {np.round(g_['delta_ci'], 2)} | {g_['outcome']}")
            g2 = gaps.get("no_rasam", {})
            if g2:
                print(f"     no RASAM: AUC word>split {g2['auc_word_gt_split']:.2f} {np.round(g2['auc_word_gt_split_ci'], 3)} | {g2['outcome']}")
        for k, v in st.items():
            if k != "prevalence":
                print(f"  2.4 {k:12s} n {v['letters']:8.0f} rate {v['rate']:.3f} sep {v['separate_rate']:.3f} MH ratio {v['ratio']:.2f} {np.round(v['ratio_ci'], 2)}")
        for t, v in fr.get("stacking_2_5", {}).items():
            if "mh_ratio" in v.get("del_or_transposition", {}):
                e = v["del_or_transposition"]
                print(f"  2.5 t={t} stacked {v['stacked_pairs']} rate {e['rate_stacked']:.3f} vs {e['rate_not_stacked']:.3f} MH {e['mh_ratio']:.2f} {np.round(e['mh_ci'], 2)}"
                      f" | transp MH {v['transposition']['mh_ratio']:.2f} {np.round(v['transposition']['mh_ci'], 2)}")
    # BH over style x family
    rej, adj = bh(p_style)
    for (F, c), r, q in zip(p_cells, rej, adj):
        v = res["families"][F]["dot_styles_2_4"][c]
        v["q"], v["fdr_survives"] = float(q), bool(r)
        v["keep"] = bool(r and v["ratio"] >= 2 and v["ratio_ci"][0] > 1)
    keep = defaultdict(list)
    for (F, c) in p_cells:
        if res["families"][F]["dot_styles_2_4"][c]["keep"]:
            keep[c].append(F)
    res["dot_styles_kept"] = {c: fs for c, fs in keep.items() if len(fs) >= 2}
    (OUT / "image_studies.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "gallery_picks.json").write_text(json.dumps(picks, ensure_ascii=False), encoding="utf-8")
    print("\nkept styles:", res["dot_styles_kept"])


if __name__ == "__main__":
    main()
