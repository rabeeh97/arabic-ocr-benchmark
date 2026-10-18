"""Text-side studies on every reading (checklist 2.3 skeleton classes and the tooth-count detector, 2.13 similarity-choice
model of confusions, P-L4). Rules were written first: results/characterize/dots/prereg.json.

    .geomvenv/bin/python scripts/dots_text.py      # results/characterize/dots/text_studies.json, text_picks.json

Letters view (canonical, harakat and tatweel stripped). Panel lines of all six test sets; families as analyze_dots.fam.
Intervals: line-cluster bootstrap (shared line weights across families, so cross-family statistics pair up).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

from dots_common import DATASETS, OUT, analyze_letters, ci, load_store, rasm_of, readings
from eval_stats import bh, boot_p

LOOPS = set("فقومهعصضطظ")
TOOTH1, TOOTH3 = set("بتثنيئ"), set("سش")
TAILS = set("رزونيع")
VERTS = set("الكط")
ALPHA = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويةىأإآؤئء")
AI = {c: k for k, c in enumerate(ALPHA)}
REPS = 1000
SCM_REPS = 100


def classes(ch: str, form: str) -> list[str]:
    out = []
    if ch in LOOPS:
        out.append("loops")
    if ch in TOOTH1 | TOOTH3 and form in ("initial", "medial"):
        out.append("teeth")
    if ch in TAILS and form in ("final", "isolated"):
        out.append("tails")
    if ch in VERTS:
        out.append("verticals")
    return out


def tooth_runs(LI):
    """Maximal runs of consecutive joined ini/med tooth letters in one PAW: [(first index, last index, teeth)]."""
    runs, cur = [], []
    for li in LI:
        is_t = li.letter in TOOTH1 | TOOTH3 and li.position in ("initial", "medial")
        if is_t and cur and cur[-1].paw == li.paw and li.index == cur[-1].index + 1:
            cur.append(li)
            continue
        if cur:
            runs.append(cur)
        cur = [li] if is_t else []
    if cur:
        runs.append(cur)
    return [(r[0].index, r[-1].index, sum(3 if x.letter in TOOTH3 else 1 for x in r), any(x.letter in TOOTH3 for x in r)) for r in runs]


def ci_basic(point: float, boots: np.ndarray) -> list[float]:
    """Basic (reverse-percentile) bootstrap interval. Used for the SCM statistics: resampled lines leave more empty
    confusion cells, which biases the percentile interval of sparse-count statistics away from the estimate."""
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(2 * point - hi), float(2 * point - lo)]


def scm_fit(N: np.ndarray, x0=None):
    """Luce similarity-choice model by maximum likelihood. N: counts GT i -> read j. Returns (log b, log eta, x)."""
    n = N.shape[0]
    iu = np.triu_indices(n, 1)
    Np = N + 0.01  # tiny pseudo-count keeps unseen pairs finite
    rows = Np.sum(1)

    def unpack(x):
        lb = np.concatenate([[0.0], x[: n - 1]])
        le = np.zeros((n, n))
        le[iu] = x[n - 1 :]
        le = le + le.T
        return lb, le

    def f(x):
        lb, le = unpack(x)
        A = lb[None, :] + le
        m = A.max(1, keepdims=True)
        lz = m[:, 0] + np.log(np.exp(A - m).sum(1))
        P = np.exp(A - lz[:, None])
        ll = (Np * (A - lz[:, None])).sum()
        R = Np - rows[:, None] * P
        gb = R.sum(0)[1:]
        ge = (R + R.T)[iu]
        return -ll, -np.concatenate([gb, ge])
    if x0 is None:
        x0 = np.full(n - 1 + len(iu[0]), -3.0)
        x0[: n - 1] = 0.0
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(-10, 10)] * (n - 1) + [(-15, 0)] * len(iu[0]), options={"maxiter": 500})
    lb, le = unpack(r.x)
    return lb, le, r.x


def main() -> None:
    S, fams = load_store(DATASETS + ("nakba_blind",))
    keys = sorted({r["sample_key"] for rows in fams.values() for r in rows})
    lix = {k: i for i, k in enumerate(keys)}
    nl = len(keys)
    rng = np.random.default_rng(0)
    Wt = np.zeros((REPS + 1, nl), np.float32)
    Wt[0] = 1
    for r in range(REPS):
        Wt[r + 1] = np.bincount(rng.integers(0, nl, nl), minlength=nl)
    LIc = {}
    res = {"prereg": "results/characterize/dots/prereg.json", "families": {}}
    picks = {"teeth": []}
    rnd = random.Random(0)
    conf_all, scm = {}, {}
    p_cls, cells = [], []
    p_tooth, tcells = [], []
    freq = Counter()
    for F, rows in fams.items():
        if len({r["sample_key"] for r in rows}) < 150:
            continue
        cl_n = defaultdict(lambda: np.zeros(nl))
        cl_e = defaultdict(lambda: np.zeros(nl))
        conf = Counter()
        cls_conf = Counter()
        tb = defaultdict(lambda: np.zeros((3, nl)))  # bucket -> (runs, events, teeth)
        CM = defaultdict(Counter)  # line -> (i*36+j) counts
        n_read = 0
        for key, run, g, read, ins in readings(S, rows):
            i = lix[key]
            n_read += 1
            if key not in LIc:
                LIc[key] = analyze_letters(g)
            LI = LIc[key]
            for li in LI:
                if li.index >= len(read):
                    continue
                h = read[li.index]
                err = h != li.letter
                cl_n["all"][i] += 1
                cl_e["all"][i] += err
                for c in classes(li.letter, li.position):
                    cl_n[c][i] += 1
                    cl_e[c][i] += err
                if h is not None and h != " " and h != li.letter:
                    conf[(li.letter, h)] += 1
                    cg = classes(li.letter, li.position)
                    ch_ = classes(h, li.position)
                    tag = "within" if set(cg) & set(ch_) else "between" if cg and ch_ else "to/from other"
                    for c in cg or ["none"]:
                        cls_conf[(c, tag)] += 1
                if li.letter in AI and h in AI:
                    CM[i][AI[li.letter] * len(ALPHA) + AI[h]] += 1
                    if F == "CRNN" and run == "crnn/base":
                        freq[li.letter] += 1
            for a, b, teeth, has3 in tooth_runs(LI):
                ev = any(read[k] is None for k in range(a, b + 1))
                ev |= any(any(c in TOOTH1 | TOOTH3 | {"ى"} for c in ins.get(k, "")) for k in range(a, b + 2))
                bk = min(teeth, 5)
                tb[bk][0, i] += 1
                tb[bk][1, i] += ev
                tb[bk][2, i] += teeth
                if not has3:  # sensitivity: one-tooth letters only, so teeth = letters (no س / ش)
                    k1 = f"1t-{min(b - a + 1, 4)}"
                    tb[k1][0, i] += 1
                    tb[k1][1, i] += ev
                    tb[k1][2, i] += b - a + 1
                if teeth >= 4 and rnd.random() < 0.02:
                    picks["teeth"].append({"key": key, "run": run, "family": F, "a": a, "b": b, "teeth": teeth, "event": bool(ev)})
        fr = {"readings": n_read}
        # ---- class error ratios ----
        base = (Wt @ cl_e["all"]) / np.maximum(1, Wt @ cl_n["all"])
        cr = {"all_letter_error": float(base[0])}
        for c in ("loops", "teeth", "tails", "verticals"):
            if cl_n[c].sum() < 2000:
                continue
            rate = (Wt @ cl_e[c]) / np.maximum(1, Wt @ cl_n[c])
            ratio = rate / base
            p = boot_p(float(np.log(ratio[0])), np.log(ratio[1:]))
            cr[c] = {"letters": float(cl_n[c].sum()), "error": float(rate[0]), "error_ci": ci(rate[1:]), "ratio": float(ratio[0]),
                     "ratio_ci": ci(ratio[1:]), "p": p}
            p_cls.append(p)
            cells.append((F, c))
        fr["classes_2_3"] = cr
        tot = sum(v for (c, t), v in cls_conf.items())
        fr["confusion_classes"] = {f"{c}|{t}": v / max(1, tot) for (c, t), v in sorted(cls_conf.items())}
        fr["top_confusions"] = [[a, b, n] for (a, b), n in conf.most_common(15)]
        # ---- tooth detector ----
        tt = {}
        for bk in sorted(tb, key=str):
            rr = (Wt @ tb[bk][1]) / np.maximum(1, Wt @ tb[bk][0])
            pt = (Wt @ tb[bk][1]) / np.maximum(1, Wt @ tb[bk][2])
            tt[str(bk)] = {"runs": float(tb[bk][0].sum()), "event_rate_per_run": float(rr[0]), "per_run_ci": ci(rr[1:]),
                           "per_tooth": float(pt[0]), "per_tooth_ci": ci(pt[1:])}
        if 1 in tb and (tb.get(4) is not None or tb.get(5) is not None):
            hi = sum(tb[k] for k in (4, 5) if k in tb)
            p_hi = (Wt @ hi[1]) / np.maximum(1, Wt @ hi[2])
            p_1 = (Wt @ tb[1][1]) / np.maximum(1, Wt @ tb[1][2])
            ratio = p_hi / np.maximum(1e-9, p_1)
            p = boot_p(float(np.log(max(ratio[0], 1e-9))), np.log(np.clip(ratio[1:], 1e-9, None)))
            tt["per_tooth_ratio_ge4_vs_1"] = {"ratio": float(ratio[0]), "ci": ci(ratio[1:]), "p": p}
            p_tooth.append(p)
            tcells.append(F)
        if "1t-1" in tb and ("1t-3" in tb or "1t-4" in tb):
            hi = sum(tb[k] for k in ("1t-3", "1t-4") if k in tb)
            r_ = ((Wt @ hi[1]) / np.maximum(1, Wt @ hi[2])) / np.maximum(1e-9, (Wt @ tb["1t-1"][1]) / np.maximum(1, Wt @ tb["1t-1"][2]))
            tt["sensitivity_one_tooth_letters_ge3_vs_1_per_letter"] = {"ratio": float(r_[0]), "ci": ci(r_[1:])}
            print("  one-tooth-letter runs, per-letter ratio >=3 vs 1:", round(float(r_[0]), 2), np.round(ci(r_[1:]), 2))
        fr["tooth_count_2_3"] = tt
        res["families"][F] = fr
        conf_all[F] = CM
        print(f"\n{F}: readings {n_read}; all-letter error {cr['all_letter_error']:.3f}")
        for c in ("loops", "teeth", "tails", "verticals"):
            if c in cr:
                print(f"  {c:9s} err {cr[c]['error']:.3f} ratio {cr[c]['ratio']:.2f} {np.round(cr[c]['ratio_ci'], 2)}")
        for bk, v in tt.items():
            if bk.isdigit() or bk.startswith("1t"):
                print(f"  teeth={bk}: runs {v['runs']:.0f} per-run {v['event_rate_per_run']:.3f} per-tooth {v['per_tooth']:.4f} {np.round(v['per_tooth_ci'], 4)}")
        if "per_tooth_ratio_ge4_vs_1" in tt:
            print("  per-tooth ratio >=4 vs 1:", tt["per_tooth_ratio_ge4_vs_1"])
        print("  top confusions:", " ".join(f"{a}→{b}:{n}" for a, b, n in fr["top_confusions"][:10]))
    # BH
    rej, adj = bh(p_cls)
    for (F, c), r, q in zip(cells, rej, adj):
        v = res["families"][F]["classes_2_3"][c]
        v["q"], v["fdr_survives"] = float(q), bool(r)
    findings = {}
    for c in ("loops", "teeth", "tails", "verticals"):
        fs = [F for F in res["families"] if c in res["families"][F]["classes_2_3"]]
        ok = [F for F in fs if res["families"][F]["classes_2_3"][c]["ratio_ci"][0] > 1 and res["families"][F]["classes_2_3"][c]["fdr_survives"]]
        findings[c] = {"families": len(fs), "above_baseline": ok, "T3_finding": len(ok) == len(fs) and len(fs) >= 3}
    res["class_findings"] = findings
    rej, adj = bh(p_tooth)
    for F, r, q in zip(tcells, rej, adj):
        v = res["families"][F]["tooth_count_2_3"]["per_tooth_ratio_ge4_vs_1"]
        v["q"], v["fdr_survives"] = float(q), bool(r)
    res["tooth_minim_analogue"] = all(res["families"][F]["tooth_count_2_3"]["per_tooth_ratio_ge4_vs_1"]["ci"][0] > 1 and
                                      res["families"][F]["tooth_count_2_3"]["per_tooth_ratio_ge4_vs_1"]["fdr_survives"] for F in tcells)
    print("\nclass findings", findings, "\ntooth minim analogue", res["tooth_minim_analogue"])

    # ---- 2.13 similarity-choice model ----
    nA = len(ALPHA)
    dotpair = np.zeros((nA, nA), bool)
    for a in ALPHA:
        for b in ALPHA:
            if a != b and any(rasm_of(a, p) is not None and rasm_of(a, p) == rasm_of(b, p) for p in ("initial", "medial", "final", "isolated")):
                dotpair[AI[a], AI[b]] = True
    iu = np.triu_indices(nA, 1)
    fit = {}
    for F, CM in conf_all.items():
        lines = np.array(list(CM.keys()))
        mats = np.zeros((len(lines), nA * nA), np.float32)
        for r_, i in enumerate(lines):
            for c, v in CM[i].items():
                mats[r_, c] = v
        Wl = Wt[: SCM_REPS + 1][:, lines]
        Ns = (Wl @ mats).reshape(-1, nA, nA)
        les, lbs, x = [], [], None
        for r_ in range(SCM_REPS + 1):
            lb, le, x = scm_fit(Ns[r_].astype(float), None if x is None else x.copy())
            les.append(le[iu])
            lbs.append(lb)
        fit[F] = (np.array(les), np.array(lbs), Ns[0])
        print("SCM fitted", F, flush=True)
    pooled = sum(v[2] for v in fit.values())
    sel = (pooled + pooled.T)[iu] >= 5  # pairs with any real confusion
    Fs = list(fit)
    rho = {}
    for a in range(len(Fs)):
        for b in range(a + 1, len(Fs)):
            ea, eb = fit[Fs[a]][0][:, sel], fit[Fs[b]][0][:, sel]
            rs = np.array([spearmanr(ea[r_], eb[r_])[0] for r_ in range(SCM_REPS + 1)])
            rho[f"{Fs[a]} ~ {Fs[b]}"] = {"rho": float(rs[0]), "ci": ci_basic(rs[0], rs[1:]), "ci_percentile": ci(rs[1:])}
    dp = dotpair[iu][sel]
    dots = {}
    for F in Fs:
        e = fit[F][0][:, sel]
        ratio = np.exp(e[:, dp].mean(1) - e[:, ~dp].mean(1))
        lb = fit[F][1]
        fv = np.array([freq.get(c, 0) for c in ALPHA], float)
        rb = spearmanr(lb[0], fv)[0]
        top = np.argsort(-fit[F][0][0])[:12]
        dots[F] = {"dot_pair_eta_ratio": float(ratio[0]), "ci": ci_basic(ratio[0], ratio[1:]), "ci_percentile": ci(ratio[1:]), "bias_vs_letter_frequency_rho": float(rb),
                   "bias_sd_log": float(np.std(lb[0])),
                   "top_similar_pairs": [[ALPHA[iu[0][k]], ALPHA[iu[1][k]], float(np.exp(fit[F][0][0][k]))] for k in top]}
    shared = all(v["ci"][0] > 0.4 for v in rho.values())
    res["scm_2_13"] = {"pairs_used": int(sel.sum()), "dot_pairs_used": int(dp.sum()), "rho_between_families": rho, "per_family": dots,
                       "shared_similarity": shared, "dots_main_axis": all(v["ci"][0] > 1 for v in dots.values()),
                       "human_comparison": "not done: no human Arabic letter-confusion matrix in hand"}
    print(json.dumps(res["scm_2_13"], ensure_ascii=False, indent=1)[:3000])
    (OUT / "text_studies.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "text_picks.json").write_text(json.dumps(picks, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
