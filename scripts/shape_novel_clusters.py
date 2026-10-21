"""Novel-shape clusters, atlas v0 (checklist 2.2; shape-programme step 1; map novel-clusters). CPU.

    .geomvenv/bin/python scripts/shape_novel_clusters.py --feats results/characterize/shape/letter_feats.npz \
        --out results/characterize/shape/clusters.json --html results/characterize/shape/atlas_v0.html

Input: shape_letter_feats.py (X-G10 letter embedding: mean aligner-CRNN CNN column in the CTC span, PCA 128, cosine).
1. Per letter x form with >= 150 training-bank letters and >= 40 KHATT + NAKBA letters: spherical k-means (k-means++,
   5 restarts) on a fit sample of the bank plus KHATT and NAKBA letters (caps 3,000 / 1,000 / 1,000 per cell, so NAKBA does
   not dominate), plus the Muharaf test letters (cap 1,000) on the same footing, so the in-domain control can also create its
   own clusters (a deviation from the design, which fits on bank + KHATT + NAKBA only: without it the control would be
   biased towards zero flags). k in 2..10 by the mean silhouette (cosine) on <= 2,000 fit points; every letter is then
   assigned to its nearest centroid.
2. Enrichment of cluster c for set D (KHATT v1.0 test; NAKBA = test + blind sample; Muharaf test = control):
   R = share of D's cell letters in c / share of the bank's cell letters in c (+0.5 smoothing).
   Poisson line bootstrap (1,000 reps, lines resampled in every set at once) for the 95 % interval of log R and a p-value
   (eval_stats.boot_p); BH-FDR q = 0.1 over all (cluster, set) tests (eval_stats.bh).
   A "missing allograph" for D: survives FDR, R lower bound >= 2, and holds >= 5 % of D's letters in that cell.
3. Link to error: within each cell and set, letter error (greedy reading, as X-G10) in flagged clusters vs the rest of the
   cell; Mantel-Haenszel odds ratio pooled over cells, line-bootstrap 95 % interval.

Decision rule (fixed before the run; the design gives only the "done when" atlas condition):
  The atlas v0 page exists with clusters per letter x form, dataset mix and medoids  -> step 1 DONE.
  "Missing allographs" are a real, usable finding (feeds step 3 policy (b)) when all three hold:
   (a) KHATT or NAKBA has a flagged cluster in >= 5 letter x form cells;
   (b) the Muharaf-test control has flagged clusters in <= 1/5 as many cells as that set (the FDR false-positive check);
   (c) the pooled MH odds ratio of letter error, flagged vs same-cell rest, has its 95 % lower bound > 1 in that set.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from eval_stats import Q_FDR, bh, boot_p  # noqa: E402

SETS = ["train", "khatt", "nakba", "muharaf_test"]
TARGETS = ["khatt", "nakba", "muharaf_test"]
NAMES = {"train": "Muharaf train (bank)", "khatt": "KHATT v1.0 test", "nakba": "NAKBA (test + blind sample)", "muharaf_test": "Muharaf test (control)"}
COLORS = {"train": "#7a8590", "khatt": "#d1495b", "nakba": "#2e86ab", "muharaf_test": "#edae49"}
FORMS = ["iso", "ini", "med", "fin"]
CAP = {"train": 3000, "khatt": 1000, "nakba": 1000, "muharaf_test": 1000}


def kmeans(X, k, rng, iters=40, restarts=5):
    best = None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        for _ in range(k - 1):  # k-means++ on cosine distance
            d = np.clip(1 - np.max(X @ np.stack(C).T, axis=1), 0, None)
            C.append(X[rng.choice(len(X), p=d / d.sum()) if d.sum() > 0 else rng.integers(len(X))])
        C = np.stack(C)
        for _ in range(iters):
            lab = np.argmax(X @ C.T, axis=1)
            newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            newC /= np.linalg.norm(newC, axis=1, keepdims=True) + 1e-8
            if np.allclose(newC, C, atol=1e-5):
                break
            C = newC
        lab = np.argmax(X @ C.T, axis=1)
        inertia = float((1 - (X * C[lab]).sum(1)).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


def silhouette(X, lab):
    D = np.clip(1 - X @ X.T, 0, None)
    ks = np.unique(lab)
    if len(ks) < 2:
        return -1.0
    M = np.stack([D[:, lab == j].mean(1) for j in ks], 1)  # mean distance to each cluster (own cluster includes self: fix below)
    n = np.array([(lab == j).sum() for j in ks])
    own = np.searchsorted(ks, lab)
    a = M[np.arange(len(X)), own] * n[own] / np.maximum(n[own] - 1, 1)
    M[np.arange(len(X)), own] = np.inf
    b = M.min(1)
    s = np.where(n[own] > 1, (b - a) / np.maximum(a, b), 0)
    return float(s.mean())


def png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def crop(img, x0, x1, h=40, pad=4):
    c = img[:, max(0, x0 - pad): min(img.shape[1], x1 + pad)]
    if c.shape[1] < 2:
        c = np.full((64, 8), 255, np.uint8)
    w = max(4, round(c.shape[1] * h / 64))
    return np.asarray(Image.fromarray(c).resize((w, h), Image.LANCZOS))


def strip(crops, gap=6, h=40):
    W = sum(c.shape[1] for c in crops) + gap * (len(crops) - 1)
    out = np.full((h, max(W, 1)), 255, np.uint8)
    x = 0
    for c in crops:
        out[:, x: x + c.shape[1]] = c
        x += c.shape[1] + gap
    return out


def line_with_box(img, x0, x1, maxw=1400):
    rgb = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(rgb)
    d.rectangle([x0, 1, x1 - 1, img.shape[0] - 2], outline=(209, 73, 91), width=2)
    if rgb.width > maxw:
        rgb = rgb.resize((maxw, round(rgb.height * maxw / rgb.width)), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--min-train", type=int, default=150)
    ap.add_argument("--min-target", type=int, default=40)
    a = ap.parse_args()
    d = np.load(a.feats)
    Z = d["z"].astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    line, ch, form, x0, x1, ok = d["line"], d["ch"], d["form"], d["x0"], d["x1"], d["ok"]
    line_set = np.array([("nakba" if s.startswith("nakba") else s) for s in d["line_set"]])
    line_id, line_label, line_hyp = d["line_id"], d["line_label"], d["line_hyp"]
    imgs = pickle.load(open(a.feats.with_name("letter_lines.pkl"), "rb"))
    lset = line_set[line]
    sidx = np.array([SETS.index(s) for s in lset])
    n_lines = len(line_id)
    print("letters", len(Z), {s: int((lset == s).sum()) for s in SETS}, flush=True)
    rng = np.random.default_rng(0)

    cells = sorted({(c, f) for c, f in zip(ch, form)})
    cluster = np.full(len(Z), -1, np.int32)  # global cluster id
    cinfo = []  # per global cluster: cell, k-index, medoid letter index
    cell_info = {}
    skipped = []
    for c, f in cells:
        m = np.where((ch == c) & (form == f))[0]
        cnt = {s: int((lset[m] == s).sum()) for s in SETS}
        if cnt["train"] < a.min_train or cnt["khatt"] + cnt["nakba"] < a.min_target:
            skipped.append({"cell": f"{c}|{f}", **cnt})
            continue
        fit = np.concatenate([rng.permutation(m[lset[m] == s])[: CAP[s]] for s in ("train", "khatt", "nakba", "muharaf_test")])
        Xf = Z[fit]
        sub = rng.permutation(len(fit))[:2000]
        sil = {}
        cands = {}
        for k in range(2, 11):
            C, lab = kmeans(Xf, k, rng)
            sil[k] = silhouette(Xf[sub], lab[sub])
            cands[k] = C
        kbest = max(sil, key=sil.get)
        C = cands[kbest]
        lab = np.argmax(Z[m] @ C.T, axis=1)
        base = len(cinfo)
        cluster[m] = base + lab
        for j in range(kbest):
            mem = m[lab == j]
            fit_mem = mem[np.isin(lset[mem], ["train", "khatt", "nakba", "muharaf_test"])]
            pool = fit_mem if len(fit_mem) else mem
            med = int(pool[np.argmax(Z[pool] @ C[j])]) if len(pool) else -1
            cinfo.append({"cell": f"{c}|{f}", "k": j, "medoid": med})
        cell_info[f"{c}|{f}"] = {"k": kbest, "silhouette": {str(k): round(v, 4) for k, v in sil.items()}, "counts": cnt,
                                 "clusters": list(range(base, base + kbest))}
        print(c, f, cnt, "k", kbest, round(sil[kbest], 3), flush=True)
    K = len(cinfo)
    cell_of_cluster = np.array([list(cell_info).index(ci["cell"]) for ci in cinfo])
    valid = cluster >= 0
    # counts per (set, cluster) and per (set, cell), point + Poisson line bootstrap
    key = sidx[valid] * K + cluster[valid]
    lv = line[valid]

    def counts(w=None):
        lw = None if w is None else w[lv]
        cc = np.bincount(key, weights=lw, minlength=len(SETS) * K).reshape(len(SETS), K)
        cell = np.zeros((len(SETS), len(cell_info)))
        np.add.at(cell.T, cell_of_cluster, cc.T)
        return cc, cell

    def logR(cc, cell):
        share = (cc + 0.5) / (cell[:, cell_of_cluster] + 0.5 * np.bincount(cell_of_cluster)[cell_of_cluster])
        return np.log(share[1:] / share[0])  # targets x K

    cc0, cell0 = counts()
    lr0 = logR(cc0, cell0)
    boots = np.empty((a.reps, *lr0.shape), np.float32)
    # error link: flagged decided after; store per-rep weights seed so we can recompute
    for r in range(a.reps):
        w = np.random.default_rng(1000 + r).poisson(1.0, n_lines).astype(float)
        boots[r] = logR(*counts(w))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    pv = np.array([[boot_p(lr0[t, j], boots[:, t, j]) for j in range(K)] for t in range(len(TARGETS))])
    share_t = cc0[1:] / np.maximum(cell0[1:, cell_of_cluster], 1)
    share_b = cc0[0] / np.maximum(cell0[0, cell_of_cluster], 1)
    testable = (cell0[1:, cell_of_cluster] >= 20)  # need >= 20 letters of the set in the cell
    p_flat = np.where(testable & (lr0 > 0), pv, np.nan).ravel()  # one-sided interest: enriched clusters only
    rej, qv = bh(p_flat, Q_FDR)
    rej, qv = rej.reshape(pv.shape), qv.reshape(pv.shape)
    flagged = rej & (lo >= np.log(2)) & (share_t >= 0.05)
    print("flagged per set", dict(zip(TARGETS, flagged.sum(1).tolist())), flush=True)
    cells_flagged = {t: sorted({cinfo[j]["cell"] for j in np.where(flagged[i])[0]}) for i, t in enumerate(TARGETS)}

    # error link: MH odds ratio, flagged vs rest of the same cell, per target set; line bootstrap
    err = (~ok).astype(float)

    def mh(t_i, w=None):
        t = TARGETS[t_i]
        s = valid & (lset == t)
        if w is None:
            ww = np.ones(s.sum())
        else:
            ww = w[line[s]]
        fl = flagged[t_i][cluster[s]]
        cell = cell_of_cluster[cluster[s]]
        e = err[s]
        nc = len(cell_info)
        A = np.bincount(cell, ww * fl * e, nc)          # flagged, error
        B = np.bincount(cell, ww * fl * (1 - e), nc)    # flagged, correct
        Cc = np.bincount(cell, ww * (1 - fl) * e, nc)   # rest, error
        D = np.bincount(cell, ww * (1 - fl) * (1 - e), nc)
        N = A + B + Cc + D
        use = (A + B > 0) & (Cc + D > 0)
        num = (A * D / np.maximum(N, 1))[use].sum()
        den = (B * Cc / np.maximum(N, 1))[use].sum()
        er_f = A[use].sum() / max((A + B)[use].sum(), 1)
        er_r = Cc[use].sum() / max((Cc + D)[use].sum(), 1)
        return (num / den if den > 0 else np.nan), er_f, er_r, int((A + B)[use].sum())

    link = {}
    for i, t in enumerate(TARGETS):
        if not flagged[i].any():
            link[t] = None
            continue
        o, ef, er, nf = mh(i)
        bo = np.array([mh(i, np.random.default_rng(5000 + r).poisson(1.0, n_lines).astype(float))[0] for r in range(a.reps)])
        bo = bo[np.isfinite(bo)]
        link[t] = {"mh_or": float(o), "ci": [float(x) for x in np.percentile(bo, [2.5, 97.5])], "err_flagged": float(ef),
                   "err_rest_same_cells": float(er), "letters_flagged": nf}
        print(t, link[t], flush=True)
    # decision
    dec = {}
    for t in ("khatt", "nakba"):
        n_t = len(cells_flagged[t])
        n_c = len(cells_flagged["muharaf_test"])
        lk = link[t]
        dec[t] = {"cells_flagged": n_t, "control_cells_flagged": n_c, "a": n_t >= 5, "b": n_c <= n_t / 5,
                  "c": bool(lk and lk["ci"][0] > 1)}
        dec[t]["pass"] = dec[t]["a"] and dec[t]["b"] and dec[t]["c"]
    print("decision", dec, flush=True)

    clusters_out = []
    for j, ci in enumerate(cinfo):
        row = {**ci, "n": {s: int(cc0[si, j]) for si, s in enumerate(SETS)},
               "share": {s: round(float(cc0[si, j] / max(cell0[si, cell_of_cluster[j]], 1)), 4) for si, s in enumerate(SETS)}}
        for ti, t in enumerate(TARGETS):
            row[f"R_{t}"] = [round(float(np.exp(lr0[ti, j])), 3), round(float(np.exp(lo[ti, j])), 3), round(float(np.exp(hi[ti, j])), 3)]
            row[f"q_{t}"] = None if np.isnan(qv[ti, j]) else round(float(qv[ti, j]), 5)
            row[f"flag_{t}"] = bool(flagged[ti, j])
        er = {}
        for s in SETS:
            mm = (cluster == j) & (lset == s)
            er[s] = None if mm.sum() < 10 else round(float(err[mm].mean()), 4)
        row["letter_error"] = er
        clusters_out.append(row)
    res = {"decision_rule": __doc__.split("Decision rule")[1].strip(), "letters": int(len(Z)), "clustered_letters": int(valid.sum()),
           "pca_explained": float(d["pca_explained"]), "cells": cell_info, "skipped_cells": skipped, "clusters": clusters_out,
           "flagged_cells": cells_flagged, "flag_counts": dict(zip(TARGETS, flagged.sum(1).tolist())), "error_link": link,
           "decision": dec, "reps": a.reps}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- atlas v0 page (one self-contained HTML; crops embedded)
    rs = np.random.default_rng(7)
    H = []
    H.append("""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shape Atlas v0</title><style>
:root{--bg:#fbfaf7;--fg:#1d2329;--mut:#5f6b76;--card:#fff;--line:#e3e0d8;--flag:#d1495b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181b;--fg:#e6e8ea;--mut:#9aa5ae;--card:#1d2125;--line:#30363c}}
:root[data-theme="dark"]{--bg:#15181b;--fg:#e6e8ea;--mut:#9aa5ae;--card:#1d2125;--line:#30363c}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1500px}
h1{font-size:22px;margin:4px 0}h2{font-size:18px;margin:26px 0 8px;border-bottom:1px solid var(--line)}h3{font-size:15px;margin:14px 0 4px}
.mut{color:var(--mut)}table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;vertical-align:middle}
.crop{background:#fff;border-radius:3px;max-width:100%}.bar{display:flex;height:10px;width:160px;border-radius:2px;overflow:hidden}
.flag{color:var(--flag);font-weight:600}.ar{font-family:'Noto Naskh Arabic','Geeza Pro',serif;font-size:17px;direction:rtl}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:8px 0;overflow-x:auto}
.leg span{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;border-radius:2px}
img.line{max-width:100%;background:#fff}details summary{cursor:pointer;font-weight:600}
</style></head><body>""")
    H.append("<h1>Shape atlas v0: novel letter-shape clusters</h1>")
    H.append(f"<p class='mut'>Checklist 2.2 / shape-programme step 1. Letter embedding as X-G10 (aligner CRNN, mean CNN column in the CTC span, PCA 128 "
             f"keeps {float(d['pca_explained']):.0%} of variance). Spherical k-means per letter × form, k by silhouette. {len(Z):,} letters; "
             f"{len(cell_info)} cells clustered, {len(skipped)} skipped (too few letters).</p>")
    H.append("<p class='leg'>Dataset mix:" + "".join(f"<span style='background:{COLORS[s]}'></span>{NAMES[s]}" for s in SETS) + "</p>")
    H.append("<h2>Decision</h2><div class='card'><pre style='white-space:pre-wrap;margin:0'>" + html.escape(res["decision_rule"]) + "</pre><table><tr><th>set</th><th>cells with a missing allograph</th><th>control cells (Muharaf test)</th><th>(a)</th><th>(b)</th><th>(c)</th><th>pass</th></tr>")
    for t, v in dec.items():
        H.append(f"<tr><td>{NAMES[t]}</td><td>{v['cells_flagged']}</td><td>{v['control_cells_flagged']}</td><td>{v['a']}</td><td>{v['b']}</td><td>{v['c']}</td><td><b>{v['pass']}</b></td></tr>")
    H.append("</table><h3>Letter error, flagged clusters vs rest of the same cells (Mantel–Haenszel, line bootstrap)</h3><table><tr><th>set</th><th>letters in flagged clusters</th><th>error flagged</th><th>error rest</th><th>MH OR [95 % CI]</th></tr>")
    for t, v in link.items():
        if v:
            H.append(f"<tr><td>{NAMES[t]}</td><td>{v['letters_flagged']}</td><td>{v['err_flagged']:.3f}</td><td>{v['err_rest_same_cells']:.3f}</td><td>{v['mh_or']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]</td></tr>")
        else:
            H.append(f"<tr><td>{NAMES[t]}</td><td>0</td><td colspan=3>no flagged clusters</td></tr>")
    H.append("</table></div>")

    # flagged clusters gallery with line context (best = read right, worst = misread, random)
    H.append("<h2>Missing allographs (flagged clusters) with line context</h2><p class='mut'>For each flagged cluster: medoid and members from the flagged set and, "
             "for comparison, the bank's closest letters; then lines where a member was misread, read right, and random ones (box = CTC span). GT and the aligner CRNN's greedy reading.</p>")
    order = sorted([(ti, j) for ti in range(2) for j in range(K) if flagged[ti, j]], key=lambda x: -lr0[x[0], x[1]])
    for ti, j in order[:40]:
        t = TARGETS[ti]
        ci = clusters_out[j]
        mem = np.where((cluster == j) & (lset == t))[0]
        mem_t = np.where((cluster == j) & (lset == "train"))[0]
        R = ci[f"R_{t}"]
        H.append(f"<div class='card'><h3><span class='ar'>{html.escape(ci['cell'].split('|')[0])}</span> {ci['cell'].split('|')[1]} · cluster {ci['k']} · "
                 f"<span class='flag'>{NAMES[t]}</span>: share {ci['share'][t]:.1%} vs bank {ci['share']['train']:.1%}, R = {R[0]:.1f} [{R[1]:.1f}, {R[2]:.1f}], q = {ci[f'q_{t}']}; "
                 f"letter error {ci['letter_error'][t]} vs bank-cluster {ci['letter_error']['train']}</h3>")
        sel = rs.permutation(mem)[:12]
        H.append(f"<div class='mut'>{t} members</div><img class='crop' src='data:image/png;base64,{png_b64(strip([crop(imgs[line[i]], x0[i], x1[i]) for i in sel]))}'>")
        if len(mem_t):
            selt = rs.permutation(mem_t)[:12]
            H.append(f"<div class='mut'>bank members ({len(mem_t)})</div><img class='crop' src='data:image/png;base64,{png_b64(strip([crop(imgs[line[i]], x0[i], x1[i]) for i in selt]))}'>")
        else:
            H.append("<div class='mut'>no bank members</div>")
        wrong = rs.permutation(mem[~ok[mem]])[:2]
        right = rs.permutation(mem[ok[mem]])[:2]
        rnd = rs.permutation(mem)[:1]
        H.append("<details><summary>lines: misread (worst) · read right (best) · random</summary>")
        for kind, ss in (("misread", wrong), ("read right", right), ("random", rnd)):
            for i in ss:
                li = line[i]
                H.append(f"<div style='margin:6px 0'><span class='mut'>{kind} · {html.escape(str(line_id[li]))} · letter #{int(d['pos'][i])}</span><br>"
                         f"<img class='line' src='data:image/png;base64,{line_with_box(imgs[li], x0[i], x1[i])}'><div class='ar'>GT: {html.escape(str(line_label[li]))}</div>"
                         f"<div class='ar'>CRNN: {html.escape(str(line_hyp[li]))}</div></div>")
        H.append("</details></div>")

    # the atlas proper: every cell, every cluster
    H.append("<h2>Atlas: clusters per letter × form</h2><p class='mut'>Per cluster: medoid (left, boxed) then 7 random members; bar = share of each set's cell letters in the cluster, "
             "normalised so a cluster used equally by every set shows equal segments; R = enrichment over the bank.</p>")
    for cell, info in cell_info.items():
        c, f = cell.split("|")
        H.append(f"<div class='card'><h3><span class='ar'>{html.escape(c)}</span> {f} · k = {info['k']} (silhouette {info['silhouette'][str(info['k'])]:.3f}) · "
                 + " · ".join(f"{s} {info['counts'][s]}" for s in SETS) + "</h3><table>")
        for j in info["clusters"]:
            ci = clusters_out[j]
            mem = np.where(cluster == j)[0]
            med = ci["medoid"]
            sel = rs.permutation(mem)[:7]
            crops = [crop(imgs[line[med]], x0[med], x1[med])] if med >= 0 else []
            if crops:
                crops[0] = crops[0].copy()
                crops[0][[0, -1], :] = 0
                crops[0][:, [0, -1]] = 0
            crops += [crop(imgs[line[i]], x0[i], x1[i]) for i in sel]
            sh = np.array([ci["share"][s] for s in SETS])
            tot = sh.sum() or 1
            bar = "".join(f"<div style='width:{100 * v / tot:.1f}%;background:{COLORS[s]}' title='{s} {v:.1%}'></div>" for s, v in zip(SETS, sh))
            flags = " ".join(f"<span class='flag'>missing in train: {t}</span>" for t in TARGETS if ci[f"flag_{t}"])
            H.append(f"<tr><td>{ci['k']}</td><td><img class='crop' src='data:image/png;base64,{png_b64(strip(crops))}'></td><td><div class='bar'>{bar}</div></td>"
                     f"<td>n {sum(ci['n'].values())}<br>R KHATT {ci['R_khatt'][0]:.2f} · NAKBA {ci['R_nakba'][0]:.2f} · Muh {ci['R_muharaf_test'][0]:.2f}</td><td>{flags}</td></tr>")
        H.append("</table></div>")
    H.append("<h2>Cells not clustered</h2><p class='mut'>Fewer than 150 bank letters or 40 KHATT + NAKBA letters. Cells with KHATT/NAKBA letters but almost no bank letters are missing forms in their own right.</p><table><tr><th>cell</th>"
             + "".join(f"<th>{s}</th>" for s in SETS) + "</tr>")
    for s in sorted(skipped, key=lambda r: -(r["khatt"] + r["nakba"]))[:80]:
        H.append(f"<tr><td class='ar'>{html.escape(s['cell'])}</td>" + "".join(f"<td>{s[x]}</td>" for x in SETS) + "</tr>")
    H.append("</table></body></html>")
    a.html.write_text("\n".join(H), encoding="utf-8")
    print("wrote", a.html, f"{a.html.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
