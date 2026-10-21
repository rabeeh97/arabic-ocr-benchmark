"""Checklist 2.12: IRT 2PL line difficulty and discrimination (R3.2), challenge subsets as dataset filters (R3.3), and
length-stratified reporting (X5-len).

    .geomvenv/bin/python scripts/audit_difficulty.py [--out results/audit]

R3.2  Per test set, systems (audit_common.load_systems; >= 50 % of the set's lines) are the test-takers and lines the items.
      A response is correct when the line CER at no_diacritics is <= 10 %. 2PL: P = sigmoid(a_j (theta_i - b_j)), MAP fit
      (theta ~ N(0,1), b ~ N(0, 2), log a ~ N(0, 0.7)) with torch LBFGS on CPU, missing cells masked. Reliability: split the
      systems into random halves 20 times, fit each half, Spearman of b (and of a) between halves; the same for the
      consensus difficulty (median line CER). Negative / near-zero discrimination lines (better systems fail more often) are
      exported as label-review (R4.1) candidates.
R3.3  OUT/challenge_subsets.json: {subset: {definition, keys: [sample_key]}}, usable as a dataset filter
      (audit_common.subset_keys). Subsets: hard_core, crnn_only_failure, vlm_only_failure (A3 matrix), irt_top_quintile,
      irt_low_discrimination, dot_dense (top quintile of dotted letters per letter, per set), oov_heavy (>= 30 % of words
      absent from the lexicon), fragments (<= 20 characters, or digits only, or one word), loose_crops (ink extent < 0.8),
      slant_outliers (store features), leaked (audit_leakage).
X5-len CER per length band (<= 20, 21-40, 41-60, > 60 characters of the cleaned GT) per system and set, line bootstrap
      95 % CI; fragment - full-line CER difference per system with a bootstrap p, BH-FDR q = 0.1 across systems x sets.

Decision rules (written before the run):
  R3.2  IRT difficulty replaces the consensus median CER as the line-difficulty measure only if its mean split-half
        Spearman >= 0.80 AND >= the consensus measure's; discrimination is reported per line only if its split-half
        Spearman >= 0.50 (else only the negative-discrimination list is used, as R4.1 candidates).
  X5-len Fragments become a separate leaderboard subset if fragment CER exceeds full-line CER for a majority of
        systems in a set after FDR; a set's headline then shows CER(full lines) and CER(fragments) beside the micro CER.
Outputs: OUT/difficulty.json, OUT/challenge_subsets.json, OUT/length_bands.json, OUT/difficulty_gallery.html.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import HTML_HEAD, OUT, W, load_samples, load_systems, panel_images, png_b64  # noqa: E402
from audit_spelling import LEX, tokens  # noqa: E402
from eval_stats import bh, boot_p  # noqa: E402
from arabic_ocr_bench.characterize.text import DOTTED, analyze_letters  # noqa: E402

OKCER = 0.10
BANDS = [(0, 20, "≤20"), (21, 40, "21–40"), (41, 60, "41–60"), (61, 10**6, ">60")]
torch.set_num_threads(4)


def fit_2pl(Y: np.ndarray, M: np.ndarray, iters: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y, M: systems x items (response, observed mask). Returns theta, a, b."""
    y, m = torch.tensor(Y, dtype=torch.float64), torch.tensor(M, dtype=torch.float64)
    S, I = Y.shape
    th = torch.zeros(S, dtype=torch.float64, requires_grad=True)
    la = torch.zeros(I, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(I, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th, la, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = la.exp()[None, :] * (th[:, None] - b[None, :])
        ll = -(torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum()
        prior = -(th ** 2).sum() / 2 - (b ** 2).sum() / (2 * 4.0) - (la ** 2).sum() / (2 * 0.49)
        loss = -(ll + prior)
        loss.backward()
        return loss

    opt.step(closure)
    return th.detach().numpy(), la.exp().detach().numpy(), b.detach().numpy()


def lexicon(words: set[str]) -> set[str]:
    out = set()
    with gzip.open(LEX, "rt") as f:
        for line in f:
            w = line.split("\t", 1)[0]
            if w in words:
                out.add(w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()
    samples = load_samples()
    systems = load_systems(("e_no_diacritics", "n_no_diacritics", "e_crnn_native", "n_crnn_native"))

    # ---------------------------------------------------------------- R3.2
    irt, line_irt = {}, {}
    for ds, runs in sorted(systems.items()):
        nk = sum(1 for s in samples.values() if s["dataset"] == ds)
        cols = [r for r in runs if len(runs[r]) >= 0.5 * nk]
        keys = sorted({k for r in cols for k in runs[r]})
        keys = [k for k in keys if sum(k in runs[r] for r in cols) >= max(3, 0.5 * len(cols))]
        if len(cols) < 6:
            irt[ds] = {"note": f"only {len(cols)} systems"}
            continue
        C = np.full((len(cols), len(keys)), np.nan)
        for i, r in enumerate(cols):
            for j, k in enumerate(keys):
                if k in runs[r]:
                    x = runs[r][k]
                    C[i, j] = min(1.0, x["e_no_diacritics"] / max(1, x["n_no_diacritics"]))
        M = ~np.isnan(C)
        Y = np.where(M, (np.nan_to_num(C, nan=1) <= OKCER).astype(float), 0.0)
        th, aa, bb = fit_2pl(Y, M.astype(float))
        cons = np.nanmedian(C, 0)
        rng = np.random.default_rng(0)
        rb, ra, rc = [], [], []
        for _ in range(a.splits):
            p = rng.permutation(len(cols))
            h1, h2 = p[: len(cols) // 2], p[len(cols) // 2:]
            ok = (M[h1].sum(0) >= 2) & (M[h2].sum(0) >= 2)
            _, a1, b1 = fit_2pl(Y[h1], M[h1].astype(float))
            _, a2, b2 = fit_2pl(Y[h2], M[h2].astype(float))
            rb.append(spearmanr(b1[ok], b2[ok])[0])
            ra.append(spearmanr(a1[ok], a2[ok])[0])
            c1, c2 = np.nanmedian(C[h1][:, ok], 0), np.nanmedian(C[h2][:, ok], 0)
            rc.append(spearmanr(c1, c2)[0])
        # sensitivity (added after the first run showed a 10 % threshold leaves most RASAM / KHATT lines all-wrong): threshold =
        # the set's median cell CER, so about half the responses are correct
        tmed = float(np.nanmedian(C))
        Ym = np.where(M, (np.nan_to_num(C, nan=1) <= tmed).astype(float), 0.0)
        rbm, ram = [], []
        rng2 = np.random.default_rng(0)
        for _ in range(a.splits):
            p = rng2.permutation(len(cols))
            h1, h2 = p[: len(cols) // 2], p[len(cols) // 2:]
            ok = (M[h1].sum(0) >= 2) & (M[h2].sum(0) >= 2)
            _, a1, b1 = fit_2pl(Ym[h1], M[h1].astype(float))
            _, a2, b2 = fit_2pl(Ym[h2], M[h2].astype(float))
            rbm.append(spearmanr(b1[ok], b2[ok])[0])
            ram.append(spearmanr(a1[ok], a2[ok])[0])
        sens = {"threshold": tmed, "split_half_spearman_b": float(np.mean(rbm)), "split_half_spearman_a": float(np.mean(ram)),
                "all_wrong_lines": int((Ym.sum(0) == 0).sum()), "all_correct_lines": int((Ym.sum(0) == M.sum(0)).sum()),
                "would_adopt": float(np.mean(rbm)) >= 0.80 and float(np.mean(rbm)) >= float(np.mean(rc))}
        # negative / low discrimination: the item's correctness correlates negatively with system ability
        pb = np.array([np.corrcoef(Y[M[:, j], j], th[M[:, j]])[0, 1] if Y[M[:, j], j].std() > 0 else np.nan for j in range(len(keys))])
        low = [keys[j] for j in range(len(keys)) if not np.isnan(pb[j]) and pb[j] < 0 and Y[M[:, j], j].mean() not in (0, 1)]
        adopt = float(np.mean(rb)) >= 0.80 and float(np.mean(rb)) >= float(np.mean(rc))
        irt[ds] = {"systems": len(cols), "lines": len(keys), "split_half_spearman_b": [float(np.mean(rb)), float(np.min(rb)), float(np.max(rb))],
                   "split_half_spearman_a": [float(np.mean(ra)), float(np.min(ra)), float(np.max(ra))],
                   "split_half_spearman_consensus": [float(np.mean(rc)), float(np.min(rc)), float(np.max(rc))],
                   "spearman_b_vs_consensus": float(spearmanr(bb, cons)[0]),
                   "all_correct_lines": int((Y.sum(0) == M.sum(0)).sum()), "all_wrong_lines": int((Y.sum(0) == 0).sum()),
                   "negative_discrimination_lines": len(low),
                   "adopt_irt_difficulty": adopt, "report_discrimination": float(np.mean(ra)) >= 0.50, "sensitivity_median_threshold": sens,
                   "theta": dict(zip(cols, map(float, th)))}
        for j, k in enumerate(keys):
            line_irt[k] = {"b": float(bb[j]), "a": float(aa[j]), "pb": None if np.isnan(pb[j]) else float(pb[j]), "consensus": float(cons[j]),
                           "n_sys": int(M[:, j].sum()), "p_correct": float(Y[M[:, j], j].mean())}
        irt[ds]["negative_discrimination_keys"] = low
        print(ds, {k: v for k, v in irt[ds].items() if k not in ("theta", "negative_discrimination_keys")}, flush=True)

    # ---------------------------------------------------------------- R3.3
    subsets = defaultdict(lambda: {"definition": "", "keys": []})
    for ds in ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2"):
        f = W / f"results/analysis/A/matrix_{ds}.json"
        if f.exists():
            lab = json.load(open(f))["labels"]
            for name in ("hard_core", "crnn_only_failure", "vlm_only_failure"):
                subsets[name]["keys"] += lab.get(name, [])
    subsets["hard_core"]["definition"] = "every matrix system > 25 % line CER at no_diacritics (analyze_lines A3)"
    subsets["crnn_only_failure"]["definition"] = "all CRNNs > 25 %, median VLM <= 15 % (A3)"
    subsets["vlm_only_failure"]["definition"] = "all VLMs > 25 %, best CRNN <= 15 % (A3)"
    by_ds = defaultdict(list)
    for k, s in samples.items():
        by_ds[s["dataset"]].append(k)
    for ds, ks in by_ds.items():
        bs = [(line_irt[k]["b"], k) for k in ks if k in line_irt]
        if bs:
            thr = np.quantile([x for x, _ in bs], 0.8)
            subsets["irt_top_quintile"]["keys"] += [k for x, k in bs if x >= thr]
        if ds in irt and "negative_discrimination_keys" in irt[ds]:
            subsets["irt_low_discrimination"]["keys"] += irt[ds]["negative_discrimination_keys"]
    subsets["irt_top_quintile"]["definition"] = "top 20 % of IRT 2PL difficulty b within the set"
    subsets["irt_low_discrimination"]["definition"] = "point-biserial of line correctness with system ability < 0 (better systems fail it more); R4.1 candidates"
    words = set()
    tok = {}
    for k, s in samples.items():
        tok[k] = tokens(s["gt_raw"])
        words.update(tok[k])
    lex = lexicon(words)
    dd = {}
    for k, s in samples.items():
        L = analyze_letters(s["gt_raw"])
        dd[k] = sum(li.letter in DOTTED for li in L) / max(1, len(L))
    for ds, ks in by_ds.items():
        thr = np.quantile([dd[k] for k in ks], 0.8)
        subsets["dot_dense"]["keys"] += [k for k in ks if dd[k] >= thr and dd[k] > 0]
    subsets["dot_dense"]["definition"] = "top 20 % of dotted letters per letter within the set"
    subsets["oov_heavy"]["keys"] = [k for k in samples if tok[k] and sum(w not in lex for w in tok[k]) / len(tok[k]) >= 0.3]
    subsets["oov_heavy"]["definition"] = ">= 30 % of the line's words absent from the lexicon (/Volumes/PortableSSD/data/lexicon, surface form)"
    frag = {k for k, s in samples.items() if len(s["gt_clean"]) <= 20 or s["gt_clean"].replace(" ", "").isdigit() or len(s["gt_clean"].split()) <= 1}
    subsets["fragments"]["keys"] = sorted(frag)
    subsets["fragments"]["definition"] = "cleaned GT <= 20 characters, or digits only, or a single word"
    subsets["loose_crops"]["keys"] = [k for k, s in samples.items() if s["ink_extent"] is not None and s["ink_extent"] < 0.8]
    subsets["loose_crops"]["definition"] = "ink covers < 80 % of the crop height (samples.ink_extent)"
    feats = pq.read_table(W / "results/store/features.parquet", columns=["sample_key", "slant_outlier"]).to_pylist()
    subsets["slant_outliers"]["keys"] = [r["sample_key"] for r in feats if r["slant_outlier"] and r["sample_key"] in samples]
    subsets["slant_outliers"]["definition"] = "store_features slant_outlier"
    lf = OUT / "leakage_flags.parquet"
    if lf.exists():
        subsets["leaked"]["keys"] = [r["sample_key"] for r in pq.read_table(lf, columns=["sample_key", "leaked"]).to_pylist() if r["leaked"]]
        subsets["leaked"]["definition"] = "text or image found in the set's own training pool (audit_leakage)"
    reg = {}
    for name, v in subsets.items():
        ks = sorted(set(v["keys"]))
        cnt = defaultdict(int)
        for k in ks:
            cnt[samples[k]["dataset"]] += 1
        reg[name] = {"definition": v["definition"], "count": len(ks), "per_set": {ds: {"n": cnt[ds], "share": cnt[ds] / len(by_ds[ds])} for ds in by_ds if cnt[ds]}, "keys": ks}
    # CER on each subset vs the rest, pooled over the systems per set (macro over systems)
    for name, v in reg.items():
        S = set(v["keys"])
        eff = {}
        for ds, runs in systems.items():
            ins = [k for k in by_ds.get(ds, []) if k in S]
            if len(ins) < 10:
                continue
            d = []
            for r, m in runs.items():
                e1 = sum(m[k]["e_no_diacritics"] for k in ins if k in m); n1 = sum(max(1, m[k]["n_no_diacritics"]) for k in ins if k in m)
                e0 = sum(x["e_no_diacritics"] for k, x in m.items() if k not in S); n0 = sum(max(1, x["n_no_diacritics"]) for k, x in m.items() if k not in S)
                if n1 and n0:
                    d.append(e1 / n1 - e0 / n0)
            if d:
                eff[ds] = {"median_cer_gap_vs_rest": float(np.median(d)), "systems": len(d)}
        v["cer_gap"] = eff
    json.dump(reg, open(a.out / "challenge_subsets.json", "w"), ensure_ascii=False)

    # ---------------------------------------------------------------- X5-len
    lb, pv, pk = {}, [], []
    for ds, runs in sorted(systems.items()):
        lb[ds] = {}
        for r, m in sorted(runs.items()):
            keys = list(m)
            e = np.array([m[k]["e_no_diacritics"] for k in keys], float)
            n = np.array([max(1, m[k]["n_no_diacritics"]) for k in keys], float)
            L = np.array([len(samples[k]["gt_clean"]) for k in keys])
            fr = np.array([k in frag for k in keys])
            row = {"lines": len(keys), "micro": float(e.sum() / n.sum())}
            rng = np.random.default_rng(0)
            for lo, hi, nm in BANDS:
                sel = (L >= lo) & (L <= hi)
                if sel.sum() >= 5:
                    es, ns = e[sel], n[sel]
                    bs = [es[i].sum() / ns[i].sum() for i in (rng.integers(0, len(es), len(es)) for _ in range(500))]
                    row[nm] = {"n": int(sel.sum()), "cer": float(es.sum() / ns.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            if fr.sum() >= 10 and (~fr).sum() >= 10:
                d0 = e[fr].sum() / n[fr].sum() - e[~fr].sum() / n[~fr].sum()
                bs = []
                for _ in range(a.reps):
                    i = rng.integers(0, len(keys), len(keys))
                    f_, ee, nn = fr[i], e[i], n[i]
                    if f_.sum() and (~f_).sum():
                        bs.append(ee[f_].sum() / nn[f_].sum() - ee[~f_].sum() / nn[~f_].sum())
                bs = np.array(bs)
                row["fragments"] = {"n": int(fr.sum()), "cer": float(e[fr].sum() / n[fr].sum()), "full_cer": float(e[~fr].sum() / n[~fr].sum()),
                                    "delta": float(d0), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": boot_p(d0, bs)}
                pv.append(row["fragments"]["p"])
                pk.append((ds, r))
            lb[ds][r] = row
    rej, adj = bh(pv)
    for (ds, r), rj, q in zip(pk, rej, adj):
        lb[ds][r]["fragments"]["q_bh"] = float(q)
        lb[ds][r]["fragments"]["survives_fdr"] = bool(rj)
    len_dec = {}
    for ds in lb:
        fr_rows = [v["fragments"] for v in lb[ds].values() if "fragments" in v]
        worse = sum(f["survives_fdr"] and f["delta"] > 0 for f in fr_rows)
        better = sum(f["survives_fdr"] and f["delta"] < 0 for f in fr_rows)
        len_dec[ds] = {"systems": len(fr_rows), "fragments_worse_fdr": worse, "fragments_better_fdr": better,
                       "separate_subset": bool(fr_rows) and worse > len(fr_rows) / 2,
                       "fragment_share": sum(k in frag for k in by_ds.get(ds, [])) / max(1, len(by_ds.get(ds, [])))}
    json.dump({"bands": [b[2] for b in BANDS], "per_set": lb, "decision": len_dec}, open(a.out / "length_bands.json", "w"), ensure_ascii=False)
    json.dump({"decision_rules": __doc__.split("Decision rules (written before the run):")[1].split("Outputs:")[0].strip(), "R3.2": irt,
               "subsets": {k: {kk: vv for kk, vv in v.items() if kk != "keys"} for k, v in reg.items()}, "X5_len_decision": len_dec},
              open(a.out / "difficulty.json", "w"), ensure_ascii=False, indent=1)
    pq_rows = [{"sample_key": k, **v} for k, v in line_irt.items()]
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(pq_rows), a.out / "irt_lines.parquet")
    print(json.dumps(len_dec, indent=1))
    print(json.dumps({k: (v["count"], v.get("cer_gap")) for k, v in reg.items()}, ensure_ascii=False)[:3000])

    # ---------------------------------------------------------------- gallery
    rnd = random.Random(0)
    pick = []
    for ds in irt:
        if "negative_discrimination_keys" not in irt[ds]:
            continue
        ks = [k for k in line_irt if samples[k]["dataset"] == ds]
        neg = sorted(irt[ds]["negative_discrimination_keys"], key=lambda k: line_irt[k]["pb"])[:4]
        hard = sorted(ks, key=lambda k: -line_irt[k]["b"])[:3]
        easy = sorted(ks, key=lambda k: line_irt[k]["b"])[:2]
        fr = [k for k in ks if k in frag]
        pick += [(ds, k, "negative discrimination (R4.1 candidate)") for k in neg] + [(ds, k, "hardest (b)") for k in hard] + \
                [(ds, k, "easiest (b)") for k in easy] + [(ds, k, "random fragment") for k in rnd.sample(fr, min(3, len(fr)))] + \
                [(ds, k, "random") for k in rnd.sample(ks, 2)]
    imgs = panel_images({k for _, k, _ in pick})
    html = [HTML_HEAD.format(title="Line difficulty"), "<h1>IRT difficulty, challenge subsets, length bands (checklist 2.12)</h1>",
            "<h2>R3.2 reliability</h2><table><tr><th>set</th><th>systems</th><th>lines</th><th>split-half ρ(b)</th><th>split-half ρ(a)</th><th>split-half ρ(consensus)</th><th>ρ(b, consensus)</th><th>neg. discr. lines</th><th>adopt IRT b?</th></tr>"]
    for ds, r in irt.items():
        if "systems" in r:
            html.append(f"<tr><td>{ds}</td><td>{r['systems']}</td><td>{r['lines']}</td><td>{r['split_half_spearman_b'][0]:.3f}</td><td>{r['split_half_spearman_a'][0]:.3f}</td>"
                        f"<td>{r['split_half_spearman_consensus'][0]:.3f}</td><td>{r['spearman_b_vs_consensus']:.3f}</td><td>{r['negative_discrimination_lines']}</td><td>{r['adopt_irt_difficulty']}</td></tr>")
    html.append("</table><h2>R3.3 challenge subsets</h2><table><tr><th>subset</th><th>definition</th><th>lines</th><th>median CER gap vs rest, per set</th></tr>")
    for n, v in reg.items():
        gaps = ", ".join(f"{d} {x['median_cer_gap_vs_rest'] * 100:+.1f}" for d, x in v["cer_gap"].items())
        html.append(f"<tr><td>{n}</td><td>{v['definition']}</td><td>{v['count']}</td><td>{gaps}</td></tr>")
    html.append("</table><h2>X5-len: CER by length band (no_diacritics)</h2>")
    for ds, rows in lb.items():
        html.append(f"<h3>{ds} — fragments {len_dec[ds]['fragment_share']:.1%} of lines; separate subset: {len_dec[ds]['separate_subset']}</h3><table><tr><th>system</th><th>micro</th>" +
                    "".join(f"<th>{b[2]}</th>" for b in BANDS) + "<th>fragments − full [95% CI] q</th></tr>")
        for r, v in sorted(rows.items(), key=lambda kv: kv[1]["micro"]):
            cells = "".join(f"<td>{v[b[2]]['cer']*100:.1f} (n={v[b[2]]['n']})</td>" if b[2] in v else "<td>–</td>" for b in BANDS)
            f = v.get("fragments")
            fc = f"{f['delta']*100:+.1f} [{f['ci95'][0]*100:+.1f}, {f['ci95'][1]*100:+.1f}] q={f['q_bh']:.3f}" if f else "–"
            html.append(f"<tr><td>{r}</td><td>{v['micro']*100:.1f}</td>{cells}<td>{fc}</td></tr>")
        html.append("</table>")
    html.append("<h2>Lines</h2>")
    for ds, k, why in pick:
        if k in imgs:
            li = line_irt[k]
            ex = ""
            th = irt[ds]["theta"]
            best = max(th, key=th.get)
            if k in systems[ds].get(best, {}):
                ex = f"<div class=ar>{best}: {systems[ds][best][k]['hyp']}</div>"
            html.append(f"<div class=card><div class=mut>{why} · {k} · b={li['b']:.2f} a={li['a']:.2f} pb={li['pb']} p(correct)={li['p_correct']:.2f}</div>"
                        f"<img src='{png_b64(imgs[k])}'><div class=ar>GT: {samples[k]['gt_raw']}</div>{ex}</div>")
    html.append("</body></html>")
    (a.out / "difficulty_gallery.html").write_text("\n".join(html))


if __name__ == "__main__":
    main()
