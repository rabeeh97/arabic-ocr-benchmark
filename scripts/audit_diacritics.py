"""Checklist 2.8 / R2.7 (the diacritic scoring rule) and 2.11a / R1.3 (diacritic-policy share per set).

    .geomvenv/bin/python scripts/audit_diacritics.py [--out results/audit]

R1.3  For every label set (the panel test sets, the full KHATT v1.0 test and the raw HF training labels): share of lines with
      at least one haraka, harakat per 100 letters, and the per-line level (none / sparse / partial / full, text.diacritic_level).
      Policy: "undiacritised" when < 2 % of lines carry marks, "sparse" when marks are present but < 15 % letter coverage on
      average among marked lines, "diacritised" otherwise.
R2.7  Per test set, the systems that read >= 80 % of its lines (common lines only) ranked by micro CER at the orthographic
      rung (marks kept, every other fold applied) and at no_diacritics (marks stripped). Kendall tau and Spearman rho with a
      line-bootstrap 95 % interval; the largest rank moves; per system the diacritic tax (CER_orth - CER_no_diac) and the
      marks it emits per 100 letters where the GT has none (errors.diacritic_mismatch "inserted").

Decision rule (written before the run):
  Marks are IMMATERIAL to the ranking of a set when the lower 95 % bound of Kendall tau(orth, no_diac) >= 0.90; otherwise
  scoring with marks re-ranks systems. Headline rung: no_diacritics for "undiacritised" and "sparse" sets (keeping marks there
  scores the model's mark habit against labels that mostly omit them); orthographic (marks kept) as the headline only for a
  "diacritised" set, with no_diacritics beside it. If marks re-rank an undiacritised/sparse set, marks-kept numbers must not be
  quoted for it at all.
Outputs: OUT/diacritics.json, OUT/diacritics_gallery.html.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import HTML_HEAD, OUT, load_samples, load_systems, panel_images, png_b64  # noqa: E402
from audit_leakage import TEXT_COL, TRAIN_TEXT, files  # noqa: E402
from arabic_ocr_bench.characterize import errors  # noqa: E402
from arabic_ocr_bench.characterize.text import analyze_letters, diacritic_level, diacritics  # noqa: E402

COVER = 0.8


def line_diac(raw: str) -> tuple[int, int, float]:
    L = analyze_letters(raw or "")
    d = diacritics(raw or "", L)
    return d["marks"], len(L), d["coverage"]


def profile(texts: list[str]) -> dict:
    marks, letters, lev = 0, 0, Counter()
    cov_marked = []
    with_marks = 0
    for t in texts:
        m, n, c = line_diac(t)
        marks += m
        letters += n
        lev[diacritic_level(c)] += 1
        if m:
            with_marks += 1
            cov_marked.append(c)
    n = max(1, len(texts))
    share = with_marks / n
    mean_cov = float(np.mean(cov_marked)) if cov_marked else 0.0
    policy = "undiacritised" if share < 0.02 else ("sparse" if mean_cov < 0.15 else "diacritised")
    return {"lines": len(texts), "share_lines_with_marks": share, "harakat_per_100_letters": 100 * marks / max(1, letters),
            "mean_coverage_of_marked_lines": mean_cov, "levels": {k: v / n for k, v in lev.items()}, "policy": policy}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    samples = load_samples()

    # ---- R1.3
    prof = {}
    for ds in sorted({s["dataset"] for s in samples.values()}):
        prof[ds] = profile([s["gt_raw"] for s in samples.values() if s["dataset"] == ds])
    for pool, gl in TRAIN_TEXT.items():
        col = TEXT_COL.get(pool, "text")
        prof[pool] = profile([t or "" for f in files(gl) for t in pq.read_table(f, columns=[col]).column(col).to_pylist()])
    print(json.dumps({k: (round(v["share_lines_with_marks"], 3), round(v["harakat_per_100_letters"], 2), v["policy"]) for k, v in prof.items()}))

    # ---- R2.7
    systems = load_systems(("e_orthographic", "n_orthographic", "e_no_diacritics", "n_no_diacritics", "e_raw", "n_raw"))
    res, gal = {}, {}
    # leaderboard view: every system on its own lines (>= 50), as the leaderboard ranks them; lines resampled per system
    wide = {}
    for ds, runs in sorted(systems.items()):
        rs = sorted(runs)
        arr = [(np.array([runs[r][k]["e_orthographic"] for k in runs[r]], float), np.array([runs[r][k]["e_no_diacritics"] for k in runs[r]], float),
                np.array([max(1, runs[r][k]["n_orthographic"]) for k in runs[r]], float), np.array([max(1, runs[r][k]["n_no_diacritics"]) for k in runs[r]], float)) for r in rs]
        c1 = np.array([x[0].sum() / x[2].sum() for x in arr])
        c2 = np.array([x[1].sum() / x[3].sum() for x in arr])
        rng = np.random.default_rng(1)
        bt = []
        for _ in range(a.reps // 2):
            b1, b2 = [], []
            for e1, e2, n1, n2 in arr:
                i = rng.integers(0, len(e1), len(e1))
                b1.append(e1[i].sum() / n1[i].sum())
                b2.append(e2[i].sum() / n2[i].sum())
            bt.append(kendalltau(b1, b2)[0])
        r1, r2 = np.argsort(np.argsort(c1)), np.argsort(np.argsort(c2))
        lo = float(np.nanpercentile(bt, 2.5))
        wide[ds] = {"systems": len(rs), "kendall_tau": float(kendalltau(c1, c2)[0]), "tau_ci95": [lo, float(np.nanpercentile(bt, 97.5))],
                    "max_rank_move": int(np.abs(r1 - r2).max()), "marks_immaterial_to_ranking": lo >= 0.90,
                    "movers": sorted([(rs[i], int(r1[i]) + 1, int(r2[i]) + 1) for i in range(len(rs)) if r1[i] != r2[i]], key=lambda x: -abs(x[1] - x[2]))[:8]}
        print("leaderboard view", ds, json.dumps({k: v for k, v in wide[ds].items() if k != "movers"}))
    for ds, runs in sorted(systems.items()):
        nkeys = sum(1 for s in samples.values() if s["dataset"] == ds)
        chosen = [r for r in runs if len(runs[r]) >= COVER * nkeys]
        keys = sorted(set.intersection(*[set(runs[r]) for r in chosen])) if chosen else []
        if len(chosen) < 4 or len(keys) < 50:
            res[ds] = {"note": f"{len(chosen)} systems cover >= 80 % of lines; {len(keys)} common lines", "policy": prof[ds]["policy"]}
            continue
        E = {rg: np.array([[runs[r][k][f"e_{rg}"] for k in keys] for r in chosen], float) for rg in ("orthographic", "no_diacritics", "raw")}
        N = {rg: np.array([max(1, runs[chosen[0]][k][f"n_{rg}"]) for k in keys], float) for rg in ("orthographic", "no_diacritics", "raw")}
        cer = {rg: E[rg].sum(1) / N[rg].sum() for rg in E}
        tau = kendalltau(cer["orthographic"], cer["no_diacritics"])[0]
        rho = spearmanr(cer["orthographic"], cer["no_diacritics"])[0]
        rng = np.random.default_rng(0)
        bt, br = [], []
        for _ in range(a.reps):
            w = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
            c1 = E["orthographic"] @ w / (N["orthographic"] @ w)
            c2 = E["no_diacritics"] @ w / (N["no_diacritics"] @ w)
            bt.append(kendalltau(c1, c2)[0])
            br.append(spearmanr(c1, c2)[0])
        r1 = np.argsort(np.argsort(cer["orthographic"]))
        r2 = np.argsort(np.argsort(cer["no_diacritics"]))
        gt_marked = np.array([line_diac(samples[k]["gt_raw"])[0] > 0 for k in keys])
        orth = {k: errors.ladder_strings(samples[k]["gt_raw"])[3] for k in keys}
        per = []
        for i, r in enumerate(chosen):
            ins, gl = 0, 0
            for k in keys:
                dm = errors.diacritic_mismatch(orth[k], errors.ladder_strings(runs[r][k]["hyp"] or "")[3])
                ins += dm["inserted"]
                gl += dm["aligned_letters"]
            tax_line = (E["orthographic"][i] - E["no_diacritics"][i])
            lo, hi = np.percentile([tax_line[idx].sum() / N["orthographic"][idx].sum() for idx in (rng.integers(0, len(keys), len(keys)) for _ in range(500))], [2.5, 97.5])
            per.append({"run": r, "cer_raw": float(cer["raw"][i]), "cer_orth": float(cer["orthographic"][i]), "cer_no_diac": float(cer["no_diacritics"][i]),
                        "diacritic_tax": float(cer["orthographic"][i] - cer["no_diacritics"][i]), "tax_ci95": [float(lo), float(hi)],
                        "rank_orth": int(r1[i]) + 1, "rank_no_diac": int(r2[i]) + 1, "inserted_marks_per_100_letters": 100 * ins / max(1, gl)})
        per.sort(key=lambda x: x["rank_no_diac"])
        tau_lo = float(np.nanpercentile(bt, 2.5))
        immaterial = tau_lo >= 0.90
        pol = prof[ds]["policy"]
        headline = "no_diacritics" if pol in ("undiacritised", "sparse") else "orthographic (marks kept), no_diacritics beside it"
        res[ds] = {"policy": pol, "systems": len(chosen), "lines": len(keys), "lines_with_gt_marks": float(gt_marked.mean()),
                   "kendall_tau": float(tau), "tau_ci95": [tau_lo, float(np.nanpercentile(bt, 97.5))],
                   "spearman_rho": float(rho), "rho_ci95": [float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))],
                   "max_rank_move": int(np.abs(r1 - r2).max()), "systems_moving_ge2": [p["run"] for p in per if abs(p["rank_orth"] - p["rank_no_diac"]) >= 2],
                   "top5_set_changed": {p["run"] for p in per if p["rank_no_diac"] <= 5} != {p["run"] for p in per if p["rank_orth"] <= 5},
                   "marks_immaterial_to_ranking": immaterial, "headline_rung": headline,
                   "marks_kept_quotable": bool(immaterial or pol == "diacritised"), "per_system": per}
        # gallery material: the system with the biggest tax; its lines with the largest / zero tax on marked GT
        worst = max(per, key=lambda p: p["diacritic_tax"])
        i = chosen.index(worst["run"])
        tl = (E["orthographic"][i] - E["no_diacritics"][i])
        order = np.argsort(-tl)
        good = [j for j in range(len(keys)) if gt_marked[j] and tl[j] == 0 and E["no_diacritics"][i][j] == 0]
        rnd = random.Random(0)
        gal[ds] = (worst["run"], [(keys[j], int(tl[j]), "worst") for j in order[:5]] + [(keys[j], 0, "best") for j in rnd.sample(good, min(3, len(good)))] +
                   [(keys[j], int(tl[j]), "random") for j in rnd.sample(range(len(keys)), 4)])

    # sparse-marks note (Muharaf): do fine-tunes trained on stripped targets add marks? share of marked lines in test vs train
    note = {"muharaf_test": prof.get("muharaf_lines"), "muharaf_train_raw": prof.get("muharaf_train"),
            "text": "Muharaf labels carry sparse marks (see share_lines_with_marks); the CRNN / aligner training targets strip them "
                    "(clean()), so CRNN readings never carry marks; the marks-kept rung charges every CRNN the GT marks."}
    out = {"decision_rule": __doc__.split("Decision rule (written before the run):")[1].split("Outputs:")[0].strip(),
           "R1.3_profile": prof, "R2.7": res, "R2.7_leaderboard_view": wide, "sparse_marks_note": note}
    json.dump(out, open(a.out / "diacritics.json", "w"), ensure_ascii=False, indent=1)
    for ds, r in res.items():
        if "kendall_tau" in r:
            print(ds, r["policy"], r["systems"], r["lines"], "tau %.3f [%.3f,%.3f]" % (r["kendall_tau"], *r["tau_ci95"]), "maxmove", r["max_rank_move"],
                  "immaterial", r["marks_immaterial_to_ranking"], "max tax %.3f" % max(p["diacritic_tax"] for p in r["per_system"]))
        else:
            print(ds, r)

    # ---- gallery
    imgs = panel_images({k for _, L in gal.values() for k, _, _ in L})
    html = [HTML_HEAD.format(title="Diacritic scoring rule"), "<h1>Diacritic scoring rule (checklist 2.8, R2.7) and policy share (R1.3)</h1>",
            f"<p class=mut>{out['decision_rule']}</p><h2>R1.3 policy fingerprint</h2><table><tr><th>set</th><th>lines</th><th>lines with marks</th>"
            "<th>harakat / 100 letters</th><th>coverage of marked lines</th><th>policy</th></tr>"]
    for k, v in prof.items():
        html.append(f"<tr><td>{k}</td><td>{v['lines']}</td><td>{v['share_lines_with_marks']:.1%}</td><td>{v['harakat_per_100_letters']:.2f}</td>"
                    f"<td>{v['mean_coverage_of_marked_lines']:.1%}</td><td>{v['policy']}</td></tr>")
    html.append("</table><h2>Rank agreement, marks kept vs stripped</h2><table><tr><th>set</th><th>policy</th><th>systems</th><th>lines</th><th>Kendall τ [95% CI]</th>"
                "<th>Spearman ρ</th><th>max rank move</th><th>immaterial?</th><th>headline rung</th></tr>")
    for ds, r in res.items():
        if "kendall_tau" in r:
            html.append(f"<tr><td>{ds}</td><td>{r['policy']}</td><td>{r['systems']}</td><td>{r['lines']}</td><td>{r['kendall_tau']:.3f} [{r['tau_ci95'][0]:.3f}, {r['tau_ci95'][1]:.3f}]</td>"
                        f"<td>{r['spearman_rho']:.3f}</td><td>{r['max_rank_move']}</td><td>{r['marks_immaterial_to_ranking']}</td><td>{r['headline_rung']}</td></tr>")
    html.append("</table><h2>Leaderboard view (every system on its own lines)</h2><table><tr><th>set</th><th>systems</th><th>Kendall τ [95% CI]</th><th>max move</th><th>immaterial?</th><th>movers (orth → no-diac rank)</th></tr>")
    for ds, r in wide.items():
        html.append(f"<tr><td>{ds}</td><td>{r['systems']}</td><td>{r['kendall_tau']:.3f} [{r['tau_ci95'][0]:.3f}, {r['tau_ci95'][1]:.3f}]</td><td>{r['max_rank_move']}</td>"
                    f"<td>{r['marks_immaterial_to_ranking']}</td><td>{'; '.join(f'{m[0]} {m[1]}→{m[2]}' for m in r['movers'])}</td></tr>")
    html.append("</table>")
    for ds, r in res.items():
        if "per_system" not in r:
            continue
        html.append(f"<h3>{ds}</h3><table><tr><th>system</th><th>CER orth</th><th>CER no-diac</th><th>tax [95% CI]</th><th>rank orth → no-diac</th><th>inserted marks /100 letters</th></tr>")
        for p in r["per_system"]:
            html.append(f"<tr><td>{p['run']}</td><td>{p['cer_orth']:.3f}</td><td>{p['cer_no_diac']:.3f}</td><td>{p['diacritic_tax']*100:.2f} pt [{p['tax_ci95'][0]*100:.2f}, {p['tax_ci95'][1]*100:.2f}]</td>"
                        f"<td>{p['rank_orth']} → {p['rank_no_diac']}</td><td>{p['inserted_marks_per_100_letters']:.2f}</td></tr>")
        html.append("</table>")
    for ds, (run, L) in gal.items():
        html.append(f"<h2>{ds}: lines for {run} (largest diacritic tax = worst; marked GT read exactly = best; random)</h2>")
        for k, t, kind in L:
            if k not in imgs:
                continue
            hyp = systems[ds][run][k]["hyp"]
            html.append(f"<div class=card><div class=mut>{kind} · {k} · tax {t} edits</div><img src='{png_b64(imgs[k])}'><div class=ar>GT: {samples[k]['gt_raw']}</div>"
                        f"<div class=ar>{run}: {hyp}</div></div>")
    html.append("</body></html>")
    (a.out / "diacritics_gallery.html").write_text("\n".join(html))


if __name__ == "__main__":
    main()
