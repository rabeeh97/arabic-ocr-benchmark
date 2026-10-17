"""Pattern detectors over the per-line store (plan 2026-10-13, §3.5).

    .geomvenv/bin/python scripts/run_detectors.py [--store results/store] [--out results/analysis/D]

Each detector flags a line (sample-side: image or text) or a reading (prediction-side) by a documented rule. For every
run with ≥ 50 lines on a dataset (preferred pack), the output gives per detector: the share of lines flagged, the share
of the run's edits on flagged lines, mean line CER flagged vs not, and the **excess edits explained** = Σ over flagged
lines of (edits − n_chars × CER of the unflagged lines) / total edits. Lines above the run's 80th percentile of line
CER that no detector flags form the **residual**, the feed for new detectors (D/residual_<dataset>.json).
Each detector row also gets a two-sided Mann–Whitney p-value (line CER flagged vs unflagged); the detectors of one run ×
dataset are one false-discovery family, Benjamini–Hochberg at q = 0.1 (eval_stats.bh), and `fdr_survives` marks the rows
that pass (evaluation-standards item 5).
Rung: no_diacritics. Detectors needing image measurements beyond ink extent (slant, rules, stroke) come with
store_features.py (D2, D3, D5, D6) and detectors_extra.py (D4, results/store/intrusion.parquet).
Checklist 1.4 / 1.5 added D4 neighbour-line intrusion, D21 lam-alef ligature, D23 memorisation signature and D24 seed or
numeric instability (seed_instability.py); rules in detectors_extra.py and seed_instability.py. Two extra tables:
  d21_ligature  per run × dataset, the per-letter misread rate of lam and alef inside a ligature vs elsewhere (two-proportion z,
                BH over the table);
  d23_runs      the C6 check, per run × dataset: whitespace-quirk and quirk-word reproduction rates and the exact-line
                rate, each against the median of the 5 runs nearest in CER (other lineages); `signature` when the exact
                rate is ≥ 3× that median and ≥ 5 lines are exact (data-and-label-audit §9). AIN rows stay "unverified" (CORRECTIONS C6).
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from analyze_letters import outcomes, words_view  # noqa: E402
from arabic_ocr_bench.characterize.samples import _letters_view  # noqa: E402
from eval_stats import Q_FDR, bh  # noqa: E402
from scipy.stats import mannwhitneyu, norm  # noqa: E402
from detectors_extra import lam_alef, lam_alef_letters, lam_and_alef_elsewhere, quirk_hits, ws_quirks  # noqa: E402
import seed_instability  # noqa: E402

RUNG = "no_diacritics"
HARAKAT = re.compile("[ً-ْٰ]")
LATIN_CJK = re.compile("[A-Za-z぀-ヿ一-鿿]")
DIGITS = re.compile("[0-9٠-٩۰-۹]")


def lc(r):
    return r[f"e_{RUNG}"] / max(1, r[f"n_{RUNG}"])


# ---- sample-side (depend only on the line) -------------------------------------------------------------
def d1_loose_crop(s, ctx):
    return s["ink_extent"] < 0.35


def d7_fragment(s, ctx):
    g = _letters_view(s["gt_raw"])
    return len(g) <= 20 or bool(DIGITS.fullmatch(g.replace(" ", ""))) or bool(LATIN_CJK.search(g))


def d8_letter_list(s, ctx):
    return sum(len(w) == 1 for w in words_view(s["gt_raw"])) >= 4


def d9_unseen_text(s, ctx):
    w = words_view(s["gt_raw"])
    return bool(w) and sum(x not in ctx["vocab"] for x in w) / len(w) >= 0.5


def d10_label_suspect(s, ctx):
    return s["sample_key"] in ctx["label_suspects"]


def d10b_label_tail(s, ctx):
    return s["sample_key"] in ctx["tail_suspects"]


def dx_ctc_filtered(s, ctx):
    return not s["ctc_ok"]


# ---- prediction-side (depend on the reading) ------------------------------------------------------------
def d11_truncation(s, r, ctx):
    g, h = _letters_view(s["gt_raw"]), _letters_view(r["hyp"])
    if len(g) < 10:
        return False
    ops = Levenshtein.opcodes(g, h)
    tail = ops[-1]
    head = ops[0]
    cut_tail = tail.tag == "delete" and tail.src_end - tail.src_start >= 0.1 * len(g)
    cut_head = head.tag == "delete" and head.src_end - head.src_start >= 0.1 * len(g)
    return cut_tail or cut_head


def d12_loop(s, r, ctx):
    h = r["hyp"]
    w = h.split()
    for n in (1, 2, 3, 4):
        for i in range(len(w) - 3 * n + 1):
            if w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n]:
                return True
    if re.search(r"(.{4})\1{4,}", h):
        return True
    return len(h) > 1.5 * max(1, len(s["gt_raw"])) + 10


def d13_whole_line(s, r, ctx):
    return lc(r) > 0.8


def d14_script_mismatch(s, r, ctx):
    h = r["hyp"]
    return bool(h) and not LATIN_CJK.search(s["gt_raw"]) and len(LATIN_CJK.findall(h)) >= 0.2 * len(h.replace(" ", ""))


def d15_fluent_rewrite(s, r, ctx):
    if lc(r) <= 0.3:
        return False
    gw, hw = words_view(s["gt_raw"]), words_view(r["hyp"])
    wrong = [hw[j] for op in Levenshtein.opcodes(gw, hw) if op.tag != "equal" for j in range(op.dest_start, op.dest_end)]
    return len(wrong) >= 2 and sum(x in ctx["vocab"] for x in wrong) >= 0.5 * len(wrong)


def d16_diacritic_insertion(s, r, ctx):
    n = max(1, len(_letters_view(s["gt_raw"])))
    return 100 * (len(HARAKAT.findall(r["hyp"])) - len(HARAKAT.findall(s["gt_raw"]))) / n > 2


def d17_orthographic_only(s, r, ctx):
    e0 = r["e_tatweel_digits"]
    return e0 > 0 and (e0 - r["e_orthographic"]) >= 0.5 * e0


def d18_dot_only(s, r, ctx):
    e0 = r["e_no_diacritics"]
    return e0 > 0 and (e0 - r["e_rasm"]) >= 0.5 * e0


def d19_maghrebi_dots(s, r, ctx):
    return any(o["oc"] == "sub" and (o["ch"], o["read"]) in (("ق", "ف"), ("ف", "ب")) for o in outcomes(s["gt_raw"], r["hyp"])[0])


def d20_paw_gap_space(s, r, ctx):
    return outcomes(s["gt_raw"], r["hyp"])[1]["space_ins_at_paw_gap"] >= 1


def d22_reversed(s, r, ctx):
    return bool(r["reversed_suspect"])


def _feat(s, k):
    f = s.get("feat") or {}
    return bool(f.get("ok")) and bool(f.get(k))


def d2_slant_outlier(s, ctx):
    return _feat(s, "slant_outlier")


def d3_ruled(s, ctx):
    return _feat(s, "ruled")


def d5_low_res(s, ctx):
    return _feat(s, "low_res")


def d6_stroke_pen_outlier(s, ctx):
    return _feat(s, "stroke_outlier") or _feat(s, "pen_outlier")


def d4_intrusion(s, ctx):
    return bool((s.get("intr") or {}).get("d4"))


def d24_unstable(s, ctx):
    return s["sample_key"] in ctx["seed_unstable"] or s["sample_key"] in ctx["numeric_unstable"]


def d21_lam_alef(s, r, ctx):
    return lc(r) <= 0.8 and lam_alef(s["gt_raw"], r["hyp"])[1] >= 1


def quirk_words_for(k, run, ctx) -> set[str]:
    """D23 quirk words of line k, leaving the run itself out of the references."""
    rs = ctx["refstats"].get(k)
    if not rs:
        return set()
    n, good, cnt, own = rs["n"], rs["good"], rs["count"], rs["own"].get(run)
    if own is not None:
        n, good = n - 1, good - own[1]
    if n < 5 or good < 0.6 * n:
        return set()
    return {w for w in ctx["gt_words"][k] if len(w) >= 2 and cnt.get(w, 0) - (own is not None and w in own[0]) <= 0.1 * n}


def d23_memorisation(s, r, ctx):
    k = s["sample_key"]
    if quirk_hits(ctx["ws_quirks"][k], r["hyp"])[1] >= 1:
        return True
    return bool(quirk_words_for(k, r["run_id"], ctx) & set(words_view(r["hyp"])))


SAMPLE_DETECTORS = {"D1 loose crop (ink < 35 % of height)": d1_loose_crop, "D2 slant outside training p2.5–97.5": d2_slant_outlier,
                    "D3 ruled lines (> 2 % of ink)": d3_ruled, "D4 neighbour-line intrusion (edge ink or two baselines)": d4_intrusion, "D5 low resolution (px per char < train p5, or dots < 2 px)": d5_low_res,
                    "D6 stroke width or pen contrast outside training": d6_stroke_pen_outlier, "D7 fragment line": d7_fragment, "D8 letter-list drill": d8_letter_list,
                    "D9 mostly unseen words": d9_unseen_text, "D10 label suspect (strong models agree, GT differs)": d10_label_suspect,
                    "D10b label runs past the image (systems agree the tail is missing)": d10b_label_tail,
                    "D24 seed or numeric instability (CRNN seeds / fp16 vs fp32)": d24_unstable,
                    "Dx dropped by the old CTC filter": dx_ctc_filtered}
PRED_DETECTORS = {"D11 truncation (head or tail cut)": d11_truncation, "D12 repetition loop / runaway": d12_loop, "D13 whole-line failure (CER > 0.8)": d13_whole_line,
                  "D14 script mismatch": d14_script_mismatch, "D15 fluent rewrite": d15_fluent_rewrite, "D16 diacritic insertion": d16_diacritic_insertion,
                  "D17 orthographic only": d17_orthographic_only, "D18 dot only": d18_dot_only, "D19 Maghrebi dot swap (ق→ف, ف→ب)": d19_maghrebi_dots,
                  "D20 space at a PAW gap": d20_paw_gap_space, "D21 lam-alef ligature misread": d21_lam_alef, "D22 reversed reading": d22_reversed,
                  "D23 memorisation signature (GT quirk reproduced)": d23_memorisation}


def label_suspects(idx, S) -> set[str]:
    """D10: ≥ 3 competent VLM runs read the line alike (pairwise CER < 0.1) and all miss the GT (CER > 0.3)."""
    by_key = defaultdict(list)
    for (run, tr, pack, ds), m in idx.items():
        if tr != "none" or not run.startswith("vlm/") or any(x in run for x in ("ain", "paddle", "trocr")) or len(m) < 50:
            continue
        cer = np.mean([lc(r) for r in m.values()])
        if cer > 0.35:
            continue
        for k, r in m.items():
            by_key[k].append(r)
    out = set()
    for k, rs in by_key.items():
        miss = [r for r in rs if lc(r) > 0.3]
        if len(miss) < 3:
            continue
        hs = [_letters_view(r["hyp"]) for r in miss]
        agree = sum(Levenshtein.normalized_distance(a, b) < 0.1 for i, a in enumerate(hs) for b in hs[i + 1:])
        if agree >= 3:
            out.add(k)
    return out


def competent_vlm_runs(idx):
    """(run, pack, dataset) of the competent VLM readings (as D10): transform none, not AIN / PaddleOCR / TrOCR, ≥ 50 lines,
    mean line CER ≤ 0.35."""
    out = []
    for (run, tr, pack, ds), m in idx.items():
        if tr != "none" or not run.startswith("vlm/") or any(x in run for x in ("ain", "paddle", "trocr")) or len(m) < 50:
            continue
        if np.mean([lc(r) for r in m.values()]) <= 0.35:
            out.append((run, pack, ds))
    return out


def ref_stats(idx, S) -> dict:
    """D23: per line, the competent VLM references: count, how many read the line well (CER ≤ 0.3), how many contain each
    word, and each reference's own (words, good) so that a run can be left out of its own reference set."""
    out = {}
    for run, pack, ds in competent_vlm_runs(idx):
        for k, r in idx[(run, "none", pack, ds)].items():
            o = out.setdefault(k, {"n": 0, "good": 0, "count": defaultdict(int), "own": {}})
            ws = set(words_view(r["hyp"]))
            good = lc(r) <= 0.3
            o["n"] += 1
            o["good"] += good
            for w in ws:
                o["count"][w] += 1
            o["own"][run] = (ws, good)
    return out


def lineage(run: str) -> str:
    """Runs of one model family (AIN-7B and its fine-tunes) are not each other's peers in the C6 check."""
    name = run.split("@")[0]
    return "ain" if "ain" in name.split("/")[-1] else name


def d23_runs(chosen, ctx) -> list[dict]:
    """The C6 check per run × dataset (transform none)."""
    base = {}
    for (spec, ds), (_, pack, m) in chosen.items():
        run, tr = spec.split("@")
        if tr != "none":
            continue
        e = sum(r[f"e_{RUNG}"] for r in m.values())
        n = sum(r[f"n_{RUNG}"] for r in m.values())
        wo = wr = qo = qr = 0
        for k, r in m.items():
            o, x = quirk_hits(ctx["ws_quirks"][k], r["hyp"])
            wo, wr = wo + o, wr + x
            qw = quirk_words_for(k, run, ctx)
            qo, qr = qo + len(qw), qr + len(qw & set(words_view(r["hyp"])))
        base[(spec, ds)] = {"run": spec, "dataset": ds, "lines": len(m), "cer": e / max(1, n),
                            "exact_rate": sum(r[f"e_{RUNG}"] == 0 for r in m.values()) / len(m),
                            "ws_quirk_opportunities": wo, "ws_quirk_reproduced": wr, "ws_quirk_rate": wr / wo if wo else None,
                            "quirk_words": qo, "quirk_words_reproduced": qr, "quirk_word_rate": qr / qo if qo else None}
    rows = []
    for (spec, ds), v in base.items():
        others = [x for (s2, d2), x in base.items() if d2 == ds and lineage(s2) != lineage(spec)]
        peers = sorted(others, key=lambda x: abs(np.log(max(x["cer"], 1e-4) / max(v["cer"], 1e-4))))[:5]
        row = dict(v, peers=[x["run"] for x in peers])
        for key in ("exact_rate", "ws_quirk_rate", "quirk_word_rate"):
            pv = [x[key] for x in peers if x[key] is not None]
            med = float(np.median(pv)) if pv else None
            row[f"{key}_peer_median"] = med
            floor = 0.5 / v["lines"]  # half a line, so that a zero median does not divide by zero
            row[f"{key}_ratio"] = None if med is None or v[key] is None else v[key] / max(med, floor)
        row["signature"] = bool(row["exact_rate_ratio"] is not None and row["exact_rate_ratio"] >= 3
                                and v["exact_rate"] * v["lines"] >= 5)  # ≥ 5 exact lines, so one or two lucky lines do not count
        row["status"] = "unverified (C6)" if lineage(spec) == "ain" else ("signature" if row["signature"] else "none")
        rows.append(row)
    return sorted(rows, key=lambda x: (x["dataset"], x["run"]))


def d21_ligature(chosen, S) -> list[dict]:
    """Per run × dataset: lam and alef letters inside a lam-alef ligature misread vs lam and alef letters elsewhere misread
    (per letter, two-proportion z, BH)."""
    rows = []
    for (spec, ds), (_, pack, m) in chosen.items():
        lp = le = op = oe = 0
        for k, r in m.items():
            a_, b_ = lam_alef_letters(S[k]["gt_raw"], r["hyp"])
            c_, d_ = lam_and_alef_elsewhere(S[k]["gt_raw"], r["hyp"])
            lp, le, op, oe = lp + a_, le + b_, op + c_, oe + d_
        if not lp or not op:
            continue
        p1, p2 = le / lp, oe / op
        pp = (le + oe) / (lp + op)
        se = np.sqrt(pp * (1 - pp) * (1 / lp + 1 / op))
        rows.append({"run": spec, "dataset": ds, "ligature_letters": lp, "ligature_misread": p1, "lam_alef_elsewhere": op, "elsewhere_misread": p2,
                     "ratio": p1 / p2 if p2 else None, "p": float(2 * norm.sf(abs(p1 - p2) / se)) if se > 0 else 1.0})
    rej, adj = bh([r["p"] for r in rows], Q_FDR)
    for r, j, q in zip(rows, rej, adj):
        r["q_bh"], r["fdr_survives"] = float(q), bool(j)
    return sorted(rows, key=lambda x: (x["dataset"], x["run"]))


def tail_cut(gt: str, hyp: str) -> float:
    """Share of the GT (letters view) deleted at its end, after the last matching block."""
    g, h = _letters_view(gt), _letters_view(hyp)
    if not g:
        return 0.0
    last = max((op.src_end for op in Levenshtein.opcodes(g, h) if op.tag == "equal"), default=0)
    return (len(g) - last) / len(g)


def tail_suspects(idx, S) -> set[str]:
    """D10b (CORRECTIONS C8): ≥ 5 competent systems read the line, and ≥ 70 % of them leave the last ≥ 15 % of the GT
    unmatched while matching earlier parts — the label most likely continues past what the image shows."""
    by_key = defaultdict(list)
    for (run, tr, pack, ds), m in idx.items():
        if tr != "none" or any(x in run for x in ("ain", "paddle", "trocr")) or len(m) < 50:
            continue
        if not (run.startswith("vlm/") or run in ("crnn/base", "crnn/rep2")):
            continue
        if np.mean([lc(r) for r in m.values()]) > 0.35:
            continue
        for k, r in m.items():
            by_key[k].append(r)
    out = set()
    for k, rs in by_key.items():
        if len(rs) < 5 or d7_fragment(S[k], None):  # short, digit-only or Latin lines give spurious tails
            continue
        cut = [tail_cut(S[k]["gt_raw"], r["hyp"]) >= 0.15 and lc(r) < 0.9 for r in rs]
        if sum(cut) >= 0.7 * len(rs):
            out.add(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--pairs", type=Path, default=W / "configs/analysis/pairs.json")
    ap.add_argument("--vocab-pkl", type=Path, default=W / "results/panel/aligner_data_standard.pkl")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/D")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    packs = json.loads(a.pairs.read_text(encoding="utf-8"))["pack_pref"]
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    if (a.store / "features.parquet").exists():  # image measurements from store_features.py
        for f in pq.read_table(a.store / "features.parquet").to_pylist():
            if f["sample_key"] in S:
                S[f["sample_key"]]["feat"] = f
    if (a.store / "intrusion.parquet").exists():  # D4 measurements from detectors_extra.py
        for f in pq.read_table(a.store / "intrusion.parquet").to_pylist():
            if f["sample_key"] in S:
                S[f["sample_key"]]["intr"] = f
    idx = defaultdict(dict)
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        if r.get("in_train") == "yes":  # never score lines a run trained on (CORRECTIONS C4)
            continue
        idx[(r["run_id"], r["transform"], r["pack"], r["dataset"])][r["sample_key"]] = r
    ctx = {"vocab": {w for r in pickle.load(open(a.vocab_pkl, "rb"))["rows"] if r["role"] == "train" for w in words_view(r["label"])}}
    ctx["label_suspects"] = label_suspects(idx, S)
    ctx["tail_suspects"] = tail_suspects(idx, S)
    print("D10b tail suspects:", len(ctx["tail_suspects"]), sorted(ctx["tail_suspects"])[:12])
    inst = seed_instability.instability(seed_instability.load(a.store))
    ctx["seed_unstable"], ctx["numeric_unstable"] = inst["seed"], inst["numeric"]
    ctx["ws_quirks"] = {k: ws_quirks(s["gt_raw"]) for k, s in S.items()}
    ctx["gt_words"] = {k: set(words_view(s["gt_raw"])) for k, s in S.items()}
    ctx["refstats"] = ref_stats(idx, S)
    print("D24 seed-unstable", len(ctx["seed_unstable"]), "numeric-unstable", len(ctx["numeric_unstable"]))
    sflags = {name: {k for k, s in S.items() if f(s, ctx)} for name, f in SAMPLE_DETECTORS.items()}
    prevalence = {ds: {name: sum(S[k]["dataset"] == ds for k in v) / sum(s["dataset"] == ds for s in S.values()) for name, v in sflags.items()}
                  for ds in sorted({s["dataset"] for s in S.values()})}

    chosen = {}
    for (run, tr, pack, ds), m in idx.items():
        if len(m) < 50:
            continue
        key = (f"{run}@{tr}", ds)
        rank = packs.index(pack) if pack in packs else 99
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, pack, m)
    report, residual = {}, defaultdict(dict)
    for (spec, ds), (_, pack, m) in sorted(chosen.items()):
        keys = list(m)
        e = np.array([m[k][f"e_{RUNG}"] for k in keys], float)
        n = np.array([max(1, m[k][f"n_{RUNG}"]) for k in keys], float)
        flags = {name: np.array([k in v for k in keys]) for name, v in sflags.items()}
        for name, f in PRED_DETECTORS.items():
            flags[name] = np.array([bool(f(S[k], m[k], ctx)) for k in keys])
        rows = {}
        cer = e / n
        for name, fl in flags.items():
            if not fl.any():
                rows[name] = {"share_lines": 0.0}
                continue
            base = e[~fl].sum() / n[~fl].sum() if (~fl).any() else 0.0
            rows[name] = {"share_lines": float(fl.mean()), "share_edits": float(e[fl].sum() / max(1, e.sum())),
                          "cer_flagged": float(e[fl].sum() / n[fl].sum()), "cer_unflagged": float(base),
                          "excess_edits_explained": float((e[fl] - n[fl] * base).sum() / max(1, e.sum())),
                          "example_ids": [k for k, f_ in zip(keys, fl) if f_][:6]}
            if (~fl).any():
                rows[name]["p"] = float(mannwhitneyu(cer[fl], cer[~fl], alternative="two-sided").pvalue)
        tested = [d for d, x in rows.items() if "p" in x]
        rej, adj = bh([rows[d]["p"] for d in tested], Q_FDR)
        for d, r_, q_ in zip(tested, rej, adj):
            rows[d]["q_bh"], rows[d]["fdr_survives"] = float(q_), bool(r_)
        any_flag = np.zeros(len(keys), bool)
        for fl in flags.values():
            any_flag |= fl
        hi = cer >= np.percentile(cer, 80)
        res = [k for k, h, f_ in zip(keys, hi, any_flag) if h and not f_]
        residual[ds][spec] = res
        report[f"{spec}|{ds}"] = {"pack": pack, "lines": len(keys), "cer": float(e.sum() / n.sum()), "detectors": rows,
                                  "fdr_q": Q_FDR, "n_detectors_tested": len(tested), "n_detectors_fdr": int(rej.sum()),
                                  "residual_lines": len(res), "residual_share_of_top20": len(res) / max(1, int(hi.sum()))}
    t21, t23 = d21_ligature(chosen, S), d23_runs(chosen, ctx)
    (a.out / "detectors.json").write_text(json.dumps({"prevalence": prevalence, "runs": report, "label_suspects": sorted(ctx["label_suspects"]),
                                                      "d21_ligature": t21, "d23_runs": t23},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    for ds, v in residual.items():
        (a.out / f"residual_{ds}.json").write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")

    print("prevalence (sample-side):")
    for ds, v in prevalence.items():
        print(f"  {ds:14s} " + "  ".join(f"{k.split(' ')[0]} {100 * x:.1f}%" for k, x in v.items()))
    print(f"label suspects: {len(ctx['label_suspects'])}")
    show = ["crnn/base@none", "crnn/base@inkcrop", "vlm/suite-v1-qwen3vl8b-ft-bf16@none", "vlm/qwen3-vl-8b-bf16@none", "crnn+llm/e8-qwen3-0.6b@none", "vlm/gemini-flash@none"]
    for k, v in report.items():
        spec, ds = k.split("|")
        if spec not in show or ds not in ("kitab_khatt", "nakba_lines", "rasam1"):
            continue
        top = sorted(((d, x.get("excess_edits_explained", 0)) for d, x in v["detectors"].items()), key=lambda t: -t[1])[:4]
        print(f"\n{k} lines {v['lines']} CER {100 * v['cer']:.1f}%  residual {v['residual_lines']} ({100 * v['residual_share_of_top20']:.0f}% of the worst fifth unexplained)")
        for d, x in top:
            dd = v["detectors"][d]
            print(f"   {d:48s} lines {100 * dd['share_lines']:5.1f}%  edits {100 * dd.get('share_edits', 0):5.1f}%  excess explained {100 * x:5.1f}%"
                  f"  q {dd.get('q_bh', float('nan')):.3g}{' *' if dd.get('fdr_survives') else ''}")
    new = [d for d in [*SAMPLE_DETECTORS, *PRED_DETECTORS] if d.startswith(("D4 ", "D21 ", "D23 ", "D24 "))]
    print("\nnew detectors (1.4 / 1.5): runs × datasets with hits, and rows surviving BH")
    for d in new:
        rows = [v["detectors"][d] for v in report.values() if d in v["detectors"]]
        hit = [x for x in rows if x.get("share_lines", 0) > 0]
        print(f"   {d:62s} hit in {len(hit):3d}/{len(rows)}  FDR survivors {sum(bool(x.get('fdr_survives')) for x in rows):3d}"
              f"  median share {100 * np.median([x['share_lines'] for x in hit]) if hit else 0:5.1f}%")
    sig = [x for x in t21 if x["fdr_survives"]]
    print(f"\nD21 ligature vs lam/alef elsewhere: {len(sig)}/{len(t21)} rows survive BH; ligature harder in {sum(x['ratio'] and x['ratio'] > 1 for x in sig)}")
    print("C6 / D23 per run (exact-line, whitespace-quirk ratios vs the 5 nearest-CER peers):")
    for x in t23:
        if x["status"] != "none" or x["dataset"] == "kitab_khatt":
            f = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
            print(f"   {x['run']:44s} {x['dataset']:13s} CER {100 * x['cer']:5.1f}%  exact {100 * x['exact_rate']:5.1f}% ×{f(x['exact_rate_ratio'])}"
                  f"  ws quirk {x['ws_quirk_reproduced']}/{x['ws_quirk_opportunities']} ×{f(x['ws_quirk_rate_ratio'])}"
                  f"  quirk words {x['quirk_words_reproduced']}/{x['quirk_words']}  {x['status']}")


if __name__ == "__main__":
    main()
