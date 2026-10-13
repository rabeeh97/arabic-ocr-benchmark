"""Pattern detectors over the per-line store (plan 2026-10-13, §3.5).

    .geomvenv/bin/python scripts/run_detectors.py [--store results/store] [--out results/analysis/D]

Each detector flags a line (sample-side: image or text) or a reading (prediction-side) by a documented rule. For every
run with ≥ 50 lines on a dataset (preferred pack), the output gives per detector: the share of lines flagged, the share
of the run's edits on flagged lines, mean line CER flagged vs not, and the **excess edits explained** = Σ over flagged
lines of (edits − n_chars × CER of the unflagged lines) / total edits. Lines above the run's 80th percentile of line
CER that no detector flags form the **residual**, the feed for new detectors (D/residual_<dataset>.json).
Rung: no_diacritics. Detectors needing image measurements beyond ink extent (slant, rules, stroke) come with
store_features.py; D2–D6 are listed but not run yet.
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


SAMPLE_DETECTORS = {"D1 loose crop (ink < 35 % of height)": d1_loose_crop, "D7 fragment line": d7_fragment, "D8 letter-list drill": d8_letter_list,
                    "D9 mostly unseen words": d9_unseen_text, "D10 label suspect (strong models agree, GT differs)": d10_label_suspect,
                    "D10b label runs past the image (systems agree the tail is missing)": d10b_label_tail,
                    "Dx dropped by the old CTC filter": dx_ctc_filtered}
PRED_DETECTORS = {"D11 truncation (head or tail cut)": d11_truncation, "D12 repetition loop / runaway": d12_loop, "D13 whole-line failure (CER > 0.8)": d13_whole_line,
                  "D14 script mismatch": d14_script_mismatch, "D15 fluent rewrite": d15_fluent_rewrite, "D16 diacritic insertion": d16_diacritic_insertion,
                  "D17 orthographic only": d17_orthographic_only, "D18 dot only": d18_dot_only, "D19 Maghrebi dot swap (ق→ف, ف→ب)": d19_maghrebi_dots,
                  "D20 space at a PAW gap": d20_paw_gap_space, "D22 reversed reading": d22_reversed}


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
    idx = defaultdict(dict)
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        idx[(r["run_id"], r["transform"], r["pack"], r["dataset"])][r["sample_key"]] = r
    ctx = {"vocab": {w for r in pickle.load(open(a.vocab_pkl, "rb"))["rows"] if r["role"] == "train" for w in words_view(r["label"])}}
    ctx["label_suspects"] = label_suspects(idx, S)
    ctx["tail_suspects"] = tail_suspects(idx, S)
    print("D10b tail suspects:", len(ctx["tail_suspects"]), sorted(ctx["tail_suspects"])[:12])
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
        for name, fl in flags.items():
            if not fl.any():
                rows[name] = {"share_lines": 0.0}
                continue
            base = e[~fl].sum() / n[~fl].sum() if (~fl).any() else 0.0
            rows[name] = {"share_lines": float(fl.mean()), "share_edits": float(e[fl].sum() / max(1, e.sum())),
                          "cer_flagged": float(e[fl].sum() / n[fl].sum()), "cer_unflagged": float(base),
                          "excess_edits_explained": float((e[fl] - n[fl] * base).sum() / max(1, e.sum())),
                          "example_ids": [k for k, f_ in zip(keys, fl) if f_][:6]}
        any_flag = np.zeros(len(keys), bool)
        for fl in flags.values():
            any_flag |= fl
        cer = e / n
        hi = cer >= np.percentile(cer, 80)
        res = [k for k, h, f_ in zip(keys, hi, any_flag) if h and not f_]
        residual[ds][spec] = res
        report[f"{spec}|{ds}"] = {"pack": pack, "lines": len(keys), "cer": float(e.sum() / n.sum()), "detectors": rows,
                                  "residual_lines": len(res), "residual_share_of_top20": len(res) / max(1, int(hi.sum()))}
    (a.out / "detectors.json").write_text(json.dumps({"prevalence": prevalence, "runs": report, "label_suspects": sorted(ctx["label_suspects"])},
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
            print(f"   {d:48s} lines {100 * dd['share_lines']:5.1f}%  edits {100 * dd.get('share_edits', 0):5.1f}%  excess explained {100 * x:5.1f}%")


if __name__ == "__main__":
    main()
