"""Text-side tables (plan 2026-10-13, P1.6: R1.6 lexical diversity, R1.7 shape tail), from GT only.

    .geomvenv/bin/python scripts/analyze_text_tables.py [--out results/analysis/T/text_tables.json]

R1.6, per set (as alazzawi2026crossling Table 3): lines, unique lines, duplicated-line share, words, unique words,
type-token ratio on the first 5,000 words (so sets of different size compare), mean max word-Jaccard of a line to the
other lines (repetitiveness).
R1.7, per test set against each training pool (Muharaf train, NAKBA train): share of test letters whose
letter × positional form occurs fewer than 10 / 50 times in that pool (alazzawi2026gap Fig. 6), and the share of
test words never seen in the pool.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from analyze_letters import words_view  # noqa: E402
from arabic_ocr_bench.characterize.text import analyze_letters  # noqa: E402


def diversity(lines: list[str], rng) -> dict:
    words = [w for ln in lines for w in words_view(ln)]
    c = Counter(lines)
    sets = [set(words_view(ln)) for ln in lines]
    sample = rng.choice(len(lines), min(400, len(lines)), replace=False)
    jac = []
    for i in sample:
        best = 0.0
        for j, s in enumerate(sets):
            if j != i and sets[i] and s:
                best = max(best, len(sets[i] & s) / len(sets[i] | s))
        jac.append(best)
    first = words[:5000]
    return {"lines": len(lines), "unique_lines": len(c), "dup_line_share": sum(v for v in c.values() if v > 1) / len(lines), "words": len(words),
            "unique_words": len(set(words)), "ttr_5k": len(set(first)) / max(1, len(first)), "mean_max_jaccard": float(np.mean(jac))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=W / "results/analysis/T/text_tables.json")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    S = pq.read_table(W / "results/store/samples.parquet").to_pylist()
    pools = {"muharaf_train": [r["label"] for r in pickle.load(open(W / "results/panel/aligner_data_standard.pkl", "rb"))["rows"] if r["role"] == "train"],
             "nakba_train": [r["label"] for r in pickle.load(open(W / "results/panel/aligner_data_nakba.pkl", "rb"))["rows"] if r["role"] == "train"]}
    sets = {ds: [s["gt_clean"] for s in S if s["dataset"] == ds] for ds in ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2")}
    R16 = {name: diversity(lines, rng) for name, lines in {**pools, **sets}.items()}
    forms, vocab = {}, {}
    for p, lines in pools.items():
        c = Counter()
        for ln in lines:
            for li in analyze_letters(ln):
                c[(li.letter, li.position)] += 1
        forms[p], vocab[p] = c, {w for ln in lines for w in words_view(ln)}
    R17 = {}
    for ds, lines in sets.items():
        lets = [(li.letter, li.position) for ln in lines for li in analyze_letters(ln)]
        ws = [w for ln in lines for w in words_view(ln)]
        R17[ds] = {p: {"letters": len(lets), "share_form_lt10": sum(forms[p][x] < 10 for x in lets) / len(lets), "share_form_lt50": sum(forms[p][x] < 50 for x in lets) / len(lets),
                       "unseen_word_share": sum(w not in vocab[p] for w in ws) / len(ws)} for p in pools}
    a.out.write_text(json.dumps({"R1.6": R16, "R1.7": R17}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'set':14s} {'lines':>6s} {'uniq':>6s} {'dup%':>5s} {'words':>7s} {'uniqW':>6s} {'TTR5k':>6s} {'maxJac':>6s}")
    for k, v in R16.items():
        print(f"{k:14s} {v['lines']:6d} {v['unique_lines']:6d} {100 * v['dup_line_share']:5.1f} {v['words']:7d} {v['unique_words']:6d} {v['ttr_5k']:6.3f} {v['mean_max_jaccard']:6.3f}")
    print("\nshape tail and unseen words (vs Muharaf train | vs NAKBA train)")
    for ds, v in R17.items():
        m, n = v["muharaf_train"], v["nakba_train"]
        print(f"{ds:14s} forms<10 {100 * m['share_form_lt10']:5.2f}% | {100 * n['share_form_lt10']:5.2f}%   forms<50 {100 * m['share_form_lt50']:5.2f}% | {100 * n['share_form_lt50']:5.2f}%   unseen words {100 * m['unseen_word_share']:5.1f}% | {100 * n['unseen_word_share']:5.1f}%")


if __name__ == "__main__":
    main()
