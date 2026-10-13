"""Analysis B, the letter, PAW and word level (plan 2026-10-13, §3.4), on the per-line store.

    .geomvenv/bin/python scripts/analyze_letters.py [--store results/store] [--pairs configs/analysis/pairs.json] [--out results/analysis/B]

Every GT letter of every reading gets an outcome (ok / sub with the letter read / del) on the letters view (canonical,
harakat and tatweel stripped, as characterize/samples.letter_outcomes), plus its positional form, rasm class, dotted and
rasm-ambiguous flags, PAW and word index. Per run and dataset:
  B2 error rate per letter × form with Wilson 95 % intervals;
  B3 the top confusions, each tagged dot (same skeleton in that position), ortho (alef / ya / ta marbuta / hamza classes)
     or other; B4 dot-only errors per 100 dotted letters;
  B5 PAW exact-read rate by PAW length; spaces inserted inside words and the share at PAW gaps against the chance rate;
  B6 word error rate, error on words seen / unseen in the Muharaf training vocabulary, and fluent-but-wrong words
     (a wrong word that is itself a training-vocabulary word).
Per pair in pairs.json: Δ error per letter × form with a line-bootstrap interval, and the cells that move most.
Runs: every CRNN run / transform in its preferred pack, E8, and VLM rows with ≥ 50 lines (AIN, PaddleOCR and TrOCR
flagged, not dropped, see CORRECTIONS C5, C6).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from arabic_ocr_bench.characterize.errors import _ORTHO_CLASSES, is_paw_gap_space, paw_gap_base_rate  # noqa: E402
from arabic_ocr_bench.characterize.samples import _letters_view  # noqa: E402
from arabic_ocr_bench.characterize.text import DOTTED, analyze_letters, rasm_of  # noqa: E402

FLAGGED = ("ain", "paddleocr", "trocr")


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def outcomes(gt: str, hyp: str) -> tuple[list[dict], dict]:
    """Per GT letter: outcome and the letter read; plus line-level PAW / space counts."""
    g, p = _letters_view(gt), _letters_view(hyp)
    read: dict[int, str | None] = {}
    for op in Levenshtein.opcodes(g, p):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                read[i] = g[i]
        elif op.tag == "replace":
            for k, i in enumerate(range(op.src_start, op.src_end)):
                j = op.dest_start + k
                read[i] = p[j] if j < op.dest_end else None
        elif op.tag == "delete":
            for i in range(op.src_start, op.src_end):
                read[i] = None
    out = []
    for li in analyze_letters(g):
        h = read.get(li.index)
        oc = "ok" if h == li.letter else ("del" if h is None or h == " " else "sub")
        kind = None
        if oc == "sub":
            if rasm_of(h, li.position) is not None and rasm_of(h, li.position) == li.rasm:
                kind = "dot"
            elif any(li.letter in c and h in c for c in _ORTHO_CLASSES):
                kind = "ortho"
            else:
                kind = "other"
        out.append({"ch": li.letter, "form": li.position, "dotted": li.letter in DOTTED, "ambiguous": li.ambiguous, "paw": li.paw,
                    "word": li.word, "oc": oc, "read": h if oc == "sub" else None, "kind": kind})
    ins, at_paw = is_paw_gap_space(g, p)
    gaps, paw_gaps = paw_gap_base_rate(g)
    return out, {"space_ins_in_word": ins, "space_ins_at_paw_gap": at_paw, "word_gaps": gaps, "paw_gaps": paw_gaps}


def words_view(s: str) -> list[str]:
    return _letters_view(s).split()


class Agg:
    def __init__(self):
        self.cell = defaultdict(lambda: [0, 0, 0])  # (ch, form) -> [n, sub, del]
        self.conf = Counter()
        self.kind = Counter()
        self.dotted = [0, 0]  # [dotted letters, dot-only subs]
        self.paw_len = defaultdict(lambda: [0, 0])  # len -> [n, all ok]
        self.space = Counter()
        self.words = Counter()
        self.lines = 0

    def add(self, gt: str, hyp: str, vocab: set[str]):
        oc, sp = outcomes(gt, hyp)
        self.lines += 1
        for o in oc:
            c = self.cell[(o["ch"], o["form"])]
            c[0] += 1
            c[1] += o["oc"] == "sub"
            c[2] += o["oc"] == "del"
            if o["oc"] == "sub":
                self.conf[(o["ch"], o["read"])] += 1
                self.kind[o["kind"]] += 1
            if o["dotted"]:
                self.dotted[0] += 1
                self.dotted[1] += o["kind"] == "dot"
        paws = defaultdict(list)
        for o in oc:
            paws[o["paw"]].append(o["oc"] == "ok")
        for v in paws.values():
            pl = self.paw_len[min(len(v), 6)]
            pl[0] += 1
            pl[1] += all(v)
        self.space.update(sp)
        gw, hw = words_view(gt), words_view(hyp)
        ops = Levenshtein.opcodes(gw, hw)
        wrong_gt = set()
        for op in ops:
            if op.tag != "equal":
                wrong_gt.update(range(op.src_start, op.src_end))
                for j in range(op.dest_start, op.dest_end):
                    self.words["hyp_wrong"] += 1
                    self.words["hyp_wrong_in_vocab"] += hw[j] in vocab
        self.words["word_edits"] += Levenshtein.distance(gw, hw)
        self.words["gt_words"] += len(gw)
        for i, w in enumerate(gw):
            seen = w in vocab
            self.words["gt_seen" if seen else "gt_unseen"] += 1
            self.words["gt_seen_wrong" if seen else "gt_unseen_wrong"] += i in wrong_gt

    def report(self) -> dict:
        cells = {f"{ch}|{f}": {"n": n, "err": (s + d) / n, "sub": s / n, "del": d / n, "ci95": wilson(s + d, n)}
                 for (ch, f), (n, s, d) in sorted(self.cell.items()) if n >= 5}
        n_all = sum(v[0] for v in self.cell.values())
        w = self.words
        sp = self.space
        return {"lines": self.lines, "letters": n_all, "letter_error": sum(v[1] + v[2] for v in self.cell.values()) / max(1, n_all),
                "sub_kinds": dict(self.kind), "dot_only_per_100_dotted": 100 * self.dotted[1] / max(1, self.dotted[0]),
                "top_confusions": [[a, b, k] for (a, b), k in self.conf.most_common(30)],
                "paw_exact_by_len": {str(k): v[1] / v[0] for k, v in sorted(self.paw_len.items())},
                "space_in_word_per_100_gaps": 100 * sp["space_ins_in_word"] / max(1, sp["word_gaps"]),
                "space_at_paw_gap_share": sp["space_ins_at_paw_gap"] / max(1, sp["space_ins_in_word"]),
                "paw_gap_chance": sp["paw_gaps"] / max(1, sp["word_gaps"]),
                "wer": w["word_edits"] / max(1, w["gt_words"]),
                "err_seen_words": w["gt_seen_wrong"] / max(1, w["gt_seen"]), "err_unseen_words": w["gt_unseen_wrong"] / max(1, w["gt_unseen"]),
                "unseen_word_share": w["gt_unseen"] / max(1, w["gt_words"]),
                "fluent_wrong_share": w["hyp_wrong_in_vocab"] / max(1, w["hyp_wrong"]), "cells": cells}


def cell_errors(gt: str, hyp: str) -> Counter:
    c = Counter()
    for o in outcomes(gt, hyp)[0]:
        c[(o["ch"], o["form"], "n")] += 1
        c[(o["ch"], o["form"], "e")] += o["oc"] != "ok"
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--pairs", type=Path, default=W / "configs/analysis/pairs.json")
    ap.add_argument("--vocab-pkl", type=Path, default=W / "results/panel/aligner_data_standard.pkl")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/B")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.pairs.read_text(encoding="utf-8"))
    packs = cfg["pack_pref"]
    vocab = {w for r in pickle.load(open(a.vocab_pkl, "rb"))["rows"] if r["role"] == "train" for w in words_view(r["label"])}
    print("training vocabulary", len(vocab), "words")
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    idx = defaultdict(dict)
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        if r.get("in_train") == "yes":  # never score lines a run trained on (CORRECTIONS C4)
            continue
        idx[(r["run_id"], r["transform"], r["pack"], r["dataset"])][r["sample_key"]] = r

    chosen = {}  # (run@tr, ds) -> rows, preferred pack only
    for (run, tr, pack, ds), m in idx.items():
        fam = next(iter(m.values()))["family"]
        if fam in ("api",) or len(m) < 50:
            continue
        key = (f"{run}@{tr}", ds)
        rank = packs.index(pack) if pack in packs else 99
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, pack, m)
    res = {}
    for (spec, ds), (_, pack, m) in sorted(chosen.items()):
        ag = Agg()
        for k, r in m.items():
            ag.add(S[k]["gt_raw"], r["hyp"], vocab)
        rep = ag.report()
        rep["pack"], rep["flagged"] = pack, any(x in spec.lower() for x in FLAGGED)
        res[f"{spec}|{ds}"] = rep
    (a.out / "letters.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    pairs = {}
    for pr in cfg["pairs"]:
        for ds in sorted({k[3] for k in idx}):
            for p in packs:
                ra, ta = pr["A"].split("@")
                rb, tb = pr["B"].split("@")
                A, B = idx.get((ra, ta, p, ds)), idx.get((rb, tb, p, ds))
                if A and B:
                    break
            else:
                continue
            keys = sorted(set(A) & set(B))
            ca = [cell_errors(S[k]["gt_raw"], A[k]["hyp"]) for k in keys]
            cb = [cell_errors(S[k]["gt_raw"], B[k]["hyp"]) for k in keys]
            cells = sorted({(c[0], c[1]) for x in ca for c in x})
            rng = np.random.default_rng(0)
            boot = rng.integers(0, len(keys), (1000, len(keys)))
            moves = []
            for ch, f in cells:
                n = np.array([x[(ch, f, "n")] for x in ca], float)
                if n.sum() < 30:
                    continue
                ea = np.array([x[(ch, f, "e")] for x in ca], float)
                eb = np.array([x[(ch, f, "e")] for x in cb], float)
                d = (eb.sum() - ea.sum()) / n.sum()
                bs = (eb[boot].sum(1) - ea[boot].sum(1)) / np.maximum(1, n[boot].sum(1))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                moves.append({"cell": f"{ch}|{f}", "n": int(n.sum()), "err_A": float(ea.sum() / n.sum()), "err_B": float(eb.sum() / n.sum()), "delta": float(d),
                              "ci95": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0)})
            moves.sort(key=lambda x: x["delta"])
            pairs[f"{pr['name']}|{ds}"] = {"A": pr["A"], "B": pr["B"], "pack": p, "lines": len(keys), "cells": moves,
                                           "n_cells_better_ci": sum(m["excludes_0"] and m["delta"] < 0 for m in moves),
                                           "n_cells_worse_ci": sum(m["excludes_0"] and m["delta"] > 0 for m in moves)}
    (a.out / "pairs_letters.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'run|dataset':60s} {'lines':>5s} {'letErr':>6s} {'dot/100':>7s} {'WER':>5s} {'unseenW':>7s} {'fluent':>6s} {'PAWsp':>5s} {'chance':>6s}")
    for k, v in res.items():
        if v["lines"] < 100:
            continue
        print(f"{k[:60]:60s} {v['lines']:5d} {100 * v['letter_error']:6.1f} {v['dot_only_per_100_dotted']:7.2f} {100 * v['wer']:5.1f} "
              f"{100 * v['err_unseen_words']:7.1f} {100 * v['fluent_wrong_share']:6.1f} {100 * v['space_at_paw_gap_share']:5.1f} {100 * v['paw_gap_chance']:6.1f}")
    print("\npairs: letter × form cells whose interval excludes 0 (better / worse)")
    for k, v in pairs.items():
        top = ", ".join(f"{m['cell']} {100 * m['delta']:+.1f}" for m in v["cells"][:3])
        print(f"  {k[:58]:58s} {v['n_cells_better_ci']:3d} / {v['n_cells_worse_ci']:3d}   most improved: {top}")


if __name__ == "__main__":
    main()
