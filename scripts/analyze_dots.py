"""Dots and skeletons across every system (plan addition 2026-10-15, theme T2).

    .geomvenv/bin/python scripts/analyze_dots.py [--store results/store] [--out results/analysis/B/dots.json]

Per family × set and per run (lines a run trained on excluded; AIN, PaddleOCR and TrOCR left out):
  no-dots CER (the store's rasm rung: both sides mapped to their dotless skeletons), dot tax = CER − no-dots CER, the
  dots' share of CER, the space tax (no-dots CER − no-dots-no-spaces CER), exact lines, and lines that are right
  except for their dots (rasm rung exact, letters view not);
  dot errors by the GT letter's dot pattern (1, 2 or 3 dots, above or below), per 100 such letters;
and the cross-family per-line oracle: the line-wise better of the best CRNN reading and the best fine-tuned VLM reading.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from analyze_letters import outcomes  # noqa: E402

DOTS = {"ب": "1 below", "ت": "2 above", "ث": "3 above", "ج": "1 below", "خ": "1 above", "ذ": "1 above", "ز": "1 above", "ش": "3 above",
        "ض": "1 above", "ظ": "1 above", "غ": "1 above", "ف": "1 above", "ق": "2 above", "ن": "1 above", "ي": "2 below", "ة": "2 above"}
SKIP = ("ain", "paddleocr-v5", "trocr")


def fam(r) -> str | None:
    if r["run_id"].startswith("crnn/"):
        return "CRNN" if r["transform"] == "none" else "CRNN, ink-cropped" if r["transform"] == "inkcrop" else None
    if r["run_id"].startswith("crnn+llm/"):
        return "CRNN → LLM (E8)" if r["transform"] == "none" else None
    if r["transform"] != "none":
        return None
    return {"vlm-ft": "VLM fine-tuned", "vlm-base": "VLM base", "api": "API", "ocr-engine": "OCR engine"}.get(r["family"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/B/dots.json")
    a = ap.parse_args()
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    rows = [r for r in pq.read_table(a.store / "lines.parquet").to_pylist()
            if r.get("in_train") != "yes" and r["pack"] in ("panel-2026-10-13", "bench", "eval_data+heldout") and not any(x in r["run_id"] for x in SKIP)]

    def tally(group):
        e = sum(r["e_no_diacritics"] for r in group)
        n = sum(max(1, r["n_no_diacritics"]) for r in group)
        er = sum(r["e_rasm"] for r in group)
        es = sum(r["e_no_spaces"] for r in group)
        return {"lines": len(group), "cer": e / n, "no_dots_cer": er / n, "dot_tax": (e - er) / n, "dots_share": (e - er) / max(1, e),
                "no_dots_no_spaces_cer": es / n, "space_tax": (er - es) / n, "exact": sum(r["e_no_diacritics"] == 0 for r in group) / len(group),
                "right_but_dots": sum(r["e_rasm"] == 0 and r["e_no_diacritics"] > 0 for r in group) / len(group)}

    by_fam, by_run = defaultdict(list), defaultdict(list)
    for r in rows:
        f = fam(r)
        if f:
            by_fam[(f, r["dataset"])].append(r)
            by_run[(f"{r['run_id']}@{r['transform']}", r["dataset"])].append(r)
    fam_res = {f"{f}|{ds}": tally(g) for (f, ds), g in by_fam.items() if len(g) >= 150}
    run_res = {f"{k}|{ds}": tally(g) for (k, ds), g in by_run.items() if len(g) >= 100}

    # dot errors by dot pattern, per family (one representative run per family and set to keep it cheap)
    dot_types = {}
    for (f, ds), g in by_fam.items():
        if len(g) < 150 or ds not in ("kitab_khatt", "nakba_lines", "rasam1", "muharaf_lines"):
            continue
        n, wrong = Counter(), Counter()
        for r in g[:3000]:
            for o in outcomes(S[r["sample_key"]]["gt_raw"], r["hyp"])[0]:
                t = DOTS.get(o["ch"])
                if t:
                    n[t] += 1
                    wrong[t] += o["kind"] == "dot"
        dot_types[f"{f}|{ds}"] = {t: {"letters": n[t], "dot_errors_per_100": 100 * wrong[t] / n[t]} for t in n if n[t] >= 50}

    # cross-family oracle
    oracle = {}
    for ds in ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2"):
        best = defaultdict(dict)
        for r in rows:
            f = fam(r)
            if r["dataset"] != ds or f not in ("CRNN", "CRNN, ink-cropped", "VLM fine-tuned"):
                continue
            g = "CRNN" if f.startswith("CRNN") else "VLM fine-tuned"
            c = r["e_no_diacritics"] / max(1, r["n_no_diacritics"])
            best[r["sample_key"]][g] = min(c, best[r["sample_key"]].get(g, 9))
        ks = [k for k, v in best.items() if len(v) == 2]
        if len(ks) < 50:
            continue
        cr = [best[k]["CRNN"] for k in ks]
        vl = [best[k]["VLM fine-tuned"] for k in ks]
        oracle[ds] = {"lines": len(ks), "best_crnn_mean_line_cer": statistics.mean(cr), "best_vlm_ft_mean_line_cer": statistics.mean(vl),
                      "per_line_min": statistics.mean(min(x, y) for x, y in zip(cr, vl)), "crnn_better_lines": sum(x < y for x, y in zip(cr, vl)),
                      "vlm_better_lines": sum(y < x for x, y in zip(cr, vl))}
    a.out.write_text(json.dumps({"families": fam_res, "runs": run_res, "dot_types": dot_types, "cross_family_oracle": oracle}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'family|set':40s} {'lines':>6s} {'CER':>6s} {'no-dots':>8s} {'dot tax':>8s} {'share':>6s} {'space tax':>9s} {'right but dots':>14s}")
    for k, v in sorted(fam_res.items()):
        print(f"{k[:40]:40s} {v['lines']:6d} {100 * v['cer']:6.1f} {100 * v['no_dots_cer']:8.1f} {100 * v['dot_tax']:8.2f} {100 * v['dots_share']:5.1f}% {100 * v['space_tax']:9.2f} {100 * v['right_but_dots']:13.1f}%")
    print("\ndot errors per 100 letters, by dot pattern")
    for k, v in sorted(dot_types.items()):
        print(f"  {k[:38]:38s} " + "  ".join(f"{t}: {x['dot_errors_per_100']:.1f}" for t, x in sorted(v.items())))
    print("\ncross-family oracle (mean line CER, no-diacritics rung)")
    for ds, v in oracle.items():
        print(f"  {ds:14s} lines {v['lines']:5d}  best CRNN {100 * v['best_crnn_mean_line_cer']:.1f}  best fine-tuned VLM {100 * v['best_vlm_ft_mean_line_cer']:.1f}  "
              f"per-line min {100 * v['per_line_min']:.1f}  (CRNN better {v['crnn_better_lines']}, VLM better {v['vlm_better_lines']})")


if __name__ == "__main__":
    main()
