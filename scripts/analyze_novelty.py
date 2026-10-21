"""The X-G10 confound battery: does letter-shape novelty predict letter errors beyond the line it sits in? (plan P1.2)

    .geomvenv/bin/python scripts/analyze_novelty.py [--letters results/panel/letters] [--out results/analysis/V/novelty.json]

Input: per-letter dumps from xai_letter_shapes.py --dump-letters (one per model: id, tag, index in the cleaned label,
letter, form, span, ok, novelty, pixel novelty). Exposure: a letter is "novel" when its novelty is above the 95th
percentile of the in-domain Muharaf letters (the dump model's own scale), "typical" when below the 60th; letters in
between are left out. Tests, each as a Mantel–Haenszel odds ratio of misreading (novel vs typical) with a line-bootstrap
95 % interval:
  (a) crude (no stratification) and on ink-cropped lines (tag __inkcrop);
  (b) stratified by line (removes everything line-level: crop, writer, scan, text) and by letter × form;
  (c) the same exposure against the errors of other systems on the same letters: other CRNNs, the median fine-tuned VLM;
  (d) novelty from one model against errors of another (spans and features independent of the judged reading);
  (e) without letter-list drill lines.
Pass (plan): stratified-by-line OR ≥ 2 with the interval above 1 for the CRNN, and ≥ 1.5 for the median VLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from geom_inkcrop_export import clean  # noqa: E402


def mh_or(groups) -> float:
    """Mantel–Haenszel OR over strata; each stratum = (a novel-wrong, b novel-right, c typical-wrong, d typical-right)."""
    num = den = 0.0
    for a, b, c, d in groups:
        n = a + b + c + d
        if n and (a + b) and (c + d):
            num += a * d / n
            den += b * c / n
    return num / den if den else float("nan")


def ok_vector(label: str, hyp: str) -> list[bool]:
    """Per character of the cleaned label: read correctly by this reading (equal blocks of the alignment)."""
    good = [False] * len(label)
    for op in Levenshtein.opcodes(label, clean(hyp)):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                good[i] = True
    return good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", type=Path, default=W / "results/panel/letters")
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/V/novelty.json")
    ap.add_argument("--within-set", action="store_true", help="novel / typical = top 15 % / bottom 50 % of the test set's own letters")
    ap.add_argument("--field", default="novelty", help="novelty (CNN features) or novelty_pix (model-independent ink-crop pixels)")
    a = ap.parse_args()
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    letter_list = {k for k, s in S.items() if sum(len(w) == 1 for w in s["gt_clean"].split()) >= 4}
    dumps = {}
    for d in sorted(a.letters.iterdir()):
        f = d / "letters.json"
        if f.exists():
            J = json.loads(f.read_text(encoding="utf-8"))
            dumps[d.name] = [dict(zip(J["fields"], r)) for r in J["rows"]]
            if a.field != "novelty":
                for r in dumps[d.name]:
                    r["novelty"] = r[a.field]
    # readings of every system on the panel lines, for (c)
    reads = defaultdict(dict)
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        if r["transform"] != "none" or r.get("in_train") == "yes" or r["pack"] not in ("panel-2026-10-13", "bench"):
            continue
        if any(x in r["run_id"] for x in ("ain", "paddleocr", "trocr")):
            continue
        reads[r["run_id"]][r["sample_key"]] = r["hyp"]
    vlm_ft = [run for run in reads if run.startswith("vlm/") and ("-ft" in run or "baseer" in run)]
    okcache = {}

    def ok_of(run, key, i):
        ck = (run, key)
        if ck not in okcache:
            okcache[ck] = ok_vector(S[key]["gt_clean"], reads[run][key]) if key in reads[run] else None
        v = okcache[ck]
        return None if v is None or i >= len(v) else v[i]

    rng = np.random.default_rng(0)
    res = {}
    for model, rows in dumps.items():
        mu = [r["novelty"] for r in rows if r["tag"] == "muharaf_lines" and r["novelty"] is not None]
        if len(mu) < 500:
            continue
        hi_t, lo_t = np.percentile(mu, 95), np.percentile(mu, 60)
        for tag in ("kitab_khatt", "kitab_khatt__inkcrop", "nakba_lines", "rasam1", "rasam2"):
            if a.within_set:  # thresholds inside the test set: top 15 % novel, bottom 50 % typical
                v = [r["novelty"] for r in rows if r["tag"] == tag and r["novelty"] is not None]
                if len(v) < 300:
                    continue
                hi_t, lo_t = np.percentile(v, 85), np.percentile(v, 50)
            L = [r for r in rows if r["tag"] == tag and r["novelty"] is not None and (r["novelty"] > hi_t or r["novelty"] < lo_t)]
            if len(L) < 300:
                continue
            ds = tag.split("__")[0]
            by_line = defaultdict(list)
            for r in L:
                by_line[r["id"]].append(r)
            lines = list(by_line)

            def table_for(judge, keep=lambda k: True, strat="line"):
                strata = defaultdict(lambda: [0, 0, 0, 0])
                for k in lines:
                    if not keep(k):
                        continue
                    for r in by_line[k]:
                        ok = r["ok"] if judge == "self" else ok_of(judge, k, r["i"]) if not isinstance(judge, list) else None
                        if isinstance(judge, list):  # median VLM: wrong when most of the runs that read it are wrong
                            v = [ok_of(j, k, r["i"]) for j in judge]
                            v = [x for x in v if x is not None]
                            if len(v) < 2:
                                continue
                            ok = sum(v) > len(v) / 2
                        if ok is None:
                            continue
                        s = k if strat == "line" else (r["ch"], r["form"]) if strat == "letter" else "all"
                        novel = r["novelty"] > hi_t
                        strata[s][(0 if novel else 2) + (0 if not ok else 1)] += 1
                return strata

            def boot(judge, keep=lambda k: True, strat="line", reps=300):
                st = table_for(judge, keep, strat)
                point = mh_or(st.values())
                if strat != "line":
                    return [point, None, None]
                ks = [k for k in st]
                bs = []
                for _ in range(reps):
                    pick = rng.integers(0, len(ks), len(ks))
                    bs.append(mh_or([st[ks[j]] for j in pick]))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                return [point, float(lo), float(hi)]

            r = {"letters": len(L), "lines": len(lines), "novel_share": float(np.mean([x["novelty"] > hi_t for x in L])),
                 "crude": boot("self", strat="all"), "by_line": boot("self"), "by_letter_form": boot("self", strat="letter"),
                 "by_line_no_letter_lists": boot("self", keep=lambda k: k not in letter_list)}
            if "__" not in tag:
                for other in ("crnn/base", "crnn/combo", "crnn/std_s0"):
                    if other != {"aligner_run": "crnn/base"}.get(model) and S and any(k in reads.get(other, {}) for k in lines):
                        r[f"errors_of {other}"] = boot(other)
                ft = [j for j in vlm_ft if sum(k in reads[j] for k in lines) >= 0.5 * len(lines)]
                if len(ft) >= 2:
                    r["errors_of median fine-tuned VLM"] = boot(ft)
                    r["vlm_runs"] = ft
            res[f"{model}|{tag}"] = r
            print(f"{model:22s} {tag:22s} letters {len(L):6d} novel {100 * r['novel_share']:4.1f}%  crude OR {r['crude'][0]:.2f}  "
                  f"within-line OR {r['by_line'][0]:.2f} [{r['by_line'][1]:.2f},{r['by_line'][2]:.2f}]  within letter×form {r['by_letter_form'][0]:.2f}  "
                  f"no letter lists {r['by_line_no_letter_lists'][0]:.2f}" + "".join(f"  | {k.replace('errors_of ', '')} {v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" for k, v in r.items() if k.startswith("errors_of")))
    crnn = res.get("aligner_run|kitab_khatt", {})
    vlm = crnn.get("errors_of median fine-tuned VLM")
    verdict = {"crnn_within_line_or": crnn.get("by_line"), "vlm_within_line_or": vlm,
               "pass": bool(crnn and crnn["by_line"][0] >= 2 and crnn["by_line"][1] > 1 and vlm and vlm[0] >= 1.5 and vlm[1] > 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"verdict": verdict, "results": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
