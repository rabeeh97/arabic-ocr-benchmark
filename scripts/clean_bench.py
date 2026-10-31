"""Strict RASAM-1 / RASAM-2 / TariMa board on three views of the official test set, with page-clustered bootstrap error bars.

    PYTHONPATH=src:scripts python scripts/clean_bench.py [--reps 2000]

Views (all on the official Hugging Face test lines):
  official     the published crops and references, all lines
  drop-only    the published crops, the 21 test lines in results/rasam/clean/drop.json removed (no re-reads needed: every system)
  clean        the 25 broken crops replaced by our re-cuts from the page images (every system re-read them: clean_rerun_local.py,
               colab_recut_rerun.py) and the 21 test lines in results/rasam/clean/drop.json removed
  clean-indep  the same re-cuts, but only the lines flagged by sources independent of our models removed: the manual check of
               calfa-co/rasam-dataset PR #7 and our by-eye list (results/rasam/push/data_issues_curated.json). Guards against the
               automatic drop rules favouring CRNN-like readers (they were found with cross-fitted CRNN readings)
Error bars: 95 % percentile intervals of micro CER from resampling PAGES with replacement within each set (lines on a page are not
independent). Paired differences use the same resampled pages for both systems.
Writes results/rasam/clean/bench.json (+ per-system, per-view numbers) and prints the board.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

NAME = {"r1": "RASAM-1", "r2": "RASAM-2", "tm": "tarima"}
SETS = ("r1", "r2", "tm")
C = Path("results/rasam/clean")
RR = C / "rerun"
LM = Path("results/rasam/lm")
CR = Path("results/rasam/crnn")
SUITE = "/Volumes/PortableSSD/repos/arabic_ocr/suite-v1-artifacts"


def by_index(path_fmt, sets=SETS, key="lines"):
    out = {}
    for t in sets:
        p = Path(path_fmt.format(t=t))
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = d[key] if isinstance(d, dict) else d
        for i, x in enumerate(rows):
            out[f"{NAME[t]}/test/{int(x.get('i', i)):06d}"] = x["hyp"]
    return out


def by_id(path_fmt, sets=SETS, field="hyp"):
    out = {}
    for t in sets:
        p = Path(path_fmt.format(t=t))
        if p.exists():
            out.update({x["id"]: x[field] for x in json.loads(p.read_text())["lines"]})
    return out


def load_rerun(name):
    p = RR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def systems():
    """name -> (official readings {key: hyp}, re-cut readings {key: hyp} or None, kind, note)"""
    ours_lm = load_rerun("ours_lm.json") or {}
    colab = Path("results/rasam/clean/rerun/colab")
    S = {}
    S["Ours: 2-model ensemble + char LM + AraGPT2"] = (by_id(str(LM / "NB_PX_ENS_{t}__{t}__rescored.json")), ours_lm.get("ENS"), "ours", "best system")
    S["Ours: 2-model ensemble + char LM"] = (by_id(str(LM / "NB_PX_ENS_{t}__{t}__test.json")), ours_lm.get("ENS_lm"), "ours", "")
    S["Ours: 60-epoch CRNN + char LM + AraGPT2"] = (by_id(str(LM / "NB_PX_E60_{t}__{t}__rescored.json")), ours_lm.get("E60"), "ours", "single model")
    S["Ours: pretrained CRNN + char LM + AraGPT2"] = (by_id(str(LM / "NB_PX_PT_{t}__{t}__rescored.json")), ours_lm.get("PT"), "ours", "~40k extra pretraining lines")
    S["Ours: 60-epoch CRNN, greedy"] = (by_id(str(CR / "E60_dirty__{t}__test.json")), by_id(str(CR / "E60_recut__{t}__test.json")), "ours", "no LM")
    S["Ours: 60-epoch CRNN on cleaned training data, greedy"] = (by_id(str(CR / "CE60_dirty__{t}__test.json")), by_id(str(CR / "CE60_recut__{t}__test.json")), "ours",
                                                                 "same recipe, 202 bad lines dropped / 250 crops re-cut in training")
    S["AMIDDA 1.0 Line 4B (Calfa VLM)"] = (by_index("results/rasam/external/amidda_{t}__test.json"), load_rerun("colab/preds_recut_amidda.json"), "rescored", "Colab, card settings")
    S["HAFITH (642 M)"] = (by_index("results/rasam/colab_runs/job-hafith-eval-t4-10251632/content/crnn_out/hafith/preds_{t}.json", ("r1", "r2"), key=None),
                           load_rerun("colab/preds_recut_hafith.json"), "rescored", "362 RASAM-1 test lines in its training data")
    S["HATFormer (AraMS-28k, 334 M)"] = (by_index("results/rasam/external/hatformer_{t}__test.json"), load_rerun("hatformer.json"), "rescored", "zero-shot")
    S["Phoenix (Kraken)"] = (by_id_named("phoenix"), load_rerun("phoenix.json"), "rescored", "binarised crops")
    S["TariMa manuscript model (Kraken)"] = (by_id_named("tarima-manuscript"), load_rerun("tarima-manuscript.json"), "rescored", "binarised crops")
    return S


def by_id_named(name):
    out = {}
    for t in ("r1", "r2"):
        p = Path(f"results/rasam/replication/{name}__{NAME[t]}__test__bin.json")
        out.update({x["id"]: x["hyp"] for x in json.loads(p.read_text())["lines"]})
    return out


def edits(ref, hyp):
    c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp or "", "raw").strip())
    return c["sub"] + c["del"] + c["ins"], c["ref_len"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    # references and pages from our own test readings (same official order and text)
    meta = {}
    for t in SETS:
        for x in json.loads((CR / f"E60_dirty__{t}__test.json").read_text())["lines"]:
            meta[x["id"]] = {"set": t, "ref": x["ref"], "page": f"{x['manuscript']}|{x['page']}"}
    recut = {d["key"]: d for d in json.loads((RR / "recut_test.json").read_text())}
    drop = set(json.loads((C / "drop.json").read_text()))
    reasons = json.loads((C / "clean_report.json").read_text())["reasons"]
    eye = {ex["id"] for c in json.loads(Path("results/rasam/push/data_issues_curated.json").read_text())["categories"] for ex in c["examples"]}
    indep = {k for k, v in reasons.items() if "external" in v} | (eye - set(recut))
    views = {"official": (set(), False), "drop-only": (drop, False), "clean": (drop, True), "clean-indep": (indep, True)}
    S = systems()
    res = {"views": {v: {t: sum(1 for k, m in meta.items() if m["set"] == t and k not in d) for t in SETS} for v, (d, _) in views.items()},
           "systems": {}, "paired": {}}
    errs = {}  # (system, view, set) -> {page: [edits, ref_len]}
    for name, (off, rc, kind, note) in S.items():
        res["systems"][name] = {"kind": kind, "note": note}
        for v, (dset, use_rc) in views.items():
            for t in SETS:
                keys = [k for k, m in meta.items() if m["set"] == t and k not in dset]
                if not all(k in off for k in keys):
                    continue  # this system did not read this set
                if use_rc and any(k in recut for k in keys) and rc is None:
                    res["systems"][name].setdefault("missing_recut", []).append(f"{v}/{t}")
                    continue
                pg = defaultdict(lambda: [0, 0])
                for k in keys:
                    hyp = rc[k] if (use_rc and k in recut) else off[k]
                    e, n = edits(meta[k]["ref"], hyp)
                    pg[meta[k]["page"]][0] += e
                    pg[meta[k]["page"]][1] += n
                errs[name, v, t] = pg
                arr = np.array(list(pg.values()), dtype=np.float64)
                idx = rng.integers(0, len(arr), size=(a.reps, len(arr)))
                boot = arr[idx, 0].sum(1) / arr[idx, 1].sum(1) * 100
                res["systems"][name][f"{v}/{t}"] = {"cer": round(arr[:, 0].sum() / arr[:, 1].sum() * 100, 2),
                                                     "ci95": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
                                                     "n_lines": len(keys), "n_pages": len(arr)}
    # paired page-bootstrap differences for the comparisons we draw conclusions from
    best = "Ours: 2-model ensemble + char LM + AraGPT2"
    pairs = [("Ours: 60-epoch CRNN on cleaned training data, greedy", "Ours: 60-epoch CRNN, greedy"),
             ("AMIDDA 1.0 Line 4B (Calfa VLM)", best), ("HAFITH (642 M)", best), ("Ours: 60-epoch CRNN + char LM + AraGPT2", best)]
    for s1, s2 in pairs:
        for v in views:
            for t in SETS:
                if (s1, v, t) not in errs or (s2, v, t) not in errs:
                    continue
                p1, p2 = errs[s1, v, t], errs[s2, v, t]
                pages = sorted(p1)
                A = np.array([p1[p] for p in pages], dtype=np.float64)
                B = np.array([p2[p] for p in pages], dtype=np.float64)
                idx = rng.integers(0, len(pages), size=(a.reps, len(pages)))
                d = (A[idx, 0].sum(1) / A[idx, 1].sum(1) - B[idx, 0].sum(1) / B[idx, 1].sum(1)) * 100
                res["paired"][f"{s1} − {s2} | {v}/{t}"] = {"diff": round(float((A[:, 0].sum() / A[:, 1].sum() - B[:, 0].sum() / B[:, 1].sum()) * 100), 2),
                                                           "ci95": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                                                           "p_le_0": round(float((d <= 0).mean()), 3)}
    # 200-line panel (test lines 0-199 of RASAM-1/2): the suite_v1 VLM fine-tunes were only run there
    vlm = defaultdict(dict)
    for f in sorted(Path(SUITE).glob("*/bench/full/rasam*/*/predictions.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error"):
                t = {"rasam1": "r1", "rasam2": "r2"}[r["dataset"]]
                vlm[r["model"]][f"{NAME[t]}/test/{int(r['sample_id'].split('/')[-1]):06d}"] = r["cleaned_text"] or ""
    panel_sys = {**{f"VLM fine-tune: {m}": d for m, d in vlm.items()},
                 **{n: S[n][0] for n in (best, "Ours: 60-epoch CRNN, greedy", "AMIDDA 1.0 Line 4B (Calfa VLM)", "HAFITH (642 M)")}}
    res["panel"] = {}
    for name, off in panel_sys.items():
        for v, dset in (("official", set()), ("drop-only", drop)):
            for t in ("r1", "r2"):
                keys = [f"{NAME[t]}/test/{i:06d}" for i in range(200)]
                keys = [k for k in keys if k in meta and k not in dset and k in off]  # a few VLM lines errored out: score what was read
                if len(keys) < 150:
                    continue
                pg = defaultdict(lambda: [0, 0])
                for k in keys:
                    e, n = edits(meta[k]["ref"], off[k])
                    pg[meta[k]["page"]][0] += e
                    pg[meta[k]["page"]][1] += n
                arr = np.array(list(pg.values()), dtype=np.float64)
                idx = rng.integers(0, len(arr), size=(a.reps, len(arr)))
                boot = arr[idx, 0].sum(1) / arr[idx, 1].sum(1) * 100
                res["panel"].setdefault(name, {})[f"{v}/{t}"] = {"cer": round(arr[:, 0].sum() / arr[:, 1].sum() * 100, 2),
                    "ci95": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)], "n_lines": len(keys), "n_pages": len(arr)}
    res["drop_independent"] = {"lines": len(indep), "test_lines": sum(1 for k in indep if "/test/" in k)}
    (C / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("lines per view:", res["views"])
    for name, r in res["systems"].items():
        cells = []
        for v in views:
            cells.append(" ".join(f"{r[f'{v}/{t}']['cer']:6.2f}[{r[f'{v}/{t}']['ci95'][0]:.1f}-{r[f'{v}/{t}']['ci95'][1]:.1f}]" if f"{v}/{t}" in r else "    –          "
                                  for t in SETS))
        print(f"{name[:52]:52s} | " + " | ".join(cells), r.get("missing_recut", ""))
    for name, r in res["panel"].items():
        print("panel", f"{name[:50]:50s}", " ".join(f"{k}={x['cer']:.2f}[{x['ci95'][0]:.1f}-{x['ci95'][1]:.1f}]" for k, x in r.items()))
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
