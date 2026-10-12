"""Shape-targeted oversampling: baseline, targeted and random-control CRNNs on KHATT half B (never used for the selection).

    .geomvenv/bin/python scripts/shape_analyze.py results/geometry/shape OUT.json

Needs base.json, targeted.json, random.json (geom_transfer_eval.py on eval_data.pkl, with per-line readings), halves.json and the
local inkcrop_eval.pkl for the ground truth. Reports CER on half B with paired bootstrap intervals (targeted against random, and
each against the baseline) and the letter error rate per character, so the letters KHATT finds unusual can be followed.
"""
import difflib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import edit_distance  # noqa: E402

d = Path(sys.argv[1])
H = json.loads((d / "halves.json").read_text())
gt = {r["id"]: r["label"] for r in pickle.load(open("results/geometry/inkcrop_eval.pkl", "rb"))["rows"] if r["dataset"] == "kitab_khatt"}
FILES = {"base": "base.json", "targeted": "targeted_eval.json", "random": "random_eval.json"}
M = {m: json.loads((d / f).read_text(encoding="utf-8"))["per_line"]["kitab_khatt"] for m, f in FILES.items()}
rng = np.random.default_rng(0)
FOCUS = list("حخشطجظىغعهكم")


def letter_errors(ids, m):
    tot, bad = Counter(), Counter()
    for i in ids:
        lab, hyp = gt[i], M[m][i][2]
        ok = np.zeros(len(lab), bool)
        for op, i1, i2, _, _ in difflib.SequenceMatcher(None, lab, hyp, autojunk=False).get_opcodes():
            if op == "equal":
                ok[i1:i2] = True
        for k, ch in enumerate(lab):
            tot[ch] += 1
            bad[ch] += int(not ok[k])
    return tot, bad


out = {}
for half in ("half_a", "half_b"):
    ids = [i for i in H[half] if i in gt and all(i in M[m] for m in M)]
    e = {m: np.array([edit_distance(list(M[m][i][2]), list(gt[i])) for i in ids], float) for m in M}
    n = np.array([len(gt[i]) for i in ids], float)
    res = {"lines": len(ids), "cer": {m: float(e[m].sum() / n.sum()) for m in M}, "paired": {}}
    for a_, b_ in (("base", "targeted"), ("base", "random"), ("random", "targeted")):
        bs = []
        for _ in range(4000):
            s = rng.integers(0, len(ids), len(ids))
            bs.append((e[b_][s].sum() - e[a_][s].sum()) / n[s].sum())
        res["paired"][f"{b_} - {a_}"] = {"change": float((e[b_].sum() - e[a_].sum()) / n.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    per = {}
    for m in M:
        tot, bad = letter_errors(ids, m)
        per[m] = {ch: [bad[ch], tot[ch]] for ch in FOCUS if tot[ch]}
        allL = [c for c in tot if "ء" <= c <= "ي"]
        per[m]["all letters"] = [sum(bad[c] for c in allL), sum(tot[c] for c in allL)]
        per[m]["focus letters"] = [sum(bad[c] for c in FOCUS), sum(tot[c] for c in FOCUS)]
    res["letter_errors"] = per
    out[half] = res
    print(half, json.dumps({k: v for k, v in res.items() if k != "letter_errors"}))
    print("  focus letters", {m: round(per[m]["focus letters"][0] / per[m]["focus letters"][1], 3) for m in M}, " all letters", {m: round(per[m]["all letters"][0] / per[m]["all letters"][1], 3) for m in M})
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
