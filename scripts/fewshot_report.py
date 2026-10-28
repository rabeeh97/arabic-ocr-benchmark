"""Summarise the few-shot manuscript adaptation runs (rasam_fewshot_adapt.py on the 5 leave-manuscript-out folds) -> results/rasam/fewshot_report.json.

CER is micro-averaged over all 15 held-out RASAM-2 manuscripts' official test lines (character-weighted), per K, averaged over seeds.
K = -1 is the whole adaptation pool of that manuscript (one seed).
"""
import json
from collections import defaultdict
from pathlib import Path

R = Path("results/rasam/fewshot")
rows = []
for i in range(1, 6):
    rows += json.loads((R / f"fewshot_lomo{i}.json").read_text())["results"]




def chars(r):
    return sum(len(x["ref"]) for x in r["lines"])


agg = defaultdict(lambda: defaultdict(list))  # K -> ms -> [(greedy, lm)]
for r in rows:
    agg[r["k"]][r["manuscript"]].append((r["greedy"] / 100, r["lm"] / 100, chars(r)))
out = {}
for k in sorted(agg, key=lambda x: (x < 0, x)):
    g = l = w = 0.0
    per = {}
    for ms, v in agg[k].items():
        mg = sum(x[0] for x in v) / len(v); ml = sum(x[1] for x in v) / len(v); n = v[0][2] or 1
        per[ms] = round(ml * 100, 2)
        g += mg * n; l += ml * n; w += n
    out[str(k)] = {"greedy": round(g / w * 100, 2), "lm": round(l / w * 100, 2), "n_ms": len(agg[k]), "per_ms_lm": per}
    print(f"K={k:>4}  greedy {g / w * 100:6.2f}  LM {l / w * 100:6.2f}  ({len(agg[k])} manuscripts)")

# Without MS 1936 (about 50 % CER for every model; design §18 reports LOMO without it): 9.87 % unseen vs 6.69 % in-split.
lomo = json.loads(Path("results/rasam/lomo_report.json").read_text())["per_manuscript"]
ref = {r["manuscript"]: chars(r) for r in rows if r["k"] == 0}
keep = [ms for ms in ref if ms != "BULAC_MS_ARA_1936"]
W = sum(ref[m] for m in keep)
ex = {k: round(sum(out[k]["per_ms_lm"][m] * ref[m] for m in keep) / W, 2) for k in out}
ex["insplit"] = round(sum(lomo[m]["test_insplit_lm_mean"] * ref[m] for m in keep) / W, 2)
print("without MS 1936 (LM):", ex)
out["without_1936_lm"] = ex
Path("results/rasam/fewshot_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
