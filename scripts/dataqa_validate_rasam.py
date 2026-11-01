"""Compare data-QA flags on RASAM-1/2 and TariMa test with the earlier RASAM
clean lists (results/rasam/clean/clean_report.json reasons, re-cut list) and the
by-eye examples (results/rasam/push/data_issues_curated.json).

  PYTHONPATH=src python scripts/dataqa_validate_rasam.py
Writes results/dataqa/rasam_validation.json and prints a summary.
"""

import csv
import json
from collections import Counter, defaultdict

rep = json.load(open("results/rasam/clean/clean_report.json"))["reasons"]
recut = set()
for r in csv.DictReader(open("release/rasam-clean/manifest.csv")):
    if "recut" in r["action"]:
        recut.add(f"{r['config']}/{r['split']}/{int(r['index']):06d}")
cur = json.load(open("results/rasam/push/data_issues_curated.json"))["categories"]

flags = {}
n = 0
for cfg in ("RASAM-1", "RASAM-2", "tarima"):
    d = json.load(open(f"results/dataqa/{cfg}_test.json"))
    n += d["n"]
    flags.update({k: {f["check"] for f in v} for k, v in d["flags"].items()})

known = defaultdict(set)  # id -> earlier reasons
for k, rs in rep.items():
    if "/test/" in k:
        known[k] |= set(rs)
for k in recut:
    if "/test/" in k:
        known[k].add("recut")

INFO = {"neighbour_clip"}
strong = {k: v - INFO for k, v in flags.items() if v - INFO}
by_reason = {}
for r in sorted({x for v in known.values() for x in v}):
    ids = [k for k, v in known.items() if r in v]
    hit = [k for k in ids if k in strong]
    by_reason[r] = {"lines": len(ids), "flagged": len(hit),
                    "by_check": dict(Counter(c for k in hit for c in strong[k]).most_common())}
by_check = {}
for c in sorted({x for v in strong.values() for x in v}):
    ids = [k for k, v in strong.items() if c in v]
    by_check[c] = {"lines": len(ids), "in_earlier_lists": sum(k in known for k in ids)}
curated = []
for c in cur:
    for e in c["examples"]:
        curated.append({"kind": c["key"], "id": e["id"], "flags": sorted(flags.get(e["id"], []))})
out = {"n_test_lines": n, "earlier_flagged": len(known), "qa_flagged": len(strong),
       "both": len(set(known) & set(strong)), "by_earlier_reason": by_reason, "by_check": by_check,
       "curated": curated,
       "curated_caught": sum(bool(set(x["flags"]) - INFO) for x in curated), "curated_n": len(curated)}
json.dump(out, open("results/dataqa/rasam_validation.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "curated"}, ensure_ascii=False, indent=1))
for x in curated:
    print(x["kind"], x["id"], x["flags"])
