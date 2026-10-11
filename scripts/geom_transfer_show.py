"""Print the per-feature result of a geom_transfer run."""
import json
import sys

r = json.load(open(sys.argv[1] + "/transfer.json"))
for p, v in r["pairs"].items():
    print(p, {k: round(x, 2) for k, x in v["params"].items()})
    for k in v["z_before"]:
        print(f"  {k:18} z {v['z_before'][k]:6.2f} -> {v['z_after'][k]:6.2f}   med src {v['median_before'][k]:7.2f} out {v['median_after'][k]:7.2f} tgt {v['median_target'][k]:7.2f}")
