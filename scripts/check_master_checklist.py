"""Prove that MASTER-CHECKLIST.md lists every unfinished item of the research map (and every ID such an item groups).

    .geomvenv/bin/python scripts/check_master_checklist.py

Reads docs/research/research_map.json and docs/research/MASTER-CHECKLIST.md. An item counts as listed when its map ID appears
in a `map:` tag of the checklist. Exits non-zero when anything is missing, so it can gate a commit.
"""
import json
import re
import sys
from pathlib import Path

W = Path(__file__).resolve().parent.parent
R = json.loads((W / "docs/research/research_map.json").read_text(encoding="utf-8"))
md = (W / "docs/research/MASTER-CHECKLIST.md").read_text(encoding="utf-8")
tags = set()
for m in re.finditer(r"map: ((?:`[^`]+`(?:, )?)+)", md):
    tags |= set(re.findall(r"`([^`]+)`", m.group(1)))
open_items = [it for it in R["items"] if it["status"] != "done"]
missing = [it["id"] for it in open_items if it["id"] not in tags]
print(f"unfinished map items: {len(open_items)}; listed in the checklist: {len(open_items) - len(missing)}; missing: {len(missing)}")
if missing:
    print("missing:", ", ".join(missing))
unknown = sorted(t for t in tags if t not in {it["id"] for it in R["items"]} and not t.startswith("aug-"))
if unknown:
    print("tags not in the map (check spelling):", ", ".join(unknown))
boxes = len(re.findall(r"^\s*- \[ \]", md, flags=re.M))
print(f"checkboxes: {boxes}")
sys.exit(1 if missing else 0)
