"""Check that the research map places every idea, task and experiment from every source (nothing silently dropped).

    .geomvenv/bin/python scripts/research_map_coverage.py [--map docs/research/research_map.json]

Sources: the ledger's EXP and QUEUE arrays, ideas log rows, idea-pool rows, augmentation catalogue rows and paper records
(next tests); the main checkout's NEXT_STEPS.md task IDs; the 2026-10-13 plan's detector IDs; and the chat items listed
below. An ID counts as placed when some map item names it in `id`, `covers`, `src` or `title`. Prints what is missing.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

W = Path(__file__).resolve().parent.parent
M = Path("/Volumes/PortableSSD/repos/arabic_ocr")
CHAT = ["makharij", "PAW-gap-width", "novel-clusters", "shape-nakba", "E6", "lm-decoding", "nakba-indomain", "vlm-geometry", "R9", "cluster-ci",
        "R4.1", "X-G8", "D10b", "C5", "std-split", "hard-example-control", "box-cleanup"]


def sources() -> dict[str, list[str]]:
    h = (W / "docs/research/paper-ledger.html").read_text(encoding="utf-8")
    out = {}

    def arr(name):
        i = h.index(f"const {name} = ["); j = h.index("\n];", i)
        return h[i:j]
    out["EXP"] = re.findall(r'\n\s*\["([^"]+)",', arr("EXP"))
    out["QUEUE"] = [x for row in re.findall(r'"(?:cpu|gpu)", "([^"]+)"\]', arr("QUEUE")) for x in re.split(r"\s*·\s*", row) if x != "new"]
    out["AUG"] = [f"aug-{n}" for n in re.findall(r'\n\s*\["(\d+)",', arr("AUG"))]
    i = h.find("const P = ["); j = h.find("\n];", i)
    out["PAPERS"] = re.findall(r'key:"([^"]+)"', h[i:j])
    out["IDEAS"] = [f"ideas #{n}" for n in range(1, 16)]
    out["POOL"] = [f"pool #{n}" for n in range(1, 15)]
    ns = (M / "docs/research/NEXT_STEPS.md").read_text(encoding="utf-8")
    out["NEXT_STEPS"] = sorted(set(re.findall(r"\*\*((?:R\d+(?:\.\d+)?|L\d|E\d′?|C\d|X\d|S\d|P-L\d|A1|E1\.\d))\b", ns)) | set(re.findall(r"- \[[ x]\] (R\d\.\d|L\d)\b", ns)))
    out["DETECTORS"] = [f"D{n}" for n in range(1, 25)]
    out["CHAT"] = CHAT
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=W / "docs/research/research_map.json")
    a = ap.parse_args()
    R = json.loads(a.map.read_text(encoding="utf-8"))
    hay = []
    for it in R["items"]:
        hay.append(" | ".join([it["id"], " ".join(it.get("covers", [])), it.get("src", ""), it["title"]]))
    blob = "\n".join(hay)

    def placed(x: str) -> bool:
        return re.search(r"(?<![\w.-])" + re.escape(x) + r"(?![\w.′-]|\.\d)", blob) is not None

    total = missing = 0
    for src, ids in sources().items():
        miss = [x for x in ids if not placed(x)]
        total += len(ids)
        missing += len(miss)
        print(f"{src:11s} {len(ids):3d} ids, {len(miss):3d} not placed" + (": " + ", ".join(miss) if miss else ""))
    print(f"TOTAL {total} ids, {missing} not placed")


if __name__ == "__main__":
    main()
