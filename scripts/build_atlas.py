"""Build the Arabic Script Atlas page (docs/research/atlas/index.html).

Reads the newest results/characterize/<stamp>/summary.json, the newest
error decomposition (results/characterize/<stamp>/errors.json, from
scripts/decompose_errors.py) and the research tracker docs/research/progress.json, and inlines both into
docs/research/atlas/template.html. Re-run after every profiler run or tracker
edit; the page is then republished to the same artifact URL.

    .venv/bin/python scripts/build_atlas.py [--summary path/to/summary.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_ocr_bench.characterize.text import joining_type

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/research/atlas/template.html"
OUT = ROOT / "docs/research/atlas/index.html"
PROGRESS = ROOT / "docs/research/progress.json"

# dotless display glyphs for rasm classes that are written with a dot
_DOTLESS = {"ب": "ٮ", "ف": "ڡ"}


def latest_summary() -> Path:
    runs = sorted((ROOT / "results/characterize").glob("*/summary.json"))
    if not runs:
        raise SystemExit("no summary.json yet -- run scripts/profile_datasets.py first")
    return runs[-1]


def latest_errors() -> dict | None:
    runs = sorted((ROOT / "results/characterize").glob("*/errors.json"))
    if not runs:
        return None
    doc = json.loads(runs[-1].read_text(encoding="utf-8"))
    keep = ("model", "dataset", "n_scored", "n_error", "fine_tuned", "in_train", "small")
    doc["pairs"] = [
        {**{k: p[k] for k in keep},
         "cer": {k: round(v, 4) for k, v in p["ladder"]["cer"].items()},
         "tax": p["taxonomy_share"],
         "ins100": round(p["diacritics"]["inserted_per_100"], 2),
         "drop": round(p["diacritics"]["dropped_share"], 3),
         "gt_marks": p["diacritics"].get("gt_marks", 0),
         "paw": p["word_internal_spaces"]}
        for p in doc["pairs"] if "ladder" in p
    ]
    doc["path"] = str(runs[-1].relative_to(ROOT))
    return doc


def enrich(summary: dict) -> dict:
    for ds in summary["datasets"].values():
        for ex in ds["examples"]:
            for li in ex["letters"]:
                li["jt"] = joining_type(li["ch"])
                li["rd"] = _DOTLESS.get(li["rasm"], li["rasm"]) if li["rasm"] else li["ch"]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()
    path = args.summary or latest_summary()
    summary = enrich(json.loads(path.read_text(encoding="utf-8")))
    summary["summary_path"] = str(path.relative_to(ROOT))
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    payload = json.dumps({"summary": summary, "progress": progress, "errors": latest_errors()},
                         ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # keep the inline <script> intact
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__ATLAS_DATA__*/null", payload)
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}  ({len(html) / 1024:.0f} KB, from {summary['summary_path']})")


if __name__ == "__main__":
    main()
