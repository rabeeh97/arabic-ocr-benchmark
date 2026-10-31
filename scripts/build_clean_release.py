"""Build release/rasam-clean/: the corrected RASAM-1 / RASAM-2 / TariMa line lists (automatic, no hand labelling of our own).

    PYTHONPATH=src:scripts python scripts/build_clean_release.py

Writes
  manifest.csv   one row per line we act on: HF config/split/index, manuscript, page, action (recut | drop | recut+drop), reasons, reference
  recut/<config>/<split>/<index>.png   the 250 crops re-cut from the page images (rasam_recut_broken.py, crop check v2)
  drop.json      the 202 lines removed from the clean view, keyed <config>/<split>/<index>
  score.py       standalone scorer: official vs clean CER from a predictions file keyed by HF index (copied from scripts/release_score.py)
  README.md      what was changed and why, and how to use it
"""
from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path

REL = Path("release/rasam-clean")
PACK = "/Volumes/PortableSSD/models/rasam-replication/crnn-data/rasam_tm_h96_rep2.pkl"
RULES = {
    "external": "marked invalid (corrupted image or wrong/missing label) in the manual check of calfa-co/rasam-dataset PR #7 (RASAM-1 only)",
    "garbled": "reference far less probable under a character LM than what two cross-fitted readers read (reference text does not match the image)",
    "garbled_page": "on a page whose references are, as a whole, much less probable than the readings (MS 1977 pages 0017, 0071, 0158)",
    "blank": "crop contains no ink, or reference is empty",
    "shifted": "reference belongs to a neighbouring line (another line of the page reads it)",
    "vertical": "crop is a vertical / marginal strip that cannot be read as the reference",
    "truncated": "crop is cut short; the page image does not hold the full reference either",
    "doubled": "crop holds two lines",
}


def main() -> None:
    REL.mkdir(parents=True, exist_ok=True)
    rows = {r["key"]: r for r in pickle.load(open(PACK, "rb"))["rows"]}
    reps = pickle.load(open("results/rasam/repairs/rasam_recut_v2.pkl", "rb"))
    reps = {f"{n}/{s}/{i:06d}": v for (n, s, i), v in reps.items()}
    drop = json.loads(Path("results/rasam/clean/drop.json").read_text())
    reasons = json.loads(Path("results/rasam/clean/clean_report.json").read_text())["reasons"]
    out = []
    for k in sorted(set(reps) | set(drop)):
        cfg, split, idx = k.split("/")
        r = rows[k]
        act = "+".join(a for a, on in (("recut", k in reps), ("drop", k in drop)) if on)
        if k in reps:
            p = REL / "recut" / cfg / split / f"{idx}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(reps[k]["bytes"])
        out.append({"config": cfg, "split": split, "index": int(idx), "manuscript": r["manuscript"], "page": r["page"], "action": act,
                    "reasons": ";".join(reasons.get(k, [])), "xml_line": reps.get(k, {}).get("xml_line", ""),
                    "recut_png": f"recut/{cfg}/{split}/{idx}.png" if k in reps else "", "reference": r["label"]})
    with open(REL / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (REL / "drop.json").write_text(json.dumps(sorted(drop), indent=0))
    shutil.copy("scripts/release_score.py", REL / "score.py")
    # counts for the README
    from collections import Counter
    c = Counter((o["split"], o["action"]) for o in out)
    by_rule = Counter(x for k in drop for x in reasons.get(k, []))
    test = Counter((o["config"], o["action"]) for o in out if o["split"] == "test")
    print("rows", len(out), dict(c))
    print("rules", dict(by_rule))
    print("test", dict(test))
    (REL / "counts.json").write_text(json.dumps({"by_split_action": {f"{a}|{b}": n for (a, b), n in c.items()},
                                                 "drop_by_rule": dict(by_rule), "test_by_config_action": {f"{a}|{b}": n for (a, b), n in test.items()},
                                                 "rules": RULES}, indent=1))


if __name__ == "__main__":
    main()
