"""Export an error-focused panel (worst lines + easy controls) for the attribution spike.

    .venv/bin/python scripts/xai_export_error_lines.py SCORES PREDS OUT_DIR [--worst 15] [--control 8]

SCORES/PREDS are the scores.jsonl / predictions.jsonl of one eval run (Muharaf test).
"Worst" = highest CER among lines whose output length is within 0.5-1.5x the GT length
and whose GT is 15-45 chars, so we look at *misreadings*, not runaway or truncated
generations. Controls = lowest-CER lines of the same length range. Writes
OUT_DIR/<i>.png + manifest.json ({id, dataset, gt, stored_hyp, cer, group, file}).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from datasets import Dataset

HF = Path.home() / ".cache/huggingface/datasets"
ARROW = {"muharaf_lines": ("aamijar___muharaf-public", "muharaf-public-test.arrow")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", type=Path)
    ap.add_argument("preds", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--worst", type=int, default=15)
    ap.add_argument("--control", type=int, default=8)
    a = ap.parse_args()
    cer = {}
    for l in a.scores.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        cer[r["sample_id"]] = r["cer_raw"]
    rows = []
    for l in a.preds.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        gt, hyp = r["ground_truth"].strip(), (r.get("raw_text") or "").strip()
        if not 15 <= len(gt) <= 45 or "\n" in gt or not 0.5 <= len(hyp) / len(gt) <= 1.5:
            continue
        rows.append((cer[r["sample_id"]], r["sample_id"], r["dataset"], gt, hyp))
    rows.sort()
    pick = [(*r, "worst") for r in rows[::-1][: a.worst]] + [(*r, "control") for r in rows[: a.control]]
    a.out.mkdir(parents=True, exist_ok=True)
    arrows: dict[str, Dataset] = {}
    manifest = []
    for c, sid, ds, gt, hyp, group in pick:
        folder, arrow = ARROW[ds]
        if ds not in arrows:
            (path,) = glob.glob(str(HF / folder / "*" / "*" / "*" / arrow))
            arrows[ds] = Dataset.from_file(path)
        fn = f"{len(manifest):03d}.png"
        arrows[ds][int(sid.rsplit("/", 1)[-1])]["image"].convert("RGB").save(a.out / fn)
        manifest.append({"id": sid, "dataset": ds, "gt": gt, "stored_hyp": hyp, "cer": c, "group": group, "file": fn})
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(rows), "eligible;", len(manifest), "exported")


if __name__ == "__main__":
    main()
