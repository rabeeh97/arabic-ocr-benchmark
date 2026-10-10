"""Export a small, letter-targeted panel of line images for the attribution spike.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/xai_export_lines.py OUT_DIR [--n 10]

Picks short lines (14-40 chars) per dataset that contain the confusable letters
we want to look at (alef/lam, and the jim/ha/kha family), writes `OUT_DIR/<i>.png`
and `OUT_DIR/manifest.json` ({id, dataset, gt, file}). Reads the cached Arrow files directly
(offline; the datasets cache-hash lookup fails for RASAM-1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import glob

from datasets import Dataset

# name -> (hf_id, split)
HF = Path.home() / ".cache/huggingface/datasets"
DATASETS = {"rasam1": ("calfa-ai___rasam-1", "rasam-1-test.arrow", "test"),
            "muharaf_lines": ("aamijar___muharaf-public", "muharaf-public-test.arrow", "test")}
ALEF_LAM = set("اآأإل")
JHKH = set("جحخ")
TEXT_KEYS = ("text", "answer", "transcription", "label", "gt", "ground_truth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=10, help="lines per dataset")
    ap.add_argument("--scan", type=int, default=600)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (folder, arrow, split) in DATASETS.items():
        (path,) = glob.glob(str(HF / folder / "*" / "*" / "*" / arrow))
        ds = Dataset.from_file(path)
        got = 0
        for i in range(min(a.scan, len(ds))):
            row = ds[i]
            gt = next((row[k] for k in TEXT_KEYS if isinstance(row.get(k), str)), "").strip()
            if not 14 <= len(gt) <= 40 or "\n" in gt:
                continue
            if not (ALEF_LAM & set(gt) and JHKH & set(gt)):
                continue
            fn = f"{len(manifest):03d}.png"
            row["image"].convert("RGB").save(a.out / fn)
            manifest.append({"id": f"{name}/{split}/{i:06d}", "dataset": name, "gt": gt, "file": fn})
            got += 1
            if got >= a.n:
                break
        print(name, got)
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
