"""Pack held-out lines from datasets the aligner never saw (NAKBA, KHATT) for a generalisation check.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/xai_prepare_eval_data.py OUT.pkl [--n 600]

Same preprocessing and label cleaning as xai_prepare_aligner_data.py. Rows get role "eval".
"""
from __future__ import annotations

import argparse
import glob
import pickle
import random
from pathlib import Path

from datasets import Dataset

from xai_prepare_aligner_data import clean, prep  # noqa: E402  (run from scripts/ or with PYTHONPATH=scripts)

HF = Path.home() / ".cache/huggingface/datasets"
SOURCES = [  # (folder, file glob, text column, tag)
    ("U4RASD___omar-al-saleh-manuscripts-segments", "*-test.arrow", None, "nakba_lines"),
    ("U4RASD___omar-al-saleh-manuscripts-segments", "*-blind_test.arrow", None, "nakba_blind"),
    ("ahmedheakl___arocrbench_khatt", "*.arrow", "text", "kitab_khatt"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    rng = random.Random(0)
    rows = []
    for folder, pat, col, tag in SOURCES:
        (path,) = sorted(glob.glob(str(HF / folder / "*" / "*" / "*" / pat)))[:1]
        ds = Dataset.from_file(path)
        if col is None:  # find the text column
            col = next(c for c in ("text", "transcription", "answer", "label") if c in ds.features)
        print(tag, len(ds), list(ds.features), "text column:", col, flush=True)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        kept = 0
        for i in idx:
            row = ds[i]
            label = clean(row[col] or "")
            if not 2 <= len(label) <= 120:
                continue
            arr = prep(row["image"])
            if arr.shape[1] // 4 < len(label) + 4:
                continue
            rows.append({"id": f"{tag}/{i:06d}", "dataset": tag, "role": "eval", "label": label, "img": arr})
            kept += 1
            if kept >= a.n:
                break
        print(" ", tag, "kept", kept, flush=True)
    pickle.dump({"H": 64, "rows": rows}, open(a.out, "wb"), protocol=4)
    print(len(rows), "rows ->", a.out, f"{a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
