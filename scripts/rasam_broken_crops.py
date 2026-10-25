"""Count degenerate line crops (almost no height) in every RASAM split and measure how much of our test error they cause.

    PYTHONPATH=src python scripts/rasam_broken_crops.py
Writes results/rasam/broken_crops.json.
"""
from __future__ import annotations

import glob
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
MIN_H = 8  # a text line needs more than a few pixels of height


def main() -> None:
    out = {}
    for ds, tag in (("RASAM-1", "r1"), ("RASAM-2", "r2"), ("tarima", "tm")):
        out[ds] = {}
        for split in ("train", "validation", "test"):
            hs, ms = [], Counter()
            for f in sorted(glob.glob(f"{ROOT}{ds}/snapshots/*/data/{split}-*.parquet")):
                for r in pq.read_table(f, columns=["image", "manuscript_name"]).to_pylist():
                    h = Image.open(io.BytesIO(r["image"]["bytes"])).size[1]
                    hs.append(h)
                    if h < MIN_H:
                        ms[r["manuscript_name"]] += 1
            bad = [i for i, h in enumerate(hs) if h < MIN_H]
            out[ds][split] = {"lines": len(hs), "broken": len(bad), "by_manuscript": dict(ms), "test_index": bad if split == "test" else None}
            print(f"{ds:8s} {split:10s} {len(hs):5d} lines, {len(bad):4d} crops under {MIN_H} px tall  {dict(ms)}")
    # error share of the broken test lines for our best systems
    for tag, ds, f in (("r1", "RASAM-1", "results/rasam/lm/R2_h96_lm_r1__r1__test.json"), ("r2", "RASAM-2", "results/rasam/lm/R2_h96_lm_r2__r2__test.json")):
        lines = json.load(open(f))["lines"]
        bad = set(out[ds]["test"]["test_index"])
        e_all = n_all = e_ok = n_ok = 0
        for i, l in enumerate(lines):
            c = char_edit_counts(normalize(l["ref"], "raw").strip(), normalize(l["hyp"], "raw").strip())
            e, n = c["sub"] + c["del"] + c["ins"], c["ref_len"]
            e_all += e
            n_all += n
            if i not in bad:
                e_ok += e
                n_ok += n
        print(f"{ds} test, best system: CER {e_all / n_all * 100:.2f} % on all {len(lines)} lines; {e_ok / n_ok * 100:.2f} % without the {len(bad)} broken crops "
              f"(they hold {(e_all - e_ok) / max(1, e_all):.0%} of the errors)")
        out[ds]["best_system_cer_without_broken"] = round(e_ok / n_ok * 100, 2)
    Path("results/rasam/broken_crops.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
