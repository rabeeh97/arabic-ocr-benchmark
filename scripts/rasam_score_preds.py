"""Score HAFITH's predictions and ours on the official RASAM test sets with the project metric, on all lines and on the CLEAN subset
(official test lines whose text does not occur in HAFITH's train or validation split, so HAFITH cannot have memorised them).

    PYTHONPATH=src python scripts/rasam_score_preds.py HAFITH_DIR [OUR_RESULT.json ...]

HAFITH_DIR holds preds_r1.json, preds_r2.json and hafith_lines_overlap.json (from colab_hafith_eval.py). Our result files are
results/rasam/{crnn,lm}/<name>__r1__test.json / __r2__test.json; their line order is the official test order (test lines are never dropped).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

BIDI = re.compile("[​-‏‪-‮⁦-⁩﻿؜]")


def norm(t: str) -> str:
    return " ".join(BIDI.sub("", unicodedata.normalize("NFC", t or "")).split())


def micro(pairs) -> tuple[float, int]:
    e = n = 0
    for ref, hyp in pairs:
        c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp, "raw").strip())
        e += c["sub"] + c["del"] + c["ins"]
        n += c["ref_len"]
    return e / max(1, n), len(pairs)


def main() -> None:
    hd = Path(sys.argv[1])
    ov = json.loads((hd / "hafith_lines_overlap.json").read_text())
    seen = set(ov.get("train", [])) | set(ov.get("val", []))
    out = {}
    for tag in ("r1", "r2"):
        p = hd / f"preds_{tag}.json"
        if not p.exists():
            continue
        preds = json.loads(p.read_text())
        clean_idx = {d["i"] for d in preds if norm(d["ref"]) not in seen}
        rows = {"HAFITH": [(d["ref"], d["hyp"]) for d in preds]}
        for f in sys.argv[2:]:
            j = json.loads(Path(f).read_text())
            if j.get("dataset") != tag:
                continue
            rows[Path(f).stem.replace("__test", "")] = [(l["ref"], l["hyp"]) for l in j["lines"]]
        res = {}
        for name, pairs in rows.items():
            if len(pairs) != len(preds):
                continue
            a, na = micro(pairs)
            c, nc = micro([pairs[i] for i in sorted(clean_idx)])
            res[name] = {"all": round(a * 100, 2), "n_all": na, "clean": round(c * 100, 2), "n_clean": nc}
            print(f"{tag} {name:40s} all {a * 100:6.2f} % (n={na})   clean (not in HAFITH train/val) {c * 100:6.2f} % (n={nc})")
        out[tag] = res
    Path("results/rasam/hafith_comparison.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
