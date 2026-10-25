"""Score existing predictions under the data adjustments other papers made, on the official RASAM test sets (no retraining).

    PYTHONPATH=src python scripts/rasam_protocols.py

Protocols (applied to the same predictions):
  official            all lines, raw labels
  standardised labels Lucas et al. 2022 standardised the character set (54 -> 38 classes); approximated by our diac_insensitive profile
                      (diacritics and tatweel removed, alef / hamza / yaa / taa-marbuta variants and digits folded)
  no coloured lines   CHR 2024 reports MS 1947 without its red/blue lines; here: lines whose red/blue-ink share is in the top 10 % are removed
  repaired crops      RASAM-2 only: the 7 degenerate MS 1936 test crops re-cut from the page images (rasam_recut_broken.py)
  main text only      catchwords and marginal notes removed
  all adjustments     standardised labels + no coloured lines + repaired + main text only
Line features come from results/rasam/hard_lines/{r1,r2}_features.json (same official test order).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

SYSTEMS = {  # name: (RASAM-1 file, RASAM-2 official file, RASAM-2 repaired file)
    "96 px + TariMa + LM (best)": ("results/rasam/lm/TM_h96_lm_r1__r1__test.json", "results/rasam/lm/TM_h96_lm_r2__r2__test.json", "results/rasam/lm/TM_h96_lm_repairedtest__r2__test.json"),
    "96 px + LM": ("results/rasam/lm/R2_h96_lm_r1__r1__test.json", "results/rasam/lm/R2_h96_lm_r2__r2__test.json", "results/rasam/lm/R2_h96_lm_repairedtest__r2__test.json"),
    "96 px greedy": ("results/rasam/crnn/R2_h96_w05_s0_best__r1__test.json", "results/rasam/crnn/R2_h96_w05_s0_best__r2__test.json", "results/rasam/crnn/R2_h96_repairedtest__r2__test.json"),
}


def cer(pairs, profile: str) -> float:
    e = n = 0
    for ref, hyp in pairs:
        c = char_edit_counts(normalize(ref, profile).strip(), normalize(hyp, profile).strip())
        e += c["sub"] + c["del"] + c["ins"]
        n += c["ref_len"]
    return e / max(1, n) * 100


def main() -> None:
    feats = {t: json.loads(Path(f"results/rasam/hard_lines/{t}_features.json").read_text()) for t in ("r1", "r2")}
    out = {}
    for name, (f1, f2, f2r) in SYSTEMS.items():
        res = {}
        for tag, f, frep in (("RASAM-1", f1, None), ("RASAM-2", f2, f2r)):
            fe = feats["r1" if tag == "RASAM-1" else "r2"]
            lines = json.loads(Path(f).read_text())["lines"]
            rep = json.loads(Path(frep).read_text())["lines"] if frep and Path(frep).exists() else None
            col_cut = np.quantile([x["colored_frac"] for x in fe], 0.9)
            keep_col = [x["colored_frac"] <= col_cut for x in fe]
            keep_main = [x["region"] in ("MainText", "MainZone") for x in fe]
            pairs = [(l["ref"], l["hyp"]) for l in lines]
            pr = [(l["ref"], l["hyp"]) for l in rep] if rep else pairs
            row = {
                "official": cer(pairs, "raw"),
                "standardised labels": cer(pairs, "diac_insensitive"),
                "no coloured lines": cer([p for p, k in zip(pairs, keep_col) if k], "raw"),
                "main text only": cer([p for p, k in zip(pairs, keep_main) if k], "raw"),
            }
            if rep:
                row["repaired crops"] = cer(pr, "raw")
            row["all adjustments"] = cer([p for p, kc, km in zip(pr, keep_col, keep_main) if kc and km], "diac_insensitive")
            row["lines kept (all adjustments)"] = sum(1 for kc, km in zip(keep_col, keep_main) if kc and km)
            res[tag] = {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}
        out[name] = res
        for tag, row in res.items():
            print(f"{name:28s} {tag}: " + " | ".join(f"{k} {v}" for k, v in row.items()))
    Path("results/rasam/protocols.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
